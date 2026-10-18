"""
ProactiveWatcher — the assistant that speaks up only when it matters, under a strict budget.

It reads the briefing cache (never the network) on a timer and raises at most a handful of interruptions a day:

    meeting_30      a meeting starts in ~30 minutes
    meeting_5       a meeting starts in ~5 minutes
    priority_item   a new unread item scored above a high threshold (mail / Slack / Messages)
    stale_collector a source has not synced for over 2 hours

Every candidate passes the same gates, in this order, and a candidate that is held back is DROPPED, not queued — quiet
means quiet; the morning briefing is where backlog goes:

    dedupe    the same item/meeting-stage/source never fires twice (persisted)
    DND       `vesper dnd on` (a flag file) silences everything
    quiet     quiet hours (default 23:00-08:00 Asia/Kolkata)
    budget    at most `max_per_day` interruptions per local day
    cooldown  at most one interruption per kind-group per cooldown window

What an interruption may CONTAIN is fixed by `build_payload`: a source label, a sender display name (sanitised, and only
if it does not read like instructions to an AI) or the minutes until a meeting. Never a subject, snippet, message text or
(by default) a meeting title — all of that is third-party text.

Delivery: HUD (an event on the bus) and a macOS notification; spoken only when voice mode is on. Nothing here sends,
posts or writes to any external account. Default OFF (`proactive.watcher.enabled`), docs/PROACTIVE.md.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
from zoneinfo import ZoneInfo

from briefing.cache import BriefingCache
from briefing.config import BriefingConfig
from briefing.sanitize import clean_text, display_name, looks_like_instructions
from utils.logger import get_logger

logger = get_logger(__name__)

SOURCE_LABELS = {"gmail": "Email", "slack": "Slack", "messages": "Messages", "calendar": "Calendar"}
KIND_GROUP = {"meeting_30": "meeting", "meeting_5": "meeting", "priority_item": "priority_item",
              "stale_collector": "stale_collector"}
#: Order candidates are considered in: time-critical first.
KIND_ORDER = ("meeting_5", "meeting_30", "priority_item", "stale_collector")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEN_TTL_HOURS = 36
DEFAULT_STATE = "data/proactive_state.json"
DEFAULT_DND = "data/proactive_dnd.json"


# ----------------------------------------------------------------------------- config

@dataclass
class WatcherConfig:
    enabled: bool = False
    tick_seconds: float = 30.0
    max_per_day: int = 6
    quiet_start: str = "23:00"
    quiet_end: str = "08:00"
    timezone: str = "Asia/Kolkata"
    meeting_30: bool = True
    meeting_5: bool = True
    priority_enabled: bool = True
    priority_threshold: float = 60.0
    priority_max_age_minutes: float = 30.0
    stale_enabled: bool = True
    stale_after_minutes: float = 120.0
    cooldown_minutes: Dict[str, float] = field(default_factory=lambda: {
        "meeting": 2.0, "priority_item": 15.0, "stale_collector": 360.0})
    hud: bool = True
    macos_notification: bool = True
    speak: bool = True
    include_meeting_title: bool = False
    state_path: str = str(PROJECT_ROOT / DEFAULT_STATE)
    dnd_path: str = str(PROJECT_ROOT / DEFAULT_DND)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "WatcherConfig":
        d = data or {}
        cfg = cls()
        for key in ("enabled", "include_meeting_title"):
            if key in d:
                setattr(cfg, key, bool(d[key]))
        if "tick_seconds" in d:
            cfg.tick_seconds = max(1.0, float(d["tick_seconds"]))
        if "max_per_day" in d:
            cfg.max_per_day = max(0, int(d["max_per_day"]))
        q = d.get("quiet_hours") or {}
        cfg.quiet_start = str(q.get("start", cfg.quiet_start))
        cfg.quiet_end = str(q.get("end", cfg.quiet_end))
        cfg.timezone = str(q.get("timezone", cfg.timezone))
        ev = d.get("events") or {}
        cfg.meeting_30 = bool((ev.get("meeting_30min", {}) if isinstance(ev.get("meeting_30min"), dict) else {"enabled": ev.get("meeting_30min", True)}).get("enabled", True))
        cfg.meeting_5 = bool((ev.get("meeting_5min", {}) if isinstance(ev.get("meeting_5min"), dict) else {"enabled": ev.get("meeting_5min", True)}).get("enabled", True))
        pr = ev.get("priority_item") or {}
        cfg.priority_enabled = bool(pr.get("enabled", True))
        cfg.priority_threshold = float(pr.get("threshold", cfg.priority_threshold))
        cfg.priority_max_age_minutes = float(pr.get("max_age_minutes", cfg.priority_max_age_minutes))
        st = ev.get("stale_collector") or {}
        cfg.stale_enabled = bool(st.get("enabled", True))
        cfg.stale_after_minutes = float(st.get("after_minutes", cfg.stale_after_minutes))
        for k, v in (d.get("cooldown_minutes") or {}).items():
            cfg.cooldown_minutes[str(k)] = max(0.0, float(v))
        dl = d.get("delivery") or {}
        cfg.hud = bool(dl.get("hud", True))
        cfg.macos_notification = bool(dl.get("macos_notification", True))
        cfg.speak = bool(dl.get("speak", True))
        # relative paths mean "inside the project", so `vesper dnd` and the gateway agree whatever the cwd is
        cfg.state_path = str(_abs(d.get("state_path", cfg.state_path)))
        cfg.dnd_path = str(_abs(d.get("dnd_path", cfg.dnd_path)))
        ZoneInfo(cfg.timezone)               # fail loudly at load time, not at 3 a.m.
        _parse_hhmm(cfg.quiet_start)
        _parse_hhmm(cfg.quiet_end)
        return cfg


def _abs(path: object) -> Path:
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def _parse_hhmm(text: str) -> Tuple[int, int]:
    m = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", str(text).strip())
    if not m:
        raise ValueError(f"expected HH:MM, got {text!r}")
    return int(m.group(1)), int(m.group(2))


# ----------------------------------------------------------------------------- payloads

@dataclass(frozen=True)
class Payload:
    """Everything an interruption may say. There is no field for message text."""

    title: str          # "Vesper · Email"
    body: str           # "New message from Priya Sharma"
    speech: str         # the same thing, as a sentence
    kind: str
    source: str


_SAFE_NAME = re.compile(r"[^A-Za-z0-9 .'\-]")


def safe_name(raw: object, limit: int = 28) -> Optional[str]:
    """A sender display name fit for a notification: sanitised, short, no addresses or digits-heavy ids — or None when it
    reads like instructions to an AI (the interruption is then withheld, not shown with the name removed)."""
    if looks_like_instructions(raw):
        return None
    text = str(raw or "")
    # A From header ("Name <addr>") needs the header parser; a bare display name (Slack, Messages) must NOT go through
    # it — parseaddr would keep only the first word.
    name = display_name(text, limit) if ("<" in text or "@" in text) else clean_text(text, limit)
    name = _SAFE_NAME.sub("", name).strip()
    if sum(c.isdigit() for c in name) >= 5:
        return "an unknown number"
    return name or "someone"


def build_payload(kind: str, *, source: str = "", name: Optional[str] = None, minutes: Optional[int] = None,
                  address: str = "Sir") -> Payload:
    label = SOURCE_LABELS.get(source, "Vesper")
    if kind == "meeting_30" or kind == "meeting_5":
        n = int(minutes if minutes is not None else (30 if kind == "meeting_30" else 5))
        body = f"Meeting in {n} minutes"
        return Payload("Vesper · Calendar", body, f"{body}, {address}.", kind, "calendar")
    if kind == "priority_item":
        who = name or "someone"
        body = f"New message from {who}"
        return Payload(f"Vesper · {label}", body, f"{body} on {label}, {address}.", kind, source)
    if kind == "stale_collector":
        body = f"{label} hasn't synced for a while"
        return Payload("Vesper · Heads up", body, f"{body}, {address}.", kind, source)
    raise ValueError(f"unknown interruption kind {kind!r}")


@dataclass
class Candidate:
    kind: str
    key: str                   # dedupe key; contains ids, never text
    payload: Optional[Payload]
    rank: float = 0.0          # higher first within a kind
    withheld: Optional[str] = None      # reason this candidate may not be shown at all (e.g. "unsafe_sender")


# ----------------------------------------------------------------------------- state

class State:
    """Persisted counters. Plain JSON, no content: ids, kinds, timestamps and counts."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: Dict[str, Any] = {"day": "", "count": 0, "suppressed": {}, "delivered": {}, "last": {}, "seen": {}}
        try:
            loaded = json.loads(self.path.read_text())
            if isinstance(loaded, dict):
                self.data.update({k: v for k, v in loaded.items() if k in self.data})
        except (OSError, ValueError):
            pass

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".proactive-", suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(self.data, f)
            os.replace(tmp, self.path)
        except OSError as exc:
            logger.warning(f"[Proactive] could not save state: {exc}")

    def roll_day(self, day: str) -> None:
        if self.data.get("day") != day:
            self.data.update(day=day, count=0, suppressed={}, delivered={})

    def prune_seen(self, now_ts: float) -> None:
        cutoff = now_ts - SEEN_TTL_HOURS * 3600
        self.data["seen"] = {k: v for k, v in self.data["seen"].items() if v >= cutoff}


# ----------------------------------------------------------------------------- DND

def read_dnd(path: str, now: datetime) -> Dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {"on": False}
    if not isinstance(data, dict) or not data.get("on"):
        return {"on": False}
    until = data.get("until")
    if until:
        try:
            if datetime.fromisoformat(until) <= now:
                return {"on": False, "expired": True}
        except ValueError:
            pass
    return data


def write_dnd(path: str, on: bool, now: datetime, minutes: Optional[float] = None) -> Dict[str, Any]:
    p = Path(path)
    if not on:
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        return {"on": False}
    data: Dict[str, Any] = {"on": True, "since": now.isoformat()}
    if minutes:
        data["until"] = (now + timedelta(minutes=float(minutes))).isoformat()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    return data


# ----------------------------------------------------------------------------- delivery

class Notifier(Protocol):
    async def notify(self, title: str, body: str) -> None: ...


class MacNotifier:
    """A macOS banner via osascript. The script is constant; title and body travel as argv (never interpolated)."""

    SCRIPT = ("on run argv\n"
              "display notification (item 1 of argv) with title (item 2 of argv)\n"
              "end run")

    async def notify(self, title: str, body: str) -> None:
        import subprocess
        await asyncio.to_thread(
            subprocess.run, ["osascript", "-e", self.SCRIPT, body, title],
            capture_output=True, timeout=10, check=False, shell=False)


# ----------------------------------------------------------------------------- the watcher

@dataclass
class Delivery:
    kind: str
    key: str
    payload: Payload
    channels: List[str]


class ProactiveWatcher:
    def __init__(self, cache: BriefingCache, cfg: WatcherConfig, *, briefing_cfg: Optional[BriefingConfig] = None,
                 event_bus: Any = None, notifier: Optional[Notifier] = None,
                 clock: Optional[Callable[[], datetime]] = None, voice_mode: Callable[[], bool] = lambda: False,
                 address_as: str = "Sir", sleep: Callable[[float], Any] = asyncio.sleep):
        self._cache, self.cfg = cache, cfg
        self._bcfg = briefing_cfg or BriefingConfig()
        self._bus, self._notifier = event_bus, notifier
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._voice_mode, self._address, self._sleep = voice_mode, address_as, sleep
        self._tz = ZoneInfo(cfg.timezone)
        self._state = State(Path(cfg.state_path))
        self._task: Optional["asyncio.Task[None]"] = None
        self.delivered: List[Delivery] = []          # what this process delivered (tests, `status`)

    # ------------------------------------------------------------ time helpers
    def _now(self) -> datetime:
        n = self._clock()
        return n if n.tzinfo else n.replace(tzinfo=timezone.utc)

    def in_quiet_hours(self, now: Optional[datetime] = None) -> bool:
        local = (now or self._now()).astimezone(self._tz)
        s, e = _parse_hhmm(self.cfg.quiet_start), _parse_hhmm(self.cfg.quiet_end)
        t = (local.hour, local.minute)
        if s == e:
            return False
        return (t >= s or t < e) if s > e else (s <= t < e)

    # ------------------------------------------------------------ candidates
    def candidates(self, now: datetime) -> List[Candidate]:
        out: List[Candidate] = []
        out += self._meeting_candidates(now)
        if self.cfg.priority_enabled:
            out += self._priority_candidates(now)
        if self.cfg.stale_enabled:
            out += self._stale_candidates(now)
        order = {k: i for i, k in enumerate(KIND_ORDER)}
        return sorted(out, key=lambda c: (order[c.kind], -c.rank))

    def _meeting_candidates(self, now: datetime) -> List[Candidate]:
        out = []
        for it in self._cache.active_items("calendar"):
            if it.all_day or it.start_at is None:
                continue
            minutes = (it.start_at - now.timestamp()) / 60.0
            if self.cfg.meeting_5 and 0 < minutes <= 5:
                out.append(Candidate("meeting_5", f"{it.id}:5", build_payload("meeting_5", minutes=max(1, round(minutes)),
                                     address=self._address), rank=-minutes))
            elif self.cfg.meeting_30 and 25 <= minutes <= 30:
                out.append(Candidate("meeting_30", f"{it.id}:30", build_payload("meeting_30", minutes=round(minutes),
                                     address=self._address), rank=-minutes))
        return out

    def _priority_candidates(self, now: datetime) -> List[Candidate]:
        from briefing.builder import scoring_context
        from briefing.scorer import score_item

        ts = now.timestamp()
        ctx = scoring_context(self._cache, ts, self._bcfg)
        out = []
        for source in ("gmail", "slack", "messages"):
            for it in self._cache.active_items(source):
                if not it.unread or ts - it.timestamp > self.cfg.priority_max_age_minutes * 60 or it.timestamp > ts + 60:
                    continue
                scored = score_item(it, ctx, self._bcfg)
                if scored.excluded or scored.bulk or scored.score < self.cfg.priority_threshold:
                    continue
                if scored.suspicious:
                    out.append(Candidate("priority_item", f"item:{it.id}", None, withheld="unsafe_text"))
                    continue
                name = safe_name(it.sender)
                if name is None:
                    out.append(Candidate("priority_item", f"item:{it.id}", None, withheld="unsafe_sender"))
                    continue
                out.append(Candidate("priority_item", f"item:{it.id}",
                                     build_payload("priority_item", source=source, name=name, address=self._address),
                                     rank=float(scored.score)))
        return out

    def _stale_candidates(self, now: datetime) -> List[Candidate]:
        out, stale_keys = [], set()
        limit = self.cfg.stale_after_minutes * 60
        for source in ("gmail", "calendar", "slack", "messages"):
            h = self._cache.health(source)
            if h.last_attempt is None and h.last_success is None:
                continue                  # never configured/run (an optional source that is off): not "stale"
            age = h.age_seconds(now.timestamp())
            if age is None or age > limit:
                key = f"stale:{source}"
                stale_keys.add(key)
                out.append(Candidate("stale_collector", key, build_payload("stale_collector", source=source,
                                                                           address=self._address), rank=age or 1e9))
        # a source that has recovered may warn again the next time it goes stale
        recovered = [k for k in self._state.data["seen"] if k.startswith("stale:") and k not in stale_keys]
        for k in recovered:
            del self._state.data["seen"][k]
        return out

    # ------------------------------------------------------------ the gates
    def _gate(self, c: Candidate, now: datetime, dnd: bool, quiet: bool) -> Optional[str]:
        """The reason this candidate must not be delivered, or None."""
        if c.withheld:
            return c.withheld
        if dnd:
            return "dnd"
        if quiet:
            return "quiet_hours"
        if self._state.data["count"] >= self.cfg.max_per_day:
            return "budget"
        group = KIND_GROUP[c.kind]
        last = self._state.data["last"].get(group)
        cooldown = self.cfg.cooldown_minutes.get(group, 0.0) * 60
        if last is not None and now.timestamp() - last < cooldown:
            return "cooldown"
        return None

    # ------------------------------------------------------------ one tick
    async def tick(self) -> List[Delivery]:
        now = self._now()
        ts = now.timestamp()
        self._state.roll_day(now.astimezone(self._tz).date().isoformat())
        self._state.prune_seen(ts)
        dnd = bool(read_dnd(self.cfg.dnd_path, now).get("on"))
        quiet = self.in_quiet_hours(now)
        done: List[Delivery] = []
        for c in self.candidates(now):
            if c.key in self._state.data["seen"]:
                continue                                   # dedupe: never twice
            self._state.data["seen"][c.key] = ts           # considered once; held-back candidates are dropped, not queued
            reason = self._gate(c, now, dnd, quiet)
            if reason is not None:
                sup = self._state.data["suppressed"]
                sup[reason] = sup.get(reason, 0) + 1
                continue
            assert c.payload is not None
            channels = await self._deliver(c.payload)
            self._state.data["count"] += 1
            self._state.data["last"][KIND_GROUP[c.kind]] = ts
            dl = self._state.data["delivered"]
            dl[c.kind] = dl.get(c.kind, 0) + 1
            delivery = Delivery(c.kind, c.key, c.payload, channels)
            done.append(delivery)
            self.delivered.append(delivery)
        self._state.save()
        return done

    async def _deliver(self, p: Payload) -> List[str]:
        channels: List[str] = []
        speak = self.cfg.speak and bool(self._voice_mode())
        if self._bus is not None:
            from schemas.events import ProactiveNotificationEvent, VoiceOutputEvent
            # Delivery policy (utils/notifications.py): ALWAYS the HUD; additionally speak when voice mode is on (quiet hours are
            # already gated before we get here); the macOS banner below is best-effort on top and never the only channel.
            if self.cfg.hud:
                await self._bus.emit(ProactiveNotificationEvent(title=p.title, body=p.body, kind=p.kind,
                                                                origin=p.source, source="ProactiveWatcher"))
                channels.append("hud")
            if speak:
                await self._bus.emit(VoiceOutputEvent(text=p.speech, source="ProactiveWatcher"))
                channels.append("voice")
        from utils import notify_path as np_

        # osascript is the FALLBACK: when the HUD is connected (and can post banners) the HUD event above already carries the banner,
        # posted under the HUD's own app identity.
        if self.cfg.hud and self._bus is not None and np_.PATH.active() == np_.HUD:
            channels.append("macos")
        elif self.cfg.macos_notification and self._notifier is not None:
            try:
                await self._notifier.notify(p.title, p.body)
                channels.append("macos")
            except Exception as exc:          # a failed banner must never break the loop
                logger.warning(f"[Proactive] notification failed: {type(exc).__name__}")
        return channels

    # ------------------------------------------------------------ status + loop
    def status(self) -> Dict[str, Any]:
        now = self._now()
        self._state.roll_day(now.astimezone(self._tz).date().isoformat())
        return {"day": self._state.data["day"], "delivered_today": self._state.data["count"],
                "max_per_day": self.cfg.max_per_day, "by_kind": dict(self._state.data["delivered"]),
                "suppressed": dict(self._state.data["suppressed"]), "quiet_hours_now": self.in_quiet_hours(now),
                "dnd": bool(read_dnd(self.cfg.dnd_path, now).get("on"))}

    def start(self) -> None:
        if self.cfg.enabled and (self._task is None or self._task.done()):
            self._task = asyncio.ensure_future(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:              # the watcher must never be able to take the process down
                logger.warning(f"[Proactive] tick failed: {type(exc).__name__}: {exc}")
            await self._sleep(self.cfg.tick_seconds)
