"""
proactive/morning_routine.py — the routine that defines Vesper (PC4).

No new integrations: this COMPOSES tools that already exist. Triggered by the
Proactive Engine (wake-time cron, or first app activity after a configured hour),
the routine, in sequence:

  a. Gathers, each source guarded independently — weather, today's calendar,
     reminders due, unread-email triage, GitHub notifications, Slack mentions,
     and the day plan. A source that fails is recorded as a gap; it never blocks
     the rest.
  b. Delivers ONE composed briefing in Vesper's voice (spoken via VoiceOutputEvent
     → TTS + HUD): weather in a line, what matters today, the three items most
     needing action, then the day plan — with any gap named in a single clause.
  c. OFFERS to set up the environment (start the focus playlist + open the work
     apps) as a SINGLE confirmation — the confirm-tier setup_workspace tool. If
     the routine is marked pre_approved it runs the set directly; otherwise the
     whole set is approved at once, or not at all.

Anti-annoyance rules are enforced here (not by prompting): at most once per
calendar day; deferred while the user is mid-conversation; a "not now" pushes it
by defer_minutes and a second "not now" cancels it for the day.

An optional evening counterpart (run_evening) gives a two-line close: what got
done (today's commits) and the next commitment.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Awaitable, Dict, List, Optional, Tuple

from bus.event_bus import EventBus, get_event_bus
from guardian.gate import VerdictType
from proactive.briefing import _as_list
from proactive.day_planner import DEFAULT_NOTION_TASKS_DB, assemble_day_plan_text
from schemas.events import VoiceOutputEvent
from tools.registry import ToolSpec, get_registry
from utils.logger import get_logger

logger = get_logger(__name__)

SETUP_WORKSPACE_TOOL = "setup_workspace"

# gate outcomes
_RUN = "run"
_SKIP = "skip"
_DEFER_CONVERSATION = "defer_conversation"

MORNING_STYLE_NOTE = (
    "(Deliver Vesper's morning briefing now — ONE composed piece, spoken aloud. "
    "Structure it, restrained: open with the weather in a single line; then what "
    "matters today; then the THREE items most needing action; then the time-blocked "
    "day plan. One salutation at most, no filler. Keep it tight.)"
)

EVENING_STYLE_NOTE = (
    "(Give Vesper's evening close — at most TWO lines. First: what got done today. "
    "Second: the first commitment coming up. Flat and brief, no fluff.)"
)


# =============================================================================
# setup_workspace — the environment-setup tool the routine offers as one action.
# =============================================================================
async def _open_app(app: str) -> bool:
    """Open a macOS app by name. Module-level so tests can monkeypatch it."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "open", "-a", app, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        return (await asyncio.wait_for(proc.wait(), timeout=15)) == 0
    except Exception as exc:
        logger.warning(f"[setup_workspace] could not open {app!r}: {exc}")
        return False


async def _setup_workspace_handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    playlist = str(arguments.get("playlist", "") or "").strip()
    apps = [str(a) for a in (arguments.get("apps") or [])]
    done: List[str] = []

    if playlist:
        play = get_registry().get("play")  # Spotify MCP tool, if connected
        if play is not None and play.handler is not None:
            try:
                await play.handler({"query": playlist}, {})
                done.append(f"started {playlist}")
            except Exception as exc:
                logger.warning(f"[setup_workspace] playlist start failed: {exc}")

    opened = [a for a in apps if await _open_app(a)]
    if opened:
        done.append("opened " + ", ".join(opened))
    return "Workspace ready — " + ("; ".join(done) if done else "nothing to do") + "."


async def _setup_confirm_summary(arguments: Dict[str, Any]) -> str:
    playlist = str(arguments.get("playlist", "") or "").strip()
    apps = [str(a) for a in (arguments.get("apps") or [])]
    bits: List[str] = []
    if playlist:
        bits.append(f"start {playlist}")
    if apps:
        bits.append("open " + ", ".join(apps))
    return "Set up your workspace — " + (" and ".join(bits) if bits else "nothing") + "?"


def register_setup_workspace(registry=None) -> None:
    registry = registry or get_registry()
    if registry.get(SETUP_WORKSPACE_TOOL) is not None:
        return
    registry.register(ToolSpec(
        name=SETUP_WORKSPACE_TOOL,
        description=(
            "Set up the work environment in one go: start the focus playlist and open "
            "the configured work apps. A single confirmation covers the whole set."
        ),
        parameters={"type": "object", "properties": {
            "playlist": {"type": "string", "description": "Playlist/query to start."},
            "apps": {"type": "array", "items": {"type": "string"}, "description": "App names to open."},
        }},
        tier="confirm", handler=_setup_workspace_handler, confirm_summary=_setup_confirm_summary, category="routine",
    ))


# =============================================================================
# Evening data — git commits today (module-level so tests can monkeypatch).
# =============================================================================
async def _git_commits_today() -> List[str]:
    proc = await asyncio.create_subprocess_exec(
        "git", "log", "--since=midnight", "--oneline",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    return [ln for ln in out.decode(errors="replace").splitlines() if ln.strip()]


@dataclass
class RunResult:
    text: str
    gaps: List[str] = field(default_factory=list)
    sections: Dict[str, Any] = field(default_factory=dict)


class MorningRoutine:
    """Composes existing tools into the once-a-day morning (and evening) routine."""

    def __init__(
        self,
        *,
        planner: Any,
        guardian: Any,
        registry=None,
        event_bus: Optional[EventBus] = None,
        calendar_sensor: Any = None,
        memory_agent: Any = None,
        config: Optional[Dict[str, Any]] = None,
        clock=None,
    ):
        self._planner = planner
        self._guardian = guardian
        self._registry = registry or get_registry()
        self._event_bus = event_bus or get_event_bus()
        self._calendar_sensor = calendar_sensor
        self._memory_agent = memory_agent
        self._config = config or {}
        self._clock = clock or datetime.now

        # anti-annoyance state
        self._completed_date: Optional[date] = None
        self._cancelled_date: Optional[date] = None
        self._deferred_until: Optional[datetime] = None
        self._defer_count: int = 0
        self._defer_count_date: Optional[date] = None

        register_setup_workspace(self._registry)

    # -------------------------------- config -------------------------------
    def _cfg(self, key: str, default: Any = None) -> Any:
        value: Any = self._config
        for part in key.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(part)
            if value is None:
                return default
        return value

    def _playlist(self) -> str:
        return str(self._cfg("proactive.morning_routine.playlist", "your focus playlist"))

    def _work_apps(self) -> List[str]:
        return [str(a) for a in (self._cfg("proactive.morning_routine.work_apps", []) or [])]

    def _pre_approved(self) -> bool:
        return bool(self._cfg("proactive.morning_routine.pre_approved", False))

    def _defer_minutes(self) -> int:
        return int(self._cfg("proactive.morning_routine.defer_minutes", 60))

    def _conversation_retry_min(self) -> int:
        return int(self._cfg("proactive.morning_routine.conversation_retry_minutes", 5))

    # ---------------------------- anti-annoyance ---------------------------
    def _gate(self, now: datetime, in_conversation: bool) -> Tuple[str, str]:
        today = now.date()
        if self._cancelled_date == today:
            return _SKIP, "cancelled for today"
        if self._completed_date == today:
            return _SKIP, "already ran today"
        if self._deferred_until is not None and now < self._deferred_until:
            return _SKIP, "deferred"
        if in_conversation:
            return _DEFER_CONVERSATION, "user mid-conversation"
        return _RUN, "ok"

    def should_run(self, now: Optional[datetime] = None, in_conversation: bool = False) -> bool:
        now = now or self._clock()
        return self._gate(now, in_conversation)[0] == _RUN

    def not_now(self, now: Optional[datetime] = None) -> str:
        """Record a 'not now'. First today → defer by defer_minutes; second → cancel
        for the day."""
        now = now or self._clock()
        today = now.date()
        if self._defer_count_date != today:
            self._defer_count_date = today
            self._defer_count = 0
        self._defer_count += 1
        if self._defer_count >= 2:
            self._cancelled_date = today
            self._deferred_until = None
            self._completed_date = None
            logger.info("[morning] cancelled for the day (second 'not now')")
            return "cancelled"
        self._deferred_until = now + timedelta(minutes=self._defer_minutes())
        self._completed_date = None  # let it re-run once the defer window elapses
        logger.info(f"[morning] deferred until {self._deferred_until.isoformat()}")
        return "deferred"

    # -------------------------------- run ----------------------------------
    async def run(self, now: Optional[datetime] = None, in_conversation: bool = False) -> Optional[RunResult]:
        now = now or self._clock()
        decision, reason = self._gate(now, in_conversation)
        if decision != _RUN:
            if decision == _DEFER_CONVERSATION:
                self._deferred_until = now + timedelta(minutes=self._conversation_retry_min())
            logger.info(f"[morning] not running: {reason}")
            return None

        sections, gaps = await self._gather()
        prompt = self._compose_prompt(sections, gaps)
        result = await self._planner.run(user_text=prompt, purpose="planning")
        text = getattr(result, "text", "") or ""
        await self._event_bus.emit(VoiceOutputEvent(text=text, source="MorningRoutine"))

        # It has delivered — that satisfies once-per-day regardless of the offer.
        self._completed_date = now.date()
        await self._offer_setup()
        return RunResult(text=text, gaps=gaps, sections=sections)

    async def _guarded(self, name: str, coro: Awaitable[Any], gaps: List[str]) -> Any:
        try:
            return await coro
        except Exception as exc:
            logger.warning(f"[morning] source '{name}' failed: {exc}")
            gaps.append(name)
            return None

    async def _gather(self) -> Tuple[Dict[str, Any], List[str]]:
        """Compose existing tools. Each source is guarded independently — a failure
        becomes a gap and never blocks the others."""
        sections: Dict[str, Any] = {}
        gaps: List[str] = []
        reg = self._registry

        weather = reg.get("current_weather")
        if weather is not None and weather.handler is not None:
            loc = self._cfg("proactive.morning_routine.weather_location", "")
            sections["weather"] = await self._guarded(
                "weather", self._weather(weather, loc), gaps
            )

        if self._calendar_sensor is not None:
            sections["calendar"] = await self._guarded(
                "calendar", self._calendar_sensor.fetch_todays_events(), gaps
            )

        reminders = reg.get("reminders_due")
        if reminders is not None and reminders.handler is not None:
            sections["reminders"] = await self._guarded(
                "reminders", self._as_list_of(reminders), gaps
            )

        inbox = reg.get("list_unread")
        if inbox is not None and inbox.handler is not None:
            sections["inbox"] = await self._guarded("email", self._inbox(inbox), gaps)

        notif = reg.get("list_notifications")
        prs = reg.get("list_pull_requests")
        if (notif and notif.handler) or (prs and prs.handler):
            sections["dev"] = await self._guarded("github", self._dev(notif, prs), gaps)

        mentions = reg.get("get_mentions")
        if mentions is not None and mentions.handler is not None and mentions.category == "mcp:slack":
            sections["slack"] = await self._guarded("slack", self._slack(mentions), gaps)

        sections["day_plan"] = await self._guarded(
            "day plan",
            assemble_day_plan_text(reg, self._memory_agent, str(self._cfg("proactive.day_plan.notion_db", DEFAULT_NOTION_TASKS_DB))),
            gaps,
        )
        return sections, gaps

    @staticmethod
    async def _weather(tool: ToolSpec, loc: Any) -> str:
        return (await tool.handler({"location": loc}, {})).strip()

    @staticmethod
    async def _as_list_of(tool: ToolSpec) -> List[Any]:
        return _as_list(await tool.handler({}, {}))

    @staticmethod
    async def _inbox(tool: ToolSpec) -> List[Any]:
        raw = await tool.handler({"max_n": 20}, {})
        return json.loads(raw) if isinstance(raw, str) else (raw or [])

    @staticmethod
    async def _dev(notif: Optional[ToolSpec], prs: Optional[ToolSpec]) -> str:
        parts: List[str] = []
        if notif is not None and notif.handler is not None:
            parts.append(f"{len(_as_list(await notif.handler({}, {})))} GitHub notifications")
        if prs is not None and prs.handler is not None:
            parts.append(f"{len(_as_list(await prs.handler({}, {})))} PRs awaiting review")
        return "; ".join(parts)

    @staticmethod
    async def _slack(tool: ToolSpec) -> str:
        items = _as_list(await tool.handler({}, {}))
        return f"{len(items)} Slack mentions awaiting reply" if items else ""

    # ------------------------------ compose --------------------------------
    def _compose_prompt(self, sections: Dict[str, Any], gaps: List[str]) -> str:
        L: List[str] = []
        weather = sections.get("weather")
        L.append(f"Weather: {weather}" if weather else "Weather: unavailable.")
        L.append("")

        calendar = sections.get("calendar")
        if calendar:
            L.append("Today's calendar:")
            for title, start in calendar:
                when = start.strftime("%H:%M") if hasattr(start, "strftime") else str(start)
                L.append(f"  - {title} at {when}")
        else:
            L.append("Today's calendar: nothing scheduled (or unavailable).")

        reminders = sections.get("reminders")
        if reminders:
            L.append("Reminders due:")
            for r in reminders[:10]:
                L.append(f"  - {r if isinstance(r, str) else r.get('title', r)}")

        inbox = sections.get("inbox")
        if inbox:
            L.append(f"Unread email: {len(inbox)}.")
            for m in inbox[:8]:
                if isinstance(m, dict):
                    L.append(f"  - From {m.get('sender', '')}: {m.get('subject', '')}")

        if sections.get("dev"):
            L.append(sections["dev"])
        if sections.get("slack"):
            L.append(sections["slack"])

        if sections.get("day_plan"):
            L.append("")
            L.append(str(sections["day_plan"]))

        data = "\n".join(L)

        gap_note = ""
        if gaps:
            gap_note = (
                "\n\n(These sources were unavailable: " + ", ".join(sorted(set(gaps))) +
                ". Name the gap in a single clause and brief on the rest — never skip "
                "the briefing over a broken source.)"
            )

        playlist, apps = self._playlist(), self._work_apps()
        setup_desc = f"start {playlist}" + (f" and open {', '.join(apps)}" if apps else "")
        if self._pre_approved():
            offer = f"\n\n(You are setting up his workspace now — {setup_desc}. Say so in one line.)"
        else:
            offer = f"\n\n(Close by OFFERING to set up his workspace — {setup_desc} — which he confirms once. Offer; do not assume yes.)"

        return f"{MORNING_STYLE_NOTE}\n\n{data}{gap_note}{offer}"

    # ------------------------------- offer ---------------------------------
    async def _offer_setup(self) -> None:
        spec = self._registry.get(SETUP_WORKSPACE_TOOL)
        if spec is None:
            return
        args = {"playlist": self._playlist(), "apps": self._work_apps()}
        if not args["playlist"] and not args["apps"]:
            return

        if self._pre_approved():
            try:
                await spec.handler(args, {})
            except Exception as exc:
                logger.warning(f"[morning] pre-approved setup failed: {exc}")
            return

        if self._guardian is None:
            return
        verdict = await self._guardian.check(spec, args)
        if verdict.outcome == VerdictType.NEEDS_CONFIRMATION:
            verdict = await self._guardian.await_resolution(verdict.request_id)
        if verdict.outcome == VerdictType.ALLOW:
            try:
                await spec.handler(args, {})
            except Exception as exc:
                logger.warning(f"[morning] setup failed after approval: {exc}")

    # ------------------------------ evening --------------------------------
    async def run_evening(self, now: Optional[datetime] = None, in_conversation: bool = False) -> Optional[RunResult]:
        now = now or self._clock()
        if in_conversation:
            logger.info("[evening] deferring — user mid-conversation")
            return None
        gaps: List[str] = []
        commits = await self._guarded("commits", _git_commits_today(), gaps)

        upcoming = None
        up = self._registry.get("upcoming")
        if up is not None and up.handler is not None:
            upcoming = await self._guarded("upcoming", up.handler({}, {}), gaps)

        prompt = self._compose_evening_prompt(commits, upcoming, gaps)
        result = await self._planner.run(user_text=prompt, purpose="planning")
        text = getattr(result, "text", "") or ""
        await self._event_bus.emit(VoiceOutputEvent(text=text, source="MorningRoutine"))
        return RunResult(text=text, gaps=gaps)

    def _compose_evening_prompt(self, commits: Optional[List[str]], upcoming: Any, gaps: List[str]) -> str:
        L: List[str] = []
        if commits is None:
            L.append("Commits today: unavailable.")
        else:
            L.append(f"Commits today: {len(commits)}" + (f" ({'; '.join(commits[:5])})" if commits else "."))
        L.append(f"Next commitment: {upcoming}" if upcoming else "Next commitment: unknown.")
        gap_note = ""
        if gaps:
            gap_note = f"\n\n(Unavailable: {', '.join(sorted(set(gaps)))} — note briefly.)"
        return f"{EVENING_STYLE_NOTE}\n\n" + "\n".join(L) + gap_note
