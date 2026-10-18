"""Notifications, honestly.

`osascript -e 'display notification …'` exits 0 whether or not anything appears: the terminal app may lack notification permission,
a Focus mode may be on, or the banner may be filed under "Script Editor". So Vesper never says a notification was shown. What it can
say is "I asked macOS to show one" — unless the user has CONFIRMED, with `vesper notify-test`, that banners really appear from this app.

This module holds the small pieces: the confirmation state file (data/notifications.json: status, timestamp, which terminal app — never any
message text), the delivery policy shared by the proactive watcher and the notify tool, and the notify-test flow.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = PROJECT_ROOT / "data" / "notifications.json"
CONFIRMED, UNCONFIRMED, FAILED = "confirmed", "unconfirmed", "failed"

#: TERM_PROGRAM -> the name System Settings lists.
_TERMINALS = {"Apple_Terminal": "Terminal", "iTerm.app": "iTerm", "ghostty": "Ghostty", "vscode": "Visual Studio Code",
              "WarpTerminal": "Warp", "Hyper": "Hyper", "WezTerm": "WezTerm", "kitty": "kitty", "Alacritty": "Alacritty"}


def terminal_app(env: Optional[Dict[str, str]] = None) -> str:
    env = os.environ if env is None else env
    return _TERMINALS.get(env.get("TERM_PROGRAM", ""), env.get("TERM_PROGRAM") or "the terminal app you start Vesper from")


class NotifyState:
    """{"status": confirmed|failed, "app": "...", "at": iso} — one record, overwritten by `vesper notify-test`."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else STATE_PATH      # resolved at call time (tests redirect STATE_PATH)

    def load(self) -> Dict[str, Any]:
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def save(self, status: str, app: str, now: Optional[datetime] = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"status": status, "app": app, "at": (now or datetime.now().astimezone()).isoformat(timespec="seconds")}
        self.path.write_text(json.dumps(rec))

    def status(self, app: Optional[str] = None) -> str:
        """confirmed only if the user confirmed it FROM THIS APP; a confirmation made from another terminal does not count."""
        rec = self.load()
        st = rec.get("status")
        if st == CONFIRMED:
            return CONFIRMED if (app is None or rec.get("app") == app) else UNCONFIRMED
        if st == FAILED:
            return FAILED if (app is None or rec.get("app") == app) else UNCONFIRMED
        return UNCONFIRMED


SETTINGS_PATH = "System Settings > Notifications > {app} > Allow Notifications"


def fix_hint(app: str) -> str:
    return (f"{SETTINGS_PATH.format(app=app)} (macOS may list the banner under 'Script Editor' instead), and make sure no Focus mode "
            "is silencing it; then run `vesper notify-test`")


# --------------------------------------------------------------------------- delivery policy

@dataclass(frozen=True)
class Plan:
    hud: bool
    voice: bool
    macos: bool


def delivery_plan(*, hud_enabled: bool = True, voice_on: bool = False, quiet: bool = False, speak_enabled: bool = True,
                  macos_enabled: bool = True) -> Plan:
    """ALWAYS the HUD; speak when voice mode is on and it is not quiet hours; the macOS banner is best-effort on top."""
    return Plan(hud=hud_enabled, voice=bool(voice_on and speak_enabled and not quiet), macos=macos_enabled)


def _hhmm(s: str) -> dtime:
    h, m = str(s).split(":")
    return dtime(int(h), int(m))


def in_quiet_hours(start: str, end: str, tz: str, now: Optional[datetime] = None) -> bool:
    from zoneinfo import ZoneInfo

    local = (now or datetime.now().astimezone()).astimezone(ZoneInfo(tz)).time()
    s, e = _hhmm(start), _hhmm(end)
    return (s <= local or local < e) if s > e else (s <= local < e)


def policy_from_config(cfg: Any, now: Optional[datetime] = None) -> Plan:
    """Voice mode = voice output AND input enabled in settings; quiet hours from proactive.watcher.quiet_hours."""
    def g(path: str, default: Any = None) -> Any:
        cur: Any = cfg
        for part in path.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
            if cur is None:
                return default
        return cur

    voice_on = bool(g("voice.output.enabled", False)) and bool(g("voice.input.enabled", False))
    try:
        quiet = in_quiet_hours(str(g("proactive.watcher.quiet_hours.start", "23:00")), str(g("proactive.watcher.quiet_hours.end", "08:00")),
                               str(g("proactive.watcher.timezone", "Asia/Kolkata")), now)
    except Exception:  # noqa: BLE001
        quiet = False
    return delivery_plan(voice_on=voice_on, quiet=quiet)


# --------------------------------------------------------------------------- notify-test

def run_notify_test(post: Callable[[str, str], Tuple[bool, str]], ask: Callable[[str], str], state: NotifyState, app: str,
                    out: Callable[[str], None] = print, now: Optional[datetime] = None) -> int:
    """Post ONE notification, ask whether it was seen, record the answer. Exit 0 = confirmed."""
    ok, err = post("Vesper notification test", "If you can read this, notifications work. You can dismiss it.")
    if not ok:
        state.save(FAILED, app, now)
        out(f"macOS refused the request ({(err or 'no detail')[:80]}). Notifications: failed for {app}.")
        out("Fix: " + fix_hint(app))
        return 1
    try:
        answer = ask("A test notification was requested. Did you see it? (y/n) ").strip().lower()
    except EOFError:
        answer = ""
    if answer.startswith("y"):
        state.save(CONFIRMED, app, now)
        out(f"Confirmed: notifications appear from {app}. Vesper may now say a notification was shown.")
        return 0
    if answer.startswith("n"):
        state.save(FAILED, app, now)
        out(f"Recorded: the notification did not appear for {app}; Vesper will not claim notifications.")
        out("Fix: " + fix_hint(app))
        return 1
    out("No answer recorded: notifications stay unconfirmed.")
    return 2
