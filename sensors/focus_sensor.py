"""
FocusSensor — polls the frontmost macOS application and publishes
AppFocusChangedEvent on change.

Local only: reads the frontmost app's name via NSWorkspace (pyobjc), with
an AppleScript fallback. Nothing observed leaves the machine except as
short context lines handed to the LLM planner.

NSWorkspace quirk: `frontmostApplication()` is populated reactively from
workspace-activation notifications, which are delivered only to the main
thread's Cocoa run loop, and only while that run loop is actually
spinning. A plain asyncio poll loop never spins one (and pumping some
*other* thread's run loop, e.g. via run_in_executor, does not receive
those notifications either — confirmed empirically), so without a main-
thread pump the property freezes at whatever was frontmost when the
process started and never updates. The fix is a brief run-loop pump on
the main thread immediately before each read; this blocks the event loop
for ~0.3s once per poll cycle, an acceptable cost at a multi-second poll
interval.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Deque, List, Optional, Tuple

from schemas.events import AppFocusChangedEvent
from sensors.base_sensor import BaseSensor
from utils.applescript import run_applescript

try:
    from AppKit import NSWorkspace
    from Foundation import NSDate, NSRunLoop

    NSWORKSPACE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    NSWorkspace = None  # type: ignore
    NSDate = None  # type: ignore
    NSRunLoop = None  # type: ignore
    NSWORKSPACE_AVAILABLE = False

_FRONTMOST_APPLESCRIPT = (
    'tell application "System Events" to get name of first process whose frontmost is true'
)

#: How long to pump the run loop before reading frontmostApplication(),
#: to let a just-delivered workspace-activation notification land.
_RUN_LOOP_PUMP_SECONDS = 0.3

_SWITCH_HISTORY_MAXLEN = 200


class FocusSensor(BaseSensor):
    """Polls the frontmost application every few seconds."""

    config_key = "sensors.focus"
    default_poll_interval_seconds = 5.0

    def __init__(self, event_bus=None, config=None):
        super().__init__(event_bus=event_bus, config=config)
        self._previous_app: Optional[str] = None
        self._switch_history: Deque[Tuple[datetime, str]] = deque(maxlen=_SWITCH_HISTORY_MAXLEN)

    async def poll(self) -> None:
        current_app = await self._get_frontmost_app()
        if not current_app or current_app == self._previous_app:
            return

        previous_app = self._previous_app
        now = datetime.now(timezone.utc)
        self._previous_app = current_app
        self._switch_history.append((now, current_app))

        if previous_app is None:
            # First observation — nothing to report a "switch" from yet.
            return

        await self._event_bus.emit(
            AppFocusChangedEvent(
                app_name=current_app,
                previous_app=previous_app,
                timestamp=now,
                source="FocusSensor",
            )
        )

    async def _get_frontmost_app(self) -> Optional[str]:
        if NSWORKSPACE_AVAILABLE:
            try:
                # Deliberately synchronous, on this (main) thread — see the
                # module docstring on why run_in_executor doesn't work here.
                name = self._query_nsworkspace_frontmost()
                if name:
                    return name
            except Exception as exc:
                self._logger.debug(f"NSWorkspace lookup failed, falling back to AppleScript: {exc}")

        try:
            name = run_applescript(_FRONTMOST_APPLESCRIPT)
            return name or None
        except Exception as exc:
            self._logger.debug(f"AppleScript frontmost-app lookup failed: {exc}")
            return None

    @staticmethod
    def _query_nsworkspace_frontmost() -> Optional[str]:
        """Pump the main thread's run loop briefly, then read frontmostApplication()."""
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(_RUN_LOOP_PUMP_SECONDS))
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return None
        name = app.localizedName()
        return str(name) if name else None

    def recent_switches(self, window_minutes: float = 30.0) -> List[Tuple[datetime, str]]:
        """Rolling window of recent (timestamp, app_name) switches."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        return [(ts, app) for ts, app in self._switch_history if ts >= cutoff]
