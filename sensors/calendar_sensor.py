"""
CalendarSensor — polls upcoming calendar events and publishes
UpcomingMeetingEvent as a meeting approaches.

Tries EventKit (pyobjc) first, falling back to the `icalBuddy` CLI if
installed. Local only: nothing observed leaves the machine except as
short context lines handed to the LLM planner.

The icalBuddy fallback's output parsing is a best-effort defense against
its plain-text format (icalBuddy has no stable machine-readable mode);
the EventKit path is the primary, exact source of truth.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import threading
from datetime import date, datetime, time as dt_time, timedelta, timezone
from typing import List, Optional, Tuple

from schemas.events import UpcomingMeetingEvent
from sensors.base_sensor import BaseSensor

try:
    import EventKit
    import Foundation

    EVENTKIT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    EventKit = None  # type: ignore
    Foundation = None  # type: ignore
    EVENTKIT_AVAILABLE = False

#: Minutes-before-start checkpoints at which a reminder should fire.
_REMINDER_CHECKPOINTS_MIN = (15, 5)
#: Tolerance band around each checkpoint, to absorb the poll cadence.
_CHECKPOINT_TOLERANCE_MIN = 1.5

_ICALBUDDY_ARGS = ["-nc", "-po", "datetime,title", "-ps", "|:|", "-tf", "%H:%M", "-df", "%Y-%m-%d"]
_ICALBUDDY_DATETIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2}).{0,12}?(\d{2}:\d{2})\s*:?\s*(.*)$")


class CalendarSensor(BaseSensor):
    """Polls upcoming meetings and reminds at 15 and 5 minutes before."""

    config_key = "sensors.calendar"
    default_poll_interval_seconds = 300.0  # 5 minutes

    def __init__(self, event_bus=None, config=None):
        super().__init__(event_bus=event_bus, config=config)
        self._event_store = None
        self._eventkit_authorized = False
        # (title, start_time.isoformat(), checkpoint) already fired — a
        # checkpoint is never re-announced on a later poll.
        self._fired_checkpoints: set = set()

    @property
    def lookahead_minutes(self) -> float:
        return float(self._get_config(f"{self.config_key}.lookahead_minutes", 120))

    async def poll(self) -> None:
        end = datetime.now() + timedelta(minutes=self.lookahead_minutes)
        events = await self._fetch_upcoming_events(end)
        now = datetime.now(timezone.utc)

        for title, start_time in events:
            minutes_until = (start_time - now).total_seconds() / 60.0
            for checkpoint in _REMINDER_CHECKPOINTS_MIN:
                if abs(minutes_until - checkpoint) > _CHECKPOINT_TOLERANCE_MIN:
                    continue
                key = (title, start_time.isoformat(), checkpoint)
                if key in self._fired_checkpoints:
                    continue
                self._fired_checkpoints.add(key)
                await self._event_bus.emit(
                    UpcomingMeetingEvent(
                        title=title,
                        start_time=start_time,
                        minutes_until=int(round(minutes_until)),
                        source="CalendarSensor",
                    )
                )

    async def fetch_todays_events(self) -> List[Tuple[str, datetime]]:
        """All of today's events from now until midnight — for the morning briefing."""
        end = datetime.combine(date.today(), dt_time.max)
        return await self._fetch_upcoming_events(end)

    # =========================================================================
    # Backend selection
    # =========================================================================

    async def _fetch_upcoming_events(self, end: datetime) -> List[Tuple[str, datetime]]:
        if EVENTKIT_AVAILABLE:
            try:
                return await self._fetch_via_eventkit(end)
            except Exception as exc:
                self._logger.warning(f"EventKit lookup failed: {exc}")

        try:
            return await self._fetch_via_icalbuddy(end)
        except FileNotFoundError:
            self._logger.debug("icalBuddy not installed; no calendar backend available")
        except Exception as exc:
            self._logger.warning(f"icalBuddy lookup failed: {exc}")

        return []

    # =========================================================================
    # EventKit backend
    # =========================================================================

    async def _fetch_via_eventkit(self, end: datetime) -> List[Tuple[str, datetime]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_via_eventkit_sync, end)

    def _fetch_via_eventkit_sync(self, end: datetime) -> List[Tuple[str, datetime]]:
        if self._event_store is None:
            self._event_store = EventKit.EKEventStore.alloc().init()

        if not self._eventkit_authorized:
            self._eventkit_authorized = self._request_eventkit_access()
            if not self._eventkit_authorized:
                return []

        now = datetime.now()

        start_date = Foundation.NSDate.dateWithTimeIntervalSince1970_(now.timestamp())
        end_date = Foundation.NSDate.dateWithTimeIntervalSince1970_(end.timestamp())

        predicate = self._event_store.predicateForEventsWithStartDate_endDate_calendars_(
            start_date, end_date, None,
        )
        events = list(self._event_store.eventsMatchingPredicate_(predicate) or [])

        results: List[Tuple[str, datetime]] = []
        for item in events:
            title = str(item.title() or "Untitled")
            start = datetime.fromtimestamp(item.startDate().timeIntervalSince1970(), tz=timezone.utc)
            results.append((title, start))
        return results

    def _request_eventkit_access(self) -> bool:
        done_event = threading.Event()
        granted = {"value": False}

        def completion(success, error):
            granted["value"] = bool(success)
            done_event.set()

        if hasattr(self._event_store, "requestFullAccessToEventsWithCompletion_"):
            self._event_store.requestFullAccessToEventsWithCompletion_(completion)
        elif hasattr(self._event_store, "requestAccessToEntityType_completion_"):
            self._event_store.requestAccessToEntityType_completion_(EventKit.EKEntityTypeEvent, completion)
        else:
            return False

        done_event.wait(timeout=10)
        return granted["value"]

    # =========================================================================
    # icalBuddy backend (fallback)
    # =========================================================================

    async def _fetch_via_icalbuddy(self, end: datetime) -> List[Tuple[str, datetime]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_via_icalbuddy_sync, end)

    def _fetch_via_icalbuddy_sync(self, end: datetime) -> List[Tuple[str, datetime]]:
        minutes_ahead = max(1.0, (end - datetime.now()).total_seconds() / 60.0)
        days_ahead = max(1, int(minutes_ahead // (24 * 60)) + 1)
        result = subprocess.run(
            ["icalBuddy", *_ICALBUDDY_ARGS, f"eventsToday+{days_ahead}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return []
        return self._parse_icalbuddy_output(result.stdout)

    @staticmethod
    def _parse_icalbuddy_output(output: str) -> List[Tuple[str, datetime]]:
        results: List[Tuple[str, datetime]] = []
        for raw_line in output.splitlines():
            line = raw_line.strip().lstrip("•").strip()
            match = _ICALBUDDY_DATETIME_RE.search(line)
            if not match:
                continue
            date_str, time_str, title = match.group(1), match.group(2), match.group(3).strip()
            if not title:
                continue
            try:
                start = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            results.append((title, start))
        return results
