"""
eventkit_client — Calendar + Reminders access for the apple_pim MCP
server. EventKit (pyobjc) is the primary backend; a best-effort
AppleScript (osascript) fallback covers reads when EventKit access
isn't available/authorized.

Runs inside the apple_pim MCP server's own Python 3.10+ virtualenv
(mcp_servers/.venv) — separate from the main app's Python 3.9, exactly
like the Gmail MCP server (see mcp_servers/gmail/gmail_client.py).

Writes (add_reminder, create_event) are EventKit-only: AppleScript date
literals are locale-format-dependent to construct safely, and creation
is already gated behind Guardian's confirm tier, so a fragile secondary
path isn't worth the risk — they raise a clear error if EventKit isn't
authorized rather than silently falling back.
"""

from __future__ import annotations

import asyncio
import subprocess
import threading
from datetime import date, datetime, time as dt_time, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    import EventKit
    import Foundation

    EVENTKIT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    EventKit = None  # type: ignore
    Foundation = None  # type: ignore
    EVENTKIT_AVAILABLE = False

_ACCESS_TIMEOUT_SECONDS = 10.0
_APPLESCRIPT_TIMEOUT_SECONDS = 15.0
_FIELD_SEP = "|:|"

_event_store = None  # lazily built EKEventStore, cached for the process lifetime
_events_authorized = False
_reminders_authorized = False


# =============================================================================
# EventKit access
# =============================================================================

def _get_store():
    global _event_store
    if _event_store is None:
        _event_store = EventKit.EKEventStore.alloc().init()
    return _event_store


def _request_access_sync(entity_type: Any, full_access_selector: str) -> bool:
    store = _get_store()
    done = threading.Event()
    granted = {"value": False}

    def completion(success, error):
        granted["value"] = bool(success)
        done.set()

    if hasattr(store, full_access_selector):
        getattr(store, full_access_selector)(completion)
    elif hasattr(store, "requestAccessToEntityType_completion_"):
        store.requestAccessToEntityType_completion_(entity_type, completion)
    else:
        return False

    done.wait(timeout=_ACCESS_TIMEOUT_SECONDS)
    return granted["value"]


def _ensure_events_access() -> bool:
    global _events_authorized
    if not _events_authorized:
        _events_authorized = _request_access_sync(
            EventKit.EKEntityTypeEvent, "requestFullAccessToEventsWithCompletion_"
        )
    return _events_authorized


def _ensure_reminders_access() -> bool:
    global _reminders_authorized
    if not _reminders_authorized:
        _reminders_authorized = _request_access_sync(
            EventKit.EKEntityTypeReminder, "requestFullAccessToRemindersWithCompletion_"
        )
    return _reminders_authorized


# =============================================================================
# EventKit reads
# =============================================================================

def _summarize_event(item: Any) -> Dict[str, Any]:
    start = datetime.fromtimestamp(item.startDate().timeIntervalSince1970(), tz=timezone.utc)
    end_date = item.endDate()
    end = datetime.fromtimestamp(end_date.timeIntervalSince1970(), tz=timezone.utc) if end_date else None
    calendar = item.calendar()
    return {
        "title": str(item.title() or "Untitled"),
        "start": start.isoformat(),
        "end": end.isoformat() if end else None,
        "calendar": str(calendar.title()) if calendar else None,
        "notes": str(item.notes()) if item.notes() else None,
        "all_day": bool(item.isAllDay()),
    }


def _fetch_events_sync(start: datetime, end: datetime) -> List[Dict[str, Any]]:
    if EVENTKIT_AVAILABLE and _ensure_events_access():
        store = _get_store()
        start_date = Foundation.NSDate.dateWithTimeIntervalSince1970_(start.timestamp())
        end_date = Foundation.NSDate.dateWithTimeIntervalSince1970_(end.timestamp())
        predicate = store.predicateForEventsWithStartDate_endDate_calendars_(start_date, end_date, None)
        events = list(store.eventsMatchingPredicate_(predicate) or [])
        events.sort(key=lambda e: e.startDate().timeIntervalSince1970())
        return [_summarize_event(e) for e in events]

    return _fetch_events_via_applescript(start, end)


def _summarize_reminder(item: Any) -> Dict[str, Any]:
    due_iso = None
    due_components = item.dueDateComponents()
    if due_components is not None:
        calendar = Foundation.NSCalendar.currentCalendar()
        due_date = calendar.dateFromComponents_(due_components)
        if due_date is not None:
            due_iso = datetime.fromtimestamp(due_date.timeIntervalSince1970(), tz=timezone.utc).isoformat()
    reminder_calendar = item.calendar()
    return {
        "id": str(item.calendarItemIdentifier()),
        "title": str(item.title() or "Untitled"),
        "due": due_iso,
        "notes": str(item.notes()) if item.notes() else None,
        "list": str(reminder_calendar.title()) if reminder_calendar else None,
    }


def _fetch_reminders_sync() -> List[Dict[str, Any]]:
    """All incomplete reminders across every list."""
    if EVENTKIT_AVAILABLE and _ensure_reminders_access():
        store = _get_store()
        predicate = store.predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_(
            None, None, None
        )

        done = threading.Event()
        result: Dict[str, List[Any]] = {"reminders": []}

        def completion(reminders):
            result["reminders"] = list(reminders or [])
            done.set()

        store.fetchRemindersMatchingPredicate_completion_(predicate, completion)
        done.wait(timeout=_ACCESS_TIMEOUT_SECONDS)
        return [_summarize_reminder(r) for r in result["reminders"]]

    return _fetch_reminders_via_applescript()


def _filter_reminders_by_scope(reminders: List[Dict[str, Any]], scope: str) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    today_end = datetime.combine(date.today(), dt_time.max, tzinfo=timezone.utc)

    def due_dt(r: Dict[str, Any]) -> Optional[datetime]:
        return datetime.fromisoformat(r["due"]) if r.get("due") else None

    if scope == "overdue":
        return [r for r in reminders if (d := due_dt(r)) is not None and d < now]
    if scope == "today":
        return [r for r in reminders if (d := due_dt(r)) is not None and d <= today_end]
    if scope == "no_date":
        return [r for r in reminders if not r.get("due")]
    return reminders  # "all"


# =============================================================================
# EventKit writes (confirm tier — no AppleScript fallback, see module docstring)
# =============================================================================

def _add_reminder_sync(title: str, due: Optional[str]) -> Dict[str, Any]:
    if not EVENTKIT_AVAILABLE or not _ensure_reminders_access():
        raise RuntimeError("Reminders access is not available or was not authorized")

    store = _get_store()
    reminder = EventKit.EKReminder.reminderWithEventStore_(store)
    reminder.setTitle_(title)
    reminder.setCalendar_(store.defaultCalendarForNewReminders())

    if due:
        due_dt = datetime.fromisoformat(due)
        ns_date = Foundation.NSDate.dateWithTimeIntervalSince1970_(due_dt.timestamp())
        units = (
            Foundation.NSCalendarUnitYear
            | Foundation.NSCalendarUnitMonth
            | Foundation.NSCalendarUnitDay
            | Foundation.NSCalendarUnitHour
            | Foundation.NSCalendarUnitMinute
        )
        components = Foundation.NSCalendar.currentCalendar().components_fromDate_(units, ns_date)
        reminder.setDueDateComponents_(components)

    success, error = store.saveReminder_commit_error_(reminder, True, None)
    if not success:
        raise RuntimeError(f"Failed to save reminder: {error}")
    return {"id": str(reminder.calendarItemIdentifier()), "title": title, "due": due}


def _create_event_sync(
    title: str, start: str, end: str, calendar_name: Optional[str], notes: Optional[str]
) -> Dict[str, Any]:
    if not EVENTKIT_AVAILABLE or not _ensure_events_access():
        raise RuntimeError("Calendar access is not available or was not authorized")

    store = _get_store()
    event = EventKit.EKEvent.eventWithEventStore_(store)
    event.setTitle_(title)
    event.setStartDate_(Foundation.NSDate.dateWithTimeIntervalSince1970_(datetime.fromisoformat(start).timestamp()))
    event.setEndDate_(Foundation.NSDate.dateWithTimeIntervalSince1970_(datetime.fromisoformat(end).timestamp()))
    if notes:
        event.setNotes_(notes)

    target_calendar = None
    if calendar_name:
        for cal in store.calendarsForEntityType_(EventKit.EKEntityTypeEvent) or []:
            if str(cal.title()) == calendar_name:
                target_calendar = cal
                break
    event.setCalendar_(target_calendar or store.defaultCalendarForNewEvents())

    success, error = store.saveEvent_span_error_(event, EventKit.EKSpanThisEvent, None)
    if not success:
        raise RuntimeError(f"Failed to save event: {error}")
    return {"id": str(event.eventIdentifier()), "title": title, "start": start, "end": end}


# =============================================================================
# AppleScript fallback (reads only)
# =============================================================================
#
# Dates cross in and out of AppleScript as seconds-since-epoch arithmetic
# (`(some date) - (date "1/1/1970")`), never as locale-formatted text, so
# this is safe regardless of the machine's date/time locale settings.

def _run_osascript(script: str) -> str:
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=_APPLESCRIPT_TIMEOUT_SECONDS, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"osascript failed: {result.stderr.strip()}")
    return result.stdout


def _fetch_events_via_applescript(start: datetime, end: datetime) -> List[Dict[str, Any]]:
    script = f"""
    set outputList to {{}}
    tell application "Calendar"
        repeat with cal in calendars
            set calEvents to (every event of cal whose start date ≥ (current date) and start date ≤ ((current date) + (86400 * {max(1, int((end - start).total_seconds() // 86400) + 1)})))
            repeat with evt in calEvents
                set startEpoch to ((start date of evt) - (date "1/1/1970"))
                set endEpoch to ((end date of evt) - (date "1/1/1970"))
                set outputList to outputList & {{(summary of evt as string) & "{_FIELD_SEP}" & (startEpoch as string) & "{_FIELD_SEP}" & (endEpoch as string) & "{_FIELD_SEP}" & (name of cal as string)}}
            end repeat
        end repeat
    end tell
    set AppleScript's text item delimiters to linefeed
    return outputList as string
    """
    try:
        output = _run_osascript(script)
    except (RuntimeError, FileNotFoundError):
        return []

    results: List[Dict[str, Any]] = []
    for line in output.splitlines():
        parts = line.split(_FIELD_SEP)
        if len(parts) != 4:
            continue
        title, start_epoch, end_epoch, calendar_name = parts
        try:
            event_start = datetime.fromtimestamp(float(start_epoch), tz=timezone.utc)
            event_end = datetime.fromtimestamp(float(end_epoch), tz=timezone.utc)
        except ValueError:
            continue
        results.append({
            "title": title, "start": event_start.isoformat(), "end": event_end.isoformat(),
            "calendar": calendar_name, "notes": None, "all_day": False,
        })
    return results


def _fetch_reminders_via_applescript() -> List[Dict[str, Any]]:
    script = f"""
    set outputList to {{}}
    tell application "Reminders"
        repeat with lst in lists
            set theReminders to (every reminder of lst whose completed is false)
            repeat with r in theReminders
                set dueStr to ""
                try
                    set dueStr to ((due date of r) - (date "1/1/1970")) as string
                end try
                set outputList to outputList & {{(name of r as string) & "{_FIELD_SEP}" & dueStr & "{_FIELD_SEP}" & (name of lst as string)}}
            end repeat
        end repeat
    end tell
    set AppleScript's text item delimiters to linefeed
    return outputList as string
    """
    try:
        output = _run_osascript(script)
    except (RuntimeError, FileNotFoundError):
        return []

    results: List[Dict[str, Any]] = []
    for line in output.splitlines():
        parts = line.split(_FIELD_SEP)
        if len(parts) != 3:
            continue
        title, due_epoch, list_name = parts
        due_iso = None
        if due_epoch.strip():
            try:
                due_iso = datetime.fromtimestamp(float(due_epoch), tz=timezone.utc).isoformat()
            except ValueError:
                due_iso = None
        results.append({"id": "", "title": title, "due": due_iso, "notes": None, "list": list_name})
    return results


# =============================================================================
# Public async API
# =============================================================================

async def today_events() -> List[Dict[str, Any]]:
    end = datetime.combine(date.today(), dt_time.max, tzinfo=timezone.utc)
    return await asyncio.to_thread(_fetch_events_sync, datetime.now(timezone.utc), end)


async def upcoming(days: int = 7) -> List[Dict[str, Any]]:
    start = datetime.now(timezone.utc)
    end = start + timedelta(days=max(1, days))
    return await asyncio.to_thread(_fetch_events_sync, start, end)


async def reminders_due(scope: str = "today") -> List[Dict[str, Any]]:
    reminders = await asyncio.to_thread(_fetch_reminders_sync)
    return _filter_reminders_by_scope(reminders, scope)


async def add_reminder(title: str, due: Optional[str] = None) -> Dict[str, Any]:
    return await asyncio.to_thread(_add_reminder_sync, title, due)


async def create_event(
    title: str, start: str, end: str, calendar_name: Optional[str] = None, notes: Optional[str] = None
) -> Dict[str, Any]:
    return await asyncio.to_thread(_create_event_sync, title, start, end, calendar_name, notes)
