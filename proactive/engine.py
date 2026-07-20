"""
Proactive Engine — scheduled jobs and rule evaluation that make Vesper
proactive instead of purely reactive.

Two independent mechanisms:
    - A cron-style scheduler (APScheduler) that fires configured jobs
      (e.g. a morning briefing) by publishing events onto the bus.
    - A rule evaluator subscribed to sensor events (AppFocusChangedEvent,
      UpcomingMeetingEvent) that emits ObservationEvent when a rule's
      threshold is met — gated by a per-kind cooldown so the same
      observation can never be raised more than once within its window.
      This mechanically enforces the passive call-out doctrine (see
      config/persona.md) rather than relying on prompting alone.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from bus.event_bus import EventBus, SubscriptionToken, get_event_bus
from schemas.events import (
    AppFocusChangedEvent,
    BriefingRequestedEvent,
    ObservationEvent,
    UpcomingMeetingEvent,
)
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONTEXT_SWITCH_THRESHOLD = 3
DEFAULT_CONTEXT_SWITCH_WINDOW_MIN = 30
DEFAULT_CONTEXT_SWITCH_COOLDOWN_MIN = 90
DEFAULT_MEETING_REMINDER_COOLDOWN_MIN = 10
DEFAULT_MORNING_BRIEFING_TIME = "08:30"
#: Above this many minutes-until-start, a meeting reminder is the sensor's
#: 15-minute checkpoint, not the near-term one this rule calls out on.
_MEETING_CALLOUT_MAX_MINUTES = 7

#: Maximum tracked (timestamp, app_name) entries; pruned by window anyway,
#: this just bounds memory if the window is misconfigured very large.
_SWITCH_HISTORY_MAXLEN = 500

Clock = Callable[[], datetime]


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


class ProactiveEngine:
    """Scheduled jobs + cooldown-gated rule evaluation."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict[str, Any]] = None,
        clock: Optional[Clock] = None,
    ):
        self._event_bus = event_bus or get_event_bus()
        self._config = config or {}
        self._clock: Clock = clock or _default_clock
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._subscriptions: List[SubscriptionToken] = []

        # Rolling window of (timestamp, app_name) switches into configured work apps.
        self._context_switch_history: Deque[Tuple[datetime, str]] = deque(maxlen=_SWITCH_HISTORY_MAXLEN)
        # Last-fired time per observation kind, for cooldown enforcement.
        self._last_fired: Dict[str, datetime] = {}

    def _get_config(self, key: str, default: Any = None) -> Any:
        """Dot-path lookup against the full app config."""
        value: Any = self._config
        for part in key.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(part)
            if value is None:
                return default
        return value

    async def start(self) -> None:
        """Subscribe to sensor events and start the cron scheduler."""
        self._subscriptions.append(
            self._event_bus.subscribe(AppFocusChangedEvent, self._on_app_focus_changed)
        )
        self._subscriptions.append(
            self._event_bus.subscribe(UpcomingMeetingEvent, self._on_upcoming_meeting)
        )

        self._scheduler = AsyncIOScheduler()
        self._register_scheduled_jobs()
        self._scheduler.start()
        logger.info("ProactiveEngine started")

    async def stop(self) -> None:
        """Unsubscribe and stop the cron scheduler."""
        for token in self._subscriptions:
            token.unsubscribe()
        self._subscriptions.clear()

        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
        logger.info("ProactiveEngine stopped")

    # =========================================================================
    # Scheduled jobs
    # =========================================================================

    def _register_scheduled_jobs(self) -> None:
        if self._scheduler is None:
            return
        if not self._get_config("proactive.schedule.morning_briefing.enabled", True):
            return

        time_str = str(
            self._get_config("proactive.schedule.morning_briefing.time", DEFAULT_MORNING_BRIEFING_TIME)
        )
        try:
            hour_str, minute_str = time_str.split(":")
            hour, minute = int(hour_str), int(minute_str)
        except ValueError:
            logger.warning(
                f"Invalid proactive.schedule.morning_briefing.time={time_str!r}; "
                f"using default {DEFAULT_MORNING_BRIEFING_TIME}"
            )
            hour, minute = 8, 30

        self._scheduler.add_job(
            self._fire_morning_briefing,
            CronTrigger(hour=hour, minute=minute),
            id="morning_briefing",
            replace_existing=True,
        )

    async def _fire_morning_briefing(self) -> None:
        await self._event_bus.emit(
            BriefingRequestedEvent(schedule_name="morning_briefing", source="ProactiveEngine")
        )

    # =========================================================================
    # Rule: context_switch
    # =========================================================================

    async def _on_app_focus_changed(self, event: AppFocusChangedEvent) -> None:
        if not self._get_config("proactive.rules.context_switch.enabled", True):
            return

        work_apps = self._get_config("proactive.rules.context_switch.work_apps", []) or []
        if work_apps and event.app_name not in work_apps:
            return

        now = self._clock()
        window_min = float(
            self._get_config("proactive.rules.context_switch.window_min", DEFAULT_CONTEXT_SWITCH_WINDOW_MIN)
        )
        threshold = int(
            self._get_config("proactive.rules.context_switch.threshold", DEFAULT_CONTEXT_SWITCH_THRESHOLD)
        )
        cooldown_min = float(
            self._get_config("proactive.rules.context_switch.cooldown_min", DEFAULT_CONTEXT_SWITCH_COOLDOWN_MIN)
        )

        self._context_switch_history.append((now, event.app_name))
        cutoff = now - timedelta(minutes=window_min)
        while self._context_switch_history and self._context_switch_history[0][0] < cutoff:
            self._context_switch_history.popleft()

        distinct_apps = {app for _, app in self._context_switch_history}
        if len(distinct_apps) < threshold:
            return

        if not self._cooldown_elapsed("context_switch", cooldown_min, now):
            return

        detail = (
            f"That's {len(distinct_apps)} different apps in the last {int(window_min)} minutes"
            f" ({', '.join(sorted(distinct_apps))})."
        )
        await self._emit_observation("context_switch", detail, now)

    # =========================================================================
    # Rule: meeting_reminder
    # =========================================================================

    async def _on_upcoming_meeting(self, event: UpcomingMeetingEvent) -> None:
        if not self._get_config("proactive.rules.meeting_reminder.enabled", True):
            return
        if event.minutes_until > _MEETING_CALLOUT_MAX_MINUTES:
            # Only the near-term (5-minute) checkpoint calls out proactively;
            # the sensor's 15-minute checkpoint is informational only.
            return

        now = self._clock()
        cooldown_min = float(
            self._get_config("proactive.rules.meeting_reminder.cooldown_min", DEFAULT_MEETING_REMINDER_COOLDOWN_MIN)
        )
        if not self._cooldown_elapsed("meeting_soon", cooldown_min, now):
            return

        detail = f"Your meeting '{event.title}' starts in {event.minutes_until} minutes."
        await self._emit_observation("meeting_soon", detail, now)

    # =========================================================================
    # Shared helpers
    # =========================================================================

    def _cooldown_elapsed(self, kind: str, cooldown_min: float, now: datetime) -> bool:
        last_fired = self._last_fired.get(kind)
        if last_fired is None:
            return True
        return (now - last_fired) >= timedelta(minutes=cooldown_min)

    async def _emit_observation(self, kind: str, detail: str, now: datetime) -> None:
        self._last_fired[kind] = now
        await self._event_bus.emit(ObservationEvent(kind=kind, detail=detail, source="ProactiveEngine"))
