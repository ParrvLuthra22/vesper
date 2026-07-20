"""
Tests for proactive/engine.py: rule triggering thresholds and cooldown
suppression.

A FakeClock lets tests control "now" directly instead of sleeping for
real minutes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from bus.event_bus import EventBus
from proactive.engine import ProactiveEngine
from schemas.events import AppFocusChangedEvent, ObservationEvent, UpcomingMeetingEvent


@pytest.fixture(autouse=True)
def _fresh_bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


class FakeClock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


def _config(**overrides: Any) -> Dict[str, Any]:
    context_switch = {
        "enabled": True,
        "threshold": 3,
        "window_min": 30,
        "cooldown_min": 90,
        "work_apps": [],  # empty = no filter, unless a test overrides it
    }
    meeting_reminder = {"enabled": True, "cooldown_min": 10}
    context_switch.update(overrides.get("context_switch", {}))
    meeting_reminder.update(overrides.get("meeting_reminder", {}))
    return {
        "proactive": {
            "rules": {"context_switch": context_switch, "meeting_reminder": meeting_reminder},
            "schedule": {"morning_briefing": {"enabled": False}},
        }
    }


async def _make_engine(bus: EventBus, clock: FakeClock, **overrides: Any) -> ProactiveEngine:
    engine = ProactiveEngine(event_bus=bus, config=_config(**overrides), clock=clock)
    await engine.start()
    return engine


def _observation_collector(bus: EventBus) -> List[ObservationEvent]:
    observations: List[ObservationEvent] = []

    async def on_observation(event: ObservationEvent) -> None:
        observations.append(event)

    bus.subscribe(ObservationEvent, on_observation)
    return observations


# =============================================================================
# context_switch: threshold
# =============================================================================

@pytest.mark.asyncio
async def test_context_switch_fires_at_threshold() -> None:
    bus = EventBus()
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    observations = _observation_collector(bus)
    engine = await _make_engine(bus, clock)

    await bus.emit(AppFocusChangedEvent(app_name="A", previous_app=""))
    assert observations == []
    await bus.emit(AppFocusChangedEvent(app_name="B", previous_app="A"))
    assert observations == []
    await bus.emit(AppFocusChangedEvent(app_name="C", previous_app="B"))

    assert len(observations) == 1
    assert observations[0].kind == "context_switch"
    assert "3" in observations[0].detail

    await engine.stop()


@pytest.mark.asyncio
async def test_context_switch_does_not_fire_below_threshold() -> None:
    bus = EventBus()
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    observations = _observation_collector(bus)
    engine = await _make_engine(bus, clock)

    await bus.emit(AppFocusChangedEvent(app_name="A", previous_app=""))
    await bus.emit(AppFocusChangedEvent(app_name="B", previous_app="A"))
    # Repeated switches back to an already-seen app don't add a new distinct app.
    await bus.emit(AppFocusChangedEvent(app_name="A", previous_app="B"))

    assert observations == []
    await engine.stop()


@pytest.mark.asyncio
async def test_context_switch_ignores_non_work_apps() -> None:
    bus = EventBus()
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    observations = _observation_collector(bus)
    engine = await _make_engine(bus, clock, context_switch={"work_apps": ["A", "B", "C"]})

    await bus.emit(AppFocusChangedEvent(app_name="A", previous_app=""))
    await bus.emit(AppFocusChangedEvent(app_name="Spotify", previous_app="A"))  # not a work app
    await bus.emit(AppFocusChangedEvent(app_name="B", previous_app="Spotify"))
    assert observations == []  # only 2 distinct work-app switches so far (A, B)

    await bus.emit(AppFocusChangedEvent(app_name="C", previous_app="B"))
    assert len(observations) == 1

    await engine.stop()


@pytest.mark.asyncio
async def test_context_switch_respects_window() -> None:
    bus = EventBus()
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    observations = _observation_collector(bus)
    engine = await _make_engine(bus, clock, context_switch={"window_min": 30})

    await bus.emit(AppFocusChangedEvent(app_name="A", previous_app=""))
    clock.advance(minutes=40)  # older than the 30-minute window
    await bus.emit(AppFocusChangedEvent(app_name="B", previous_app="A"))
    await bus.emit(AppFocusChangedEvent(app_name="C", previous_app="B"))

    # "A" should have aged out of the window, leaving only B and C (2 < threshold 3).
    assert observations == []

    await engine.stop()


@pytest.mark.asyncio
async def test_context_switch_disabled_via_config() -> None:
    bus = EventBus()
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    observations = _observation_collector(bus)
    engine = await _make_engine(bus, clock, context_switch={"enabled": False})

    for app in ("A", "B", "C", "D"):
        await bus.emit(AppFocusChangedEvent(app_name=app, previous_app=""))

    assert observations == []
    await engine.stop()


# =============================================================================
# context_switch: cooldown suppression
# =============================================================================

@pytest.mark.asyncio
async def test_context_switch_cooldown_suppresses_repeat_then_fires_again() -> None:
    bus = EventBus()
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    observations = _observation_collector(bus)
    engine = await _make_engine(bus, clock, context_switch={"cooldown_min": 90, "window_min": 30})

    await bus.emit(AppFocusChangedEvent(app_name="A", previous_app=""))
    await bus.emit(AppFocusChangedEvent(app_name="B", previous_app="A"))
    await bus.emit(AppFocusChangedEvent(app_name="C", previous_app="B"))
    assert len(observations) == 1  # rule fires

    clock.advance(minutes=1)
    await bus.emit(AppFocusChangedEvent(app_name="D", previous_app="C"))
    assert len(observations) == 1  # still within the 90-min cooldown: suppressed

    # Advance past the cooldown. The window has also moved on, so a fresh
    # run of 3 distinct switches is needed to re-meet the threshold.
    clock.advance(minutes=95)
    await bus.emit(AppFocusChangedEvent(app_name="E", previous_app="D"))
    await bus.emit(AppFocusChangedEvent(app_name="F", previous_app="E"))
    assert len(observations) == 1
    await bus.emit(AppFocusChangedEvent(app_name="G", previous_app="F"))

    assert len(observations) == 2  # cooldown elapsed, threshold met again: fires

    await engine.stop()


# =============================================================================
# meeting_reminder
# =============================================================================

@pytest.mark.asyncio
async def test_meeting_reminder_fires_near_five_minutes() -> None:
    bus = EventBus()
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    observations = _observation_collector(bus)
    engine = await _make_engine(bus, clock)

    await bus.emit(
        UpcomingMeetingEvent(title="1:1 with manager", start_time=clock.now, minutes_until=5)
    )

    assert len(observations) == 1
    assert observations[0].kind == "meeting_soon"
    assert "1:1 with manager" in observations[0].detail
    assert "5" in observations[0].detail

    await engine.stop()


@pytest.mark.asyncio
async def test_meeting_reminder_ignores_fifteen_minute_checkpoint() -> None:
    bus = EventBus()
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    observations = _observation_collector(bus)
    engine = await _make_engine(bus, clock)

    await bus.emit(UpcomingMeetingEvent(title="Standup", start_time=clock.now, minutes_until=15))

    assert observations == []
    await engine.stop()


@pytest.mark.asyncio
async def test_meeting_reminder_cooldown_suppresses_repeat() -> None:
    bus = EventBus()
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    observations = _observation_collector(bus)
    engine = await _make_engine(bus, clock, meeting_reminder={"cooldown_min": 10})

    await bus.emit(UpcomingMeetingEvent(title="Standup", start_time=clock.now, minutes_until=5))
    assert len(observations) == 1

    clock.advance(minutes=2)
    await bus.emit(UpcomingMeetingEvent(title="Standup", start_time=clock.now, minutes_until=4))
    assert len(observations) == 1  # still within cooldown

    clock.advance(minutes=15)
    await bus.emit(UpcomingMeetingEvent(title="Next meeting", start_time=clock.now, minutes_until=5))
    assert len(observations) == 2

    await engine.stop()


@pytest.mark.asyncio
async def test_meeting_reminder_disabled_via_config() -> None:
    bus = EventBus()
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    observations = _observation_collector(bus)
    engine = await _make_engine(bus, clock, meeting_reminder={"enabled": False})

    await bus.emit(UpcomingMeetingEvent(title="Standup", start_time=clock.now, minutes_until=5))

    assert observations == []
    await engine.stop()
