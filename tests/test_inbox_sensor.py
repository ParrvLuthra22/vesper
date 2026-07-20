"""Tests for sensors/inbox_sensor.py: surge detection with cooldown."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from bus.event_bus import EventBus
from schemas.events import ObservationEvent
from sensors.inbox_sensor import InboxSensor
from tools.registry import ToolRegistry, ToolSpec


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


def _registry_with_unread_count(counts: List[int]) -> ToolRegistry:
    """Each call to list_unread returns the next count in `counts` as that many fake messages."""
    registry = ToolRegistry()
    remaining = list(counts)

    async def handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        n = remaining.pop(0) if remaining else counts[-1]
        return json.dumps([{"id": str(i)} for i in range(n)])

    registry.register(ToolSpec(name="list_unread", description="d", handler=handler, tier="safe"))
    return registry


def _observation_collector(bus: EventBus) -> List[ObservationEvent]:
    observations: List[ObservationEvent] = []

    async def on_observation(event: ObservationEvent) -> None:
        observations.append(event)

    bus.subscribe(ObservationEvent, on_observation)
    return observations


CONFIG = {"sensors": {"inbox": {"enabled": True, "surge_threshold": 5, "surge_cooldown_min": 60}}}


@pytest.mark.asyncio
async def test_no_observation_on_first_poll() -> None:
    bus = EventBus()
    observations = _observation_collector(bus)
    registry = _registry_with_unread_count([3])
    sensor = InboxSensor(event_bus=bus, config=CONFIG, registry=registry)

    await sensor.poll()

    assert observations == []


@pytest.mark.asyncio
async def test_no_observation_below_threshold() -> None:
    bus = EventBus()
    observations = _observation_collector(bus)
    registry = _registry_with_unread_count([3, 3, 6])  # jump of 3, below threshold 5
    sensor = InboxSensor(event_bus=bus, config=CONFIG, registry=registry)

    await sensor.poll()
    await sensor.poll()

    assert observations == []


@pytest.mark.asyncio
async def test_surge_fires_at_threshold() -> None:
    bus = EventBus()
    observations = _observation_collector(bus)
    registry = _registry_with_unread_count([3, 8])  # jump of 5 == threshold
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    sensor = InboxSensor(event_bus=bus, config=CONFIG, registry=registry, clock=clock)

    await sensor.poll()  # baseline: 3
    await sensor.poll()  # 8, jump of 5 -> fires

    assert len(observations) == 1
    assert observations[0].kind == "inbox_surge"
    assert "5" in observations[0].detail


@pytest.mark.asyncio
async def test_surge_cooldown_suppresses_repeat_then_fires_again() -> None:
    bus = EventBus()
    observations = _observation_collector(bus)
    registry = _registry_with_unread_count([0, 10, 15, 30])
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    sensor = InboxSensor(event_bus=bus, config=CONFIG, registry=registry, clock=clock)

    await sensor.poll()  # baseline: 0
    await sensor.poll()  # 10, jump 10 -> fires
    assert len(observations) == 1

    clock.advance(minutes=10)
    await sensor.poll()  # 15, jump 5 -> within 60 min cooldown, suppressed
    assert len(observations) == 1

    clock.advance(minutes=61)
    await sensor.poll()  # 30, jump 15 -> cooldown elapsed, fires again
    assert len(observations) == 2


@pytest.mark.asyncio
async def test_no_op_when_list_unread_not_registered() -> None:
    bus = EventBus()
    observations = _observation_collector(bus)
    registry = ToolRegistry()  # Gmail not enabled -> no list_unread tool
    sensor = InboxSensor(event_bus=bus, config=CONFIG, registry=registry)

    await sensor.poll()
    await sensor.poll()

    assert observations == []
