"""
D1 regression: the event bus MUST be a single process-wide instance.

If two EventBus objects ever exist at once, events published on one are
invisible to handlers subscribed on the other — the exact split-bus bug the
PF0 audit flagged. These tests pin the singleton guarantee so it cannot
silently regress.
"""

from __future__ import annotations

import pytest

from bus.event_bus import EventBus, get_event_bus


@pytest.fixture(autouse=True)
def _fresh_bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


def test_get_event_bus_returns_same_object_every_call():
    a = get_event_bus()
    b = get_event_bus()
    assert a is b
    assert id(a) == id(b)


def test_direct_construction_returns_the_same_singleton():
    """`EventBus()` and `get_event_bus()` must be the identical object."""
    from_accessor = get_event_bus()
    from_constructor = EventBus()
    assert from_constructor is from_accessor


def test_reset_instance_rebuilds_a_single_consistent_instance():
    """After reset, the accessor and constructor still agree on one object."""
    first = get_event_bus()
    EventBus.reset_instance()
    second = get_event_bus()
    assert second is not first          # reset really made a new one
    assert EventBus() is second          # ...and everyone still shares it


@pytest.mark.asyncio
async def test_publish_on_accessor_reaches_handler_subscribed_via_constructor():
    """The functional guarantee: no split bus. A handler subscribed through
    one reference sees an event published through another reference."""
    from schemas.events import BaseEvent

    class _Ping(BaseEvent):
        pass

    received = []

    EventBus().subscribe(_Ping, lambda e: received.append(e))
    await get_event_bus().publish(_Ping(source="test"))

    assert len(received) == 1
