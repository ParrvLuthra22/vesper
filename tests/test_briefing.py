"""Tests for proactive/briefing.py: assembly with mocked data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agents.memory_agent import MemoryAgent
from bus.event_bus import EventBus
from proactive.briefing import (
    assemble_briefing_text,
    gather_calendar,
    gather_inbox_triage,
    render_briefing_data,
)
from tools.registry import ToolRegistry, ToolSpec


@pytest.fixture(autouse=True)
def _fresh_bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


def _make_registry_with_list_unread(messages: List[Dict[str, Any]]) -> ToolRegistry:
    registry = ToolRegistry()

    async def handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        return json.dumps(messages)

    registry.register(ToolSpec(name="list_unread", description="d", handler=handler, tier="safe"))
    return registry


class _FakeCalendarSensor:
    def __init__(self, events: List[Any]):
        self._events = events

    async def fetch_todays_events(self):
        return self._events


async def _started_memory_agent(tmp_path: Path) -> MemoryAgent:
    bus = EventBus()
    agent = MemoryAgent(
        event_bus=bus,
        config={"memory": {"sqlite": {"database_path": str(tmp_path / "memory.db")}}},
    )
    await agent.start()
    return agent


# =============================================================================
# gather_inbox_triage
# =============================================================================

@pytest.mark.asyncio
async def test_gather_inbox_triage_returns_messages() -> None:
    messages = [
        {"id": "1", "sender": "alice@x.com", "subject": "Invoice", "snippet": "Please pay..."},
        {"id": "2", "sender": "bob@x.com", "subject": "Meeting", "snippet": "Let's sync..."},
    ]
    registry = _make_registry_with_list_unread(messages)

    result = await gather_inbox_triage(registry)

    assert result["available"] is True
    assert result["count"] == 2
    assert result["messages"] == messages


@pytest.mark.asyncio
async def test_gather_inbox_triage_unavailable_when_tool_missing() -> None:
    registry = ToolRegistry()  # no list_unread registered (Gmail not enabled)

    result = await gather_inbox_triage(registry)

    assert result["available"] is False
    assert result["count"] == 0
    assert result["messages"] == []


# =============================================================================
# gather_calendar
# =============================================================================

@pytest.mark.asyncio
async def test_gather_calendar_with_sensor() -> None:
    from datetime import datetime

    sensor = _FakeCalendarSensor([("Standup", datetime(2026, 1, 1, 9, 0))])

    result = await gather_calendar(sensor)

    assert result == [{"title": "Standup", "start_time": "2026-01-01T09:00:00"}]


@pytest.mark.asyncio
async def test_gather_calendar_none_sensor_returns_empty() -> None:
    result = await gather_calendar(None)
    assert result == []


# =============================================================================
# render_briefing_data
# =============================================================================

def test_render_briefing_data_includes_all_sections() -> None:
    inbox = {"available": True, "count": 2, "messages": [{"id": "1", "sender": "a", "subject": "S", "snippet": "..."}]}
    calendar_events = [{"title": "Standup", "start_time": "2026-01-01T09:00:00"}]

    text = render_briefing_data(inbox, calendar_events, carried_over_count=0)

    assert "Unread email: 2 total." in text
    assert "Standup" in text
    assert "carried" not in text.lower()


def test_render_briefing_data_includes_carried_over_note() -> None:
    inbox = {"available": False, "count": 0, "messages": []}
    text = render_briefing_data(inbox, [], carried_over_count=2)

    assert "2 of today's top unread items" in text


# =============================================================================
# assemble_briefing_text end-to-end (mocked registry/sensor, real memory)
# =============================================================================

@pytest.mark.asyncio
async def test_assemble_briefing_text_end_to_end(tmp_path: Path) -> None:
    messages = [{"id": "1", "sender": "a@x.com", "subject": "Urgent", "snippet": "..."}]
    registry = _make_registry_with_list_unread(messages)
    calendar_sensor = _FakeCalendarSensor([])
    memory_agent = await _started_memory_agent(tmp_path)

    text = await assemble_briefing_text(registry, calendar_sensor, memory_agent)

    assert "Unread email: 1 total." in text
    assert "Urgent" in text
    assert "Today's calendar: nothing scheduled" in text


@pytest.mark.asyncio
async def test_assemble_briefing_text_flags_carried_over_items(tmp_path: Path) -> None:
    memory_agent = await _started_memory_agent(tmp_path)
    calendar_sensor = _FakeCalendarSensor([])

    day_one_messages = [{"id": "1", "sender": "a", "subject": "Still open", "snippet": "..."}]
    registry_day_one = _make_registry_with_list_unread(day_one_messages)
    await assemble_briefing_text(registry_day_one, calendar_sensor, memory_agent)

    # Same top message is still unread the next day.
    registry_day_two = _make_registry_with_list_unread(day_one_messages)
    text_day_two = await assemble_briefing_text(registry_day_two, calendar_sensor, memory_agent)

    assert "also in" in text_day_two and "yesterday's briefing" in text_day_two


@pytest.mark.asyncio
async def test_assemble_briefing_text_no_carried_over_for_new_items(tmp_path: Path) -> None:
    memory_agent = await _started_memory_agent(tmp_path)
    calendar_sensor = _FakeCalendarSensor([])

    registry_day_one = _make_registry_with_list_unread([{"id": "1", "sender": "a", "subject": "S1", "snippet": ""}])
    await assemble_briefing_text(registry_day_one, calendar_sensor, memory_agent)

    registry_day_two = _make_registry_with_list_unread([{"id": "2", "sender": "b", "subject": "S2", "snippet": ""}])
    text_day_two = await assemble_briefing_text(registry_day_two, calendar_sensor, memory_agent)

    assert "yesterday's briefing" not in text_day_two
