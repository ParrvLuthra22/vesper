"""Tests for proactive/day_planner.py: day-plan assembly with mocked sources (P09)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from proactive.day_planner import (
    assemble_day_plan_text,
    gather_calendar_events,
    gather_email_pressure,
    gather_notion_tasks,
    gather_protected_blocks,
    gather_reminders,
    make_plan_my_day_handler,
    render_day_plan_data,
)
from tools.registry import ToolRegistry, ToolSpec


def _registry_with(name: str, result: Any) -> ToolRegistry:
    registry = ToolRegistry()

    async def handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        return result if isinstance(result, str) else json.dumps(result)

    registry.register(ToolSpec(name=name, description="d", handler=handler, tier="safe"))
    return registry


class _FakeMemoryAgent:
    def __init__(self, memories: List[str]):
        self._memories = [{"text": m} for m in memories]

    async def semantic_retrieve(self, query: str, top_k: int = 8, **kwargs: Any):
        return self._memories[:top_k]


# =============================================================================
# Individual gatherers
# =============================================================================

@pytest.mark.asyncio
async def test_gather_calendar_events_parses_tool_result() -> None:
    events = [{"title": "Standup", "start": "2026-07-21T09:00:00", "end": "2026-07-21T09:15:00"}]
    registry = _registry_with("today_events", events)

    result = await gather_calendar_events(registry)

    assert result == events


@pytest.mark.asyncio
async def test_gather_calendar_events_missing_tool_returns_empty() -> None:
    result = await gather_calendar_events(ToolRegistry())
    assert result == []


@pytest.mark.asyncio
async def test_gather_reminders_parses_tool_result() -> None:
    reminders = [{"id": "1", "title": "Buy milk", "due": "2026-07-21T18:00:00"}]
    registry = _registry_with("reminders_due", reminders)

    result = await gather_reminders(registry)

    assert result == reminders


@pytest.mark.asyncio
async def test_gather_notion_tasks_parses_tool_result() -> None:
    tasks = [{"id": "abc", "url": "https://notion.so/abc", "properties": {"Name": "Ship P09"}}]
    registry = _registry_with("notion_get_database_rows", tasks)

    result = await gather_notion_tasks(registry, db_id="tasks")

    assert result == tasks


@pytest.mark.asyncio
async def test_gather_notion_tasks_unavailable_returns_empty() -> None:
    result = await gather_notion_tasks(ToolRegistry())
    assert result == []


@pytest.mark.asyncio
async def test_gather_email_pressure_parses_int() -> None:
    registry = _registry_with("unread_count", "42")
    result = await gather_email_pressure(registry)
    assert result == 42


@pytest.mark.asyncio
async def test_gather_email_pressure_unavailable_returns_none() -> None:
    result = await gather_email_pressure(ToolRegistry())
    assert result is None


@pytest.mark.asyncio
async def test_gather_protected_blocks_uses_semantic_retrieve() -> None:
    memory_agent = _FakeMemoryAgent(["The gym slot is non-negotiable, always protect it."])
    result = await gather_protected_blocks(memory_agent)
    assert result == ["The gym slot is non-negotiable, always protect it."]


@pytest.mark.asyncio
async def test_gather_protected_blocks_none_agent_returns_empty() -> None:
    result = await gather_protected_blocks(None)
    assert result == []


# =============================================================================
# Rendering
# =============================================================================

def test_render_day_plan_data_includes_all_sections() -> None:
    text = render_day_plan_data(
        calendar_events=[{"title": "Standup", "start": "09:00", "end": "09:15"}],
        reminders=[{"title": "Buy milk", "due": "18:00"}],
        notion_tasks=[{"id": "1", "properties": {"Name": "Ship P09"}}],
        unread_count=5,
        protected_blocks=["The gym slot is non-negotiable, always protect it."],
    )

    assert "Standup" in text
    assert "Buy milk" in text
    assert "Ship P09" in text
    assert "Unread email: 5" in text
    assert "gym slot is non-negotiable" in text
    assert "route around" in text.lower()


def test_render_day_plan_data_handles_all_empty() -> None:
    text = render_day_plan_data(
        calendar_events=[], reminders=[], notion_tasks=[], unread_count=None, protected_blocks=[],
    )

    assert "nothing scheduled" in text.lower()
    assert "none" in text.lower()
    assert "unavailable" in text.lower()


# =============================================================================
# End-to-end assembly + tool handler
# =============================================================================

@pytest.mark.asyncio
async def test_assemble_day_plan_text_end_to_end() -> None:
    registry = ToolRegistry()

    async def today_events_handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        return json.dumps([{"title": "Standup", "start": "2026-07-21T09:00:00", "end": None}])

    async def reminders_handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        return json.dumps([])

    async def unread_handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        return "3"

    registry.register(ToolSpec(name="today_events", description="d", handler=today_events_handler, tier="safe"))
    registry.register(ToolSpec(name="reminders_due", description="d", handler=reminders_handler, tier="safe"))
    registry.register(ToolSpec(name="unread_count", description="d", handler=unread_handler, tier="safe"))

    memory_agent = _FakeMemoryAgent(["The gym slot is non-negotiable, always protect it."])

    text = await assemble_day_plan_text(registry, memory_agent)

    assert "Standup" in text
    assert "Reminders due: none" in text
    assert "Unread email: 3" in text
    assert "gym slot is non-negotiable" in text


@pytest.mark.asyncio
async def test_assemble_day_plan_text_degrades_gracefully_with_no_sources() -> None:
    text = await assemble_day_plan_text(ToolRegistry(), None)

    assert "nothing scheduled" in text.lower()
    assert "Notion tasks: none available." in text


@pytest.mark.asyncio
async def test_plan_my_day_handler_returns_assembled_text() -> None:
    registry = _registry_with("today_events", [{"title": "Standup", "start": "09:00", "end": None}])
    handler = make_plan_my_day_handler(registry=registry, memory_agent=None)

    result = await handler({}, {})

    assert "Standup" in result
