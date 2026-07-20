"""
Tests for orchestrator/brain.py: the enable_voice_agent flag and
get_agents_status() (P08).

Not a full Brain.start() integration test -- that would require mocking
every agent plus the LLM router; those are exercised by each
component's own test file (test_planner.py, test_mcp_bridge.py,
test_inbox_sensor.py, ...). This just targets what P08 actually added.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agents.memory_agent import MemoryAgent
from bus.event_bus import EventBus
from orchestrator.brain import Brain
from tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _fresh_bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


class _FakeAgent:
    """Minimal BaseAgent stand-in -- avoids Mock's reserved `name` kwarg."""

    def __init__(self, name: str, healthy: bool = True):
        self._name = name
        self._healthy = healthy

    @property
    def name(self) -> str:
        return self._name

    def is_healthy(self) -> bool:
        return self._healthy


@pytest.mark.asyncio
async def test_enable_voice_agent_false_skips_voice_agent() -> None:
    brain = Brain(config={}, enable_voice_agent=False)
    await brain._register_default_agents()

    assert brain.get_agent("VoiceAgent") is None
    assert brain.get_agent("MemoryAgent") is not None


@pytest.mark.asyncio
async def test_enable_voice_agent_true_registers_voice_agent() -> None:
    brain = Brain(config={}, enable_voice_agent=True)
    await brain._register_default_agents()

    assert brain.get_agent("VoiceAgent") is not None


def test_get_agents_status_reports_health() -> None:
    brain = Brain(config={}, enable_voice_agent=False)
    brain.register_agent(_FakeAgent("Healthy", healthy=True))
    brain.register_agent(_FakeAgent("Unhealthy", healthy=False))

    statuses = brain.get_agents_status()

    by_name = {s["name"]: s for s in statuses}
    assert by_name["Healthy"]["healthy"] is True
    assert by_name["Unhealthy"]["healthy"] is False
    assert by_name["Healthy"]["error_count"] == 0


def test_get_router_returns_configured_router() -> None:
    brain = Brain(config={}, enable_voice_agent=False)
    assert brain.get_router() is not None


# =============================================================================
# Semantic memory retrieval per turn (P09)
# =============================================================================

@pytest.mark.asyncio
async def test_retrieve_relevant_memories_returns_semantic_matches(tmp_path: Path) -> None:
    brain = Brain(config={}, enable_voice_agent=False)
    memory_agent = MemoryAgent(
        event_bus=brain._event_bus,
        config={
            "memory": {
                "sqlite": {"database_path": str(tmp_path / "memory.db")},
                "vector_store": {"enabled": True, "persist_directory": str(tmp_path / "chroma")},
            }
        },
    )
    await memory_agent.start()
    brain.register_agent(memory_agent)

    try:
        await memory_agent.index_memory_text(
            text="The gym slot is non-negotiable, always protect it.",
            intent="reflection_preference",
            metadata={"kind": "preference"},
        )

        memories = await brain._retrieve_relevant_memories("plan my day around the gym")

        assert any("gym" in m.lower() for m in memories)
    finally:
        await memory_agent.stop()


@pytest.mark.asyncio
async def test_retrieve_relevant_memories_without_memory_agent_returns_empty() -> None:
    brain = Brain(config={}, enable_voice_agent=False)
    memories = await brain._retrieve_relevant_memories("anything")
    assert memories == []


# =============================================================================
# forget_memory tool (P09)
# =============================================================================

@pytest.mark.asyncio
async def test_forget_memory_tool_deletes_matching_memory(tmp_path: Path) -> None:
    brain = Brain(config={}, enable_voice_agent=False)
    memory_agent = MemoryAgent(
        event_bus=brain._event_bus,
        config={
            "memory": {
                "sqlite": {"database_path": str(tmp_path / "memory.db")},
                "vector_store": {"enabled": True, "persist_directory": str(tmp_path / "chroma")},
            }
        },
    )
    await memory_agent.start()
    brain.register_agent(memory_agent)

    try:
        await memory_agent.index_memory_text(
            text="The gym slot is non-negotiable, always protect it.",
            intent="reflection_preference",
            metadata={"kind": "preference"},
        )

        handler = brain._make_forget_memory_handler()
        result = await handler({"query": "gym"}, {})

        assert "Forgotten:" in result
        assert "gym" in result.lower()

        after = await memory_agent.semantic_retrieve(query="gym", top_k=5)
        assert not any("gym" in m["text"].lower() for m in after)
    finally:
        await memory_agent.stop()


@pytest.mark.asyncio
async def test_forget_memory_tool_no_match_is_graceful(tmp_path: Path) -> None:
    brain = Brain(config={}, enable_voice_agent=False)
    memory_agent = MemoryAgent(
        event_bus=brain._event_bus,
        config={
            "memory": {
                "sqlite": {"database_path": str(tmp_path / "memory.db")},
                "vector_store": {"enabled": True, "persist_directory": str(tmp_path / "chroma")},
            }
        },
    )
    await memory_agent.start()
    brain.register_agent(memory_agent)

    try:
        handler = brain._make_forget_memory_handler()
        result = await handler({"query": "something that was never stored"}, {})
        assert "couldn't find" in result.lower()
    finally:
        await memory_agent.stop()


@pytest.mark.asyncio
async def test_forget_memory_tool_without_memory_agent_is_graceful() -> None:
    brain = Brain(config={}, enable_voice_agent=False)
    handler = brain._make_forget_memory_handler()
    result = await handler({"query": "anything"}, {})
    assert "isn't available" in result.lower()


def test_register_memory_tools_uses_confirm_tier() -> None:
    fake_registry = ToolRegistry()
    brain = Brain(config={}, enable_voice_agent=False)

    with patch("orchestrator.brain.get_registry", return_value=fake_registry):
        brain._register_memory_tools()

    spec = fake_registry.get("forget_memory")
    assert spec is not None
    assert spec.tier == "confirm"
