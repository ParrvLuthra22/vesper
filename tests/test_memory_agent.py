"""Tests for agents/memory_agent.py: the "forget X" flow (P09)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.memory_agent import MemoryAgent
from bus.event_bus import EventBus


@pytest.fixture(autouse=True)
def _fresh_bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


async def _started_memory_agent(tmp_path: Path, semantic: bool = True) -> MemoryAgent:
    bus = EventBus()
    memory_config = {"sqlite": {"database_path": str(tmp_path / "memory.db")}}
    if semantic:
        memory_config["vector_store"] = {
            "enabled": True,
            "persist_directory": str(tmp_path / "chroma"),
            "collection_name": "test_memory",
        }
    agent = MemoryAgent(event_bus=bus, config={"memory": memory_config})
    await agent.start()
    return agent


@pytest.mark.asyncio
async def test_forget_deletes_matching_memory_only(tmp_path: Path) -> None:
    agent = await _started_memory_agent(tmp_path)
    try:
        await agent.index_memory_text(
            text="The gym slot is non-negotiable, always protect it.",
            intent="reflection_preference",
            metadata={"kind": "preference"},
        )
        await agent.index_memory_text(
            text="User prefers dark roast coffee.",
            intent="reflection_fact",
            metadata={"kind": "fact"},
        )

        before = await agent.semantic_retrieve(query="gym", top_k=5)
        assert any("gym" in m["text"].lower() for m in before)

        forgotten = await agent.forget(query="gym", top_k=1)
        assert forgotten
        assert any("gym" in t.lower() for t in forgotten)

        after = await agent.semantic_retrieve(query="gym", top_k=5)
        assert not any("gym" in m["text"].lower() for m in after)

        # The unrelated memory must survive.
        remaining = await agent.semantic_retrieve(query="coffee", top_k=5)
        assert any("coffee" in m["text"].lower() for m in remaining)
    finally:
        await agent.stop()


@pytest.mark.asyncio
async def test_forget_without_semantic_memory_returns_empty(tmp_path: Path) -> None:
    agent = await _started_memory_agent(tmp_path, semantic=False)
    try:
        forgotten = await agent.forget(query="anything")
        assert forgotten == []
    finally:
        await agent.stop()
