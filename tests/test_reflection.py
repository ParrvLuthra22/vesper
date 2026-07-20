"""Tests for proactive/reflection.py: extraction format + memory writes (P09)."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from llm.types import LLMResponse, RouterError
from proactive.reflection import (
    ReflectionItem,
    _parse_reflection_output,
    _render_transcript,
    run_reflection,
)


def _mock_router(text: str) -> AsyncMock:
    router = AsyncMock()
    router.complete = AsyncMock(return_value=LLMResponse(text=text))
    return router


class _FakeMemoryAgent:
    def __init__(self) -> None:
        self.indexed: List[Dict[str, Any]] = []

    async def index_memory_text(
        self, text: str, memory_type: str, intent: str, metadata: Dict[str, Any], salience: float
    ) -> int:
        self.indexed.append(
            {"text": text, "memory_type": memory_type, "intent": intent, "metadata": metadata, "salience": salience}
        )
        return 1


# =============================================================================
# Transcript rendering
# =============================================================================

def test_render_transcript_includes_user_and_response() -> None:
    turns = [{"user": "what's my day look like", "response": "You have one meeting, Sir."}]
    text = _render_transcript(turns)
    assert "User: what's my day look like" in text
    assert "Vesper: You have one meeting, Sir." in text


# =============================================================================
# Extraction-format parsing
# =============================================================================

def test_parse_reflection_output_well_formed() -> None:
    text = (
        "preference: The gym slot is non-negotiable, always protect it.\n"
        "fact: User works at Newton School of Technology.\n"
        "pattern: User frequently gets internship offer emails."
    )
    items = _parse_reflection_output(text)

    assert items == [
        ReflectionItem(kind="preference", text="The gym slot is non-negotiable, always protect it."),
        ReflectionItem(kind="fact", text="User works at Newton School of Technology."),
        ReflectionItem(kind="pattern", text="User frequently gets internship offer emails."),
    ]


def test_parse_reflection_output_skips_malformed_lines() -> None:
    text = (
        "preference: A valid one.\n"
        "just some prose with no kind prefix\n"
        "notakind: should be skipped\n"
        "fact: Another valid one."
    )
    items = _parse_reflection_output(text)

    assert [i.text for i in items] == ["A valid one.", "Another valid one."]


def test_parse_reflection_output_empty_response() -> None:
    assert _parse_reflection_output("") == []
    assert _parse_reflection_output("   \n  ") == []


def test_parse_reflection_output_caps_at_max_items() -> None:
    text = "\n".join(f"fact: item {i}" for i in range(10))
    items = _parse_reflection_output(text)
    assert len(items) == 5


# =============================================================================
# run_reflection end-to-end
# =============================================================================

@pytest.mark.asyncio
async def test_run_reflection_stores_extracted_items() -> None:
    router = _mock_router("preference: The gym slot is non-negotiable, always protect it.")
    memory_agent = _FakeMemoryAgent()
    turns = [{"user": "the gym slot is non-negotiable, always protect it", "response": "Understood, Sir."}]

    items = await run_reflection(turns=turns, router=router, memory_agent=memory_agent)

    assert len(items) == 1
    assert items[0].kind == "preference"
    assert len(memory_agent.indexed) == 1
    stored = memory_agent.indexed[0]
    assert stored["text"] == "The gym slot is non-negotiable, always protect it."
    assert stored["memory_type"] == "long_term"
    assert stored["metadata"]["kind"] == "preference"
    assert stored["metadata"]["source"] == "reflection"


@pytest.mark.asyncio
async def test_run_reflection_empty_turns_returns_empty() -> None:
    router = _mock_router("preference: should never be called")
    memory_agent = _FakeMemoryAgent()

    items = await run_reflection(turns=[], router=router, memory_agent=memory_agent)

    assert items == []
    assert memory_agent.indexed == []
    router.complete.assert_not_called()


@pytest.mark.asyncio
async def test_run_reflection_no_memory_agent_returns_empty() -> None:
    router = _mock_router("preference: should never be called")
    turns = [{"user": "hello", "response": "Good evening, Sir."}]

    items = await run_reflection(turns=turns, router=router, memory_agent=None)

    assert items == []
    router.complete.assert_not_called()


@pytest.mark.asyncio
async def test_run_reflection_router_error_returns_empty() -> None:
    router = AsyncMock()
    router.complete = AsyncMock(return_value=RouterError(user_message="down"))
    memory_agent = _FakeMemoryAgent()
    turns = [{"user": "hello", "response": "Good evening, Sir."}]

    items = await run_reflection(turns=turns, router=router, memory_agent=memory_agent)

    assert items == []
    assert memory_agent.indexed == []


@pytest.mark.asyncio
async def test_run_reflection_nothing_worth_remembering_returns_empty() -> None:
    router = _mock_router("")  # model correctly found nothing worth remembering
    memory_agent = _FakeMemoryAgent()
    turns = [{"user": "what time is it", "response": "Half past three, Sir."}]

    items = await run_reflection(turns=turns, router=router, memory_agent=memory_agent)

    assert items == []
    assert memory_agent.indexed == []
