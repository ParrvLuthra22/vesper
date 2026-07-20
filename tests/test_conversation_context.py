"""
Tests for orchestrator/brain.py's ConversationContext observation queue:
Proactive Engine observations should be offered to the Planner exactly
once, then consumed regardless of whether the model chose to voice them.
"""

from __future__ import annotations

from orchestrator.brain import ConversationContext


def test_add_and_get_pending_observations() -> None:
    context = ConversationContext()

    context.add_observation(kind="context_switch", detail="3 apps in 30 minutes.", observation_id="obs-1")
    context.add_observation(kind="meeting_soon", detail="Standup in 5 minutes.", observation_id="obs-2")

    assert context.get_pending_observations() == [
        "3 apps in 30 minutes.",
        "Standup in 5 minutes.",
    ]


def test_no_pending_observations_by_default() -> None:
    context = ConversationContext()
    assert context.get_pending_observations() == []


def test_consume_observations_clears_pending() -> None:
    context = ConversationContext()
    context.add_observation(kind="context_switch", detail="detail", observation_id="obs-1")

    context.consume_observations()

    assert context.get_pending_observations() == []


def test_consume_observations_is_idempotent() -> None:
    context = ConversationContext()
    context.add_observation(kind="context_switch", detail="detail", observation_id="obs-1")
    context.consume_observations()
    context.consume_observations()  # calling again must not raise

    assert context.get_pending_observations() == []


def test_observation_offered_once_then_new_ones_still_flow() -> None:
    """An observation consumed after one turn must not resurface, but a
    genuinely new observation added afterward must still be offered."""
    context = ConversationContext()
    context.add_observation(kind="context_switch", detail="first", observation_id="obs-1")

    # Simulate one Planner turn: read, then consume.
    offered = context.get_pending_observations()
    context.consume_observations()
    assert offered == ["first"]

    # No new observation yet -> nothing pending on the next turn.
    assert context.get_pending_observations() == []

    # A new observation arrives later -> it should be offered.
    context.add_observation(kind="meeting_soon", detail="second", observation_id="obs-2")
    assert context.get_pending_observations() == ["second"]


def test_clear_also_clears_pending_observations() -> None:
    context = ConversationContext()
    context.add_observation(kind="context_switch", detail="detail", observation_id="obs-1")

    context.clear()

    assert context.get_pending_observations() == []
