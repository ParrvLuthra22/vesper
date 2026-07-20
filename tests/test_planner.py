"""
Tests for orchestrator/planner.py (the Planner LLM tool-calling loop).

The ModelRouter is mocked (an AsyncMock standing in for ModelRouter.complete)
so these tests exercise the Planner's own loop, Guardian integration, and
bus-routed tool execution — not any real LLM or provider.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from bus.event_bus import EventBus
from guardian.gate import Guardian
from llm.types import LLMResponse, RouterError, ToolCall
from orchestrator.planner import MAX_ITERATIONS_MESSAGE, Planner
from schemas.events import (
    ActionRequestEvent,
    ActionResultEvent,
    ConfirmationRequestedEvent,
    ConfirmationResponseEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from tools.registry import ToolRegistry, ToolSpec
from tracing.tracer import Tracer

#: These tests aren't exercising tracing itself (see test_tracer.py for
#: that) -- disable it so test runs never write to the real
#: data/traces/traces.jsonl.
_NO_TRACING = {"tracing": {"enabled": False}}


@pytest.fixture(autouse=True)
def _fresh_bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


def _mock_router(responses: List[Any]) -> AsyncMock:
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=responses)
    return router


def _install_bus_responder(bus: EventBus, action: str, result: Any = "ok", success: bool = True, error: str = "") -> None:
    """Subscribe a fake agent that answers ActionRequestEvent(action=...) immediately."""

    async def on_request(event: ActionRequestEvent) -> None:
        if event.action != action:
            return
        await bus.emit(
            ActionResultEvent(
                source="TestAgent",
                action=event.action,
                success=success,
                result=result,
                error=error,
                plan_id=event.plan_id,
                step_number=event.step_number,
                correlation_id=event.correlation_id,
            )
        )

    bus.subscribe(ActionRequestEvent, on_request)


def _make_planner(
    router: AsyncMock,
    registry: ToolRegistry,
    bus: EventBus,
    **kwargs: Any,
) -> Planner:
    guardian = Guardian(event_bus=bus)
    tracer = Tracer(config=_NO_TRACING)
    return Planner(router=router, registry=registry, guardian=guardian, event_bus=bus, tracer=tracer, **kwargs)


async def _wait_until(predicate, attempts: int = 100) -> None:
    """Yield to the event loop until `predicate()` is true.

    EventBus.emit() dispatches handlers via asyncio.gather(), which schedules
    each handler as its own Task — that needs an extra loop tick to actually
    run, so a single `await asyncio.sleep(0)` isn't always enough to observe
    a just-emitted event's side effects.
    """
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition not met after yielding to the event loop")


# =============================================================================
# Pure conversation (no tools)
# =============================================================================

@pytest.mark.asyncio
async def test_pure_conversation_no_tools() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    router = _mock_router([LLMResponse(text="Good afternoon, Sir.", tool_calls=[])])

    planner = _make_planner(router, registry, bus)
    result = await planner.run(user_text="hi")

    assert result.text == "Good afternoon, Sir."
    assert not result.aborted
    assert result.tool_trace == []
    assert router.complete.call_count == 1


# =============================================================================
# Single tool call
# =============================================================================

@pytest.mark.asyncio
async def test_single_tool_call() -> None:
    bus = EventBus()
    _install_bus_responder(bus, "do_thing", result="did it")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="do_thing", description="d", target_agent="TestAgent", action="do_thing", tier="safe")
    )

    router = _mock_router([
        LLMResponse(text="", tool_calls=[ToolCall(name="do_thing", arguments={"x": 1}, id="call_1")]),
        LLMResponse(text="Done, Sir."),
    ])

    planner = _make_planner(router, registry, bus)
    result = await planner.run(user_text="do the thing")

    assert result.text == "Done, Sir."
    assert not result.aborted
    assert result.tool_trace == ["do_thing:ok"]
    assert router.complete.call_count == 2

    # Second call's messages must include the tool result feeding back to the model.
    second_call_messages = router.complete.call_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "did it"
    assert tool_messages[0]["tool_call_id"] == "call_1"


# =============================================================================
# Multi-step: tool -> result -> second tool -> answer
# =============================================================================

@pytest.mark.asyncio
async def test_multi_step_tool_then_second_tool_then_answer() -> None:
    bus = EventBus()
    _install_bus_responder(bus, "open_app", result="Opened Safari")
    _install_bus_responder(bus, "control_volume", result="Volume set to 30%")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="open_app", description="d", target_agent="TestAgent", action="open_app", tier="safe")
    )
    registry.register(
        ToolSpec(name="set_volume", description="d", target_agent="TestAgent", action="control_volume", tier="safe")
    )

    router = _mock_router([
        LLMResponse(tool_calls=[ToolCall(name="open_app", arguments={"app_name": "Safari"}, id="call_1")]),
        LLMResponse(tool_calls=[ToolCall(name="set_volume", arguments={"level": 30}, id="call_2")]),
        LLMResponse(text="Opened Safari and set volume to 30%, Sir."),
    ])

    planner = _make_planner(router, registry, bus)
    result = await planner.run(user_text="open safari and set volume to 30")

    assert result.text == "Opened Safari and set volume to 30%, Sir."
    assert not result.aborted
    assert result.tool_trace == ["open_app:ok", "set_volume:ok"]
    assert router.complete.call_count == 3


# =============================================================================
# Guardian confirm flow
# =============================================================================

@pytest.mark.asyncio
async def test_guardian_confirm_flow_approved() -> None:
    bus = EventBus()
    registry = ToolRegistry()

    async def close_app_handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        return f"Closed {arguments.get('app_name')}"

    registry.register(
        ToolSpec(name="close_app", description="d", handler=close_app_handler, tier="confirm")
    )

    requested: List[ConfirmationRequestedEvent] = []

    async def on_confirmation_requested(event: ConfirmationRequestedEvent) -> None:
        requested.append(event)

    bus.subscribe(ConfirmationRequestedEvent, on_confirmation_requested)

    router = _mock_router([
        LLMResponse(tool_calls=[ToolCall(name="close_app", arguments={"app_name": "Safari"}, id="call_1")]),
        LLMResponse(text="Closed Safari, Sir."),
    ])

    planner = _make_planner(router, registry, bus)
    task = asyncio.create_task(planner.run(user_text="close safari"))

    # Let the Planner reach the point of awaiting confirmation, then approve it.
    await _wait_until(lambda: len(requested) == 1)
    assert "close_app" in requested[0].summary
    assert "Safari" in requested[0].summary

    await bus.emit(
        ConfirmationResponseEvent(request_id=requested[0].request_id, approved=True, source="voice_agent")
    )

    result = await task
    assert result.text == "Closed Safari, Sir."
    assert result.tool_trace == ["close_app:ok"]


@pytest.mark.asyncio
async def test_guardian_confirm_flow_denied() -> None:
    bus = EventBus()
    registry = ToolRegistry()

    async def close_app_handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        return f"Closed {arguments.get('app_name')}"

    registry.register(
        ToolSpec(name="close_app", description="d", handler=close_app_handler, tier="confirm")
    )

    requested: List[ConfirmationRequestedEvent] = []

    async def on_confirmation_requested(event: ConfirmationRequestedEvent) -> None:
        requested.append(event)

    bus.subscribe(ConfirmationRequestedEvent, on_confirmation_requested)

    router = _mock_router([
        LLMResponse(tool_calls=[ToolCall(name="close_app", arguments={"app_name": "Safari"}, id="call_1")]),
        LLMResponse(text="Understood, I won't close it, Sir."),
    ])

    planner = _make_planner(router, registry, bus)
    task = asyncio.create_task(planner.run(user_text="close safari"))
    await _wait_until(lambda: len(requested) == 1)

    await bus.emit(
        ConfirmationResponseEvent(request_id=requested[0].request_id, approved=False, source="voice_agent")
    )

    result = await task
    assert result.text == "Understood, I won't close it, Sir."
    assert result.tool_trace == ["close_app:denied"]

    # The model must have seen the denial as the tool result.
    second_call_messages = router.complete.call_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert "Denied" in tool_messages[0]["content"]


# =============================================================================
# Tool timeout handling
# =============================================================================

@pytest.mark.asyncio
async def test_tool_timeout_handling() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    # No responder subscribed for "slow_action" -> the ActionResultEvent never arrives.
    registry.register(
        ToolSpec(name="slow_tool", description="d", target_agent="TestAgent", action="slow_action", tier="safe")
    )

    router = _mock_router([
        LLMResponse(tool_calls=[ToolCall(name="slow_tool", arguments={}, id="call_1")]),
        LLMResponse(text="Sorry Sir, that didn't work."),
    ])

    planner = _make_planner(router, registry, bus, tool_timeout_seconds=0.05)
    result = await planner.run(user_text="do the slow thing")

    assert result.text == "Sorry Sir, that didn't work."
    assert result.tool_trace == ["slow_tool:timeout"]

    second_call_messages = router.complete.call_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert "timed out" in tool_messages[0]["content"]


# =============================================================================
# Max-iteration cutoff
# =============================================================================

@pytest.mark.asyncio
async def test_max_iteration_cutoff() -> None:
    bus = EventBus()
    registry = ToolRegistry()

    async def noop_handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        return "ok"

    registry.register(ToolSpec(name="loop_tool", description="d", handler=noop_handler, tier="safe"))

    # The model keeps calling the tool forever and never gives a plain-text answer.
    always_tool_call = LLMResponse(tool_calls=[ToolCall(name="loop_tool", arguments={}, id="call_1")])
    router = _mock_router([always_tool_call] * 10)

    planner = _make_planner(router, registry, bus, max_iterations=3)
    result = await planner.run(user_text="keep going")

    assert result.aborted
    assert result.text == MAX_ITERATIONS_MESSAGE
    assert result.tool_trace == ["loop_tool:ok", "loop_tool:ok", "loop_tool:ok"]
    assert router.complete.call_count == 3


# =============================================================================
# RouterError handling
# =============================================================================

@pytest.mark.asyncio
async def test_router_error_returns_graceful_message() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    router = _mock_router([
        RouterError(user_message="Sir, I'm having trouble thinking right now.", purpose="planning")
    ])

    planner = _make_planner(router, registry, bus)
    result = await planner.run(user_text="hi")

    assert result.aborted
    assert result.text == "Sir, I'm having trouble thinking right now."


# =============================================================================
# Live tool-call events (ToolCallStartedEvent / ToolCallFinishedEvent) — P08
#
# These are the events a terminal/HUD renders live, distinct from
# tracing/tracer.py's TurnTrace (which records the same execution for
# later /trace inspection, not for live display).
# =============================================================================

@pytest.mark.asyncio
async def test_tool_call_events_emitted_on_success() -> None:
    bus = EventBus()
    registry = ToolRegistry()

    async def do_thing_handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        return "did it"

    registry.register(ToolSpec(name="do_thing", description="d", handler=do_thing_handler, tier="safe"))

    started: List[ToolCallStartedEvent] = []
    finished: List[ToolCallFinishedEvent] = []

    async def on_started(event: ToolCallStartedEvent) -> None:
        started.append(event)

    async def on_finished(event: ToolCallFinishedEvent) -> None:
        finished.append(event)

    bus.subscribe(ToolCallStartedEvent, on_started)
    bus.subscribe(ToolCallFinishedEvent, on_finished)

    router = _mock_router([
        LLMResponse(tool_calls=[ToolCall(name="do_thing", arguments={"x": 1}, id="call_1")]),
        LLMResponse(text="Done, Sir."),
    ])

    planner = _make_planner(router, registry, bus)
    await planner.run(user_text="do the thing")

    assert len(started) == 1
    assert started[0].tool_name == "do_thing"
    assert started[0].arguments == {"x": 1}

    assert len(finished) == 1
    assert finished[0].tool_name == "do_thing"
    assert finished[0].success is True
    assert finished[0].guardian_verdict == "allow"
    assert finished[0].result == "did it"


@pytest.mark.asyncio
async def test_tool_call_finished_event_for_unknown_tool() -> None:
    bus = EventBus()
    registry = ToolRegistry()  # nothing registered

    finished: List[ToolCallFinishedEvent] = []

    async def on_finished(event: ToolCallFinishedEvent) -> None:
        finished.append(event)

    bus.subscribe(ToolCallFinishedEvent, on_finished)

    router = _mock_router([
        LLMResponse(tool_calls=[ToolCall(name="ghost_tool", arguments={}, id="call_1")]),
        LLMResponse(text="Sorry, Sir."),
    ])

    planner = _make_planner(router, registry, bus)
    await planner.run(user_text="do the ghost thing")

    assert len(finished) == 1
    assert finished[0].tool_name == "ghost_tool"
    assert finished[0].success is False
    assert finished[0].guardian_verdict == "n/a"


@pytest.mark.asyncio
async def test_tool_call_started_fires_before_confirmation_resolves() -> None:
    """A confirm-tier call must show up live immediately, not only after approval."""
    bus = EventBus()
    registry = ToolRegistry()

    async def close_app_handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        return f"Closed {arguments.get('app_name')}"

    registry.register(ToolSpec(name="close_app", description="d", handler=close_app_handler, tier="confirm"))

    started: List[ToolCallStartedEvent] = []
    requested: List[ConfirmationRequestedEvent] = []

    async def on_started(event: ToolCallStartedEvent) -> None:
        started.append(event)

    async def on_confirmation_requested(event: ConfirmationRequestedEvent) -> None:
        requested.append(event)

    bus.subscribe(ToolCallStartedEvent, on_started)
    bus.subscribe(ConfirmationRequestedEvent, on_confirmation_requested)

    router = _mock_router([
        LLMResponse(tool_calls=[ToolCall(name="close_app", arguments={"app_name": "Safari"}, id="call_1")]),
        LLMResponse(text="Closed Safari, Sir."),
    ])

    planner = _make_planner(router, registry, bus)
    task = asyncio.create_task(planner.run(user_text="close safari"))

    await _wait_until(lambda: len(requested) == 1)
    assert len(started) == 1  # already fired, before the confirmation was resolved
    assert started[0].tool_name == "close_app"

    await bus.emit(
        ConfirmationResponseEvent(request_id=requested[0].request_id, approved=True, source="voice_agent")
    )
    await task


# =============================================================================
# on_token streaming pass-through — P08
# =============================================================================

@pytest.mark.asyncio
async def test_on_token_passed_through_to_router() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    router = _mock_router([LLMResponse(text="Good evening, Sir.")])

    planner = _make_planner(router, registry, bus)
    received: List[str] = []

    await planner.run(user_text="hi", on_token=received.append)

    assert router.complete.call_args.kwargs["on_token"] == received.append
