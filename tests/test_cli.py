"""
Tests for cli/app.py: Vesper's rich REPL rendering logic (P08).

Rich's Console(record=True) captures exactly what would hit the real
terminal, so these assert on rendered text rather than mocking rich
itself. The Brain is never constructed for real here -- these tests
target the event-bus subscriber methods (the CLI's entire view onto
what Vesper is doing), not Brain.start()/the Planner loop.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from bus.event_bus import EventBus
from cli.app import VesperCLI, _format_arguments, _vesper_style
from schemas.events import (
    ConfirmationRequestedEvent,
    ConfirmationResponseEvent,
    ObservationEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
    VoiceOutputEvent,
)


@pytest.fixture(autouse=True)
def _fresh_bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


def _cli(console: Console) -> VesperCLI:
    # None of the handlers under test call anything on `brain` -- only
    # _run_turn/_render_status do, and those aren't exercised here.
    return VesperCLI(brain=None, console=console)  # type: ignore[arg-type]


# =============================================================================
# Style selection
# =============================================================================

def test_vesper_style_uses_gold_on_truecolor() -> None:
    console = Console(color_system="truecolor")
    assert _vesper_style(console) == "dim #C9A96A"


def test_vesper_style_falls_back_to_yellow() -> None:
    console = Console(color_system="standard")
    assert _vesper_style(console) == "dim yellow"


# =============================================================================
# Argument formatting
# =============================================================================

def test_format_arguments_renders_key_value_pairs() -> None:
    assert _format_arguments({"id": "abc123", "max_n": 10}) == "id='abc123', max_n=10"


def test_format_arguments_truncates_long_values() -> None:
    long_value = "x" * 200
    rendered = _format_arguments({"instruction": long_value})
    assert rendered.startswith("instruction='")
    assert len(rendered) < len(long_value)
    assert rendered.endswith("…'")


# =============================================================================
# Live tool-call trace lines
# =============================================================================

@pytest.mark.asyncio
async def test_tool_started_renders_arrow_prefixed_line() -> None:
    console = Console(record=True, width=120)
    cli = _cli(console)

    await cli._on_tool_started(
        ToolCallStartedEvent(tool_name="list_unread", arguments={"max_n": 10}, source="Planner")
    )

    output = console.export_text()
    assert "▸ list_unread(max_n=10)" in output


@pytest.mark.asyncio
async def test_tool_finished_renders_success_mark() -> None:
    console = Console(record=True, width=120)
    cli = _cli(console)

    await cli._on_tool_finished(
        ToolCallFinishedEvent(
            tool_name="list_unread", success=True, guardian_verdict="allow",
            result="[]", latency_ms=42.0, source="Planner",
        )
    )

    output = console.export_text()
    assert "✓ list_unread" in output
    assert "42ms" in output


@pytest.mark.asyncio
async def test_tool_finished_renders_failure_with_error() -> None:
    console = Console(record=True, width=120)
    cli = _cli(console)

    await cli._on_tool_finished(
        ToolCallFinishedEvent(
            tool_name="archive", success=False, guardian_verdict="allow",
            result="Error: boom", error="boom", latency_ms=5.0, source="Planner",
        )
    )

    output = console.export_text()
    assert "✗ archive" in output
    assert "boom" in output


# =============================================================================
# Observations — a subtly distinct line style
# =============================================================================

@pytest.mark.asyncio
async def test_observation_renders_with_distinct_marker() -> None:
    console = Console(record=True, width=120)
    cli = _cli(console)

    await cli._on_observation(
        ObservationEvent(
            kind="inbox_surge",
            detail="Sir, 5 new emails have arrived since I last checked.",
            source="ProactiveEngine",
        )
    )

    output = console.export_text()
    assert "○ Sir, 5 new emails have arrived since I last checked." in output
    # Distinct from both the tool-trace prefix and Vesper's own reply prefix.
    assert "▸" not in output
    assert "Vesper:" not in output


# =============================================================================
# Vesper's replies vs. suppression during a REPL-driven turn
# =============================================================================

@pytest.mark.asyncio
async def test_voice_output_renders_when_not_suppressed() -> None:
    console = Console(record=True, width=120)
    cli = _cli(console)

    await cli._on_voice_output(VoiceOutputEvent(text="Good evening, Sir.", source="Brain"))

    assert "Vesper: Good evening, Sir." in console.export_text()


@pytest.mark.asyncio
async def test_voice_output_suppressed_during_repl_turn() -> None:
    console = Console(record=True, width=120)
    cli = _cli(console)
    cli._suppress_voice_output = True

    await cli._on_voice_output(VoiceOutputEvent(text="Good evening, Sir.", source="Brain"))

    assert console.export_text() == ""


# =============================================================================
# Confirmation flow — inline [y/N]
# =============================================================================

@pytest.mark.asyncio
async def test_confirmation_requested_renders_summary_and_emits_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(record=True, width=120)
    cli = _cli(console)
    bus = EventBus()

    # Patch the builtin, not console.input itself -- Console.input() prints
    # its own prompt via self.print(prompt, markup=...) before reading, and
    # that's exactly the code path that silently swallowed a literal
    # "[y/N]" as (invalid, dropped) rich markup; patching console.input
    # directly would skip over that bug entirely.
    monkeypatch.setattr("builtins.input", lambda: "y")

    responses = []

    async def on_response(event: ConfirmationResponseEvent) -> None:
        responses.append(event)

    bus.subscribe(ConfirmationResponseEvent, on_response)

    await cli._on_confirmation_requested(
        ConfirmationRequestedEvent(
            summary="archive(id='19f7f6e1817d7abd')",
            tool_name="archive",
            arguments={"id": "19f7f6e1817d7abd"},
            request_id="req-1",
            source="Guardian",
        )
    )

    output = console.export_text()
    assert "archive(id='19f7f6e1817d7abd')" in output
    assert "[y/N]" in output  # literal brackets, not swallowed as markup
    assert len(responses) == 1
    assert responses[0].approved is True
    assert responses[0].request_id == "req-1"


@pytest.mark.asyncio
async def test_confirmation_requested_denies_on_anything_but_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(record=True, width=120)
    cli = _cli(console)
    bus = EventBus()

    monkeypatch.setattr("builtins.input", lambda: "")  # bare Enter -> default No

    responses = []

    async def on_response(event: ConfirmationResponseEvent) -> None:
        responses.append(event)

    bus.subscribe(ConfirmationResponseEvent, on_response)

    await cli._on_confirmation_requested(
        ConfirmationRequestedEvent(
            summary="archive(id='x')", tool_name="archive", arguments={"id": "x"},
            request_id="req-2", source="Guardian",
        )
    )

    assert len(responses) == 1
    assert responses[0].approved is False
