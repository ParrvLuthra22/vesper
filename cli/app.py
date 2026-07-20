#!/usr/bin/env python3
"""
Vesper's terminal presence — a rich REPL.

Runs the exact same Brain main.py does (MCP bridge -> sensors ->
Proactive Engine -> Planner), just with VoiceAgent disabled: text goes
in/out via this terminal instead of the microphone/TTS. This is the
primary interface until the HUD ships.

Live plan traces (ToolCallStartedEvent/ToolCallFinishedEvent) and
observations (ObservationEvent) are rendered from the same event-bus
events a future HUD will consume — this module has no privileged access
to the Planner beyond what any other subscriber could get.

Usage:
    python -m vesper
    vesper                    (after `pip install -e .`)
    python cli/app.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from rich.console import Console
from rich.table import Table

from bus.event_bus import EventBus
from config.settings import load_config_dict
from orchestrator.brain import Brain
from schemas.events import (
    ConfirmationRequestedEvent,
    ConfirmationResponseEvent,
    ObservationEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
    VoiceOutputEvent,
)
from tools.registry import get_registry
from tracing.last_trace import print_last_turn
from utils.logger import get_logger, init_from_config

#: Vesper's brand color. Used only when the terminal reports truecolor
#: support; every other terminal falls back to plain "yellow".
VESPER_GOLD = "#C9A96A"

MAX_ARG_REPR_LEN = 60


def _vesper_style(console: Console) -> str:
    return f"dim {VESPER_GOLD}" if console.color_system == "truecolor" else "dim yellow"


def _format_arguments(arguments: Dict[str, Any]) -> str:
    parts = []
    for key, value in arguments.items():
        rendered = repr(value)
        if len(rendered) > MAX_ARG_REPR_LEN:
            rendered = rendered[: MAX_ARG_REPR_LEN - 1] + "…'"
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)


class VesperCLI:
    """Owns the REPL loop and every event-bus subscription that renders it."""

    def __init__(self, brain: Brain, console: Console):
        self._brain = brain
        self._console = console
        self._bus = EventBus()
        self._vesper_style = _vesper_style(console)
        # True for the duration of a REPL-driven turn: the reply has
        # already been rendered (streamed or printed from the return
        # value), so the VoiceOutputEvent the Brain also emits for it is
        # redundant here. Greetings and scheduled briefings fire outside
        # any turn, so this is False when they arrive and get printed.
        self._suppress_voice_output = False

    # -------------------------------------------------------------------
    # Event subscriptions — the CLI's entire view onto what Vesper is
    # doing, via the same events a future HUD will consume.
    # -------------------------------------------------------------------

    def _subscribe(self) -> None:
        self._bus.subscribe(VoiceOutputEvent, self._on_voice_output)
        self._bus.subscribe(ToolCallStartedEvent, self._on_tool_started)
        self._bus.subscribe(ToolCallFinishedEvent, self._on_tool_finished)
        self._bus.subscribe(ObservationEvent, self._on_observation)
        self._bus.subscribe(ConfirmationRequestedEvent, self._on_confirmation_requested)

    async def _on_voice_output(self, event: VoiceOutputEvent) -> None:
        if self._suppress_voice_output:
            return
        self._print_vesper_line(event.text)

    async def _on_tool_started(self, event: ToolCallStartedEvent) -> None:
        args = _format_arguments(event.arguments)
        self._console.print(f"▸ {event.tool_name}({args})", style="dim", markup=False, highlight=False)

    async def _on_tool_finished(self, event: ToolCallFinishedEvent) -> None:
        mark = "✓" if event.success else "✗"
        line = f"  {mark} {event.tool_name} [{event.latency_ms:.0f}ms]"
        if not event.success:
            detail = event.error or event.result
            if detail:
                line += f" — {detail}"
        self._console.print(line, style="dim", markup=False, highlight=False)

    async def _on_observation(self, event: ObservationEvent) -> None:
        self._console.print(f"○ {event.detail}", style="italic cyan", markup=False, highlight=False)

    async def _on_confirmation_requested(self, event: ConfirmationRequestedEvent) -> None:
        self._console.print(f"⚠  {event.summary}", style="bold yellow", markup=False, highlight=False)
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(
            None, lambda: self._console.input("   Confirm? [y/N] ", markup=False)
        )
        approved = answer.strip().lower() in ("y", "yes")
        await self._bus.emit(
            ConfirmationResponseEvent(request_id=event.request_id, approved=approved, source="cli")
        )

    # -------------------------------------------------------------------
    # Rendering helpers
    # -------------------------------------------------------------------

    def _print_vesper_line(self, text: str) -> None:
        self._console.print("Vesper: ", style=self._vesper_style, markup=False, highlight=False, end="")
        self._console.print(text, style=self._vesper_style, markup=False, highlight=False)

    # -------------------------------------------------------------------
    # One turn
    # -------------------------------------------------------------------

    async def _run_turn(self, text: str) -> None:
        streamed = {"tokens": 0}

        def on_token(token: str) -> None:
            if streamed["tokens"] == 0:
                self._console.print(
                    "Vesper: ", style=self._vesper_style, markup=False, highlight=False, end=""
                )
            streamed["tokens"] += 1
            self._console.print(token, style=self._vesper_style, markup=False, highlight=False, end="")

        self._suppress_voice_output = True
        try:
            result = await self._brain.handle_user_text(text, on_token=on_token)
        finally:
            self._suppress_voice_output = False

        if streamed["tokens"] > 0:
            self._console.print()  # close the streamed line
        else:
            # Fallback provider (or a provider that doesn't support
            # streaming) was used for the final reply — nothing was
            # printed live, so print the complete result now.
            self._print_vesper_line(result.text)

    # -------------------------------------------------------------------
    # Commands
    # -------------------------------------------------------------------

    async def _handle_command(self, text: str) -> bool:
        """Returns True if the REPL should exit."""
        command = text.strip().lower()
        if command == "/quit":
            return True
        elif command == "/trace":
            print_last_turn()
        elif command == "/tools":
            self._render_tools()
        elif command == "/status":
            await self._render_status()
        else:
            self._console.print(
                f"Unknown command: {text!r}. Try /trace, /tools, /status, /quit.",
                style="dim", markup=False, highlight=False,
            )
        return False

    def _render_tools(self) -> None:
        registry = get_registry()
        table = Table(title="Registered Tools")
        table.add_column("Tier")
        table.add_column("Name")
        table.add_column("Category")
        table.add_column("Slow")
        table.add_column("Description")
        for spec in sorted(registry.list_all(), key=lambda t: (t.tier, t.name)):
            description = spec.description.strip().split("\n")[0]
            table.add_row(spec.tier, spec.name, spec.category, "yes" if spec.slow else "", description)
        self._console.print(table)

    async def _render_status(self) -> None:
        agents_table = Table(title="Agents")
        agents_table.add_column("Agent")
        agents_table.add_column("Healthy")
        agents_table.add_column("Errors")
        for status in self._brain.get_agents_status():
            agents_table.add_row(
                status["name"],
                "✓" if status["healthy"] else "✗",
                str(status["error_count"]),
            )
        self._console.print(agents_table)

        providers_table = Table(title="LLM Providers")
        providers_table.add_column("Tier")
        providers_table.add_column("Provider")
        providers_table.add_column("Model")
        providers_table.add_column("Available")
        for entry in await self._brain.get_router().provider_status():
            providers_table.add_row(
                entry["tier"],
                entry["provider"],
                entry["model"],
                "✓" if entry["available"] else "✗",
            )
        self._console.print(providers_table)

    # -------------------------------------------------------------------
    # REPL loop
    # -------------------------------------------------------------------

    async def run(self) -> None:
        self._subscribe()
        await self._brain.start()

        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    text = await loop.run_in_executor(None, lambda: self._console.input("You: "))
                except (EOFError, KeyboardInterrupt):
                    self._console.print()
                    break

                text = text.strip()
                if not text:
                    continue
                if text.startswith("/"):
                    if await self._handle_command(text):
                        break
                    continue

                await self._run_turn(text)
        finally:
            self._console.print("[dim]Shutting down…[/dim]")
            await self._brain.stop("CLI exit")


def _load_config() -> Dict[str, Any]:
    config = load_config_dict()
    # The console handler in utils/logger.py is always attached at stdout;
    # this app's whole point is a clean terminal, so raise the bar to
    # WARNING for this process only (main.py's voice mode is unaffected —
    # nothing here touches settings.yaml).
    config.setdefault("general", {})["log_level"] = "WARNING"
    # The HUD overlay isn't shipped yet (this CLI is its stand-in) and
    # unconditionally spawns a native window subprocess on macOS — not
    # wanted for a terminal-only session.
    config.setdefault("ui", {}).setdefault("hud", {})["enabled"] = False
    return config


async def _main() -> int:
    config = _load_config()
    init_from_config(config.get("general", {}))
    logger = get_logger(__name__)

    console = Console()

    event_bus = EventBus()
    await event_bus.start()

    brain = Brain(config=config, event_bus=event_bus, enable_voice_agent=False)
    cli = VesperCLI(brain=brain, console=console)

    try:
        await cli.run()
    except Exception as exc:
        logger.error(f"CLI fatal error: {exc}", exc_info=True)
        console.print(f"[bold red]Fatal error:[/bold red] {exc}")
        return 1
    return 0


def run() -> None:
    """Entry point for `python -m vesper` / the `vesper` console script."""
    try:
        exit_code = asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    run()
