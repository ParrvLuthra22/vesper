"""
Planner — the LLM tool-calling planner that replaces Vesper's rule-based
intent switchboard (the old INTENT_ROUTING dict and _parse_multi_step_command
in orchestrator/brain.py).

Every user input goes through the Planner. There is no "unknown intent"
fallback: the model sees the full tool registry schema and either calls a
tool or just answers in plain text — an open vocabulary rather than a fixed
intent catalog.

Loop:
    1. Call ModelRouter.complete() with the running message history and the
       tool registry's LLM schema.
    2. If the response has no tool_calls, its text is Vesper's reply — stop.
    3. Otherwise, for each tool call: look up the ToolSpec, ask the Guardian
       whether it may run (safe -> allow immediately; confirm/dangerous ->
       emit ConfirmationRequestedEvent and block on the resolution), then
       execute allowed calls (a local handler, or a bus round trip via
       ActionRequestEvent/ActionResultEvent matched by correlation_id).
       Append each outcome as a tool-result message and loop.
    4. Give up gracefully after `max_iterations` without a final text answer.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from bus.event_bus import EventBus, get_event_bus
from guardian.gate import Guardian, VerdictType
from llm.router import ModelRouter
from llm.types import LLMResponse, RouterError, ToolCall
from schemas.events import ActionRequestEvent, ActionResultEvent, PlanCreatedEvent
from tools.registry import ToolRegistry, ToolSpec, get_registry
from utils.logger import get_logger

# Importing for its registration side effect: populates the default
# ToolRegistry with the wrapped SystemAgent/WebSearchAgent capabilities.
import tools.builtin  # noqa: F401

logger = get_logger(__name__)

MAX_ITERATIONS = 6
TOOL_EXECUTION_TIMEOUT_SECONDS = 30.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERSONA_PATH = PROJECT_ROOT / "config" / "persona.md"
DEFAULT_PERSONA = "You are Vesper, a capable macOS assistant. Address the user as Sir. Be concise."

MAX_ITERATIONS_MESSAGE = (
    "Sir, I've gone back and forth on that longer than I should have without "
    "landing on an answer. Could you rephrase or simplify the request?"
)

CONTEXT_PLACEHOLDER = "{context}"
GREETING_INSTRUCTION = "(Session start. Greet the user now, per your greeting instructions.)"


def _load_persona(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or DEFAULT_PERSONA
    except OSError:
        logger.warning(f"[Planner] could not read persona file {path}; using default persona")
        return DEFAULT_PERSONA


@dataclass
class PlannerResult:
    """Final outcome of one Planner.run() call."""

    text: str
    aborted: bool = False
    tool_trace: List[str] = field(default_factory=list)


class Planner:
    """LLM tool-calling planner: the sole entry point from user text to action."""

    def __init__(
        self,
        router: ModelRouter,
        registry: Optional[ToolRegistry] = None,
        guardian: Optional[Guardian] = None,
        event_bus: Optional[EventBus] = None,
        persona_path: Path = DEFAULT_PERSONA_PATH,
        max_iterations: int = MAX_ITERATIONS,
        tool_timeout_seconds: float = TOOL_EXECUTION_TIMEOUT_SECONDS,
    ):
        self._router = router
        self._registry = registry or get_registry()
        self._event_bus = event_bus or get_event_bus()
        self._guardian = guardian or Guardian(event_bus=self._event_bus)
        self._persona = _load_persona(persona_path)
        self._max_iterations = max_iterations
        self._tool_timeout_seconds = tool_timeout_seconds

    async def run(
        self,
        user_text: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        purpose: str = "planning",
        active_app: Optional[str] = None,
        observations: Optional[List[str]] = None,
    ) -> PlannerResult:
        """Run the tool-calling loop for one piece of user text."""
        system_prompt = self._render_system_prompt(active_app=active_app, observations=observations)
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._context_to_messages(recent_context or []))
        messages.append({"role": "user", "content": user_text})

        tools_schema = self._registry.to_llm_schema()
        trace: List[str] = []

        for _ in range(self._max_iterations):
            response = await self._router.complete(messages=messages, tools=tools_schema, purpose=purpose)

            if isinstance(response, RouterError):
                await self._emit_plan_trace(user_text, trace)
                return PlannerResult(text=response.user_message, aborted=True, tool_trace=trace)

            if not response.tool_calls:
                await self._emit_plan_trace(user_text, trace)
                return PlannerResult(text=response.text, tool_trace=trace)

            messages.append(self._assistant_tool_call_message(response))
            for tool_call in response.tool_calls:
                result_text = await self._execute_tool_call(tool_call, trace)
                messages.append(self._tool_result_message(tool_call, result_text))

        await self._emit_plan_trace(user_text, trace)
        return PlannerResult(text=MAX_ITERATIONS_MESSAGE, aborted=True, tool_trace=trace)

    async def greet(
        self,
        active_app: Optional[str] = None,
        observations: Optional[List[str]] = None,
        purpose: str = "planning",
    ) -> PlannerResult:
        """
        Produce the session-start greeting: a time-appropriate salutation,
        plus one observation if the context has something worth mentioning.

        This is a single plain completion (no tool calls, no conversation
        history) — the greeting behavior itself lives in the persona, and
        the current time/date/observations arrive via the context block.
        """
        system_prompt = self._render_system_prompt(active_app=active_app, observations=observations)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": GREETING_INSTRUCTION},
        ]

        response = await self._router.complete(messages=messages, purpose=purpose)
        if isinstance(response, RouterError):
            return PlannerResult(text=response.user_message, aborted=True)
        return PlannerResult(text=response.text)

    def _render_system_prompt(
        self,
        active_app: Optional[str] = None,
        observations: Optional[List[str]] = None,
    ) -> str:
        context_block = self._render_context_block(active_app=active_app, observations=observations)
        if CONTEXT_PLACEHOLDER in self._persona:
            return self._persona.replace(CONTEXT_PLACEHOLDER, context_block)
        return f"{self._persona}\n\n{context_block}"

    @staticmethod
    def _render_context_block(
        active_app: Optional[str] = None,
        observations: Optional[List[str]] = None,
    ) -> str:
        now = datetime.now()
        lines = [
            f"Current time: {now.strftime('%I:%M %p')}",
            f"Today's date: {now.strftime('%A, %B %d, %Y')}",
        ]
        if active_app:
            lines.append(f"Active app: {active_app}")
        if observations:
            lines.append("Pending observations:")
            lines.extend(f"  - {obs}" for obs in observations)
        else:
            lines.append("Pending observations: none")
        return "\n".join(lines)

    async def _execute_tool_call(self, tool_call: ToolCall, trace: List[str]) -> str:
        tool_spec = self._registry.get(tool_call.name)
        if tool_spec is None or not tool_spec.enabled:
            trace.append(f"{tool_call.name}:unknown_tool")
            return f"Error: no such tool '{tool_call.name}'"

        verdict = await self._guardian.check(tool_spec, tool_call.arguments, context={})
        if verdict.outcome == VerdictType.NEEDS_CONFIRMATION:
            verdict = await self._guardian.await_resolution(verdict.request_id)

        if verdict.outcome != VerdictType.ALLOW:
            trace.append(f"{tool_call.name}:denied")
            return f"Denied: {verdict.reason}"

        try:
            result_text = await self._run_tool(tool_spec, tool_call.arguments)
            trace.append(f"{tool_call.name}:ok")
            return result_text
        except asyncio.TimeoutError:
            trace.append(f"{tool_call.name}:timeout")
            return f"Error: '{tool_call.name}' timed out after {self._tool_timeout_seconds:.0f}s"
        except Exception as exc:
            trace.append(f"{tool_call.name}:error")
            return f"Error: {exc}"

    async def _run_tool(self, tool_spec: ToolSpec, arguments: Dict[str, Any]) -> str:
        if tool_spec.handler is not None:
            result = await tool_spec.handler(arguments, {})
            return str(result)
        return await self._run_bus_routed_tool(tool_spec, arguments)

    async def _run_bus_routed_tool(self, tool_spec: ToolSpec, arguments: Dict[str, Any]) -> str:
        correlation_id = uuid4()
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[ActionResultEvent]" = loop.create_future()

        async def on_result(event: ActionResultEvent) -> None:
            if event.correlation_id == correlation_id and not future.done():
                future.set_result(event)

        token = self._event_bus.subscribe(ActionResultEvent, on_result)
        try:
            await self._event_bus.emit(
                ActionRequestEvent(
                    action=tool_spec.action,
                    target_agent=tool_spec.target_agent,
                    parameters=arguments,
                    source="Planner",
                    correlation_id=correlation_id,
                )
            )
            result_event = await asyncio.wait_for(future, timeout=self._tool_timeout_seconds)
        finally:
            token.unsubscribe()

        if not result_event.success:
            raise RuntimeError(result_event.error or f"{tool_spec.name} failed")
        return str(result_event.result)

    async def _emit_plan_trace(self, description: str, trace: List[str]) -> None:
        await self._event_bus.emit(
            PlanCreatedEvent(
                plan_id=uuid4(),
                description=description,
                steps=list(trace),
                total_steps=len(trace),
                source="Planner",
            )
        )

    @staticmethod
    def _context_to_messages(recent_context: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        for turn in recent_context:
            user_text = turn.get("user")
            if user_text:
                messages.append({"role": "user", "content": str(user_text)})
            response_text = turn.get("response")
            if response_text:
                messages.append({"role": "assistant", "content": str(response_text)})
        return messages

    @staticmethod
    def _assistant_tool_call_message(response: LLMResponse) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": response.text or "",
            "tool_calls": [
                {
                    "id": tc.id or f"call_{i}",
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for i, tc in enumerate(response.tool_calls)
            ],
        }

    @staticmethod
    def _tool_result_message(tool_call: ToolCall, result_text: str) -> Dict[str, str]:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id or tool_call.name,
            "content": result_text,
        }
