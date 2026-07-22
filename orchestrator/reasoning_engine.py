"""LangGraph-based reasoning engine for complex multi-tool VESPER tasks."""

from __future__ import annotations

import asyncio
import json
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from uuid import UUID

from bus.event_bus import EventBus
from schemas.events import HUDGraphStateEvent, IntentRecognizedEvent, VoiceOutputEvent
from utils.api_keys import get_gemini_api_key
from utils.logger import get_logger

try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    END = "END"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    LANGGRAPH_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    LANGCHAIN_GEMINI_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    ChatGoogleGenerativeAI = None  # type: ignore[assignment]
    LANGCHAIN_GEMINI_AVAILABLE = False


logger = get_logger(__name__)


class VesperState(TypedDict):
    user_input: str
    plan: List[str]
    current_step: int
    tool_results: Annotated[List[str], operator.add]
    final_response: str
    needs_clarification: bool
    clarification_question: str


class ReasoningEngine:
    """LangGraph multi-step planner/executor/verifier for complex requests."""

    def __init__(
        self,
        event_bus: EventBus,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._event_bus = event_bus
        self._config = config or {}
        self._llm = self._build_llm()
        self._graph = self._build_graph()

        # Action -> tool mapping required by spec.
        self._tool_map = {
            "search": "web_search",
            "open": "safari",
            "message": "imessage",
            "file": "finder",
            "calendar": "calendar",
            "screenshot": "vision",
            "spotify": "spotify",
        }

    @property
    def is_available(self) -> bool:
        return self._graph is not None

    async def run(self, user_input: str, correlation_id: Optional[UUID] = None) -> VesperState:
        """Execute the reasoning graph for one complex user request."""
        initial_state: VesperState = {
            "user_input": user_input,
            "plan": [],
            "current_step": 0,
            "tool_results": [],
            "final_response": "",
            "needs_clarification": False,
            "clarification_question": "",
        }
        # Non-schema runtime metadata used by node internals.
        initial_state["correlation_id"] = correlation_id  # type: ignore[index]
        initial_state["complete"] = False  # type: ignore[index]
        initial_state["plan_objects"] = []  # type: ignore[index]

        if self._graph is None:
            fallback = "I can't run complex reasoning yet because LangGraph dependencies are unavailable."
            await self._event_bus.emit(
                VoiceOutputEvent(
                    text=fallback,
                    source="ReasoningEngine",
                    correlation_id=correlation_id,
                )
            )
            initial_state["final_response"] = fallback
            return initial_state

        result = await self._graph.ainvoke(initial_state)
        return result

    async def is_complex_request(self, text: str) -> bool:
        """
        Gemini complexity check:
        Does this require more than one tool? Reply yes/no.
        """
        if self._llm is None:
            return False

        prompt = f"Does this require more than one tool? Reply yes/no: '{text}'"
        try:
            reply = (await self._ask_gemini(prompt)).strip().lower()
            return reply.startswith("yes")
        except Exception as exc:
            logger.warning(f"Complexity check failed, defaulting to simple route: {exc}")
            return False

    def _build_llm(self):
        if not LANGCHAIN_GEMINI_AVAILABLE:
            logger.warning("langchain-google-genai unavailable; ReasoningEngine LLM disabled")
            return None

        api_key = get_gemini_api_key(self._get_config)
        if not api_key:
            logger.warning("Gemini API key not configured; ReasoningEngine disabled")
            return None

        model = self._get_config("reasoning.gemini.model", "gemini-1.5-flash")
        temperature = float(self._get_config("reasoning.gemini.temperature", 0.1))
        try:
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=temperature,
            )
        except Exception as exc:
            logger.error(f"Failed to initialize ReasoningEngine LLM: {exc}")
            return None

    def _build_graph(self):
        if not LANGGRAPH_AVAILABLE:
            logger.warning("langgraph not installed; ReasoningEngine graph disabled")
            return None

        graph = StateGraph(VesperState)
        graph.add_node("planner", self.planner_node)
        graph.add_node("tool_selector", self.tool_selector_node)
        graph.add_node("executor", self.executor_node)
        graph.add_node("verifier", self.verifier_node)
        graph.add_node("responder", self.responder_node)

        graph.set_entry_point("planner")
        graph.add_edge("planner", "tool_selector")
        graph.add_edge("tool_selector", "executor")
        graph.add_edge("executor", "verifier")
        graph.add_conditional_edges(
            "verifier",
            lambda s: "responder" if s.get("complete") else "tool_selector",
            {
                "responder": "responder",
                "tool_selector": "tool_selector",
            },
        )
        graph.add_edge("responder", END)
        return graph.compile()

    async def planner_node(self, state: VesperState) -> VesperState:
        """
        Break user request into atomic steps.
        Expected output format:
          [{action, tool, params}, ...]
        """
        prompt = (
            "Break this request into atomic steps for a macOS assistant: "
            f"'{state['user_input']}'. Return a JSON list of steps. "
            "Each step: {action, tool, params}"
        )

        steps_raw: List[Dict[str, Any]] = []
        try:
            planner_text = await self._ask_gemini(prompt)
            parsed = self._parse_json(planner_text)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        steps_raw.append(item)
        except Exception as exc:
            logger.warning(f"Planner node failed: {exc}")

        if not steps_raw:
            state["needs_clarification"] = True
            state["clarification_question"] = (
                "I need a little clarification to execute that. Could you restate it in smaller steps?"
            )
            state["plan"] = []
            state["plan_objects"] = []  # type: ignore[index]
            state["complete"] = True  # type: ignore[index]
            await self._emit_hud_state("planner", state)
            return state

        state["plan_objects"] = steps_raw  # type: ignore[index]
        state["plan"] = [self._format_step(step, idx + 1) for idx, step in enumerate(steps_raw)]
        state["current_step"] = 0
        state["complete"] = False  # type: ignore[index]
        await self._emit_hud_state("planner", state)
        return state

    async def tool_selector_node(self, state: VesperState) -> VesperState:
        """Select tool based on current plan action."""
        steps: List[Dict[str, Any]] = state.get("plan_objects", [])  # type: ignore[assignment]
        idx = int(state.get("current_step", 0))

        if idx >= len(steps):
            state["selected_tool"] = ""  # type: ignore[index]
            state["complete"] = True  # type: ignore[index]
            await self._emit_hud_state("tool_selector", state)
            return state

        step = steps[idx]
        action = str(step.get("action", "")).strip().lower()
        selected = self._map_action_to_tool(action, step)

        state["selected_tool"] = selected  # type: ignore[index]
        state["selected_action"] = action  # type: ignore[index]
        state["selected_params"] = step.get("params", {})  # type: ignore[index]
        await self._emit_hud_state("tool_selector", state)
        return state

    async def executor_node(self, state: VesperState) -> VesperState:
        """Emit event for selected tool and wait for result via asyncio.Event."""
        steps: List[Dict[str, Any]] = state.get("plan_objects", [])  # type: ignore[assignment]
        idx = int(state.get("current_step", 0))
        if idx >= len(steps):
            state["complete"] = True  # type: ignore[index]
            await self._emit_hud_state("executor", state)
            return state

        selected_tool = str(state.get("selected_tool", "")).strip().lower()
        step = steps[idx]
        step_text = self._step_to_command(step)
        correlation_id = state.get("correlation_id")  # type: ignore[assignment]

        result = await self._dispatch_step(
            selected_tool=selected_tool,
            step_text=step_text,
            params=step.get("params", {}),
            correlation_id=correlation_id,
        )

        results = list(state.get("tool_results", []))
        results.append(result)
        state["tool_results"] = results
        state["current_step"] = idx + 1
        await self._emit_hud_state("executor", state)
        return state

    async def verifier_node(self, state: VesperState) -> VesperState:
        """Verify whether task is complete."""
        if state.get("needs_clarification"):
            state["complete"] = True  # type: ignore[index]
            await self._emit_hud_state("verifier", state)
            return state

        # Hard guard: once all steps are done, this should be complete.
        if int(state.get("current_step", 0)) >= len(state.get("plan", [])):
            state["complete"] = True  # type: ignore[index]
            await self._emit_hud_state("verifier", state)
            return state

        prompt = (
            "Given the plan "
            f"{state.get('plan', [])} and results so far {state.get('tool_results', [])}, "
            "is the task complete? Return JSON: {complete: bool, reason: str}"
        )
        complete = False
        try:
            verifier_text = await self._ask_gemini(prompt)
            payload = self._parse_json(verifier_text)
            if isinstance(payload, dict):
                complete = bool(payload.get("complete", False))
                state["verification_reason"] = str(payload.get("reason", "")).strip()  # type: ignore[index]
        except Exception as exc:
            logger.warning(f"Verifier node failed: {exc}")

        state["complete"] = complete  # type: ignore[index]
        await self._emit_hud_state("verifier", state)
        return state

    async def responder_node(self, state: VesperState) -> VesperState:
        """Create final spoken response and emit VoiceOutputEvent."""
        if state.get("needs_clarification"):
            final_response = state.get("clarification_question") or (
                "I need clarification before I can continue."
            )
        else:
            summary_prompt = (
                "Synthesize these execution results into a natural spoken response. "
                "Be concise and explicit about what succeeded.\n\n"
                f"User request: {state.get('user_input', '')}\n"
                f"Results: {state.get('tool_results', [])}"
            )
            try:
                final_response = await self._ask_gemini(summary_prompt)
            except Exception as exc:
                logger.warning(f"Responder synthesis failed: {exc}")
                final_response = "I completed the workflow, but I couldn't generate a polished summary."

        state["final_response"] = final_response.strip()
        correlation_id = state.get("correlation_id")  # type: ignore[assignment]

        await self._event_bus.emit(
            VoiceOutputEvent(
                text=state["final_response"],
                source="ReasoningEngine",
                correlation_id=correlation_id,
            )
        )
        await self._emit_hud_state("responder", state)
        return state

    async def _dispatch_step(
        self,
        selected_tool: str,
        step_text: str,
        params: Dict[str, Any],
        correlation_id: Optional[UUID],
    ) -> str:
        """
        Emit step request via EventBus and wait for the first VoiceOutputEvent.
        """
        completion = asyncio.Event()
        holder: Dict[str, str] = {"result": ""}

        async def _on_voice_output(event: VoiceOutputEvent) -> None:
            if correlation_id and event.correlation_id != correlation_id:
                return
            if event.source == "ReasoningEngine":
                return
            holder["result"] = (event.text or "").strip()
            completion.set()

        token = self._event_bus.subscribe(VoiceOutputEvent, _on_voice_output)
        try:
            command_text = self._build_tool_command(selected_tool, step_text, params)
            await self._emit_tool_event(selected_tool, command_text, params, correlation_id)
            await asyncio.wait_for(completion.wait(), timeout=20.0)
        except asyncio.TimeoutError:
            holder["result"] = f"Timed out waiting for result from tool '{selected_tool}'."
        except Exception as exc:
            holder["result"] = f"Tool execution error for '{selected_tool}': {exc}"
        finally:
            token.unsubscribe()

        return holder["result"] or f"Step executed with '{selected_tool}'."

    async def _emit_tool_event(
        self,
        selected_tool: str,
        step_text: str,
        params: Dict[str, Any],
        correlation_id: Optional[UUID],
    ) -> None:
        intent_name = "GENERAL_QUESTION"
        if selected_tool == "web_search":
            intent_name = "SEARCH_WEB"
        elif selected_tool in {"safari", "imessage", "finder", "calendar", "spotify"}:
            intent_name = "MACOS_CONTROL"
        elif selected_tool == "vision":
            intent_name = "VISION"

        event = IntentRecognizedEvent(
            intent=intent_name,
            confidence=1.0,
            entities=params or {},
            raw_text=step_text,
            slots={k: str(v) for k, v in (params or {}).items()},
            source="ReasoningEngine",
            correlation_id=correlation_id,
        )
        await self._event_bus.emit(event)

    async def _emit_hud_state(self, node_name: str, state: VesperState) -> None:
        plan_steps = []
        current_step = int(state.get("current_step", 0))
        for idx, step in enumerate(state.get("plan", [])):
            marker = "✓" if idx < current_step else "○"
            plan_steps.append(f"{marker} {step}")

        correlation_id = state.get("correlation_id")  # type: ignore[assignment]
        await self._event_bus.emit(
            HUDGraphStateEvent(
                current_node=node_name,
                plan_steps=plan_steps,
                tool_results=list(state.get("tool_results", [])),
                source="ReasoningEngine",
                correlation_id=correlation_id,
            )
        )

    async def _ask_gemini(self, prompt: str) -> str:
        if self._llm is None:
            raise RuntimeError("Gemini LLM is not configured")
        response = await self._llm.ainvoke(prompt)
        content = getattr(response, "content", "")
        if isinstance(content, list):
            text = " ".join(str(part) for part in content)
        else:
            text = str(content)
        return text.strip()

    def _parse_json(self, text: str) -> Any:
        candidate = (text or "").strip()
        if not candidate:
            return None
        if "```" in candidate:
            candidate = candidate.replace("```json", "").replace("```", "").strip()
        if candidate.startswith("{") or candidate.startswith("["):
            return json.loads(candidate)
        start = min([i for i in [candidate.find("["), candidate.find("{")] if i != -1], default=-1)
        if start != -1:
            maybe = candidate[start:]
            end_brace = maybe.rfind("}")
            end_bracket = maybe.rfind("]")
            end = max(end_brace, end_bracket)
            if end != -1:
                return json.loads(maybe[: end + 1])
        return None

    def _map_action_to_tool(self, action: str, step: Dict[str, Any]) -> str:
        for prefix, tool in self._tool_map.items():
            if action == prefix or action.startswith(prefix):
                return tool

        tool_hint = str(step.get("tool", "")).strip().lower()
        if tool_hint:
            return tool_hint
        return "web_search"

    @staticmethod
    def _format_step(step: Dict[str, Any], index: int) -> str:
        action = str(step.get("action", "action")).strip() or "action"
        params = step.get("params", {})
        if isinstance(params, dict) and params:
            details = ", ".join(f"{k}={v}" for k, v in params.items())
            return f"{index}. {action} ({details})"
        return f"{index}. {action}"

    @staticmethod
    def _step_to_command(step: Dict[str, Any]) -> str:
        action = str(step.get("action", "")).strip()
        params = step.get("params", {})
        if isinstance(params, dict) and params:
            serialized = ", ".join(f"{k}={v}" for k, v in params.items())
            if action:
                return f"{action} {serialized}".strip()
            return serialized
        return action or "execute step"

    @staticmethod
    def _build_tool_command(selected_tool: str, step_text: str, params: Dict[str, Any]) -> str:
        params = params or {}
        query = (
            str(params.get("query") or "")
            or str(params.get("target") or "")
            or str(params.get("item") or "")
            or str(params.get("topic") or "")
            or str(params.get("url") or "")
            or step_text
        ).strip()

        if selected_tool == "web_search":
            return f"search for {query}".strip()

        if selected_tool == "safari":
            if str(params.get("url", "")).strip():
                return f"open {params.get('url')} in safari".strip()
            return f"search for {query}".strip()

        if selected_tool == "imessage":
            recipient = str(params.get("recipient") or params.get("to") or "contact").strip()
            message = str(params.get("message") or params.get("text") or query).strip()
            return f"message {recipient} saying {message}".strip()

        if selected_tool == "finder":
            folder = str(params.get("folder") or "").strip()
            file_name = str(params.get("file") or query).strip()
            if folder:
                return f"open folder {folder}".strip()
            return f"find file {file_name}".strip()

        if selected_tool == "calendar":
            title = str(params.get("title") or params.get("event") or query).strip()
            date = str(params.get("date") or params.get("day") or "").strip()
            time = str(params.get("time") or "").strip()
            if date and time:
                return f"add event {title} on {date} at {time}".strip()
            return f"schedule {title}".strip()

        if selected_tool == "spotify":
            track = str(params.get("track") or params.get("song") or query).strip()
            return f"play {track} on spotify".strip()

        if selected_tool == "vision":
            target = str(params.get("target") or params.get("query") or "").strip()
            if target:
                return f"find {target} on screen".strip()
            return "describe screen"

        return step_text

    def _get_config(self, key: str, default: Any = None) -> Any:
        value: Any = self._config
        for part in key.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(part)
            if value is None:
                return default
        return value
