"""
LangGraph-based orchestrator.

DEPRECATED / UNUSED (P03): orchestrator/__init__.py now exports Brain from
orchestrator.brain, which delegates every user input to orchestrator.planner
(an LLM tool-calling loop) instead of the regex-pattern routing this class
uses (_select_route / _is_event_bus_intent / RAG_INTENTS / *_PATTERNS).
Nothing in the codebase imports this module anymore. Kept only in case its
LangGraph StateGraph structure is wanted again later; not wired into main.py.
"""

# =============================================================================
# LangGraph Brain (superseded — see module docstring)
# =============================================================================

from __future__ import annotations

import asyncio
import re
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph

from agents.base_agent import BaseAgent
from agents.memory_agent import MemoryAgent
try:
    from agents.vision_agent import VisionAgent

    VISION_AVAILABLE = True
except ImportError:
    VisionAgent = None  # type: ignore
    VISION_AVAILABLE = False
try:
    from agents.web_search_agent import WebSearchAgent

    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WebSearchAgent = None  # type: ignore
    WEB_SEARCH_AVAILABLE = False
try:
    from agents.macos_control_agent import MacOSControlAgent

    MACOS_CONTROL_AVAILABLE = True
except ImportError:
    MacOSControlAgent = None  # type: ignore
    MACOS_CONTROL_AVAILABLE = False
from agents.rag_agent import RAGAgent
from agents.system_agent import SystemAgent
from agents.tool_agent import ToolAgent
from agents.voice_agent import VoiceAgent
from api.health import HealthServer
from bus.event_bus import EventBus, SubscriptionToken, get_event_bus
from schemas.events import IntentRecognizedEvent, VoiceInputEvent, VoiceOutputEvent
from ui.hud_overlay import HUDOverlayController
from utils.logger import get_logger
from utils.prompts import FallbackPatterns


logger = get_logger(__name__)


class BrainState(Enum):
    """Lifecycle state for the orchestrator."""

    INITIALIZING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()


class OrchestratorState(TypedDict):
    """Shared LangGraph state flowing across all nodes."""

    correlation_id: UUID
    raw_text: str
    intent: str
    confidence: float
    entities: Dict[str, Any]
    route: str
    rag_context: Dict[str, Any]
    tool_result: Dict[str, Any]
    memory_snapshot: Dict[str, Any]
    response_text: str
    errors: List[str]
    trace: List[str]


RAG_INTENTS = {"GENERAL_QUESTION", "RECALL_MEMORY", "HELP"}
VISION_PATTERNS = [
    r"\bwhat(?:'| i)?s on my screen\b",
    r"\bread my screen\b",
    r"\bdescribe (?:my )?screen\b",
    r"\bread that\b",
    r"\bwhat does it say\b",
    r"\bfind .+ on screen\b",
    r"\bwhere is .+\b",
    r"\bclick .+\b",
]
WEB_SEARCH_PATTERNS = [
    r"\bsearch for .+\b",
    r"\blook up .+\b",
    r"\bwhat is .+\b",
    r"\btell me about .+\b",
    r"\blatest news on .+\b",
    r"\bwho is .+\b",
    r"\bhow does .+ work\b",
]
MACOS_CONTROL_PATTERNS = [
    r"\bsend message to .+\b",
    r"\btext .+\b",
    r"\bmessage .+ saying .+\b",
    r"\bopen .+ in safari\b",
    r"\bopen folder .+\b",
    r"\bfind file .+\b",
    r"\bopen downloads\b",
    r"\bcreate folder .+ in .+\b",
    r"\bmove file .+ to .+\b",
    r"\bdelete file .+\b",
    r"\badd event .+ on .+ at .+\b",
    r"\bwhat(?:'| i)?s on my calendar\b",
    r"\bschedule .+\b",
    r"\bplay .+ on spotify\b",
    r"\bpause music\b",
    r"\bnext song\b",
    r"\bplay playlist .+\b",
    r"\bwhat(?:'| i)?s playing\b",
    r"\bset volume to \d+\b",
    r"\bset brightness to \d+\b",
    r"\bturn (?:off|on) wifi\b",
    r"\benable dark mode\b",
    r"\bwhat(?:'| i)?s my battery\b",
]


class Brain:
    """
    LangGraph-based orchestrator replacing the legacy Brain runtime.

    Graph nodes:
      - IntentAgent
      - RAGAgent
      - ToolAgent
      - MemoryAgent
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.config = config or {}
        self.event_bus = event_bus or get_event_bus()
        self._state = BrainState.INITIALIZING
        self._subscriptions: List[SubscriptionToken] = []

        # Keep shape compatible with existing main.py logging
        self._agents: Dict[str, BaseAgent] = {}

        # Node adapters
        self._rag_agent: Optional[RAGAgent] = None
        self._tool_agent: Optional[ToolAgent] = None
        self._hud_overlay: Optional[HUDOverlayController] = None
        self._health_server: Optional[HealthServer] = None

        self._workflow = self._build_workflow()

    @property
    def is_running(self) -> bool:
        return self._state == BrainState.RUNNING

    async def start(self) -> None:
        """Start core agents and subscribe to voice input."""
        logger.info("[LangGraphBrain] starting")

        # Register core agents used by the graph.
        memory_agent = MemoryAgent(event_bus=self.event_bus, config=self.config)
        system_agent = SystemAgent(event_bus=self.event_bus, config=self.config)
        voice_agent = VoiceAgent(event_bus=self.event_bus, config=self.config)
        vision_agent = None
        web_search_agent = None
        macos_control_agent = None
        vision_cfg = (self.config or {}).get("vision", {})
        web_cfg = (self.config or {}).get("web_search", {})
        macos_cfg = (self.config or {}).get("macos_control", {})
        if vision_cfg.get("enabled", False) and VISION_AVAILABLE:
            vision_agent = VisionAgent(event_bus=self.event_bus, config=self.config)
        if web_cfg.get("enabled", True) and WEB_SEARCH_AVAILABLE:
            web_search_agent = WebSearchAgent(event_bus=self.event_bus, config=self.config)
        if macos_cfg.get("enabled", True) and MACOS_CONTROL_AVAILABLE:
            macos_control_agent = MacOSControlAgent(event_bus=self.event_bus, config=self.config)

        self._agents = {
            "MemoryAgent": memory_agent,
            "SystemAgent": system_agent,
            "VoiceAgent": voice_agent,
        }
        if vision_agent is not None:
            self._agents["VisionAgent"] = vision_agent
        if web_search_agent is not None:
            self._agents["WebSearchAgent"] = web_search_agent
        if macos_control_agent is not None:
            self._agents["MacOSControlAgent"] = macos_control_agent

        for agent in self._agents.values():
            await agent.start()

        self._health_server = HealthServer(lambda: self._agents)
        self._health_server.start()

        self._rag_agent = RAGAgent(memory_agent=memory_agent)
        self._tool_agent = ToolAgent(
            system_agent=system_agent,
            memory_agent=memory_agent,
            event_bus=self.event_bus,
        )

        token = self.event_bus.subscribe(VoiceInputEvent, self._on_voice_input)
        self._subscriptions.append(token)

        # HUD overlay runs in its own thread and listens to EventBus updates.
        try:
            self._hud_overlay = HUDOverlayController(event_bus=self.event_bus, config=self.config)
            await self._hud_overlay.start()
            self._hud_overlay.set_agent_health("IntentAgent", True)
        except Exception as exc:
            logger.warning(f"[LangGraphBrain] HUD overlay could not start: {exc}")
            self._hud_overlay = None

        self._state = BrainState.RUNNING

        await self.event_bus.emit(
            VoiceOutputEvent(
                text="Vesper online and ready to assist, Sir.",
                source="LangGraphBrain",
            )
        )

        logger.info(
            "[LangGraphBrain] running with nodes: IntentAgent -> (RAGAgent|ToolAgent|EventBusAgent) -> MemoryAgent"
        )

    async def stop(self, reason: str = "Normal shutdown") -> None:
        """Stop subscriptions and agents gracefully."""
        logger.info(f"[LangGraphBrain] stopping: {reason}")
        self._state = BrainState.STOPPING

        for token in self._subscriptions:
            token.unsubscribe()
        self._subscriptions.clear()

        for agent in reversed(list(self._agents.values())):
            await agent.stop(reason)

        if self._hud_overlay:
            await self._hud_overlay.stop()
            self._hud_overlay = None

        if self._health_server:
            self._health_server.stop()
            self._health_server = None

        self._agents.clear()
        self._state = BrainState.STOPPED
        logger.info("[LangGraphBrain] stopped")

    async def execute(self, text: str, correlation_id: Optional[UUID] = None) -> OrchestratorState:
        """Run one async LangGraph execution for a text input."""
        cid = correlation_id or uuid4()
        initial: OrchestratorState = {
            "correlation_id": cid,
            "raw_text": text,
            "intent": "UNKNOWN",
            "confidence": 0.0,
            "entities": {},
            "route": "ToolAgent",
            "rag_context": {},
            "tool_result": {},
            "memory_snapshot": {},
            "response_text": "",
            "errors": [],
            "trace": [],
        }
        result = await self._workflow.ainvoke(initial)
        return result

    async def _on_voice_input(self, event: VoiceInputEvent) -> None:
        """EventBus entrypoint: run graph then emit response speech event."""
        try:
            final_state = await self.execute(text=event.text, correlation_id=event.event_id)
            response_text = (final_state.get("response_text") or "").strip()
            # Event-bus delegated routes (vision/web/macos) emit their own voice output.
            if response_text:
                await self.event_bus.emit(
                    VoiceOutputEvent(
                        text=response_text,
                        source="LangGraphBrain",
                        correlation_id=event.event_id,
                    )
                )
        except Exception as exc:
            logger.error(f"[LangGraphBrain] workflow failure: {exc}", exc_info=True)
            await self.event_bus.emit(
                VoiceOutputEvent(
                    text="I hit an error while processing that request.",
                    source="LangGraphBrain",
                    correlation_id=event.event_id,
                )
            )

    def _build_workflow(self):
        """Build and compile the LangGraph workflow."""
        graph = StateGraph(OrchestratorState)
        graph.add_node("IntentAgent", self._intent_agent_node)
        graph.add_node("RAGAgent", self._rag_agent_node)
        graph.add_node("ToolAgent", self._tool_agent_node)
        graph.add_node("EventBusAgent", self._event_bus_agent_node)
        graph.add_node("MemoryAgent", self._memory_agent_node)

        graph.add_edge(START, "IntentAgent")
        graph.add_conditional_edges(
            "IntentAgent",
            self._route_after_intent,
            {
                "RAGAgent": "RAGAgent",
                "ToolAgent": "ToolAgent",
                "EventBusAgent": "EventBusAgent",
            },
        )
        graph.add_edge("RAGAgent", "MemoryAgent")
        graph.add_edge("ToolAgent", "MemoryAgent")
        graph.add_edge("EventBusAgent", "MemoryAgent")
        graph.add_edge("MemoryAgent", END)
        return graph.compile()

    async def _intent_agent_node(self, state: OrchestratorState) -> OrchestratorState:
        logger.info("[NODE] IntentAgent")

        matches = FallbackPatterns.match(state["raw_text"])
        top = matches[0] if matches else {"intent": "UNKNOWN", "confidence": 0.0, "entities": {}}

        intent = str(top.get("intent", "UNKNOWN"))
        confidence = float(top.get("confidence", 0.0))
        entities = dict(top.get("entities", {}))
        route = self._select_route(intent=intent, raw_text=state["raw_text"])

        # Broadcast only for routes handled by event-driven agents.
        if route == "EventBusAgent":
            await self.event_bus.emit(
                IntentRecognizedEvent(
                    intent=intent,
                    confidence=confidence,
                    entities=entities,
                    raw_text=state["raw_text"],
                    slots=entities,
                    source="LangGraphBrain.IntentAgent",
                    correlation_id=state["correlation_id"],
                )
            )
        return {
            **state,
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "route": route,
            "trace": [*state["trace"], f"IntentAgent:{intent}"],
        }

    def _route_after_intent(self, state: OrchestratorState) -> str:
        return state["route"]

    def _select_route(self, intent: str, raw_text: str) -> str:
        if self._is_event_bus_intent(raw_text):
            return "EventBusAgent"
        if intent in RAG_INTENTS or intent == "UNKNOWN":
            return "RAGAgent"
        return "ToolAgent"

    def _is_event_bus_intent(self, raw_text: str) -> bool:
        text = (raw_text or "").strip().lower()
        if not text:
            return False
        pattern_sets = (VISION_PATTERNS, WEB_SEARCH_PATTERNS, MACOS_CONTROL_PATTERNS)
        return any(re.search(pattern, text) for patterns in pattern_sets for pattern in patterns)

    async def _rag_agent_node(self, state: OrchestratorState) -> OrchestratorState:
        logger.info("[NODE] RAGAgent")
        if not self._rag_agent:
            return {
                **state,
                "errors": [*state["errors"], "RAGAgent unavailable"],
                "trace": [*state["trace"], "RAGAgent:missing"],
            }

        rag = await self._rag_agent.retrieve(query=state["raw_text"], intent=state["intent"])
        response = rag.get("answer") or "I found relevant context."

        return {
            **state,
            "rag_context": rag,
            "response_text": response,
            "trace": [*state["trace"], "RAGAgent:ok"],
        }

    async def _event_bus_agent_node(self, state: OrchestratorState) -> OrchestratorState:
        logger.info("[NODE] EventBusAgent")
        return {
            **state,
            "response_text": "",
            "trace": [*state["trace"], "EventBusAgent:delegated"],
        }

    async def _tool_agent_node(self, state: OrchestratorState) -> OrchestratorState:
        logger.info("[NODE] ToolAgent")
        if not self._tool_agent:
            return {
                **state,
                "errors": [*state["errors"], "ToolAgent unavailable"],
                "response_text": "I can't execute tools right now.",
                "trace": [*state["trace"], "ToolAgent:missing"],
            }

        result = await self._tool_agent.execute(
            intent=state["intent"],
            entities=state["entities"],
            raw_text=state["raw_text"],
        )

        success = bool(result.get("success"))
        response = str(result.get("result") or result.get("error") or "Done")
        next_errors = state["errors"] if success else [*state["errors"], str(result.get("error", "unknown error"))]

        return {
            **state,
            "tool_result": result,
            "response_text": response,
            "errors": next_errors,
            "trace": [*state["trace"], f"ToolAgent:{'ok' if success else 'error'}"],
        }

    async def _memory_agent_node(self, state: OrchestratorState) -> OrchestratorState:
        logger.info("[NODE] MemoryAgent")
        memory_agent = self._agents.get("MemoryAgent")
        snapshot: Dict[str, Any] = {}

        if isinstance(memory_agent, MemoryAgent):
            snapshot = {
                "last_command": memory_agent.get_last_command(),
                "recent_conversation": memory_agent.get_recent_conversation(max_turns=3),
                "frequent_apps": memory_agent.get_frequent_apps(limit=3),
            }

            # Persist this graph response for later retrieval.
            if memory_agent.store:
                memory_agent.store.store(
                    memory_type="short_term",
                    category="conversation",
                    key=f"graph_response_{uuid4()}",
                    value={
                        "role": "assistant",
                        "text": state["response_text"],
                        "intent": state["intent"],
                    },
                    ttl_seconds=7200,
                )

        return {
            **state,
            "memory_snapshot": snapshot,
            "trace": [*state["trace"], "MemoryAgent:ok"],
        }


LangGraphBrain = Brain


def create_brain(config: Optional[Dict[str, Any]] = None) -> Brain:
    """Factory compatible with previous orchestrator bootstrap style."""
    return Brain(config=config)

