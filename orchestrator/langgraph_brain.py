"""LangGraph-based production orchestrator."""

# =============================================================================
# Production LangGraph Brain (active implementation)
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

"""

LangGraph Brain Implementation - Orchestrator replacement using LangGraph.

This module provides a complete LangGraph-based orchestrator that replaces the
event-driven Brain while maintaining full compatibility with the existing event bus.

Key features:
- Deterministic state transitions via LangGraph graph
- Type-safe state schema (WorkflowState)
- Retry and fallback logic built into graph structure
- Full integration with existing event bus
- Observable execution flow

Installation:
    pip install langgraph langchain

Usage:
    from orchestrator.langgraph_brain import LangGraphBrain, build_workflow
    
    # Create orchestrator
    brain = LangGraphBrain(
        event_bus=event_bus,
        agents={
            "SystemAgent": system_agent,
            "IntentAgent": intent_agent,
            "VisionAgent": vision_agent,
            "MemoryAgent": memory_agent,
        },
    )
    
    # Subscribe and let events flow
    # When VoiceInputEvent arrives, brain processes it through the graph
"""

# from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict
from uuid import UUID, uuid4
from datetime import datetime

from langgraph.graph import StateGraph, END
CompiledGraph = Any

from bus.event_bus import EventBus
from agents.base_agent import BaseAgent
from agents.intent_agent import IntentAgent
from schemas.events import (
    ActionRequestEvent,
    ActionResultEvent,
    IntentRecognizedEvent,
    VoiceInputEvent,
    VoiceOutputEvent,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# State Schema
# =============================================================================

class WorkflowState(TypedDict):
    """
    Unified state passed through LangGraph workflow.
    
    This represents the complete context of a user request from voice input
    through response generation.
    """
    
    # Input
    user_input: str                     # Raw user text
    correlation_id: UUID                # For event tracing
    
    # Intent classification
    intent: Optional[str]               # Parsed intent (e.g., "OPEN_APP")
    intent_confidence: float            # Confidence score (0.0 - 1.0)
    entities: Dict[str, Any]            # Extracted parameters (e.g., {"app": "Safari"})
    raw_text: str                       # Original utterance for fallback
    
    # Routing decision
    target_agent: Optional[str]         # Which agent handles this
    action: Optional[str]               # Specific action for the agent
    is_multi_step: bool                 # Complex command needing a plan
    
    # Execution state
    action_results: List[Dict]          # Results from each agent call
    current_step: int                   # Current step in multi-step plan
    max_retries: int                    # Retry budget
    retry_count: int                    # Retries used
    
    # Error handling
    last_error: Optional[str]           # Last error message
    error_count: int                    # Total errors in workflow
    should_clarify: bool                # Ask user for clarification?
    clarification_options: List[str]    # Clarification prompts
    
    # Response
    response_text: str                  # Final response to user
    context_snapshot: Dict              # Conversation snapshot
    events_to_emit: List[Any]           # Events to emit


# =============================================================================
# Intent Routing Configuration (from Brain)
# =============================================================================

INTENT_ROUTING: Dict[str, Dict[str, Any]] = {
    # System agent intents
    "OPEN_APP": {"agent": "SystemAgent", "action": "open_app"},
    "CLOSE_APP": {"agent": "SystemAgent", "action": "close_app"},
    "CONTROL_VOLUME": {"agent": "SystemAgent", "action": "control_volume"},
    "SET_VOLUME": {"agent": "SystemAgent", "action": "control_volume"},
    "VOLUME_UP": {"agent": "SystemAgent", "action": "volume_up"},
    "VOLUME_DOWN": {"agent": "SystemAgent", "action": "volume_down"},
    "GET_TIME": {"agent": "SystemAgent", "action": "get_time"},
    "GET_DATE": {"agent": "SystemAgent", "action": "get_date"},
    "SYSTEM_INFO": {"agent": "SystemAgent", "action": "system_info"},
    "TAKE_SCREENSHOT": {"agent": "SystemAgent", "action": "screenshot"},
    
    # Vision agent intents
    "START_VISION": {"agent": "VisionAgent", "action": "start"},
    "STOP_VISION": {"agent": "VisionAgent", "action": "stop"},
    "RECOGNIZE_FACE": {"agent": "VisionAgent", "action": "recognize_face"},
    "ENROLL_FACE": {"agent": "VisionAgent", "action": "enroll_face"},
    
    # Memory agent intents
    "RECALL_MEMORY": {"agent": "MemoryAgent", "action": "recall"},
    "SAVE_MEMORY": {"agent": "MemoryAgent", "action": "save"},
    
    # Meta intents (handled by Brain, not delegated)
    "HELP": {"agent": "Brain", "action": "help"},
    "STATUS": {"agent": "Brain", "action": "status"},
    "GREETING": {"agent": "Brain", "action": "greeting"},
}

# Multi-step command patterns
MULTI_STEP_PATTERNS = {
    "OPEN_APP_AND_NAVIGATE": ["open_app", "open_url"],
    "RECORD_AND_SAVE": ["start_recording", "save_file"],
}


# =============================================================================
# Helper Functions
# =============================================================================

def _is_multi_step_command(intent: str, entities: Dict[str, Any]) -> bool:
    """
    Detect if this intent requires multiple steps.
    
    Multi-step examples:
    - "Open Safari and go to YouTube" (open app + navigate)
    - "Record for 5 seconds and save as meeting.wav" (record + save)
    """
    return intent in MULTI_STEP_PATTERNS


def _get_steps_for_intent(intent: str, entities: Dict[str, Any]) -> List[Dict]:
    """
    Generate step list for multi-step intent.
    
    Returns list of dicts with:
      - agent: target agent name
      - action: action to invoke
      - params: parameters for action
    """
    steps = []
    
    if intent == "OPEN_APP_AND_NAVIGATE":
        steps.append({
            "agent": "SystemAgent",
            "action": "open_app",
            "params": {"app": entities.get("app", "")},
        })
        steps.append({
            "agent": "SystemAgent",
            "action": "open_url",
            "params": {"url": entities.get("url", "")},
        })
    
    elif intent == "RECORD_AND_SAVE":
        steps.append({
            "agent": "SystemAgent",
            "action": "start_recording",
            "params": {"duration": entities.get("duration", 10)},
        })
        steps.append({
            "agent": "SystemAgent",
            "action": "save_file",
            "params": {"filename": entities.get("filename", "recording.wav")},
        })
    
    return steps


# =============================================================================
# Node Implementations
# =============================================================================

def capture_input(state: WorkflowState) -> WorkflowState:
    """Initialize workflow state from user input."""
    logger.info(f"[NODE] capture_input: {state['user_input'][:60]}...")
    
    # Ensure all fields are initialized
    return {
        **state,
        "intent": None,
        "intent_confidence": 0.0,
        "entities": {},
        "raw_text": state.get("user_input", ""),
        "target_agent": None,
        "action": None,
        "is_multi_step": False,
        "action_results": [],
        "current_step": 0,
        "max_retries": 3,
        "retry_count": 0,
        "last_error": None,
        "error_count": 0,
        "should_clarify": False,
        "clarification_options": [],
        "response_text": "",
        "context_snapshot": {},
        "events_to_emit": [],
    }


async def classify_intent(
    state: WorkflowState,
    event_bus: EventBus,
    intent_agent: IntentAgent,
    **kwargs,
) -> WorkflowState:
    """Classify user input into an intent."""
    logger.info("[NODE] classify_intent")
    
    user_input = state["user_input"]
    
    try:
        # Call IntentAgent to classify
        # (In real implementation, you'd call intent_agent.classify() directly
        #  or emit IntentRecognizedEvent and wait for response)
        
        # For now, simulate by calling a sync method
        intent_result = intent_agent.classify_sync(user_input) if hasattr(intent_agent, "classify_sync") else {
            "intent": "UNKNOWN",
            "confidence": 0.0,
            "entities": {},
        }
        
        # Update state
        state = {
            **state,
            "intent": intent_result.get("intent"),
            "intent_confidence": intent_result.get("confidence", 0.0),
            "entities": intent_result.get("entities", {}),
            "raw_text": user_input,
        }
        
        logger.info(f"Classified as: {state['intent']} (confidence: {state['intent_confidence']})")
        
        # Emit event for other agents listening to event bus
        await event_bus.emit(IntentRecognizedEvent(
            intent=state["intent"] or "",
            slots=state["entities"],
            raw_text=user_input,
            correlation_id=state["correlation_id"],
            source="LangGraphBrain",
        ))
        
        return state
        
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return {
            **state,
            "last_error": str(e),
            "error_count": state["error_count"] + 1,
            "should_clarify": True,
            "clarification_options": ["I didn't quite understand that. Could you rephrase?"],
        }


def route_intent(state: WorkflowState) -> WorkflowState:
    """Route classified intent to appropriate agent."""
    logger.info(f"[NODE] route_intent: {state.get('intent')}")
    
    if not state.get("intent"):
        return {
            **state,
            "should_clarify": True,
            "clarification_options": ["I didn't catch that. What would you like me to do?"],
        }
    
    routing = INTENT_ROUTING.get(state["intent"])
    
    if not routing:
        logger.warning(f"No routing for intent: {state['intent']}")
        return {
            **state,
            "should_clarify": True,
            "clarification_options": [f"I'm not sure how to handle '{state['intent']}'."],
        }
    
    # Check if multi-step
    is_multi_step = _is_multi_step_command(state["intent"], state["entities"])
    
    return {
        **state,
        "target_agent": routing["agent"],
        "action": routing["action"],
        "is_multi_step": is_multi_step,
    }


def plan_multi_step(state: WorkflowState) -> WorkflowState:
    """Create multi-step execution plan for complex commands."""
    logger.info(f"[NODE] plan_multi_step: {state['intent']}")
    
    steps = _get_steps_for_intent(state["intent"], state["entities"])
    
    if not steps:
        logger.warning("No steps generated for multi-step intent")
        return state
    
    # Initialize action results with pending status
    action_results = [
        {"step": i, "status": "pending", "agent": step["agent"], "action": step["action"]}
        for i, step in enumerate(steps)
    ]
    
    return {
        **state,
        "action_results": action_results,
        "current_step": 0,
    }


async def execute_action(
    state: WorkflowState,
    event_bus: EventBus,
    **kwargs,
) -> WorkflowState:
    """Execute an action by emitting ActionRequestEvent and waiting for result."""
    logger.info(f"[NODE] execute_action: {state['target_agent']}.{state['action']}")
    
    if not state["target_agent"] or not state["action"]:
        return {
            **state,
            "last_error": "No target agent or action specified",
            "error_count": state["error_count"] + 1,
        }
    
    target_agent = state["target_agent"]
    action = state["action"]
    parameters = state["entities"]
    
    # Set up async event to signal when result arrives
    result_received = asyncio.Event()
    action_result = {}
    
    async def on_action_result(event: ActionResultEvent):
        nonlocal action_result
        # Only accept result if correlation ID matches
        if event.correlation_id == state["correlation_id"]:
            action_result = {
                "action": event.action,
                "success": event.success,
                "result": event.result,
                "error": event.error,
                "agent": event.source,
            }
            result_received.set()
    
    # Subscribe to ActionResultEvent
    token = await event_bus.subscribe(ActionResultEvent, on_action_result)
    
    try:
        # Emit ActionRequestEvent
        await event_bus.emit(ActionRequestEvent(
            action=action,
            target_agent=target_agent,
            parameters=parameters,
            correlation_id=state["correlation_id"],
            source="LangGraphBrain",
        ))
        
        logger.info(f"Waiting for result from {target_agent}...")
        
        # Wait for result (with timeout)
        await asyncio.wait_for(result_received.wait(), timeout=10.0)
        
        # Append result
        new_results = state["action_results"] + [action_result]
        success = action_result.get("success", False)
        
        return {
            **state,
            "action_results": new_results,
            "response_text": str(action_result.get("result", "")),
            "last_error": action_result.get("error") if not success else None,
            "error_count": state["error_count"] + (0 if success else 1),
        }
        
    except asyncio.TimeoutError:
        logger.error(f"Action timeout: {target_agent}.{action}")
        return {
            **state,
            "last_error": f"Agent {target_agent} did not respond in time",
            "error_count": state["error_count"] + 1,
        }
    
    finally:
        token.unsubscribe()


def handle_error(state: WorkflowState) -> WorkflowState:
    """Error handling with retry logic."""
    logger.info(f"[NODE] handle_error: {state.get('last_error')}")
    
    if state["retry_count"] < state["max_retries"]:
        # Can retry
        logger.info(f"Retrying... (attempt {state['retry_count'] + 1}/{state['max_retries']})")
        return {
            **state,
            "retry_count": state["retry_count"] + 1,
            "last_error": None,
        }
    
    # Max retries exceeded - escalate
    logger.warning(f"Max retries exceeded. Error: {state['last_error']}")
    
    return {
        **state,
        "should_clarify": True,
        "clarification_options": [
            f"I'm having trouble with that ({state['last_error']}). Please try again.",
            "What would you like me to do instead?",
        ],
    }


def generate_response(state: WorkflowState) -> WorkflowState:
    """Generate natural language response for user."""
    logger.info("[NODE] generate_response")
    
    if state["should_clarify"]:
        response_text = state["clarification_options"][0] if state["clarification_options"] else "Could you rephrase that?"
    elif state["action_results"]:
        last_result = state["action_results"][-1]
        if last_result.get("success"):
            response_text = f"Done: {last_result.get('result', 'Success')}"
        else:
            response_text = f"Error: {last_result.get('error', 'Unknown error')}"
    else:
        response_text = "I'm ready to help."
    
    return {
        **state,
        "response_text": response_text,
    }


async def emit_response(
    state: WorkflowState,
    event_bus: EventBus,
    **kwargs,
) -> WorkflowState:
    """Emit VoiceOutputEvent to send response to user."""
    logger.info("[NODE] emit_response")
    
    await event_bus.emit(VoiceOutputEvent(
        text=state["response_text"],
        source="LangGraphBrain",
        correlation_id=state["correlation_id"],
    ))
    
    return state


# =============================================================================
# Conditional Routing Functions
# =============================================================================

def route_after_classify(state: WorkflowState) -> str:
    """Route after intent classification."""
    if state["error_count"] > 0:
        return "handle_error"
    if state["should_clarify"]:
        return "generate_response"
    return "route_intent"


def route_after_routing(state: WorkflowState) -> str:
    """Route after intent routing."""
    if state["should_clarify"]:
        return "generate_response"
    if state["is_multi_step"]:
        return "plan_multi_step"
    return "execute_action"


def route_after_action(state: WorkflowState) -> str:
    """Route after action execution."""
    if state["error_count"] > 0:
        return "handle_error"
    
    # Check if more steps in multi-step plan
    if state["is_multi_step"]:
        if state["current_step"] < len(state["action_results"]) - 1:
            # Move to next step
            state["current_step"] += 1
            return "execute_action"
    
    return "generate_response"


def route_after_error(state: WorkflowState) -> str:
    """Route after error handling."""
    if state["retry_count"] < state["max_retries"]:
        return "execute_action"  # Retry
    return "generate_response"  # Give up


# =============================================================================
# Graph Construction
# =============================================================================

def build_workflow() -> StateGraph:
    """Build the complete LangGraph workflow."""
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("capture_input", capture_input)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("route_intent", route_intent)
    workflow.add_node("plan_multi_step", plan_multi_step)
    workflow.add_node("execute_action", execute_action)
    workflow.add_node("handle_error", handle_error)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("emit_response", emit_response)
    
    # Set entry point
    workflow.set_entry_point("capture_input")
    
    # Add edges (deterministic transitions)
    workflow.add_edge("capture_input", "classify_intent")
    
    # Conditional edges (branching based on state)
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "route_intent": "route_intent",
            "handle_error": "handle_error",
            "generate_response": "generate_response",
        },
    )
    
    workflow.add_conditional_edges(
        "route_intent",
        route_after_routing,
        {
            "execute_action": "execute_action",
            "plan_multi_step": "plan_multi_step",
            "generate_response": "generate_response",
        },
    )
    
    workflow.add_edge("plan_multi_step", "execute_action")
    
    workflow.add_conditional_edges(
        "execute_action",
        route_after_action,
        {
            "handle_error": "handle_error",
            "execute_action": "execute_action",
            "generate_response": "generate_response",
        },
    )
    
    workflow.add_conditional_edges(
        "handle_error",
        route_after_error,
        {
            "execute_action": "execute_action",
            "generate_response": "generate_response",
        },
    )
    
    workflow.add_edge("generate_response", "emit_response")
    workflow.add_edge("emit_response", END)
    
    return workflow.compile()


# =============================================================================
# LangGraphBrain Class
# =============================================================================

class LangGraphBrain:
    """
    LangGraph-based orchestrator replacing the event-driven Brain.
    
    Maintains full event bus compatibility while providing deterministic
    state-machine routing via LangGraph.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        agents: Dict[str, BaseAgent],
        intent_agent: Optional[IntentAgent] = None,
    ):
        self.event_bus = event_bus
        self.agents = agents
        self.intent_agent = intent_agent
        
        # Build workflow graph
        self.workflow = build_workflow()
        
        # Subscribe to voice input as entry point
        self._subscription_token = None
        
        logger.info("LangGraphBrain initialized")
    
    async def start(self) -> None:
        """Start the orchestrator."""
        # Subscribe to voice input events
        self._subscription_token = await self.event_bus.subscribe(
            VoiceInputEvent,
            self._on_voice_input,
        )
        logger.info("LangGraphBrain started, listening for VoiceInputEvent")
    
    async def stop(self) -> None:
        """Stop the orchestrator."""
        if self._subscription_token:
            self._subscription_token.unsubscribe()
        logger.info("LangGraphBrain stopped")
    
    async def _on_voice_input(self, event: VoiceInputEvent) -> None:
        """
        Entry point: user speaks.
        
        Initializes workflow state and runs the graph.
        """
        logger.info(f"[BRAIN] Voice input: {event.text[:60]}...")
        
        # Initialize workflow state
        initial_state: WorkflowState = {
            "user_input": event.text,
            "correlation_id": event.event_id,
            "intent": None,
            "intent_confidence": 0.0,
            "entities": {},
            "raw_text": event.text,
            "target_agent": None,
            "action": None,
            "is_multi_step": False,
            "action_results": [],
            "current_step": 0,
            "max_retries": 3,
            "retry_count": 0,
            "last_error": None,
            "error_count": 0,
            "should_clarify": False,
            "clarification_options": [],
            "response_text": "",
            "context_snapshot": {},
            "events_to_emit": [],
        }
        
        try:
            # Run the workflow graph
            # Pass dependencies via config dict (LangGraph convention)
            final_state = await self.workflow.ainvoke(
                initial_state,
                config={
                    "event_bus": self.event_bus,
                    "intent_agent": self.intent_agent,
                    "agents": self.agents,
                },
            )
            
            logger.info(f"[BRAIN] Workflow completed: response='{final_state['response_text']}'")
            
        except Exception as e:
            logger.error(f"[BRAIN] Workflow error: {e}", exc_info=True)
            
            # Fallback response
            await self.event_bus.emit(VoiceOutputEvent(
                text="I encountered an error processing your request. Please try again.",
                source="LangGraphBrain",
                correlation_id=event.event_id,
            ))


# =============================================================================
# Pseudocode for Example Flow
# =============================================================================

"""
EXAMPLE 1: Simple Intent - "What's the time?"
==============================================================

1. User speaks: "What's the time?"

2. capture_input()
   state.user_input = "What's the time?"
   state.correlation_id = UUID(...)

3. classify_intent()
   intent_agent.classify("What's the time?")
   → intent: "GET_TIME"
   → entities: {}
   → confidence: 0.95

4. route_after_classify() → "route_intent" (no errors)

5. route_intent()
   INTENT_ROUTING["GET_TIME"] = {"agent": "SystemAgent", "action": "get_time"}
   state.target_agent = "SystemAgent"
   state.action = "get_time"
   state.is_multi_step = False

6. route_after_routing() → "execute_action" (single step)

7. execute_action()
   emit ActionRequestEvent(
     action="get_time",
     target_agent="SystemAgent",
     correlation_id=UUID(...)
   )
   wait for ActionResultEvent with matching correlation_id
   receive: {"success": true, "result": "3:45 PM"}
   state.action_results = [{"action": "get_time", "success": true, "result": "3:45 PM"}]

8. route_after_action() → "generate_response" (no more steps, no errors)

9. generate_response()
   state.response_text = "Done: 3:45 PM"

10. emit_response()
    emit VoiceOutputEvent(text="Done: 3:45 PM", ...)

11. END


EXAMPLE 2: Multi-step Intent with Error Handling
==============================================================

1. User speaks: "Open Safari and go to YouTube"

2. capture_input() [same as example 1]

3. classify_intent()
   intent: "OPEN_APP_AND_NAVIGATE"
   entities: {"app": "Safari", "url": "https://youtube.com"}
   confidence: 0.92

4. route_intent()
   is_multi_step = True (matches MULTI_STEP_PATTERNS)
   state.target_agent = "SystemAgent"
   state.action = "open_app_and_navigate"

5. route_after_routing() → "plan_multi_step" (multi-step)

6. plan_multi_step()
   _get_steps_for_intent("OPEN_APP_AND_NAVIGATE", {...})
   → [
       {"agent": "SystemAgent", "action": "open_app", "params": {"app": "Safari"}},
       {"agent": "SystemAgent", "action": "open_url", "params": {"url": "..."}}
     ]
   state.action_results = [
       {"step": 0, "status": "pending", ...},
       {"step": 1, "status": "pending", ...}
   ]

7. execute_action() [Step 1: open_app]
   emit ActionRequestEvent(action="open_app", target_agent="SystemAgent", ...)
   receive: {"success": true, "result": "Safari opened"}
   state.action_results[0] = {"action": "open_app", "success": true, ...}

8. route_after_action()
   current_step (0) < len(action_results) - 1 (1)? YES
   state.current_step = 1
   → "execute_action" (next step)

9. execute_action() [Step 2: open_url]
   emit ActionRequestEvent(action="open_url", target_agent="SystemAgent", ...)
   receive: {"success": true, "result": "YouTube loaded"}

10. route_after_action()
    current_step (1) < len(action_results) - 1 (1)? NO
    → "generate_response"

11. generate_response()
    state.response_text = "Done: YouTube loaded"

12. emit_response()

13. END


EXAMPLE 3: Error Handling with Retries
==============================================================

1. User speaks: "Open NonexistentApp"

2. capture_input()

3. classify_intent()
   intent: "OPEN_APP"
   entities: {"app": "NonexistentApp"}

4-7. route_intent() → "execute_action"

8. execute_action() [Attempt 1]
   emit ActionRequestEvent(action="open_app", parameters={"app": "NonexistentApp"})
   receive: {"success": false, "error": "App 'NonexistentApp' not found"}
   state.error_count = 1

9. route_after_action()
   error_count > 0 → "handle_error"

10. handle_error()
    retry_count (0) < max_retries (3)? YES
    state.retry_count = 1
    state.last_error = None

11. route_after_error() → "execute_action" (retry)

12. execute_action() [Attempt 2]
    emit ActionRequestEvent(same request)
    receive: {"success": false, "error": "App not found"}
    state.error_count = 2

13-14. Same retry loop...

15. After 3rd attempt:
    state.error_count = 3
    state.retry_count = 3
    route_after_action() → "handle_error"

16. handle_error()
    retry_count (3) >= max_retries (3)? YES
    state.should_clarify = True
    state.clarification_options = [
        "I couldn't find that app. What app would you like to open?"
    ]

17. generate_response()
    state.response_text = "I couldn't find that app. What app would you like to open?"

18. emit_response()

19. END
"""
