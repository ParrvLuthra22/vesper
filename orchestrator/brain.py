"""
Orchestrator Brain Module - Central planning and coordination for all agents.

The Brain is the PLANNER for the assistant. It:
    - Subscribes to high-level events (USER_SPOKE)
    - Maintains short-term conversation context
    - Delegates every user input to the Planner (orchestrator/planner.py),
      an LLM tool-calling loop that decides what to do — there is no
      rule-based intent switchboard anymore
    - Emits ACTION events instead of executing tasks directly

CRITICAL DESIGN RULES:
    The Brain is a PLANNER, not a WORKER. It MUST NOT:
    - Call OS commands directly
    - Perform speech or vision processing
    - Contain OpenCV or UI logic
    - Execute any tasks itself

    Instead, the Brain:
    - Feeds user text + recent conversation context to the Planner
    - Speaks whatever text the Planner returns
    - Maintains conversation state
    - Monitors agent health/recovery

Architecture:
    The Brain follows a microservices-inspired pattern where each agent
    is an independent service, and the Brain acts as the orchestrator
    that starts/stops/monitors them. Actual task planning (which tool to
    call, in what order, whether a result needs another tool call) lives
    entirely in the Planner, driven by an LLM with tool-calling.
"""

from __future__ import annotations

import asyncio
import signal
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional
from uuid import UUID, uuid4

from bus.event_bus import EventBus, get_event_bus
from agents.base_agent import AgentState, BaseAgent
from agents.voice_agent import VoiceAgent
from agents.system_agent import SystemAgent
from agents.macos_control_agent import MacOSControlAgent
from agents.web_search_agent import WebSearchAgent
from agents.memory_agent import MemoryAgent
from agents.plugin_agent import PluginAgent
from api.health import HealthServer
from guardian.gate import Guardian
from llm.router import ModelRouter
from orchestrator.planner import Planner, PlannerResult
from proactive.briefing import deliver_scheduled_briefing, make_get_daily_briefing_handler
from proactive.day_planner import make_plan_my_day_handler
from proactive.engine import ProactiveEngine
from proactive.reflection import run_reflection
from sensors.calendar_sensor import CalendarSensor
from sensors.focus_sensor import FocusSensor
from sensors.inbox_sensor import InboxSensor
from tasks.queue import TaskQueue
from tools.mcp_bridge import MCPBridge
from tools.registry import ToolSpec, get_registry
from tracing.tracer import Tracer

# VisionAgent is optional - only import if vision dependencies are available
try:
    from agents.vision_agent import VisionAgent
    VISION_AVAILABLE = True
except ImportError:
    VisionAgent = None  # type: ignore
    VISION_AVAILABLE = False
from schemas.events import (
    AgentErrorEvent,
    AgentHealthCheckEvent,
    AgentStartedEvent,
    AgentStoppedEvent,
    BriefingRequestedEvent,
    ContextUpdatedEvent,
    ObservationEvent,
    ReflectionRequestedEvent,
    ShutdownRequestedEvent,
    VoiceInputEvent,
    VoiceOutputEvent,
)
from ui.hud_overlay import HUDOverlayController
from utils.logger import get_logger


logger = get_logger(__name__)

#: How many semantic-memory matches to surface per turn (see
#: _retrieve_relevant_memories) — small on purpose, this rides alongside
#: the {context} block on every single turn.
MEMORY_RETRIEVAL_TOP_K = 3


# =============================================================================
# Brain State
# =============================================================================

class BrainState(Enum):
    """Possible states of the Brain."""

    INITIALIZING = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class AgentInfo:
    """
    Information about a registered agent.

    Attributes:
        agent: The agent instance
        started_at: When the agent was started
        last_health_check: Last health check result
        error_count: Number of errors since last restart
    """

    agent: BaseAgent
    started_at: Optional[datetime] = None
    last_health_check: Optional[AgentHealthCheckEvent] = None
    error_count: int = 0
    restart_count: int = 0


# =============================================================================
# Conversation Context - Short-term memory for dialogue
# =============================================================================

@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    turn_number: int
    timestamp: datetime
    user_input: str
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    response: str = ""
    action_taken: str = ""


@dataclass
class ConversationContext:
    """
    Short-term conversation context maintained by the Brain.

    This tracks the current conversation state to enable:
    - Context-aware responses
    - Pronoun resolution ("open it", "play that")
    - Follow-up questions handling
    - Multi-turn dialogues

    The Planner reads this (via get_recent_context()) to build its message
    history, and the Brain writes the Planner's reply back into it after
    each turn.

    Attributes:
        session_id: Current session identifier
        turns: Recent conversation turns (limited history)
        current_topic: The current topic of conversation
        entities: Extracted entities from recent conversation
        pending_clarification: Whether we're waiting for clarification
    """

    session_id: UUID = field(default_factory=uuid4)
    turns: Deque[ConversationTurn] = field(default_factory=lambda: deque(maxlen=10))
    current_topic: str = ""
    entities: Dict[str, Any] = field(default_factory=dict)
    pending_clarification: bool = False
    clarification_context: Dict[str, Any] = field(default_factory=dict)
    # Observations from the Proactive Engine (P05), pending until the
    # Planner has been given one chance to voice them (see add_observation
    # / get_pending_observations / consume_observations below).
    pending_observations: List[Dict[str, str]] = field(default_factory=list)

    @property
    def turn_count(self) -> int:
        """Get the number of turns in this conversation."""
        return len(self.turns)

    @property
    def last_turn(self) -> Optional[ConversationTurn]:
        """Get the most recent turn."""
        return self.turns[-1] if self.turns else None

    @property
    def last_user_input(self) -> str:
        """Get the last thing the user said."""
        return self.last_turn.user_input if self.last_turn else ""

    def add_turn(
        self,
        user_input: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Add a new turn to the conversation."""
        turn = ConversationTurn(
            turn_number=len(self.turns) + 1,
            timestamp=datetime.now(timezone.utc),
            user_input=user_input,
            intent=intent,
            entities=entities or {},
        )
        self.turns.append(turn)

        # Update global entities with any new ones
        if entities:
            self.entities.update(entities)

        return turn

    def update_last_response(self, response: str, action: str = "") -> None:
        """Update the response for the last turn."""
        if self.last_turn:
            # Since ConversationTurn is not frozen, we can modify it
            object.__setattr__(self.last_turn, 'response', response)
            object.__setattr__(self.last_turn, 'action_taken', action)

    def get_recent_context(self, num_turns: int = 3) -> List[Dict[str, Any]]:
        """Get recent conversation context for the Planner's message history."""
        recent = list(self.turns)[-num_turns:]
        return [
            {
                "turn": t.turn_number,
                "user": t.user_input,
                "intent": t.intent,
                "response": t.response,
            }
            for t in recent
        ]

    def resolve_reference(self, reference: str) -> Optional[Any]:
        """
        Resolve a pronoun or reference to an entity.

        E.g., "it", "that app", "the file" -> actual entity
        """
        reference_lower = reference.lower()

        # Check for common references
        if reference_lower in ("it", "that", "this"):
            # Return the most recently mentioned entity
            if self.last_turn and self.last_turn.entities:
                # Return first entity from last turn
                for key, value in self.last_turn.entities.items():
                    return value

        # Check for specific type references
        if "app" in reference_lower and "application" in self.entities:
            return self.entities["application"]

        if "file" in reference_lower and "file_path" in self.entities:
            return self.entities["file_path"]

        return None

    def clear(self) -> None:
        """Clear the conversation context."""
        self.turns.clear()
        self.current_topic = ""
        self.entities.clear()
        self.pending_clarification = False
        self.clarification_context.clear()
        self.pending_observations.clear()

    def add_observation(self, kind: str, detail: str, observation_id: str) -> None:
        """Queue an observation from the Proactive Engine (ObservationEvent)."""
        self.pending_observations.append(
            {"kind": kind, "detail": detail, "observation_id": observation_id}
        )

    def get_pending_observations(self) -> List[str]:
        """Detail strings for observations not yet offered to the Planner."""
        return [obs["detail"] for obs in self.pending_observations]

    def consume_observations(self) -> None:
        """
        Mark all currently-pending observations as offered.

        Called once per turn, right after the Planner has been given the
        chance to voice them — mechanically enforcing "raise it once": an
        observation is only ever included in the Planner's context a
        single time, regardless of whether the model chose to mention it.
        """
        self.pending_observations.clear()


# =============================================================================
# Brain Configuration
# =============================================================================

@dataclass
class BrainConfig:
    """
    Configuration for the Brain.

    Loaded from settings.yaml orchestrator section.
    """

    startup_order: List[str] = field(default_factory=lambda: [
        "MemoryAgent",
        "SystemAgent",
        "VoiceAgent",
    ])
    health_check_interval: int = 30
    max_agent_errors: int = 5
    max_restart_attempts: int = 3
    shutdown_timeout: int = 10
    enable_auto_recovery: bool = True
    max_context_turns: int = 10


# =============================================================================
# Brain
# =============================================================================

class Brain:
    """
    Central orchestrator for the virtual assistant.

    The Brain coordinates all agents and manages the overall system.
    It implements a supervisor pattern to handle agent failures, and
    delegates all user-text handling (voice or typed) to the Planner.

    Usage:
        brain = Brain()
        await brain.start()
        # ... assistant is running ...
        await brain.stop()

    Features:
        - Ordered agent startup based on dependencies
        - Health monitoring with automatic recovery
        - Graceful shutdown with timeout
        - Event-driven communication
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        event_bus: Optional[EventBus] = None,
        enable_voice_agent: bool = True,
    ):
        """
        Initialize the Brain.

        Args:
            config: Configuration dictionary (from settings.yaml)
            event_bus: Event bus instance (defaults to global)
            enable_voice_agent: Whether to register VoiceAgent (microphone
                capture + TTS). The CLI (cli/app.py) runs with this False —
                same Brain, no voice I/O; text goes in/out via the terminal
                instead of VoiceInputEvent/VoiceOutputEvent's audio path.
        """
        self._config = config or {}
        self._event_bus = event_bus or get_event_bus()
        self._enable_voice_agent = enable_voice_agent
        self._state = BrainState.INITIALIZING

        # Parse configuration
        orchestrator_config = self._config.get("orchestrator", {})

        default_startup_order = [
            "MemoryAgent",
            "SystemAgent",
            "VoiceAgent",
            "PluginAgent",
            "VisionAgent",
        ]

        self._brain_config = BrainConfig(
            startup_order=orchestrator_config.get("startup_order", default_startup_order),
            health_check_interval=orchestrator_config.get("health_check_interval", 30),
            max_agent_errors=orchestrator_config.get("max_agent_errors", 5),
            max_restart_attempts=orchestrator_config.get("max_restart_attempts", 3),
            shutdown_timeout=orchestrator_config.get("shutdown_timeout_seconds", 10),
            enable_auto_recovery=orchestrator_config.get("enable_auto_recovery", True),
        )

        # Agent registry
        self._agents: Dict[str, AgentInfo] = {}

        # Tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._hud_overlay: Optional[HUDOverlayController] = None
        self._health_server: Optional[HealthServer] = None

        # Session info
        self._session_id = uuid4()
        self._started_at: Optional[datetime] = None

        # Conversation context - short-term memory
        self._context = ConversationContext(session_id=self._session_id)

        # The Planner: LLM tool-calling loop that replaces rule-based intent
        # routing. Built once at startup (router + guardian + persona load).
        self._guardian = Guardian(event_bus=self._event_bus)
        self._router = ModelRouter(config=self._config)
        self._tracer = Tracer(config=self._config)
        self._task_queue = TaskQueue(event_bus=self._event_bus)
        self._planner = Planner(
            router=self._router,
            guardian=self._guardian,
            event_bus=self._event_bus,
            tracer=self._tracer,
            task_queue=self._task_queue,
        )

        # MCP bridge (P07): connects to configured MCP servers (Gmail, ...)
        # and auto-registers their tools into the ToolRegistry.
        self._mcp_bridge = MCPBridge(config=self._config)

        # Sensors (P05/P07) — local-only observation, individually toggleable
        # via config; start() no-ops for any sensor that's disabled.
        self._focus_sensor = FocusSensor(event_bus=self._event_bus, config=self._config)
        self._calendar_sensor = CalendarSensor(event_bus=self._event_bus, config=self._config)
        # Polls via the Gmail MCP bridge's list_unread tool, so it must start
        # after self._mcp_bridge (see start()).
        self._inbox_sensor = InboxSensor(event_bus=self._event_bus, config=self._config)

        # Proactive Engine (P05): scheduled jobs + cooldown-gated call-out rules.
        self._proactive_engine = ProactiveEngine(event_bus=self._event_bus, config=self._config)

        logger.info(f"Brain initialized (session={self._session_id})")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def state(self) -> BrainState:
        """Get the brain's current state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if the brain is running."""
        return self._state == BrainState.RUNNING

    @property
    def session_id(self) -> UUID:
        """Get the current session ID."""
        return self._session_id

    @property
    def context(self) -> ConversationContext:
        """Get the current conversation context."""
        return self._context

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        if self._started_at is None:
            return 0.0
        return (datetime.now(timezone.utc) - self._started_at).total_seconds()

    # =========================================================================
    # Agent Registration
    # =========================================================================

    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent with the brain.

        Args:
            agent: The agent to register
        """
        name = agent.name
        if name in self._agents:
            raise ValueError(f"Agent {name} is already registered")

        self._agents[name] = AgentInfo(agent=agent)
        logger.debug(f"Registered agent: {name}")

    def unregister_agent(self, name: str) -> bool:
        """
        Unregister an agent.

        Args:
            name: Agent name to unregister

        Returns:
            True if agent was found and unregistered
        """
        if name in self._agents:
            del self._agents[name]
            logger.debug(f"Unregistered agent: {name}")
            return True
        return False

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        info = self._agents.get(name)
        return info.agent if info else None

    def get_agents_status(self) -> List[Dict[str, Any]]:
        """Snapshot of every registered agent's health — for a /status-style display."""
        statuses: List[Dict[str, Any]] = []
        for name, info in self._agents.items():
            try:
                healthy = bool(info.agent.is_healthy())
            except Exception:
                healthy = False
            statuses.append({
                "name": name,
                "healthy": healthy,
                "started_at": info.started_at,
                "error_count": info.error_count,
                "restart_count": info.restart_count,
            })
        return statuses

    def get_router(self) -> ModelRouter:
        """The Planner's ModelRouter — for provider-health displays (e.g. /status)."""
        return self._router

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def start(self) -> None:
        """
        Start the brain and all registered agents.

        Agents are started in the order specified by startup_order config.
        If an agent fails to start, the startup process is aborted.
        """
        if self._state not in (BrainState.INITIALIZING, BrainState.STOPPED):
            raise RuntimeError(f"Cannot start brain in state {self._state.name}")

        logger.info("Starting brain...")
        self._state = BrainState.STARTING

        try:
            # Subscribe to system events
            await self._subscribe_to_events()

            # Register default agents if none registered
            if not self._agents:
                await self._register_default_agents()

            # Start agents in order
            await self._start_agents()

            # Start health API server
            self._health_server = HealthServer(
                lambda: {name: info.agent for name, info in self._agents.items()}
            )
            self._health_server.start()

            # Start HUD overlay in dedicated thread (non-blocking).
            try:
                self._hud_overlay = HUDOverlayController(event_bus=self._event_bus, config=self._config)
                await self._hud_overlay.start()
            except Exception as exc:
                logger.warning(f"HUD overlay could not start: {exc}")
                self._hud_overlay = None

            # Start health check task
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )

            # Connect to configured MCP servers (Gmail, apple_pim, Notion,
            # ...) and register their tools before anything that depends
            # on those tools existing (the briefing/day-plan tools, the
            # inbox sensor).
            await self._mcp_bridge.start()
            self._register_briefing_tool()
            self._register_day_planner_tool()
            self._register_memory_tools()

            # Start sensors and the Proactive Engine. Sensors no-op if
            # disabled by config; the engine always starts (its scheduled
            # jobs and rules are individually config-gated). inbox_sensor
            # starts after the MCP bridge since it polls list_unread.
            await self._focus_sensor.start()
            await self._calendar_sensor.start()
            await self._inbox_sensor.start()
            await self._proactive_engine.start()

            self._state = BrainState.RUNNING
            self._started_at = datetime.now(timezone.utc)

            logger.info("Brain started successfully")

            await self.greet()

        except Exception as e:
            self._state = BrainState.ERROR
            logger.error(f"Failed to start brain: {e}", exc_info=True)
            raise

    async def greet(self) -> str:
        """Generate + emit the time-appropriate greeting (plus one pending
        observation, if any). Used at startup and on wake (the cinematic
        reveal). Returns the greeting text."""
        pending_observations = self._context.get_pending_observations()
        greeting = await self._planner.greet(observations=pending_observations)
        self._context.consume_observations()
        await self._event_bus.publish(VoiceOutputEvent(text=greeting.text, source="Brain"))
        return greeting.text

    async def stop(self, reason: str = "Normal shutdown") -> None:
        """
        Stop the brain and all agents.

        Agents are stopped in reverse startup order.

        Args:
            reason: Reason for shutdown
        """
        if self._state == BrainState.STOPPED:
            return

        logger.info(f"Stopping brain: {reason}")
        self._state = BrainState.STOPPING

        # Cancel health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Stop sensors, the Proactive Engine, and the MCP bridge before
        # agents/event bus.
        await self._proactive_engine.stop()
        await self._inbox_sensor.stop()
        await self._calendar_sensor.stop()
        await self._focus_sensor.stop()
        await self._mcp_bridge.stop()

        # Extract and store durable memories from this session (see
        # proactive/reflection.py) — MUST run before ShutdownRequestedEvent
        # is announced: agents (including MemoryAgent) tear themselves down
        # in response to it, same as the explicit _stop_agents() pass below.
        await self._run_reflection(reason="session_end")

        # Announce shutdown
        await self._event_bus.publish(ShutdownRequestedEvent(
            reason=reason,
            source="Brain",
        ))

        # Give agents time to handle shutdown event
        await asyncio.sleep(0.5)

        # Stop agents in reverse order
        await self._stop_agents()

        # Stop HUD overlay after agent shutdown events are processed.
        if self._hud_overlay:
            await self._hud_overlay.stop()
            self._hud_overlay = None

        # Stop health API server
        if self._health_server:
            self._health_server.stop()
            self._health_server = None

        # Stop event bus
        await self._event_bus.stop()

        self._state = BrainState.STOPPED
        logger.info("Brain stopped")

    async def run(self) -> None:
        """
        Run the brain until interrupted.

        This is the main entry point for running the assistant.
        It handles signals for graceful shutdown.
        """
        # Set up signal handlers
        loop = asyncio.get_event_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_signal(s))
            )

        try:
            await self.start()

            # Run until stopped
            while self.is_running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Brain error: {e}", exc_info=True)
        finally:
            if self._state != BrainState.STOPPED:
                await self.stop("Application exit")

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle system signals."""
        logger.info(f"Received signal {sig.name}")
        await self.stop(f"Received {sig.name}")

    # =========================================================================
    # Briefing
    # =========================================================================

    def _register_briefing_tool(self) -> None:
        """
        Register get_daily_briefing: the on-demand "brief me" tool.

        Its handler just gathers the same raw data
        (proactive/briefing.assemble_briefing_text) the scheduled
        BriefingRequestedEvent handler uses — the calling Planner turn's
        model renders the actual structured reply on its next iteration,
        exactly like any other tool result.
        """
        memory_agent = self.get_agent("MemoryAgent")
        registry = get_registry()
        if registry.get("get_daily_briefing") is not None:
            return  # already registered (e.g. brain restarted without a fresh process)

        registry.register(
            ToolSpec(
                name="get_daily_briefing",
                description=(
                    "Gather today's briefing data: unread email triage, today's "
                    "remaining calendar events, and any carried-over items from "
                    "yesterday's briefing. Use when the user asks to be briefed "
                    "— \"brief me\", \"what's my day look like\", a status update."
                ),
                parameters={"type": "object", "properties": {}},
                tier="safe",
                handler=make_get_daily_briefing_handler(
                    registry=registry,
                    calendar_sensor=self._calendar_sensor,
                    memory_agent=memory_agent,
                ),
                category="briefing",
            )
        )

    async def _handle_briefing_requested(self, event: BriefingRequestedEvent) -> None:
        """Scheduled path (ProactiveEngine's morning_briefing cron job) — no
        ongoing turn to attach to, so start a fresh one and speak the result."""
        logger.info(f"Delivering scheduled briefing: {event.schedule_name}")
        memory_agent = self.get_agent("MemoryAgent")
        result = await deliver_scheduled_briefing(
            planner=self._planner,
            calendar_sensor=self._calendar_sensor,
            memory_agent=memory_agent,
            include_day_plan=bool(
                self._config.get("proactive", {})
                .get("schedule", {})
                .get("morning_briefing", {})
                .get("include_day_plan", True)
            ),
        )
        await self._event_bus.emit(VoiceOutputEvent(text=result.text, source="Brain"))

    # =========================================================================
    # Day planning (P09)
    # =========================================================================

    def _register_day_planner_tool(self) -> None:
        """
        Register plan_my_day: the on-demand "plan my day" tool.

        Its handler just gathers the raw data (calendar, reminders,
        Notion tasks, email pressure, protected-time memories) — the
        calling Planner turn's model renders the actual time-blocked plan
        on its next iteration, exactly like get_daily_briefing.
        """
        memory_agent = self.get_agent("MemoryAgent")
        registry = get_registry()
        if registry.get("plan_my_day") is not None:
            return  # already registered (e.g. brain restarted without a fresh process)

        registry.register(
            ToolSpec(
                name="plan_my_day",
                description=(
                    "Gather everything needed for a realistic, time-blocked day "
                    "plan: today's calendar, reminders due, Notion tasks, and "
                    "unread-email pressure — plus any remembered protected/sacred "
                    "time blocks to route around. Use when the user asks "
                    "\"plan my day\" / \"what should my day look like\" / "
                    "\"help me schedule today\"."
                ),
                parameters={"type": "object", "properties": {}},
                tier="safe",
                handler=make_plan_my_day_handler(registry=registry, memory_agent=memory_agent),
                category="day_planning",
            )
        )

    # =========================================================================
    # Memory (P09)
    # =========================================================================

    def _register_memory_tools(self) -> None:
        """Register forget_memory: deletes semantic memories matching a query."""
        registry = get_registry()
        if registry.get("forget_memory") is not None:
            return

        registry.register(
            ToolSpec(
                name="forget_memory",
                description=(
                    "Permanently delete remembered facts/preferences/patterns "
                    "matching a description. Use when the user says \"forget "
                    "that I...\" / \"forget the X preference\" / \"you don't "
                    "need to remember Y anymore\". This cannot be undone."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to forget, described the way you'd search for it.",
                        }
                    },
                    "required": ["query"],
                },
                tier="confirm",
                handler=self._make_forget_memory_handler(),
                category="memory",
            )
        )

    def _make_forget_memory_handler(self):
        async def handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
            memory_agent = self.get_agent("MemoryAgent")
            if not isinstance(memory_agent, MemoryAgent):
                return "Memory isn't available right now, Sir."
            query = str(arguments.get("query", "")).strip()
            if not query:
                return "I need something to search for before I can forget it, Sir."
            forgotten = await memory_agent.forget(query=query)
            if not forgotten:
                return f"I couldn't find anything matching {query!r} to forget, Sir."
            return "Forgotten: " + "; ".join(forgotten)

        return handler

    async def _run_reflection(self, reason: str) -> None:
        """
        Extract and store durable memories from this session's conversation
        (see proactive/reflection.py), then clear the conversation context
        — a fresh day starts a fresh short-term topic/entity state, and
        the next reflection shouldn't re-summarize turns already reflected
        on. Never raises: reflection is a background nicety.
        """
        turns = self._context.get_recent_context(num_turns=self._brain_config.max_context_turns)
        if not turns:
            return

        memory_agent = self.get_agent("MemoryAgent")
        if not isinstance(memory_agent, MemoryAgent):
            return

        try:
            items = await run_reflection(turns=turns, router=self._router, memory_agent=memory_agent)
        except Exception as exc:
            logger.warning(f"Reflection failed ({reason}), non-fatal: {exc}")
            return

        if items:
            logger.info(f"Reflection ({reason}) stored {len(items)} memory item(s)")
            self._context.clear()

    async def _handle_reflection_requested(self, event: ReflectionRequestedEvent) -> None:
        """Scheduled path (ProactiveEngine's midnight_reflection cron job)."""
        await self._run_reflection(reason=event.schedule_name)

    # =========================================================================
    # Agent Lifecycle
    # =========================================================================

    async def _register_default_agents(self) -> None:
        """Register the default set of agents.

        IntentAgent is deliberately not registered: the Planner (LLM
        tool-calling) is intent recognition now, so IntentAgent's
        rule/LLM-based classification is bypassed entirely in this flow.
        """
        logger.debug("Registering default agents...")

        # Get agent-specific configs
        voice_config = self._config.get("voice", {})
        system_config = self._config.get("system", {})
        memory_config = self._config.get("memory", {})
        vision_config = self._config.get("vision", {})
        plugins_config = self._config.get("plugins", {})
        web_search_config = self._config.get("web_search", {})

        # Create and register agents
        agents = [
            MemoryAgent(config={"memory": memory_config}),
            SystemAgent(config={"system": system_config}),
            MacOSControlAgent(config={"system": system_config}),
            WebSearchAgent(config={"web_search": web_search_config, "system": system_config}),
            PluginAgent(config={"plugins": plugins_config}),
        ]

        if self._enable_voice_agent:
            agents.append(VoiceAgent(config={"voice": voice_config}))
        else:
            logger.debug("VoiceAgent not registered (enable_voice_agent=False)")

        # Conditionally add VisionAgent if enabled and available
        if vision_config.get("enabled", False):
            if VISION_AVAILABLE:
                agents.append(VisionAgent(config={"vision": vision_config}))
                logger.info("VisionAgent registered (vision enabled)")
            else:
                logger.warning(
                    "Vision is enabled in config but dependencies are not installed. "
                    "Install with: pip install pyautogui pillow pytesseract opencv-python"
                )
        else:
            logger.debug("VisionAgent not registered (vision disabled in config)")

        for agent in agents:
            self.register_agent(agent)

    async def _start_agents(self) -> None:
        """Start all registered agents in order."""
        logger.debug("Starting agents...")

        # Build ordered list based on startup_order config
        agent_names = list(self._agents.keys())
        ordered_names = []

        # Add configured order first
        for name in self._brain_config.startup_order:
            if name in agent_names:
                ordered_names.append(name)

        # Add any remaining agents
        for name in agent_names:
            if name not in ordered_names:
                ordered_names.append(name)

        # Start agents in order
        for name in ordered_names:
            info = self._agents[name]
            try:
                logger.debug(f"Starting agent: {name}")
                await info.agent.start()
                info.started_at = datetime.now(timezone.utc)
            except Exception as e:
                logger.error(f"Failed to start agent {name}: {e}")
                raise RuntimeError(f"Agent {name} failed to start") from e

    async def _stop_agents(self) -> None:
        """Stop all agents in reverse order."""
        logger.debug("Stopping agents...")

        # Get reverse order
        agent_names = list(self._agents.keys())
        agent_names.reverse()

        # Stop with timeout
        for name in agent_names:
            info = self._agents[name]
            try:
                await asyncio.wait_for(
                    info.agent.stop("Brain shutdown"),
                    timeout=self._brain_config.shutdown_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Agent {name} did not stop within timeout")
            except Exception as e:
                logger.error(f"Error stopping agent {name}: {e}")

    # =========================================================================
    # Event Handling - The Brain subscribes to high-level events
    # =========================================================================

    async def _subscribe_to_events(self) -> None:
        """
        Subscribe to relevant events.

        The Brain subscribes to:
        - VoiceInputEvent: When user speaks (USER_SPOKE) — feeds the Planner
        - ObservationEvent: Proactive Engine call-outs — queued as pending
          observations for the Planner's {context} block to surface once
        - BriefingRequestedEvent: ProactiveEngine's morning_briefing job —
          assembles and delivers the scheduled briefing
        - ReflectionRequestedEvent: ProactiveEngine's midnight_reflection
          job — extracts and stores durable memories from the day so far
        - AgentErrorEvent: When an agent has an error
        - AgentStoppedEvent: When an agent stops
        """
        self._event_bus.subscribe(VoiceInputEvent, self._handle_voice_input)
        self._event_bus.subscribe(ObservationEvent, self._handle_observation)
        self._event_bus.subscribe(BriefingRequestedEvent, self._handle_briefing_requested)
        self._event_bus.subscribe(ReflectionRequestedEvent, self._handle_reflection_requested)
        self._event_bus.subscribe(AgentErrorEvent, self._handle_agent_error)
        self._event_bus.subscribe(AgentStoppedEvent, self._handle_agent_stopped)

    async def _handle_voice_input(self, event: VoiceInputEvent) -> None:
        """Handle voice input event (USER_SPOKE) by feeding the Planner."""
        logger.info(f"User spoke: '{event.text}'")
        await self.handle_user_text(event.text, correlation_id=event.event_id)

    async def _handle_observation(self, event: ObservationEvent) -> None:
        """Queue a Proactive Engine call-out as a pending observation."""
        logger.info(f"Observation: kind={event.kind} detail={event.detail!r}")
        self._context.add_observation(kind=event.kind, detail=event.detail, observation_id=event.observation_id)

    async def handle_user_text(
        self,
        text: str,
        correlation_id: Optional[UUID] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> PlannerResult:
        """
        Run one piece of user text (voice or typed) through the Planner and
        speak the result.

        This is the single entry point shared by every input surface —
        VoiceInputEvent today, and any future typed-input path (CLI, HUD
        text box) — so there is exactly one place that reads and writes
        ConversationContext: here. Recent context is snapshotted BEFORE
        this turn is added, so the in-progress turn (with no response yet)
        never leaks into its own history as a duplicate user message.

        Any pending observations (from the Proactive Engine) are handed to
        the Planner's {context} block and then consumed, win or lose — an
        observation is only ever offered once, regardless of whether the
        model chose to voice it.

        `on_token`, if given, is forwarded to the Planner for live text
        streaming (see Planner.run) — callers that use it should treat
        the still-emitted VoiceOutputEvent as informational only, since
        they've already rendered the reply as it streamed in.
        """
        recent_context = self._context.get_recent_context(num_turns=self._brain_config.max_context_turns)
        pending_observations = self._context.get_pending_observations()
        relevant_memories = await self._retrieve_relevant_memories(text)

        self._context.add_turn(user_input=text)
        await self._event_bus.emit(ContextUpdatedEvent(
            context_type="turn",
            context_key="user_input",
            context_value=text,
            turn_number=self._context.turn_count,
            source="Brain",
        ))

        result = await self._planner.run(
            user_text=text,
            recent_context=recent_context,
            observations=pending_observations,
            memories=relevant_memories,
            on_token=on_token,
        )
        self._context.consume_observations()

        self._context.update_last_response(result.text, action="planner")

        await self._event_bus.emit(VoiceOutputEvent(
            text=result.text,
            source="Brain",
            correlation_id=correlation_id,
        ))

        return result

    async def _retrieve_relevant_memories(self, text: str) -> List[str]:
        """Top-k semantic memory matches for `text`, for the Planner's
        {context} block — e.g. a previously stated "always protect the gym
        slot" preference surfacing on a day-planning request. Empty (not an
        error) when MemoryAgent/semantic memory isn't available."""
        memory_agent = self.get_agent("MemoryAgent")
        if not isinstance(memory_agent, MemoryAgent):
            return []
        try:
            matches = await memory_agent.semantic_retrieve(query=text, top_k=MEMORY_RETRIEVAL_TOP_K)
        except Exception as exc:
            logger.debug(f"Semantic memory retrieval failed (non-fatal): {exc}")
            return []
        return [m["text"] for m in matches]

    async def _handle_agent_error(self, event: AgentErrorEvent) -> None:
        """Handle agent error events."""
        agent_name = event.agent_name

        logger.warning(f"Agent error: {agent_name} - {event.error_message}")

        if agent_name in self._agents:
            info = self._agents[agent_name]
            info.error_count += 1

            # Check if we should attempt recovery
            if (
                self._brain_config.enable_auto_recovery
                and event.is_recoverable
                and info.error_count >= self._brain_config.max_agent_errors
            ):
                await self._attempt_agent_recovery(agent_name)

    async def _handle_agent_stopped(self, event: AgentStoppedEvent) -> None:
        """Handle agent stopped events."""
        if not event.clean_shutdown and self.is_running:
            logger.warning(f"Agent {event.agent_name} stopped unexpectedly")

            if self._brain_config.enable_auto_recovery:
                await self._attempt_agent_recovery(event.agent_name)

    # =========================================================================
    # Health Monitoring
    # =========================================================================

    async def _health_check_loop(self) -> None:
        """Periodically check agent health."""
        while self.is_running:
            try:
                await asyncio.sleep(self._brain_config.health_check_interval)
                await self._check_all_agents()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _check_all_agents(self) -> None:
        """Check health of all agents."""
        for name, info in self._agents.items():
            try:
                health = await info.agent.health_check()
                info.last_health_check = health

                if not health.is_healthy:
                    logger.warning(f"Agent {name} is unhealthy")

            except Exception as e:
                logger.error(f"Failed to check agent {name}: {e}")

    async def _attempt_agent_recovery(self, agent_name: str) -> None:
        """
        Attempt to recover a failed agent.

        Args:
            agent_name: Name of the agent to recover
        """
        if agent_name not in self._agents:
            return

        info = self._agents[agent_name]

        if info.restart_count >= self._brain_config.max_restart_attempts:
            logger.error(
                f"Agent {agent_name} exceeded max restart attempts"
            )
            return

        logger.info(f"Attempting to recover agent: {agent_name}")

        try:
            # Stop if running
            if info.agent.state == AgentState.RUNNING:
                await info.agent.stop("Recovery restart")

            # Wait a moment
            await asyncio.sleep(1)

            # Restart
            await info.agent.start()

            info.restart_count += 1
            info.error_count = 0
            info.started_at = datetime.now(timezone.utc)

            logger.info(f"Agent {agent_name} recovered successfully")

        except Exception as e:
            logger.error(f"Failed to recover agent {agent_name}: {e}")

    # =========================================================================
    # Status and Diagnostics
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get current brain status.

        Returns:
            Dictionary with status information
        """
        return {
            "state": self._state.name,
            "session_id": str(self._session_id),
            "uptime_seconds": self.uptime_seconds,
            "agents": {
                name: {
                    "state": info.agent.state.name,
                    "error_count": info.error_count,
                    "restart_count": info.restart_count,
                    "uptime_seconds": info.agent.uptime_seconds,
                }
                for name, info in self._agents.items()
            },
            "event_bus_metrics": {
                "events_published": self._event_bus.metrics.events_published,
                "events_delivered": self._event_bus.metrics.events_delivered,
                "events_failed": self._event_bus.metrics.events_failed,
            },
        }

    async def run_diagnostic(self) -> Dict[str, Any]:
        """
        Run full system diagnostic.

        Returns:
            Diagnostic report
        """
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "brain_status": self.get_status(),
            "agent_health": {},
        }

        for name, info in self._agents.items():
            try:
                health = await info.agent.health_check()
                report["agent_health"][name] = {
                    "is_healthy": health.is_healthy,
                    "pending_tasks": health.pending_tasks,
                    "metrics": {
                        "events_received": info.agent.metrics.events_received,
                        "events_processed": info.agent.metrics.events_processed,
                        "events_failed": info.agent.metrics.events_failed,
                        "success_rate": info.agent.metrics.success_rate,
                        "avg_processing_time_ms": info.agent.metrics.average_processing_time_ms,
                    },
                }
            except Exception as e:
                report["agent_health"][name] = {
                    "is_healthy": False,
                    "error": str(e),
                }

        return report


# =============================================================================
# Factory Function
# =============================================================================

def create_brain(config: Optional[Dict[str, Any]] = None) -> Brain:
    """
    Factory function to create a configured Brain instance.

    Args:
        config: Configuration dictionary

    Returns:
        Configured Brain instance
    """
    return Brain(config=config)
