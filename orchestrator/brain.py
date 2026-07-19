"""
Orchestrator Brain Module - Central planning and coordination for all agents.

The Brain is the PLANNER for the assistant. It:
    - Subscribes to high-level events (USER_SPOKE, INTENT_PARSED)
    - Maintains short-term conversation context
    - Decides which agent(s) should handle an intent
    - Supports multi-step command planning
    - Emits ACTION events instead of executing tasks directly

CRITICAL DESIGN RULES:
    The Brain is a PLANNER, not a WORKER. It MUST NOT:
    - Call OS commands directly
    - Perform speech or vision processing
    - Contain OpenCV or UI logic
    - Execute any tasks itself

    Instead, the Brain:
    - Analyzes intents and context
    - Creates execution plans
    - Delegates work via ActionRequestEvent
    - Monitors execution progress
    - Maintains conversation state

Architecture:
    The Brain follows a microservices-inspired pattern where each agent
    is an independent service, and the Brain acts as the orchestrator
    that plans and delegates work.

Responsibilities:
    1. Intent Routing: Decide which agent handles each intent
    2. Multi-Step Planning: Break complex requests into action sequences
    3. Context Management: Track conversation state and entities
    4. Plan Execution: Monitor and coordinate multi-step plans
    5. Error Handling: Recover from agent failures
"""

from __future__ import annotations

import asyncio
import signal
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Deque, Dict, List, Optional, Type
from uuid import UUID, uuid4

from bus.event_bus import EventBus, get_event_bus
from agents.base_agent import AgentState, BaseAgent
from agents.voice_agent import VoiceAgent
from agents.intent_agent import IntentAgent
from agents.system_agent import SystemAgent
from agents.macos_control_agent import MacOSControlAgent
from agents.web_search_agent import WebSearchAgent
from agents.memory_agent import MemoryAgent
from agents.plugin_agent import PluginAgent
from api.health import HealthServer

# VisionAgent is optional - only import if vision dependencies are available
try:
    from agents.vision_agent import VisionAgent
    VISION_AVAILABLE = True
except ImportError:
    VisionAgent = None  # type: ignore
    VISION_AVAILABLE = False
from schemas.events import (
    ActionRequestEvent,
    ActionResultEvent,
    AgentErrorEvent,
    AgentHealthCheckEvent,
    AgentStartedEvent,
    AgentStoppedEvent,
    ClarificationNeededEvent,
    ContextUpdatedEvent,
    IntentRecognizedEvent,
    IntentUnknownEvent,
    PlanCompletedEvent,
    PlanCreatedEvent,
    PlanStepCompletedEvent,
    ResponseGeneratedEvent,
    ShutdownRequestedEvent,
    VoiceInputEvent,
    VoiceOutputEvent,
)
from ui.hud_overlay import HUDOverlayController
from utils.logger import get_logger


logger = get_logger(__name__)


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
    
    Attributes:
        session_id: Current session identifier
        turns: Recent conversation turns (limited history)
        current_topic: The current topic of conversation
        entities: Extracted entities from recent conversation
        pending_clarification: Whether we're waiting for clarification
        active_plan: Currently executing multi-step plan
    """
    
    session_id: UUID = field(default_factory=uuid4)
    turns: Deque[ConversationTurn] = field(default_factory=lambda: deque(maxlen=10))
    current_topic: str = ""
    entities: Dict[str, Any] = field(default_factory=dict)
    pending_clarification: bool = False
    clarification_context: Dict[str, Any] = field(default_factory=dict)
    active_plan_id: Optional[UUID] = None
    
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
    
    @property
    def last_intent(self) -> Optional[str]:
        """Get the last recognized intent."""
        return self.last_turn.intent if self.last_turn else None
    
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
        """Get recent conversation context for LLM prompting."""
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
        self.active_plan_id = None


# =============================================================================
# Execution Plan - For multi-step commands
# =============================================================================

@dataclass
class PlanStep:
    """A single step in an execution plan."""
    
    step_number: int
    action: str
    target_agent: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    completed: bool = False
    success: bool = False
    result: Any = None
    error: str = ""


@dataclass
class ExecutionPlan:
    """
    A multi-step execution plan created by the Brain.
    
    Used for complex commands that require multiple actions.
    E.g., "Open Safari and go to YouTube" = [open_app, navigate_url]
    """
    
    plan_id: UUID = field(default_factory=uuid4)
    description: str = ""
    steps: List[PlanStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_step: int = 0
    completed: bool = False
    aborted: bool = False
    abort_reason: str = ""
    
    @property
    def total_steps(self) -> int:
        return len(self.steps)
    
    @property
    def is_active(self) -> bool:
        return not self.completed and not self.aborted
    
    @property
    def next_step(self) -> Optional[PlanStep]:
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None
    
    def mark_step_complete(self, success: bool, result: Any = None, error: str = "") -> None:
        """Mark the current step as complete and advance."""
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            step.completed = True
            step.success = success
            step.result = result
            step.error = error
            self.current_step += 1
            
            # Check if plan is complete
            if self.current_step >= len(self.steps):
                self.completed = True
    
    def abort(self, reason: str) -> None:
        """Abort the plan."""
        self.aborted = True
        self.abort_reason = reason


# =============================================================================
# Intent Routing - Maps intents to agents
# =============================================================================

# Mapping of intent names to target agents and actions
INTENT_ROUTING: Dict[str, Dict[str, Any]] = {
    # ==========================================================================
    # System control intents -> SystemAgent
    # ==========================================================================
    # Keys are normalized to lowercase; callers must lowercase the intent
    # string before looking it up here (see _route_intent).
    "open_application": {"agent": "SystemAgent", "action": "open_app"},
    "open_app": {"agent": "SystemAgent", "action": "open_app"},
    "close_application": {"agent": "SystemAgent", "action": "close_app"},
    "close_app": {"agent": "SystemAgent", "action": "close_app"},
    "focus_app": {"agent": "SystemAgent", "action": "focus_app"},
    "switch_app": {"agent": "SystemAgent", "action": "focus_app"},
    "list_apps": {"agent": "SystemAgent", "action": "list_apps"},

    # Volume
    "set_volume": {"agent": "SystemAgent", "action": "control_volume"},
    "control_volume": {"agent": "SystemAgent", "action": "control_volume"},
    "get_volume": {"agent": "SystemAgent", "action": "get_volume"},
    "volume_up": {"agent": "SystemAgent", "action": "volume_up"},
    "volume_down": {"agent": "SystemAgent", "action": "volume_down"},
    "mute": {"agent": "SystemAgent", "action": "mute"},
    "unmute": {"agent": "SystemAgent", "action": "unmute"},

    # Brightness
    "set_brightness": {"agent": "SystemAgent", "action": "set_brightness"},
    "control_brightness": {"agent": "SystemAgent", "action": "set_brightness"},
    "get_brightness": {"agent": "SystemAgent", "action": "get_brightness"},
    "brightness_up": {"agent": "SystemAgent", "action": "brightness_up"},
    "brightness_down": {"agent": "SystemAgent", "action": "brightness_down"},

    # Time/Date
    "get_time": {"agent": "SystemAgent", "action": "get_time"},
    "get_date": {"agent": "SystemAgent", "action": "get_date"},

    # System Info
    "system_info": {"agent": "SystemAgent", "action": "system_info"},
    "get_system_stats": {"agent": "SystemAgent", "action": "system_info"},
    "get_cpu": {"agent": "SystemAgent", "action": "get_cpu"},
    "get_memory": {"agent": "SystemAgent", "action": "get_memory"},
    "get_battery": {"agent": "SystemAgent", "action": "get_battery"},
    "get_disk": {"agent": "SystemAgent", "action": "get_disk"},

    # Screen Control
    "sleep_display": {"agent": "SystemAgent", "action": "sleep_display"},
    "lock_screen": {"agent": "SystemAgent", "action": "lock_screen"},
    "system_control": {"agent": "SystemAgent", "action": "system_control"},

    # Web
    "search_web": {"agent": "SystemAgent", "action": "search_web"},
    "open_url": {"agent": "SystemAgent", "action": "open_url"},

    # Misc
    "take_screenshot": {"agent": "SystemAgent", "action": "screenshot"},
    "show_notification": {"agent": "SystemAgent", "action": "notify"},

    # ==========================================================================
    # Voice/speech intents -> VoiceAgent (via Brain delegation)
    # ==========================================================================
    "stop_speaking": {"agent": "VoiceAgent", "action": "stop_speech"},

    # ==========================================================================
    # Memory intents -> MemoryAgent
    # ==========================================================================
    "remember": {"agent": "MemoryAgent", "action": "store"},
    "recall": {"agent": "MemoryAgent", "action": "query"},
    "forget": {"agent": "MemoryAgent", "action": "delete"},
    "set_reminder": {"agent": "MemoryAgent", "action": "store"},

    # ==========================================================================
    # Meta intents -> SystemAgent handles conversational responses
    # ==========================================================================
    "greeting": {"agent": "SystemAgent", "action": "greeting"},
    "help": {"agent": "SystemAgent", "action": "help"},
    "goodbye": {"agent": "SystemAgent", "action": "goodbye"},
    "status": {"agent": "Brain", "action": "respond"},
    "thanks": {"agent": "Brain", "action": "respond"},

    # General questions (future: route to LLM agent)
    "general_question": {"agent": "Brain", "action": "respond"},

    # ==========================================================================
    # Vision intents -> VisionAgent
    # ==========================================================================
    "toggle_vision": {"agent": "VisionAgent", "action": "toggle_vision"},
    "start_vision": {"agent": "VisionAgent", "action": "toggle_vision"},
    "stop_vision": {"agent": "VisionAgent", "action": "toggle_vision"},
    "enroll_face": {"agent": "VisionAgent", "action": "enroll_face"},
    "recognize_face": {"agent": "VisionAgent", "action": "recognize_face"},
}



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
        "IntentAgent",
        "PluginAgent",
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
    It implements a supervisor pattern to handle agent failures.
    
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
    
    TODO: Add agent hot-reload
    TODO: Add configuration hot-reload
    TODO: Add metrics export (Prometheus)
    TODO: Add distributed tracing
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """
        Initialize the Brain.
        
        Args:
            config: Configuration dictionary (from settings.yaml)
            event_bus: Event bus instance (defaults to global)
        """
        self._config = config or {}
        self._event_bus = event_bus or get_event_bus()
        self._state = BrainState.INITIALIZING
        
        # Parse configuration
        orchestrator_config = self._config.get("orchestrator", {})
        
        # Default startup order
        default_startup_order = [
            "MemoryAgent",
            "SystemAgent", 
            "VoiceAgent",
            "IntentAgent",
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
        self._main_loop_task: Optional[asyncio.Task] = None
        self._hud_overlay: Optional[HUDOverlayController] = None
        self._health_server: Optional[HealthServer] = None
        
        # Session info
        self._session_id = uuid4()
        self._started_at: Optional[datetime] = None
        
        # Conversation context - short-term memory
        self._context = ConversationContext(session_id=self._session_id)
        
        # Active execution plans (for multi-step commands)
        self._active_plans: Dict[UUID, ExecutionPlan] = {}
        
        # Response templates for meta-intents
        self._meta_responses: Dict[str, str] = {
            "greeting": "Hello Sir. I'm Vesper, your virtual assistant. How can I help you?",
            "help": "I can help you control your Mac. Try saying things like 'open Safari', 'set volume to 50', or 'take a screenshot'.",
            "status": "All systems are operational. I'm ready to assist.",
            "goodbye": "Goodbye! Have a great day.",
            "thanks": "You're welcome! Is there anything else I can help with?",
        }
        
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
            
            self._state = BrainState.RUNNING
            self._started_at = datetime.now(timezone.utc)
            
            logger.info("Brain started successfully")

            await self._event_bus.publish(VoiceOutputEvent(
                text="Vesper online and ready to assist, Sir.",
                source="Brain",
            ))

        except Exception as e:
            self._state = BrainState.ERROR
            logger.error(f"Failed to start brain: {e}", exc_info=True)
            raise
    
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
    # Agent Lifecycle
    # =========================================================================
    
    async def _register_default_agents(self) -> None:
        """Register the default set of agents."""
        logger.debug("Registering default agents...")
        
        # Get agent-specific configs
        voice_config = self._config.get("voice", {})
        intent_config = self._config.get("intent", {})
        system_config = self._config.get("system", {})
        memory_config = self._config.get("memory", {})
        vision_config = self._config.get("vision", {})
        plugins_config = self._config.get("plugins", {})
        web_search_config = self._config.get("web_search", {})
        security_config = self._config.get("security", {})
        
        # Create and register agents
        agents = [
            MemoryAgent(config={"memory": memory_config}),
            SystemAgent(config={"system": system_config}),
            MacOSControlAgent(config={"system": system_config}),
            WebSearchAgent(config={"web_search": web_search_config, "intent": intent_config, "system": system_config}),
            IntentAgent(config={"intent": intent_config, "security": security_config, "reasoning": self._config.get("reasoning", {})}),
            PluginAgent(config={"plugins": plugins_config}),
            VoiceAgent(config={"voice": voice_config}),
        ]

        # Conditionally add VisionAgent if enabled and available
        if vision_config.get("enabled", False):
            if VISION_AVAILABLE:
                agents.append(VisionAgent(config={"vision": vision_config, "intent": intent_config}))
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
        - VoiceInputEvent: When user speaks (USER_SPOKE)
        - IntentRecognizedEvent: When intent is parsed (INTENT_PARSED)
        - IntentUnknownEvent: When intent cannot be determined
        - ActionResultEvent: When an action completes
        - AgentErrorEvent: When an agent has an error
        - AgentStoppedEvent: When an agent stops
        """
        # High-level user interaction events
        self._event_bus.subscribe(VoiceInputEvent, self._handle_voice_input)
        self._event_bus.subscribe(IntentRecognizedEvent, self._handle_intent_recognized)
        self._event_bus.subscribe(IntentUnknownEvent, self._handle_unknown_intent)
        
        # Action result tracking
        self._event_bus.subscribe(ActionResultEvent, self._handle_action_result)
        
        # Agent lifecycle events
        self._event_bus.subscribe(AgentErrorEvent, self._handle_agent_error)
        self._event_bus.subscribe(AgentStoppedEvent, self._handle_agent_stopped)
        
    async def _handle_voice_input(self, event: VoiceInputEvent) -> None:
        """
        Handle voice input event (USER_SPOKE).
        
        This is the first event when a user speaks. The Brain:
        1. Updates conversation context
        2. Emits context update event
        
        Intent processing happens via IntentRecognizedEvent.
        """
        logger.info(f"User spoke: '{event.text}'")
        
        # Add to conversation context
        self._context.add_turn(user_input=event.text)
        
        # Emit context update
        await self._event_bus.emit(ContextUpdatedEvent(
            context_type="turn",
            context_key="user_input",
            context_value=event.text,
            turn_number=self._context.turn_count,
            source="Brain",
        ))
    
    async def _handle_intent_recognized(self, event: IntentRecognizedEvent) -> None:
        """
        Handle recognized intent event (INTENT_PARSED).
        
        This is the core planning logic. The Brain:
        1. Updates context with intent and entities
        2. Determines which agent should handle the intent
        3. Creates an execution plan (single or multi-step)
        4. Emits ActionRequestEvent(s) - NEVER executes directly
        """
        intent = event.intent
        entities = event.slots
        
        logger.info(f"Intent recognized: {intent} with entities: {entities}")
        
        # Update conversation context
        if self._context.last_turn:
            self._context.last_turn.intent = intent
            self._context.last_turn.entities = entities
        self._context.entities.update(entities)
        
        # Route the intent to appropriate agent
        await self._route_intent(intent, entities, event.correlation_id)
    
    async def _route_intent(
        self,
        intent: str,
        entities: Dict[str, Any],
        correlation_id: Optional[UUID] = None,
    ) -> None:
        """
        Route an intent to the appropriate agent.

        The Brain is a PLANNER - it emits ActionRequestEvents
        instead of executing tasks directly.
        """
        # Normalize at the routing boundary so callers can pass either
        # case; INTENT_ROUTING keys are lowercase-only.
        normalized_intent = intent.lower()
        routing = INTENT_ROUTING.get(normalized_intent)

        if not routing:
            # Unknown intent - ask for clarification
            logger.warning(f"No routing found for intent: {intent}")
            await self._request_clarification(
                f"I'm not sure how to handle '{intent}'.",
                [intent],
            )
            return

        target_agent = routing["agent"]
        action = routing["action"]

        # Meta intents are handled by Brain (response only)
        if target_agent == "Brain":
            await self._handle_meta_intent(normalized_intent, entities, correlation_id)
            return

        # Check if this is a multi-step command
        if self._is_multi_step_command(intent, entities):
            await self._create_and_execute_plan(intent, entities, correlation_id)
        else:
            # Single action - emit ActionRequestEvent
            await self._emit_action_request(
                action=action,
                target_agent=target_agent,
                parameters=entities,
                correlation_id=correlation_id,
            )
    
    async def _handle_meta_intent(
        self,
        intent: str,
        entities: Dict[str, Any],
        correlation_id: Optional[UUID] = None,
    ) -> None:
        """
        Handle meta intents (greeting, help, status, etc.).
        
        These don't require external agents - Brain just responds.
        """
        response = self._meta_responses.get(intent, "I'm here to help.")
        
        # Special handling for status
        if intent == "status":
            response = self._generate_status_response()
        
        # Update context with response
        self._context.update_last_response(response, f"meta:{intent}")
        
        # Emit voice output (delegation to VoiceAgent)
        await self._event_bus.emit(VoiceOutputEvent(
            text=response,
            source="Brain",
            correlation_id=correlation_id,
        ))
    
    def _generate_status_response(self) -> str:
        """Generate a status response."""
        agent_count = len(self._agents)
        running_count = sum(
            1 for info in self._agents.values()
            if info.agent.state == AgentState.RUNNING
        )
        uptime = int(self.uptime_seconds)
        
        return (
            f"All systems operational. "
            f"I have {running_count} of {agent_count} agents running. "
            f"Uptime is {uptime} seconds."
        )
    
    async def _emit_action_request(
        self,
        action: str,
        target_agent: str,
        parameters: Dict[str, Any],
        correlation_id: Optional[UUID] = None,
        plan_id: Optional[UUID] = None,
        step_number: int = 0,
    ) -> None:
        """
        Emit an ActionRequestEvent to delegate work.
        
        The Brain NEVER executes tasks directly - it always
        delegates via ActionRequestEvent.
        """
        logger.info(f"Delegating action '{action}' to {target_agent}")
        
        await self._event_bus.emit(ActionRequestEvent(
            action=action,
            target_agent=target_agent,
            parameters=parameters,
            part_of_plan=plan_id is not None,
            plan_id=plan_id,
            step_number=step_number,
            source="Brain",
            correlation_id=correlation_id,
        ))
    
    async def _handle_action_result(self, event: ActionResultEvent) -> None:
        """
        Handle action result event.
        
        Updates context and continues multi-step plans if needed.
        """
        logger.info(
            f"Action '{event.action}' completed: "
            f"success={event.success}, error={event.error or 'none'}"
        )
        
        # Update context
        self._context.update_last_response(
            response=str(event.result) if event.result else "",
            action=event.action,
        )
        
        # Check if this is part of a multi-step plan
        if event.plan_id and event.plan_id in self._active_plans:
            await self._continue_plan(event.plan_id, event.success, event.result)
    
    async def _request_clarification(
        self,
        question: str,
        options: List[str],
    ) -> None:
        """Request clarification from the user."""
        self._context.pending_clarification = True
        self._context.clarification_context = {
            "question": question,
            "options": options,
        }
        
        await self._event_bus.emit(ClarificationNeededEvent(
            original_text=self._context.last_user_input,
            question=question,
            options=options,
            source="Brain",
        ))
        
        # Also speak the question
        await self._event_bus.emit(VoiceOutputEvent(
            text=question,
            source="Brain",
        ))
    
    # =========================================================================
    # Multi-Step Planning
    # =========================================================================
    
    def _is_multi_step_command(self, intent: str, entities: Dict[str, Any]) -> bool:
        """
        Determine if a command requires multiple steps.
        
        Examples of multi-step commands:
        - "Open Safari and go to YouTube"
        - "Set volume to 50 and play music"
        """
        # Check for conjunctions in the original text
        if self._context.last_user_input:
            text = self._context.last_user_input.lower()
            if " and " in text or " then " in text:
                return True
        
        # Check for multiple applications or actions
        if "applications" in entities and len(entities.get("applications", [])) > 1:
            return True
        
        return False
    
    async def _create_and_execute_plan(
        self,
        intent: str,
        entities: Dict[str, Any],
        correlation_id: Optional[UUID] = None,
    ) -> None:
        """
        Create and begin executing a multi-step plan.
        
        The Brain creates a plan and emits ActionRequestEvents
        for each step sequentially.
        """
        plan = ExecutionPlan(
            description=f"Multi-step execution for: {self._context.last_user_input}",
        )
        
        # Parse the command into steps (simplified)
        # In a full implementation, this would use the LLM
        steps = self._parse_multi_step_command(intent, entities)
        
        for i, step_info in enumerate(steps):
            plan.steps.append(PlanStep(
                step_number=i + 1,
                action=step_info["action"],
                target_agent=step_info["agent"],
                parameters=step_info.get("parameters", {}),
                description=step_info.get("description", ""),
            ))
        
        if not plan.steps:
            logger.warning("Failed to create plan steps")
            return
        
        # Register the plan
        self._active_plans[plan.plan_id] = plan
        self._context.active_plan_id = plan.plan_id
        
        # Emit plan created event
        await self._event_bus.emit(PlanCreatedEvent(
            plan_id=plan.plan_id,
            description=plan.description,
            steps=[s.description for s in plan.steps],
            total_steps=plan.total_steps,
            source="Brain",
        ))
        
        # Start executing the first step
        await self._execute_next_plan_step(plan.plan_id)
    
    def _parse_multi_step_command(
        self,
        intent: str,
        entities: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Parse a multi-step command into individual steps.
        
        This is a simplified version - a full implementation
        would use the LLM to parse complex commands.
        """
        steps = []
        text = self._context.last_user_input.lower()
        
        # Split by "and" or "then"
        import re
        parts = re.split(r'\s+(?:and|then)\s+', text)
        
        for part in parts:
            # Try to match each part to an intent
            if "open" in part:
                # Extract app name
                app_match = re.search(r'open\s+(\w+)', part)
                if app_match:
                    steps.append({
                        "action": "open_app",
                        "agent": "SystemAgent",
                        "parameters": {"application": app_match.group(1)},
                        "description": f"Open {app_match.group(1)}",
                    })
            
            elif "volume" in part:
                # Extract volume level
                vol_match = re.search(r'(\d+)', part)
                if vol_match:
                    steps.append({
                        "action": "set_volume",
                        "agent": "SystemAgent",
                        "parameters": {"level": int(vol_match.group(1))},
                        "description": f"Set volume to {vol_match.group(1)}",
                    })
        
        return steps
    
    async def _execute_next_plan_step(self, plan_id: UUID) -> None:
        """Execute the next step in a plan."""
        plan = self._active_plans.get(plan_id)
        if not plan or not plan.is_active:
            return
        
        next_step = plan.next_step
        if not next_step:
            # Plan complete
            await self._complete_plan(plan_id, success=True)
            return
        
        logger.info(
            f"Executing plan step {next_step.step_number}/{plan.total_steps}: "
            f"{next_step.description}"
        )
        
        await self._emit_action_request(
            action=next_step.action,
            target_agent=next_step.target_agent,
            parameters=next_step.parameters,
            plan_id=plan_id,
            step_number=next_step.step_number,
        )
    
    async def _continue_plan(
        self,
        plan_id: UUID,
        step_success: bool,
        step_result: Any,
    ) -> None:
        """Continue a multi-step plan after a step completes."""
        plan = self._active_plans.get(plan_id)
        if not plan:
            return
        
        # Mark step complete
        plan.mark_step_complete(step_success, step_result)
        
        # Emit step completed event
        await self._event_bus.emit(PlanStepCompletedEvent(
            plan_id=plan_id,
            step_number=plan.current_step,
            total_steps=plan.total_steps,
            success=step_success,
            continue_plan=step_success,  # Only continue if successful
            source="Brain",
        ))
        
        if not step_success:
            # Step failed - abort plan
            await self._complete_plan(plan_id, success=False)
            return
        
        # Execute next step
        await self._execute_next_plan_step(plan_id)
    
    async def _complete_plan(self, plan_id: UUID, success: bool) -> None:
        """Complete a multi-step plan."""
        plan = self._active_plans.get(plan_id)
        if not plan:
            return
        
        plan.completed = True
        
        # Emit plan completed event
        await self._event_bus.emit(PlanCompletedEvent(
            plan_id=plan_id,
            success=success,
            steps_completed=plan.current_step,
            total_steps=plan.total_steps,
            summary=plan.description,
            source="Brain",
        ))
        
        # Clear context
        if self._context.active_plan_id == plan_id:
            self._context.active_plan_id = None
        
        # Announce completion
        if success:
            response = f"Completed: {plan.description}"
        else:
            response = f"Failed to complete: {plan.description}"
        
        await self._event_bus.emit(VoiceOutputEvent(
            text=response,
            source="Brain",
        ))
        
        # Cleanup
        del self._active_plans[plan_id]

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
    
    async def _handle_unknown_intent(self, event: IntentUnknownEvent) -> None:
        """Handle unknown intent by providing helpful response."""
        response = (
            "I'm sorry, I didn't understand that. "
            "Try saying 'help' to see what I can do."
        )
        
        # Update context
        if self._context.last_turn:
            self._context.last_turn.intent = "unknown"
        self._context.update_last_response(response, "unknown_intent")
        
        await self._event_bus.emit(VoiceOutputEvent(
            text=response,
            source="Brain",
            correlation_id=event.correlation_id,
        ))
    
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
