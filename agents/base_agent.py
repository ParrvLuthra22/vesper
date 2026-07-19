"""
Base Agent Module - Abstract base class for all agents.

This module defines the foundational Agent interface that all specialized
agents must implement. It follows the Template Method pattern to ensure
consistent lifecycle management across all agents.

CRITICAL DESIGN RULE:
    Agents MUST NOT call each other directly. All inter-agent communication
    MUST go through the EventBus. This ensures:
    - Loose coupling between agents
    - Scalability (agents can be distributed)
    - Testability (agents can be tested in isolation)
    - Observability (all communication is logged)

Design Principles:
    - Single Responsibility: Each agent handles one domain
    - Open/Closed: Extend via inheritance, not modification
    - Liskov Substitution: All agents are interchangeable via base interface
    - Interface Segregation: Minimal required interface
    - Dependency Inversion: Agents depend on abstractions (EventBus)

Agent Lifecycle:
    1. __init__: Agent created, not yet started
    2. start(): Initialize resources, subscribe to events
    3. Running: Process events via handle_event(), perform actions
    4. stop(): Cleanup resources, unsubscribe from events

Subclasses must implement:
    - capabilities: Property listing agent capabilities
    - _setup(): Initialize resources and subscribe to events
    - _teardown(): Cleanup resources
    - handle_event(): Optional override for centralized event handling
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Type, Union
from uuid import UUID, uuid4

from bus.event_bus import EventBus, SubscriptionToken, get_event_bus
from schemas.events import (
    AgentErrorEvent,
    AgentHealthCheckEvent,
    AgentStartedEvent,
    AgentStoppedEvent,
    BaseEvent,
    ShutdownRequestedEvent,
)
from utils.logger import get_logger


class AgentState(Enum):
    """
    Possible states of an agent in its lifecycle.
    
    State transitions:
        CREATED -> STARTING -> RUNNING -> STOPPING -> STOPPED
                                      -> ERROR
    """
    
    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


class AgentLogger:
    """
    Structured logger for agents with automatic context injection.
    
    Provides consistent log formatting with agent name, state, and event context.
    All log messages include the agent name prefix for easy filtering.
    """
    
    def __init__(self, agent_name: str, base_logger: logging.Logger):
        self._agent_name = agent_name
        self._logger = base_logger
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs) -> None:
        """Set persistent context that will be included in all log messages."""
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all persistent context."""
        self._context.clear()
    
    def _format_message(self, message: str, event: Optional[BaseEvent] = None) -> str:
        """Format message with agent context."""
        parts = [f"[{self._agent_name}]", message]
        
        if event:
            parts.append(f"| event_type={event.type} event_id={event.event_id}")
        
        if self._context:
            context_str = " ".join(f"{k}={v}" for k, v in self._context.items())
            parts.append(f"| {context_str}")
        
        return " ".join(parts)
    
    def debug(self, message: str, event: Optional[BaseEvent] = None) -> None:
        """Log debug message with agent context."""
        self._logger.debug(self._format_message(message, event))
    
    def info(self, message: str, event: Optional[BaseEvent] = None) -> None:
        """Log info message with agent context."""
        self._logger.info(self._format_message(message, event))
    
    def warning(self, message: str, event: Optional[BaseEvent] = None) -> None:
        """Log warning message with agent context."""
        self._logger.warning(self._format_message(message, event))
    
    def error(self, message: str, event: Optional[BaseEvent] = None, exc_info: bool = False) -> None:
        """Log error message with agent context."""
        self._logger.error(self._format_message(message, event), exc_info=exc_info)
    
    def event_received(self, event: BaseEvent) -> None:
        """Log that an event was received."""
        self._logger.debug(
            f"[{self._agent_name}] EVENT_RECEIVED | "
            f"type={event.type} | "
            f"id={event.event_id} | "
            f"source={event.source}"
        )
    
    def event_handled(self, event: BaseEvent, duration_ms: float) -> None:
        """Log that an event was successfully handled."""
        self._logger.info(
            f"[{self._agent_name}] EVENT_HANDLED | "
            f"type={event.type} | "
            f"id={event.event_id} | "
            f"duration={duration_ms:.2f}ms"
        )
    
    def event_failed(self, event: BaseEvent, error: Exception) -> None:
        """Log that event handling failed."""
        self._logger.error(
            f"[{self._agent_name}] EVENT_FAILED | "
            f"type={event.type} | "
            f"id={event.event_id} | "
            f"error={type(error).__name__}: {error}"
        )


@dataclass
class AgentCapability:
    """
    Describes a capability that an agent provides.
    
    Used for agent discovery and orchestration.
    
    Attributes:
        name: Unique capability identifier
        description: Human-readable description
        input_events: Event types this capability consumes
        output_events: Event types this capability produces
    """
    
    name: str
    description: str
    input_events: List[str] = field(default_factory=list)
    output_events: List[str] = field(default_factory=list)


@dataclass
class AgentMetrics:
    """
    Runtime metrics for an agent.
    
    TODO: Add histogram for processing times
    TODO: Add error rate calculation
    """
    
    events_received: int = 0
    events_processed: int = 0
    events_failed: int = 0
    last_event_time: Optional[datetime] = None
    processing_time_total_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of event processing."""
        if self.events_received == 0:
            return 1.0
        return self.events_processed / self.events_received
    
    @property
    def average_processing_time_ms(self) -> float:
        """Calculate average processing time per event."""
        if self.events_processed == 0:
            return 0.0
        return self.processing_time_total_ms / self.events_processed


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    This class provides:
        - Lifecycle management (start, stop)
        - Event bus integration
        - Logging infrastructure
        - Metrics collection
        - Error handling
    
    Subclasses must implement:
        - _setup(): Initialize agent-specific resources
        - _teardown(): Cleanup agent-specific resources
        - capabilities: Property returning agent capabilities
    
    Example:
        class MyAgent(BaseAgent):
            @property
            def capabilities(self) -> List[AgentCapability]:
                return [AgentCapability(
                    name="my_capability",
                    description="Does something useful"
                )]
            
            async def _setup(self) -> None:
                self._subscribe(SomeEvent, self.handle_event)
            
            async def _teardown(self) -> None:
                pass
            
            async def handle_event(self, event: BaseEvent) -> None:
                # Centralized event handling
                if isinstance(event, SomeEvent):
                    await self._process_some_event(event)
    """
    
    # Handler type alias for async event handlers
    EventHandler = Callable[[BaseEvent], Awaitable[None]]
    
    def __init__(
        self,
        name: Optional[str] = None,
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the agent.
        
        IMPORTANT: Agents must NOT hold references to other agents.
        All communication must go through the EventBus.
        
        Args:
            name: Agent name (defaults to class name)
            event_bus: Event bus instance (defaults to global instance)
            config: Agent-specific configuration
        """
        self._name = name or self.__class__.__name__
        self._event_bus = event_bus or get_event_bus()
        self._config = config or {}
        self._state = AgentState.CREATED
        self._subscriptions: List[SubscriptionToken] = []
        
        # Set up structured logging for this agent
        base_logger = get_logger(f"agents.{self._name}")
        self._logger = AgentLogger(self._name, base_logger)
        
        self._metrics = AgentMetrics()
        self._agent_id = uuid4()
        self._started_at: Optional[datetime] = None
        self._pending_tasks: Set[asyncio.Task] = set()
        
        # Async hooks for lifecycle events (can be overridden)
        self._on_start_hooks: List[Callable[[], Awaitable[None]]] = []
        self._on_stop_hooks: List[Callable[[], Awaitable[None]]] = []
        self._on_error_hooks: List[Callable[[Exception], Awaitable[None]]] = []
        
        self._logger.debug(f"Agent created (id={self._agent_id})")
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def name(self) -> str:
        """Get the agent's name."""
        return self._name
    
    @property
    def event_bus(self) -> EventBus:
        """
        Get the agent's event bus reference.
        
        Use this for emitting events. Never use it to directly
        call other agents.
        """
        return self._event_bus
    
    @property
    def agent_id(self) -> UUID:
        """Get the agent's unique identifier."""
        return self._agent_id
    
    @property
    def state(self) -> AgentState:
        """Get the agent's current state."""
        return self._state
    
    @property
    def is_running(self) -> bool:
        """Check if the agent is currently running."""
        return self._state == AgentState.RUNNING
    
    @property
    def metrics(self) -> AgentMetrics:
        """Get the agent's runtime metrics."""
        return self._metrics
    
    @property
    def uptime_seconds(self) -> float:
        """Get the agent's uptime in seconds."""
        if self._started_at is None:
            return 0.0
        return (datetime.now(timezone.utc) - self._started_at).total_seconds()

    def is_healthy(self) -> bool:
        """
        Health check hook for observability.

        Override in subclasses to perform more detailed checks.
        Default implementation assumes the agent is healthy.
        """
        return True
    
    @property
    @abstractmethod
    def capabilities(self) -> List[AgentCapability]:
        """
        Get the list of capabilities this agent provides.
        
        Must be implemented by subclasses.
        
        Returns:
            List of AgentCapability instances
        """
        pass
    
    # =========================================================================
    # Lifecycle Methods
    # =========================================================================
    
    async def start(self) -> None:
        """
        Start the agent.
        
        This method:
            1. Changes state to STARTING
            2. Subscribes to shutdown events
            3. Calls _setup() for subclass initialization
            4. Changes state to RUNNING
            5. Emits AgentStartedEvent
        
        Raises:
            RuntimeError: If agent is not in CREATED or STOPPED state
        """
        if self._state not in (AgentState.CREATED, AgentState.STOPPED):
            raise RuntimeError(
                f"Cannot start agent in state {self._state.name}"
            )
        
        self._logger.info(f"Starting agent")
        self._state = AgentState.STARTING
        
        try:
            # Subscribe to system shutdown events
            self._subscribe(ShutdownRequestedEvent, self._handle_shutdown)
            
            # Run subclass-specific setup
            await self._setup()
            
            # Run any registered start hooks
            await self._run_start_hooks()
            
            self._state = AgentState.RUNNING
            self._started_at = datetime.now(timezone.utc)
            
            # Announce that we're running (via EventBus - the only communication method)
            await self._emit(AgentStartedEvent(
                agent_name=self._name,
                agent_type=self.__class__.__name__,
                capabilities=[c.name for c in self.capabilities],
                source=self._name,
            ))
            
            self._logger.info(f"Agent started successfully")
            
        except Exception as e:
            self._state = AgentState.ERROR
            self._logger.error(f"Failed to start agent: {e}", exc_info=True)
            await self._run_error_hooks(e)
            await self._emit(AgentErrorEvent(
                agent_name=self._name,
                error_type=type(e).__name__,
                error_message=str(e),
                is_recoverable=False,
                source=self._name,
            ))
            raise
    
    async def stop(self, reason: str = "Normal shutdown") -> None:
        """
        Stop the agent.
        
        This method:
            1. Runs stop hooks
            2. Changes state to STOPPING
            3. Cancels pending tasks
            4. Calls _teardown() for subclass cleanup
            5. Unsubscribes from all events
            6. Changes state to STOPPED
            7. Emits AgentStoppedEvent
        """
        if self._state not in (AgentState.RUNNING, AgentState.ERROR):
            self._logger.warning(
                f"Attempted to stop agent in state {self._state.name}"
            )
            return
        
        self._logger.info(f"Stopping agent (reason: {reason})")
        self._state = AgentState.STOPPING
        
        try:
            # Run stop hooks before teardown
            await self._run_stop_hooks()
            
            # Cancel pending tasks
            await self._cancel_pending_tasks()
            
            # Run subclass-specific teardown
            await self._teardown()
            
            # Unsubscribe from all events
            self._unsubscribe_all()
            
            self._state = AgentState.STOPPED
            
            # Announce that we've stopped (via EventBus)
            await self._emit(AgentStoppedEvent(
                agent_name=self._name,
                reason=reason,
                clean_shutdown=True,
                source=self._name,
            ))
            
            self._logger.info(f"Agent stopped")
            
        except Exception as e:
            self._state = AgentState.ERROR
            self._logger.error(f"Error stopping agent: {e}", exc_info=True)
            await self._emit(AgentStoppedEvent(
                agent_name=self._name,
                reason=f"Error: {e}",
                clean_shutdown=False,
                source=self._name,
            ))
    
    async def health_check(self) -> AgentHealthCheckEvent:
        """
        Perform a health check on this agent.
        
        Returns:
            AgentHealthCheckEvent with health status
        """
        return AgentHealthCheckEvent(
            agent_name=self._name,
            is_healthy=self._state == AgentState.RUNNING,
            last_activity=self._metrics.last_event_time,
            pending_tasks=len(self._pending_tasks),
            source=self._name,
        )
    
    # =========================================================================
    # Abstract Methods (Must be implemented by subclasses)
    # =========================================================================
    
    @abstractmethod
    async def _setup(self) -> None:
        """
        Initialize agent-specific resources.
        
        Called during start(). Use this to:
            - Subscribe to relevant events
            - Initialize connections
            - Load models or data
        
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    async def _teardown(self) -> None:
        """
        Cleanup agent-specific resources.
        
        Called during stop(). Use this to:
            - Close connections
            - Save state
            - Release resources
        
        Must be implemented by subclasses.
        """
        pass
    
    async def handle_event(self, event: BaseEvent) -> None:
        """
        Handle an incoming event.
        
        This is the primary method for processing events. Subclasses can
        override this for centralized event handling, or use individual
        handler methods registered via _subscribe().
        
        Default implementation does nothing - override in subclasses
        if you want a single entry point for all events.
        
        Args:
            event: The event to handle
        
        Example:
            async def handle_event(self, event: BaseEvent) -> None:
                if isinstance(event, VoiceInputEvent):
                    await self._process_voice_input(event)
                elif isinstance(event, IntentRecognizedEvent):
                    await self._process_intent(event)
        """
        # Default implementation - subclasses can override
        pass
    
    # =========================================================================
    # Async Lifecycle Hooks
    # =========================================================================
    
    def add_on_start_hook(self, hook: Callable[[], Awaitable[None]]) -> None:
        """
        Add a hook to be called when the agent starts.
        
        Hooks are called after _setup() completes successfully.
        
        Args:
            hook: Async function to call on start
        """
        self._on_start_hooks.append(hook)
    
    def add_on_stop_hook(self, hook: Callable[[], Awaitable[None]]) -> None:
        """
        Add a hook to be called when the agent stops.
        
        Hooks are called before _teardown() is called.
        
        Args:
            hook: Async function to call on stop
        """
        self._on_stop_hooks.append(hook)
    
    def add_on_error_hook(self, hook: Callable[[Exception], Awaitable[None]]) -> None:
        """
        Add a hook to be called when an error occurs during event handling.
        
        Args:
            hook: Async function to call on error, receives the exception
        """
        self._on_error_hooks.append(hook)
    
    async def _run_start_hooks(self) -> None:
        """Run all registered start hooks."""
        for hook in self._on_start_hooks:
            try:
                await hook()
            except Exception as e:
                self._logger.error(f"Start hook failed: {e}", exc_info=True)
    
    async def _run_stop_hooks(self) -> None:
        """Run all registered stop hooks."""
        for hook in self._on_stop_hooks:
            try:
                await hook()
            except Exception as e:
                self._logger.error(f"Stop hook failed: {e}", exc_info=True)
    
    async def _run_error_hooks(self, error: Exception) -> None:
        """Run all registered error hooks."""
        for hook in self._on_error_hooks:
            try:
                await hook(error)
            except Exception as e:
                self._logger.error(f"Error hook failed: {e}", exc_info=True)
    
    # =========================================================================
    # Event Handling Helpers
    # =========================================================================
    
    def _subscribe(
        self,
        event_type: Type[BaseEvent],
        handler: Callable[[BaseEvent], Any],
    ) -> SubscriptionToken:
        """
        Subscribe to an event type.
        
        Automatically wraps handler with metrics tracking and error handling.
        All subscriptions go through the EventBus - this is the ONLY way
        for agents to receive events from other agents.
        
        Args:
            event_type: The event type to subscribe to
            handler: The handler function (sync or async)
        
        Returns:
            SubscriptionToken for the subscription
        """
        # Wrap handler with metrics and error handling
        wrapped_handler = self._wrap_handler(handler)
        
        token = self._event_bus.subscribe(event_type, wrapped_handler)
        self._subscriptions.append(token)
        
        self._logger.debug(f"Subscribed to {event_type.__name__}")
        
        return token
    
    def _wrap_handler(
        self,
        handler: Callable[[BaseEvent], Any],
    ) -> Callable[[BaseEvent], Any]:
        """
        Wrap a handler with metrics, structured logging, and error handling.
        
        This wrapper:
            - Checks if agent is running before processing
            - Logs event receipt and completion
            - Tracks timing metrics
            - Handles errors and runs error hooks
            - Emits error events on failure
        
        Args:
            handler: Original handler function (sync or async)
        
        Returns:
            Wrapped async handler function
        """
        async def wrapped(event: BaseEvent) -> None:
            if not self.is_running:
                self._logger.warning(f"Received event while not running", event=event)
                return
            
            # Log event receipt
            self._logger.event_received(event)
            
            self._metrics.events_received += 1
            self._metrics.last_event_time = datetime.now(timezone.utc)
            
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Execute handler (supports both sync and async)
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
                
                self._metrics.events_processed += 1
                
                # Calculate duration and log success
                duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                self._logger.event_handled(event, duration_ms)
                
            except Exception as e:
                self._metrics.events_failed += 1
                
                # Log the failure
                self._logger.event_failed(event, e)
                
                # Run error hooks
                await self._run_error_hooks(e)
                
                # Emit error event for system-wide visibility
                await self._emit(AgentErrorEvent(
                    agent_name=self._name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    is_recoverable=True,
                    source=self._name,
                ))
            
            finally:
                duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                self._metrics.processing_time_total_ms += duration_ms
        
        # Preserve function name for debugging
        wrapped.__name__ = handler.__name__
        
        return wrapped
    
    def _unsubscribe_all(self) -> None:
        """Unsubscribe from all events."""
        for token in self._subscriptions:
            token.unsubscribe()
        self._subscriptions.clear()
        self._logger.debug("Unsubscribed from all events")
    
    async def _emit(self, event: BaseEvent) -> None:
        """
        Emit an event to the event bus.
        
        This is the ONLY way for an agent to communicate with other agents.
        Never call other agents directly.
        
        Args:
            event: The event to emit
        """
        self._logger.debug(f"Emitting event: {event.type}", event=event)
        await self._event_bus.emit(event)
    
    # =========================================================================
    # Task Management
    # =========================================================================
    
    def _create_task(self, coro) -> asyncio.Task:
        """
        Create a tracked async task.
        
        Tasks created with this method will be cancelled during shutdown.
        
        Args:
            coro: Coroutine to run
        
        Returns:
            The created task
        """
        task = asyncio.create_task(coro)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task
    
    async def _cancel_pending_tasks(self) -> None:
        """Cancel all pending tasks."""
        if not self._pending_tasks:
            return
        
        self._logger.debug(f"Cancelling {len(self._pending_tasks)} pending tasks")
        
        for task in self._pending_tasks:
            task.cancel()
        
        await asyncio.gather(*self._pending_tasks, return_exceptions=True)
        self._pending_tasks.clear()
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    async def _handle_shutdown(self, event: ShutdownRequestedEvent) -> None:
        """Handle system shutdown request."""
        self._logger.info(
            f"Received shutdown request: {event.reason}"
        )
        await self.stop(reason=event.reason)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self._name!r}, "
            f"state={self._state.name}, "
            f"id={self._agent_id})"
        )
