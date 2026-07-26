"""
Event Bus Module - Core pub/sub system for agent communication.

This module implements an asynchronous event-driven architecture that allows
agents to communicate without direct coupling. It follows the Observer pattern
and supports typed events for type safety.

Architecture:
    - Publishers emit events to topics
    - Subscribers register handlers for specific event types
    - The bus routes events asynchronously to all registered handlers
    - No direct agent-to-agent calls allowed - all communication via events

Features:
    - Type-safe event handling with generics
    - Full async/await support with thread-safe operations
    - Priority-based event processing
    - Comprehensive logging for every emitted and handled event
    - Event tracing for observability (via EventTracer)
    - Event history for debugging and replay
    - Metrics tracking for performance monitoring

Observability:
    - All events are logged with structured context
    - EventTracer records event flow across agents
    - JSON logs available for machine parsing
    - Correlation IDs for request tracing
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Dict, Deque, List, Optional, Set, Type, TypeVar
from uuid import UUID, uuid4

from schemas.events import BaseEvent, EventPriority

# Import EventTracer for observability
try:
    from utils.logger import EventTracer
    _tracer_available = True
except ImportError:
    _tracer_available = False

logger = logging.getLogger(__name__)


# Type alias for event handlers
EventHandler = Callable[[BaseEvent], Awaitable[None]]
T = TypeVar("T", bound=BaseEvent)


class SubscriptionToken:
    """
    Token returned when subscribing to events.
    
    Use this token to unsubscribe from events later.
    Implements context manager for automatic cleanup.
    """
    
    def __init__(self, event_bus: "EventBus", event_type: Type[BaseEvent], handler: EventHandler):
        self._event_bus = event_bus
        self._event_type = event_type
        self._handler = handler
        self._id = uuid4()
        self._active = True
    
    @property
    def id(self) -> UUID:
        """Unique identifier for this subscription."""
        return self._id
    
    @property
    def is_active(self) -> bool:
        """Check if subscription is still active."""
        return self._active
    
    def unsubscribe(self) -> None:
        """Unsubscribe this handler from the event bus."""
        if self._active:
            self._event_bus.unsubscribe(self._event_type, self._handler)
            self._active = False
    
    async def __aenter__(self) -> "SubscriptionToken":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unsubscribe()


@dataclass
class EventMetrics:
    """
    Metrics tracking for event bus performance monitoring.
    
    Tracks:
        - Total events published and delivered
        - Failed deliveries for monitoring
        - Per-event-type counts
        - Handler registration count
        - Processing latency (average)
    """
    
    events_published: int = 0
    events_delivered: int = 0
    events_failed: int = 0
    handlers_registered: int = 0
    total_processing_time_ms: float = 0.0
    _event_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _handler_times: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    
    def record_publish(self, event_type: str) -> None:
        """Record a published event."""
        self.events_published += 1
        self._event_counts[event_type] += 1
    
    def record_delivery(self, processing_time_ms: float = 0.0) -> None:
        """Record a successful event delivery with optional timing."""
        self.events_delivered += 1
        self.total_processing_time_ms += processing_time_ms
    
    def record_failure(self, event_type: str = "") -> None:
        """Record a failed event delivery."""
        self.events_failed += 1
    
    def get_event_counts(self) -> Dict[str, int]:
        """Get counts per event type."""
        return dict(self._event_counts)
    
    @property
    def average_processing_time_ms(self) -> float:
        """Calculate average processing time per delivered event."""
        if self.events_delivered == 0:
            return 0.0
        return self.total_processing_time_ms / self.events_delivered
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        return {
            "published": self.events_published,
            "delivered": self.events_delivered,
            "failed": self.events_failed,
            "handlers": self.handlers_registered,
            "avg_processing_ms": round(self.average_processing_time_ms, 2),
            "event_counts": self.get_event_counts(),
        }


class EventBus:
    """
    Central event bus for agent-to-agent communication.
    
    This class implements a publish-subscribe pattern that allows agents
    to communicate asynchronously without direct dependencies.
    
    DESIGN PRINCIPLE: No direct agent-to-agent calls allowed.
    All inter-agent communication MUST go through the event bus.
    
    Features:
        - Type-safe event handling with generics
        - Full async/await support
        - Thread-safe subscription management
        - Priority-based event processing
        - Comprehensive logging for every emit and handle
        - Event history for debugging/replay
        - Wildcard subscriptions (subscribe to all events)
        - Metrics tracking
    
    Example:
        ```python
        bus = EventBus()
        
        async def handle_voice_input(event: VoiceInputEvent):
            print(f"Received: {event.text}")
        
        token = bus.subscribe(VoiceInputEvent, handle_voice_input)
        await bus.emit(VoiceInputEvent(text="Hello VESPER"))
        token.unsubscribe()
        ```
    """
    
    # Singleton instance
    _instance: Optional["EventBus"] = None
    _lock: threading.Lock = threading.Lock()
    
    # Event history size limit
    MAX_HISTORY_SIZE: int = 1000
    
    def __new__(cls) -> "EventBus":
        """Singleton pattern to ensure single event bus instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the event bus with all required components."""
        if self._initialized:
            return
        
        # Handler storage - maps event types to list of handlers
        self._handlers: Dict[Type[BaseEvent], List[EventHandler]] = defaultdict(list)
        self._wildcard_handlers: List[EventHandler] = []
        
        # Thread-safe lock for handler modifications
        self._handler_lock = threading.RLock()
        
        # Priority queue for event processing
        self._event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        
        # Event history for debugging and replay
        self._event_history: Deque[Dict[str, Any]] = deque(maxlen=self.MAX_HISTORY_SIZE)
        self._history_lock = threading.Lock()
        
        # State management
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._metrics = EventMetrics()
        self._initialized = True

        # Construction is logged at DEBUG so INFO shows exactly one authoritative
        # bus line — main.py's "EventBus initialized | singleton id=..." (D1).
        logger.debug(f"EventBus singleton constructed id={id(self):#x}")
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        with cls._lock:
            cls._instance = None
    
    @property
    def metrics(self) -> EventMetrics:
        """Get event bus metrics."""
        return self._metrics
    
    @property
    def event_history(self) -> List[Dict[str, Any]]:
        """Get a copy of the event history for debugging."""
        with self._history_lock:
            return list(self._event_history)
    
    def _record_event_history(self, event: BaseEvent, status: str, handler_name: str = "", error: str = "") -> None:
        """Record an event in the history for debugging/replay."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event.type,
            "event_id": str(event.event_id),
            "source": event.source,
            "status": status,
            "handler": handler_name,
            "error": error,
            "payload_summary": str(event.payload)[:200],  # Truncate for storage
        }
        with self._history_lock:
            self._event_history.append(record)
    
    def subscribe(
        self,
        event_type: Type[T],
        handler: Callable[[T], Awaitable[None]],
    ) -> SubscriptionToken:
        """
        Subscribe to events of a specific type.
        
        Thread-safe subscription that registers a handler for the given event type.
        
        Args:
            event_type: The type of event to subscribe to
            handler: Async function to handle the event
        
        Returns:
            SubscriptionToken for unsubscribing
        
        Example:
            token = bus.subscribe(VoiceInputEvent, my_handler)
        """
        with self._handler_lock:
            self._handlers[event_type].append(handler)
            self._metrics.handlers_registered += 1
        
        logger.info(
            f"[SUBSCRIBE] Handler '{handler.__name__}' subscribed to {event_type.__name__} "
            f"(total handlers: {self._metrics.handlers_registered})"
        )
        
        return SubscriptionToken(self, event_type, handler)
    
    def subscribe_all(self, handler: EventHandler) -> None:
        """
        Subscribe to all events (wildcard subscription).
        
        Thread-safe. Useful for logging, debugging, or event persistence.
        
        Args:
            handler: Async function to handle all events
        """
        with self._handler_lock:
            self._wildcard_handlers.append(handler)
        logger.info(f"[SUBSCRIBE] Wildcard handler '{handler.__name__}' subscribed to ALL events")
    
    def unsubscribe(self, event_type: Type[BaseEvent], handler: EventHandler) -> bool:
        """
        Unsubscribe a handler from an event type.
        
        Thread-safe removal of a handler.
        
        Args:
            event_type: The event type to unsubscribe from
            handler: The handler to remove
        
        Returns:
            True if handler was found and removed, False otherwise
        """
        with self._handler_lock:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
                self._metrics.handlers_registered -= 1
                logger.info(f"[UNSUBSCRIBE] Handler '{handler.__name__}' unsubscribed from {event_type.__name__}")
                return True
        return False
    
    async def emit(self, event: BaseEvent) -> None:
        """
        Emit an event to all subscribed handlers.
        
        This is the primary method for publishing events. Use this method
        instead of direct agent-to-agent calls.
        
        Events are processed by priority (CRITICAL > HIGH > NORMAL > LOW).
        All handlers for an event are executed concurrently.
        
        Logging: Every emitted event is logged with full context.
        Event Tracing: Records emission in EventTracer for observability.
        
        Args:
            event: The event to emit
        
        Example:
            await bus.emit(VoiceInputEvent(text="Hello", source="voice_agent"))
        """
        event_type = event.type
        event_id = str(event.event_id)
        self._metrics.record_publish(event_type)
        
        # Log the emission with full context
        logger.info(
            f"[EMIT] Event: {event_type} | "
            f"ID: {event_id} | "
            f"Source: {event.source} | "
            f"Priority: {event.priority.name} | "
            f"Payload: {str(event.payload)[:100]}"
        )
        
        # Record in EventTracer for observability
        if _tracer_available:
            try:
                EventTracer.get_instance().record_emit(event_type, event_id, event.source)
            except Exception:
                pass  # Don't let tracing errors break event flow
        
        # Record in history
        self._record_event_history(event, "emitted")
        
        # Get handlers for this specific event type and parent types
        with self._handler_lock:
            handlers = self._get_handlers_for_event(event)
            all_handlers = handlers + list(self._wildcard_handlers)
        
        if not all_handlers:
            # No subscribers is normal in a pub/sub bus with optional consumers
            # (UI events like PlanCreatedEvent/ContextUpdatedEvent with no HUD
            # attached, etc.). This is DEBUG, not a warning — it is not a problem.
            logger.debug(f"[EMIT] No handlers for event type: {event_type}")
            self._record_event_history(event, "no_handlers")
            return
        
        logger.debug(f"[EMIT] Dispatching to {len(all_handlers)} handler(s)")
        
        # Execute all handlers concurrently, sorted by priority
        # (CRITICAL events get processed first)
        tasks = [self._safe_execute(handler, event) for handler in all_handlers]
        await asyncio.gather(*tasks)
    
    async def publish(self, event: BaseEvent) -> None:
        """
        Publish an event to all subscribed handlers.
        
        Alias for emit() for backward compatibility.
        
        Args:
            event: The event to publish
        """
        await self.emit(event)
    
    def _get_handlers_for_event(self, event: BaseEvent) -> List[EventHandler]:
        """
        Get all handlers that should receive this event.
        
        Checks for exact type match and parent types (polymorphism support).
        """
        handlers = []
        
        # Check for exact type match and parent types
        for event_type, type_handlers in self._handlers.items():
            if isinstance(event, event_type):
                handlers.extend(type_handlers)
        
        return handlers
    
    async def _safe_execute(self, handler: EventHandler, event: BaseEvent) -> None:
        """
        Execute a handler with comprehensive error handling and logging.
        
        Features:
            - Timing measurement for performance monitoring
            - Error isolation (one failing handler doesn't affect others)
            - Full logging for every handled event
            - EventTracer integration for observability
        """
        handler_name = handler.__name__
        event_id = str(event.event_id)
        event_type = event.type
        start_time = datetime.now(timezone.utc)
        
        # Record receipt in EventTracer
        if _tracer_available:
            try:
                EventTracer.get_instance().record_receive(event_type, event_id, handler_name)
            except Exception:
                pass
        
        try:
            logger.debug(f"[HANDLE] {handler_name} processing {event_type}")
            await handler(event)
            
            # Calculate processing time
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._metrics.record_delivery(processing_time_ms)
            
            # Log successful handling
            logger.info(
                f"[HANDLED] {event_type} by {handler_name} | "
                f"Time: {processing_time_ms:.2f}ms | "
                f"Event ID: {event_id}"
            )
            
            # Record success in EventTracer
            if _tracer_available:
                try:
                    EventTracer.get_instance().record_handle(
                        event_type, event_id, handler_name, processing_time_ms
                    )
                except Exception:
                    pass
            
            # Record success in history
            self._record_event_history(event, "handled", handler_name)
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._metrics.record_failure(event_type)
            
            # Log the failure with full context
            logger.error(
                f"[HANDLE_ERROR] Handler '{handler_name}' failed for {event_type} | "
                f"Event ID: {event_id} | "
                f"Error: {str(e)}",
                exc_info=True
            )
            
            # Record failure in EventTracer
            if _tracer_available:
                try:
                    EventTracer.get_instance().record_handle(
                        event_type, event_id, handler_name, processing_time_ms, error=str(e)
                    )
                except Exception:
                    pass
            
            # Record failure in history
            self._record_event_history(event, "failed", handler_name, str(e))
    
    async def start(self) -> None:
        """
        Start the event bus background processor.
        
        This enables queued event processing for high-throughput scenarios.
        
        TODO: Implement queued processing mode
        """
        if self._running:
            logger.warning("EventBus already running")
            return
        
        self._running = True
        logger.info("[START] EventBus started and ready for events")
    
    async def stop(self) -> None:
        """
        Stop the event bus and cleanup resources.
        
        Logs final metrics before stopping.
        """
        self._running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        # Log final metrics
        metrics = self._metrics.get_summary()
        logger.info(
            f"[STOP] EventBus stopped | "
            f"Published: {metrics['published']} | "
            f"Delivered: {metrics['delivered']} | "
            f"Failed: {metrics['failed']}"
        )
    
    def get_handler_count(self, event_type: Type[BaseEvent]) -> int:
        """Get the number of handlers for a specific event type."""
        with self._handler_lock:
            return len(self._handlers.get(event_type, []))
    
    def get_all_subscriptions(self) -> Dict[str, int]:
        """Get a summary of all subscriptions by event type."""
        with self._handler_lock:
            return {
                event_type.__name__: len(handlers)
                for event_type, handlers in self._handlers.items()
                if handlers
            }
    
    def clear_all_handlers(self) -> None:
        """Remove all handlers (useful for testing)."""
        with self._handler_lock:
            self._handlers.clear()
            self._wildcard_handlers.clear()
            self._metrics.handlers_registered = 0
        logger.info("[CLEAR] All handlers removed")
    
    def clear_history(self) -> None:
        """Clear the event history."""
        with self._history_lock:
            self._event_history.clear()
        logger.debug("Event history cleared")
    
    def get_recent_events(self, count: int = 10, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent events from history, optionally filtered by type.
        
        Args:
            count: Number of recent events to return
            event_type: Optional filter by event type name
        
        Returns:
            List of event history records
        """
        with self._history_lock:
            events = list(self._event_history)
        
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        
        return events[-count:]


def get_event_bus() -> EventBus:
    """
    Get the process-wide singleton EventBus.

    This is the ONE authoritative way to obtain the bus. `EventBus()` returns
    the same object too (enforced by ``__new__``), but always prefer
    ``get_event_bus()``: it is the single, greppable entry point and — unlike
    the old module-level ``event_bus = EventBus()`` global — it constructs the
    bus lazily on first use rather than eagerly at import time (before the event
    loop even exists).

    Every agent MUST share this one instance. Two live buses would mean events
    published on one are invisible to handlers on the other.

    Returns:
        The singleton EventBus instance

    Example:
        bus = get_event_bus()
        await bus.emit(MyEvent(source="my_agent"))
    """
    return EventBus()
