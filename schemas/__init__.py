"""
JARVIS Virtual Assistant - Schemas Package.

This package defines all event schemas used for inter-agent communication.
"""

from schemas.events import (
    BaseEvent,
    EventPriority,
    EventCategory,
    VoiceInputEvent,
    VoiceOutputEvent,
    WakeWordDetectedEvent,
    ListeningStateChangedEvent,
    IntentRecognizedEvent,
    IntentUnknownEvent,
    SystemCommandEvent,
    SystemCommandResultEvent,
    MemoryStoreEvent,
    MemoryQueryEvent,
    MemoryQueryResultEvent,
    AgentStartedEvent,
    AgentStoppedEvent,
    AgentErrorEvent,
    ShutdownRequestedEvent,
    ResponseGeneratedEvent,
    HUDGraphStateEvent,
)

__all__ = [
    "BaseEvent",
    "EventPriority",
    "EventCategory",
    "VoiceInputEvent",
    "VoiceOutputEvent",
    "WakeWordDetectedEvent",
    "ListeningStateChangedEvent",
    "IntentRecognizedEvent",
    "IntentUnknownEvent",
    "SystemCommandEvent",
    "SystemCommandResultEvent",
    "MemoryStoreEvent",
    "MemoryQueryEvent",
    "MemoryQueryResultEvent",
    "AgentStartedEvent",
    "AgentStoppedEvent",
    "AgentErrorEvent",
    "ShutdownRequestedEvent",
    "ResponseGeneratedEvent",
    "HUDGraphStateEvent",
]
