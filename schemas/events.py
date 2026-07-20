"""
Event Schemas Module - Typed event definitions for the event bus.

This module defines all event types used for inter-agent communication.
Events are immutable dataclasses with automatic timestamp and ID generation.

Design Principles:
    - Events are immutable (frozen dataclasses)
    - Events carry all necessary context
    - Events are serializable for persistence/debugging
    - Events follow a clear naming convention: <Domain><Action>Event

TODO: Add event versioning for backward compatibility
TODO: Add event serialization to JSON/MessagePack
TODO: Add event validation with Pydantic
TODO: Add event compression for large payloads
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class EventPriority(Enum):
    """
    Event priority levels for processing order.
    
    Higher priority events are processed before lower priority ones.
    """
    
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()


class EventCategory(Enum):
    """
    Event categories for filtering and routing.
    
    TODO: Add more categories as system expands
    """
    
    SYSTEM = "system"
    VOICE = "voice"
    INTENT = "intent"
    MEMORY = "memory"
    UI = "ui"
    EXTERNAL = "external"
    AGENT = "agent"


@dataclass(frozen=True)
class BaseEvent:
    """
    Base class for all events in the system.
    
    All events inherit from this class and gain automatic
    ID generation and timestamp tracking.
    
    Required fields per specification:
        - type: Event type name (derived from class name)
        - payload: Event-specific data (derived from subclass fields)
        - source: The agent/component that created this event
        - timestamp: When the event was created
    
    Attributes:
        event_id: Unique identifier for this event instance
        timestamp: When the event was created
        priority: Processing priority
        source: The agent/component that created this event
        correlation_id: ID to track related events (for request/response)
        metadata: Optional dict for additional context
    
    Example:
        @dataclass(frozen=True)
        class MyCustomEvent(BaseEvent):
            my_data: str
    """
    
    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: EventPriority = field(default=EventPriority.NORMAL)
    source: str = field(default="unknown")
    correlation_id: Optional[UUID] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def type(self) -> str:
        """
        Get the event type name.
        
        Returns the class name as the event type identifier.
        This ensures every event has an explicit type.
        """
        return self.__class__.__name__
    
    @property
    def payload(self) -> Dict[str, Any]:
        """
        Get the event payload as a dictionary.
        
        Extracts all subclass-specific fields (excluding base fields)
        as a dictionary for serialization and logging.
        """
        base_fields = {'event_id', 'timestamp', 'priority', 'source', 'correlation_id', 'metadata'}
        payload_data = {}
        for f in self.__dataclass_fields__:
            if f not in base_fields:
                payload_data[f] = getattr(self, f)
        return payload_data
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the event to a dictionary.
        
        Useful for logging, debugging, and persistence.
        """
        return {
            "type": self.type,
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.name,
            "source": self.source,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "metadata": self.metadata,
            "payload": self.payload,
        }
    
    def to_json(self) -> str:
        """Serialize the event to a JSON string."""
        import json
        return json.dumps(self.to_dict(), default=str)


# =============================================================================
# Voice Events - Speech recognition and synthesis
# =============================================================================

@dataclass(frozen=True)
class VoiceInputEvent(BaseEvent):
    """
    Event emitted when voice input is recognized (USER_SPOKE).
    
    Attributes:
        text: The transcribed text from speech
        confidence: Recognition confidence (0.0 to 1.0)
        language: Detected language code (e.g., 'en-US')
        audio_duration: Duration of the audio in seconds
        is_wake_word: Whether this came from wake word detection
        is_partial: Whether this is a partial (streaming) result
    
    TODO: Add audio waveform data for advanced processing
    TODO: Add speaker identification
    """
    
    text: str = ""
    confidence: float = 0.0
    language: str = "en-US"
    audio_duration: float = 0.0
    is_wake_word: bool = False
    is_partial: bool = False
    source: str = field(default="VoiceAgent")


@dataclass(frozen=True)
class VoiceOutputEvent(BaseEvent):
    """
    Event to trigger text-to-speech output.
    
    Attributes:
        text: Text to speak
        voice_id: Voice profile to use
        speed: Speech rate multiplier
        wait_for_completion: Whether to block until speech completes
    
    TODO: Add SSML support for advanced speech control
    TODO: Add emotion/tone parameters
    """
    
    text: str = ""
    voice_id: str = "default"
    speed: float = 1.0
    wait_for_completion: bool = False


@dataclass(frozen=True)
class WakeWordDetectedEvent(BaseEvent):
    """
    Event emitted when wake word (e.g., "Hey Vesper") is detected.

    Attributes:
        wake_word: The detected wake word
        confidence: Detection confidence
    """

    wake_word: str = "vesper"
    confidence: float = 0.0
    source: str = field(default="voice_agent")


@dataclass(frozen=True)
class ListeningStateChangedEvent(BaseEvent):
    """
    Event emitted when the listening state changes.
    
    Useful for UI feedback (e.g., showing microphone indicator).
    """
    
    is_listening: bool = False
    listening_mode: str = ""  # "wake_word", "command", "idle"
    reason: str = ""
    source: str = field(default="voice_agent")


# =============================================================================
# Intent Events - Natural language understanding
# =============================================================================

@dataclass(frozen=True)
class IntentRecognizedEvent(BaseEvent):
    """
    Event emitted when user intent is identified.
    
    Attributes:
        intent: The identified intent name
        confidence: Recognition confidence
        entities: Extracted entities from the utterance
        raw_text: Original user input
        slots: Named parameters extracted from the intent
    
    TODO: Add intent disambiguation when confidence is low
    TODO: Add multi-intent support
    """
    
    intent: str = ""
    confidence: float = 0.0
    entities: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    slots: Dict[str, str] = field(default_factory=dict)
    source: str = field(default="intent_agent")


@dataclass(frozen=True)
class ParsedIntent:
    """
    Single parsed intent with entities.
    
    Used as part of MultiIntentEvent for command sequences.
    """
    intent: str
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    

@dataclass(frozen=True)
class MultiIntentEvent(BaseEvent):
    """
    Event emitted when multiple intents are detected in one utterance.
    
    Example: "Open VS Code and tell me CPU usage" becomes:
    [
        {"intent": "OPEN_APP", "entities": {"app": "VS Code"}},
        {"intent": "GET_SYSTEM_STATS", "entities": {"metric": "cpu"}}
    ]
    
    Attributes:
        intents: List of parsed intents in order
        raw_text: Original user input
        execution_mode: "sequential" or "parallel"
    """
    
    intents: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    execution_mode: str = "sequential"  # "sequential" or "parallel"
    source: str = field(default="intent_agent")


@dataclass(frozen=True)
class IntentUnknownEvent(BaseEvent):
    """
    Event emitted when intent cannot be determined.
    
    Triggers fallback behavior or clarification request.
    """
    
    raw_text: str = ""
    suggestions: List[str] = field(default_factory=list)
    source: str = field(default="intent_agent")


# =============================================================================
# Orchestrator Events - Brain planning and coordination
# =============================================================================

@dataclass(frozen=True)
class ActionRequestEvent(BaseEvent):
    """
    Event emitted by the Orchestrator to request an agent to perform an action.
    
    The Brain NEVER executes tasks directly - it emits ActionRequestEvents
    to delegate work to the appropriate agent.
    
    Attributes:
        action: The action to perform (e.g., 'open_application', 'set_volume')
        target_agent: Which agent should handle this (e.g., 'SystemAgent')
        parameters: Action-specific parameters
        priority_level: Execution priority (1=highest, 5=lowest)
        requires_confirmation: Whether user confirmation is needed before execution
        part_of_plan: Whether this is part of a multi-step plan
        plan_id: ID of the plan this action belongs to (if multi-step)
        step_number: Step number within the plan
    """
    
    action: str = ""
    target_agent: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority_level: int = 3
    requires_confirmation: bool = False
    part_of_plan: bool = False
    plan_id: Optional[UUID] = None
    step_number: int = 0
    source: str = field(default="Brain")


@dataclass(frozen=True)
class ActionResultEvent(BaseEvent):
    """
    Event emitted by an agent when an action completes.
    
    Attributes:
        action: The action that was performed
        success: Whether the action succeeded
        result: Result data from the action
        error: Error message if failed
        execution_time_ms: How long the action took
        plan_id: ID of the plan this was part of (if multi-step)
        step_number: Step number within the plan
    """
    
    action: str = ""
    success: bool = False
    result: Any = None
    error: str = ""
    execution_time_ms: float = 0.0
    plan_id: Optional[UUID] = None
    step_number: int = 0


@dataclass(frozen=True)
class PlanCreatedEvent(BaseEvent):
    """
    Event emitted when the Orchestrator creates a multi-step execution plan.
    
    Attributes:
        plan_id: Unique identifier for this plan
        description: Human-readable description of the plan
        steps: List of step descriptions
        total_steps: Total number of steps in the plan
        estimated_duration_ms: Estimated total execution time
    """
    
    plan_id: UUID = field(default_factory=uuid4)
    description: str = ""
    steps: List[str] = field(default_factory=list)
    total_steps: int = 0
    estimated_duration_ms: float = 0.0
    source: str = field(default="Brain")


@dataclass(frozen=True)
class ToolCallStartedEvent(BaseEvent):
    """
    Event emitted live by the Planner the instant it begins executing one
    tool call — before the Guardian tier check, so a confirm-tier call
    shows up immediately rather than only after approval.

    This is the live signal a terminal/HUD renders as an in-progress
    step; it is independent of tracing/tracer.py's TurnTrace (which
    records the same execution for later inspection via /trace, not for
    live display).

    Attributes:
        tool_name: Name of the tool being called.
        arguments: The tool call's arguments.
    """

    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    source: str = field(default="Planner")


@dataclass(frozen=True)
class ToolCallFinishedEvent(BaseEvent):
    """
    Event emitted live by the Planner when a tool call started by a
    matching ToolCallStartedEvent finishes (successfully, denied, timed
    out, or errored).

    Attributes:
        tool_name: Name of the tool that finished.
        success: Whether the call completed without error.
        guardian_verdict: "allow" / "deny" / "n/a" (unknown tool).
        result: Result text (or the rendered error/denial message).
        error: Error message, if any.
        latency_ms: Wall-clock time for the call.
    """

    tool_name: str = ""
    success: bool = False
    guardian_verdict: str = ""
    result: str = ""
    error: Optional[str] = None
    latency_ms: float = 0.0
    source: str = field(default="Planner")


@dataclass(frozen=True)
class PlanStepCompletedEvent(BaseEvent):
    """
    Event emitted when a step in a multi-step plan completes.
    
    Attributes:
        plan_id: ID of the plan
        step_number: Which step completed
        total_steps: Total steps in the plan
        success: Whether the step succeeded
        continue_plan: Whether to continue with the next step
    """
    
    plan_id: UUID = field(default_factory=uuid4)
    step_number: int = 0
    total_steps: int = 0
    success: bool = False
    continue_plan: bool = True


@dataclass(frozen=True)
class PlanCompletedEvent(BaseEvent):
    """
    Event emitted when a multi-step plan finishes.
    
    Attributes:
        plan_id: ID of the completed plan
        success: Whether all steps succeeded
        steps_completed: Number of steps that were completed
        total_steps: Total steps in the plan
        summary: Summary of what was accomplished
    """
    
    plan_id: UUID = field(default_factory=uuid4)
    success: bool = False
    steps_completed: int = 0
    total_steps: int = 0
    summary: str = ""
    source: str = field(default="Brain")


@dataclass(frozen=True)
class ContextUpdatedEvent(BaseEvent):
    """
    Event emitted when conversation context is updated.
    
    Allows agents to be aware of context changes for smarter responses.
    
    Attributes:
        context_type: Type of context update ('turn', 'topic', 'entity')
        context_key: What was updated
        context_value: The new value
        turn_number: Current conversation turn
    """
    
    context_type: str = ""
    context_key: str = ""
    context_value: Any = None
    turn_number: int = 0
    source: str = field(default="Brain")


@dataclass(frozen=True)
class ClarificationNeededEvent(BaseEvent):
    """
    Event emitted when the Orchestrator needs clarification from the user.
    
    Attributes:
        original_text: What the user said
        ambiguity: What is unclear
        options: Possible interpretations
        question: Question to ask the user
    """
    
    original_text: str = ""
    ambiguity: str = ""
    options: List[str] = field(default_factory=list)
    question: str = ""
    source: str = field(default="Brain")


# =============================================================================
# Sensor & Proactive Events - local-only sensing and rule-driven call-outs
# =============================================================================

@dataclass(frozen=True)
class AppFocusChangedEvent(BaseEvent):
    """
    Event emitted by FocusSensor when the frontmost macOS application
    changes.

    Attributes:
        app_name: Name of the newly-focused application.
        previous_app: Name of the previously-focused application (empty
            if this is the very first observation, i.e. not a real switch).
    """

    app_name: str = ""
    previous_app: str = ""
    source: str = field(default="FocusSensor")


@dataclass(frozen=True)
class UpcomingMeetingEvent(BaseEvent):
    """
    Event emitted by CalendarSensor as a meeting approaches (at the 15-
    and 5-minute checkpoints).

    Attributes:
        title: Meeting title.
        start_time: When the meeting starts.
        minutes_until: Minutes remaining until start_time.
    """

    title: str = ""
    start_time: Optional[datetime] = None
    minutes_until: int = 0
    source: str = field(default="CalendarSensor")


@dataclass(frozen=True)
class ObservationEvent(BaseEvent):
    """
    Event emitted by the Proactive Engine's rule evaluator when a rule
    fires (subject to its cooldown). Flows into ConversationContext as a
    pending observation for the Planner's {context} block to surface —
    the Planner may voice it once, passively, per the call-out doctrine
    in config/persona.md.

    Attributes:
        kind: Rule identifier, e.g. "context_switch", "meeting_soon".
        detail: Human-readable observation text for the Planner's context.
        observation_id: Unique id; used to track consumption once voiced.
    """

    kind: str = ""
    detail: str = ""
    observation_id: str = field(default_factory=lambda: str(uuid4()))
    source: str = field(default="ProactiveEngine")


@dataclass(frozen=True)
class BriefingRequestedEvent(BaseEvent):
    """
    Event emitted by a scheduled Proactive Engine job requesting a
    briefing. Minimal for now — P07 turns this into a rich briefing
    pipeline.

    Attributes:
        schedule_name: Which scheduled job fired this, e.g. "morning_briefing".
    """

    schedule_name: str = "morning_briefing"
    source: str = field(default="ProactiveEngine")


# =============================================================================
# System Events - macOS system interactions
# =============================================================================

@dataclass(frozen=True)
class SystemCommandEvent(BaseEvent):
    """
    Event to execute a system command.
    
    Attributes:
        command: Command type (e.g., 'open_app', 'set_volume')
        parameters: Command-specific parameters
        requires_confirmation: Whether user confirmation is needed
    
    TODO: Add command timeout
    TODO: Add sandboxing options
    """
    
    command: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False


@dataclass(frozen=True)
class MacOSCommandEvent(BaseEvent):
    """
    Event to request direct macOS control commands.

    Attributes:
        command_type: High-level command category (e.g. applescript, applescript_file)
        target: Optional command target (app/system/service)
        payload: Arbitrary command payload
    """

    command_type: str = ""
    target: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = field(default="orchestrator")


@dataclass(frozen=True)
class SystemCommandResultEvent(BaseEvent):
    """
    Event emitted when a system command completes.
    
    Attributes:
        success: Whether the command succeeded
        result: Command output or result data
        error: Error message if failed
        execution_time: How long the command took
    """
    
    success: bool = False
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    original_command: str = ""
    source: str = field(default="system_agent")


@dataclass(frozen=True)
class ApplicationLaunchedEvent(BaseEvent):
    """Event emitted when an application is launched."""
    
    app_name: str = ""
    app_bundle_id: str = ""
    success: bool = False
    source: str = field(default="system_agent")


@dataclass(frozen=True)
class ConfirmationRequestedEvent(BaseEvent):
    """
    Event emitted by the Guardian when a confirm/dangerous-tier tool call
    needs explicit approval before it may run.

    Attributes:
        summary: One-line human-readable summary of exactly what will run
            (tool name plus rendered arguments).
        tool_name: Name of the tool awaiting confirmation.
        arguments: The tool call's arguments.
        request_id: Unique ID used to match a later ConfirmationResponseEvent.
    """

    summary: str = ""
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    source: str = field(default="Guardian")


@dataclass(frozen=True)
class ConfirmationResponseEvent(BaseEvent):
    """
    Event emitted in response to a ConfirmationRequestedEvent (e.g. by voice,
    HUD, or another UI surface) to approve or deny a pending tool call.

    Attributes:
        request_id: Matches the ConfirmationRequestedEvent being resolved.
        approved: Whether the user approved the pending tool call.
    """

    request_id: str = ""
    approved: bool = False


@dataclass(frozen=True)
class SystemNotificationEvent(BaseEvent):
    """
    Event to display a system notification.
    
    TODO: Add notification actions
    TODO: Add notification grouping
    """
    
    title: str = ""
    message: str = ""
    subtitle: str = ""
    sound: bool = True


# =============================================================================
# Vision Events - Gesture detection, face recognition, visual commands
# =============================================================================

@dataclass(frozen=True)
class VisionStartEvent(BaseEvent):
    """
    Event to start the vision system.
    
    Attributes:
        enable_gestures: Enable gesture detection
        enable_faces: Enable face detection/recognition
        camera_id: Camera device ID (0 = default)
    """
    
    enable_gestures: bool = True
    enable_faces: bool = True
    camera_id: int = 0
    source: str = field(default="orchestrator")


@dataclass(frozen=True)
class VisionStopEvent(BaseEvent):
    """Event to stop the vision system."""
    
    reason: str = ""
    source: str = field(default="orchestrator")


@dataclass(frozen=True)
class GestureDetectedEvent(BaseEvent):
    """
    Event emitted when a hand gesture is detected.
    
    Supported gestures:
        - THUMBS_UP: Approval, confirm
        - THUMBS_DOWN: Disapproval, cancel
        - WAVE: Greeting, attention
        - STOP: Stop, pause
        - POINT: Pointing direction
        - FIST: Grab, select
        - OPEN_PALM: Release, clear
        - PEACE: Victory sign
        - OK: Okay gesture (thumb and index circle)
        - ONE, TWO, THREE, FOUR, FIVE: Finger counting
    
    Attributes:
        gesture: The detected gesture name
        confidence: Detection confidence (0.0 to 1.0)
        hand: Which hand ('left', 'right', 'both')
        landmarks: Optional hand landmark positions
        bounding_box: Bounding box of the hand region
    """
    
    gesture: str = ""
    confidence: float = 0.0
    hand: str = "right"
    landmarks: Dict[str, Any] = field(default_factory=dict)
    bounding_box: Dict[str, float] = field(default_factory=dict)
    source: str = field(default="VisionAgent")


@dataclass(frozen=True)
class FaceDetectedEvent(BaseEvent):
    """
    Event emitted when a face is detected.
    
    Attributes:
        face_id: Unique identifier for tracking this face
        is_recognized: Whether the face was recognized (known person)
        person_name: Name of recognized person (if is_recognized)
        confidence: Recognition confidence (0.0 to 1.0)
        bounding_box: Bounding box of the face region
        landmarks: Facial landmark positions (eyes, nose, mouth)
        emotion: Detected emotion (if enabled)
        is_looking: Whether person is looking at camera
    """
    
    face_id: str = ""
    is_recognized: bool = False
    person_name: str = ""
    confidence: float = 0.0
    bounding_box: Dict[str, float] = field(default_factory=dict)
    landmarks: Dict[str, Any] = field(default_factory=dict)
    emotion: str = ""
    is_looking: bool = False
    source: str = field(default="VisionAgent")


@dataclass(frozen=True)
class FaceLostEvent(BaseEvent):
    """
    Event emitted when a tracked face is no longer visible.
    
    Attributes:
        face_id: The face ID that was lost
        duration_seconds: How long the face was tracked
    """
    
    face_id: str = ""
    duration_seconds: float = 0.0
    source: str = field(default="VisionAgent")


@dataclass(frozen=True)
class VisionCommandEvent(BaseEvent):
    """
    Event to request a specific vision action.
    
    Commands:
        - capture_frame: Capture a single frame
        - recognize_face: Attempt to recognize current face
        - learn_face: Learn a new face with given name
        - list_known_faces: List all known face names
        - toggle_gestures: Enable/disable gesture detection
        - toggle_faces: Enable/disable face detection
    
    Attributes:
        command: The command to execute
        parameters: Command-specific parameters
    """
    
    command: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    source: str = field(default="orchestrator")


@dataclass(frozen=True)
class VisionCommandResultEvent(BaseEvent):
    """
    Result of a vision command.
    
    Attributes:
        command: The original command
        success: Whether the command succeeded
        result: Command result data
        error: Error message if failed
    """
    
    command: str = ""
    success: bool = False
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    source: str = field(default="VisionAgent")


@dataclass(frozen=True)
class ScreenshotEvent(BaseEvent):
    """
    Event to request a screenshot capture.

    bbox uses (x, y, w, h). If omitted, full screen is captured.
    """

    bbox: Optional[tuple[int, int, int, int]] = None
    save_path: str = "/tmp/vesper_screen.png"
    source: str = field(default="orchestrator")


@dataclass(frozen=True)
class PresenceChangedEvent(BaseEvent):
    """
    Event emitted when user presence changes.
    
    Triggered when user appears/disappears from camera view.
    Useful for auto-wake, screen lock, etc.
    
    Attributes:
        is_present: Whether a user is now present
        face_count: Number of faces currently visible
        duration_absent: Seconds since last presence (if becoming present)
    """
    
    is_present: bool = False
    face_count: int = 0
    duration_absent: float = 0.0
    source: str = field(default="VisionAgent")


# =============================================================================
# Memory Events - Context and history management
# =============================================================================

@dataclass(frozen=True)
class MemoryStoreEvent(BaseEvent):
    """
    Event to store information in memory.
    
    Attributes:
        key: Memory key for retrieval
        value: Data to store
        ttl: Time-to-live in seconds (None = permanent)
        memory_type: Type of memory (short_term, long_term, episodic)
    """
    
    key: str = ""
    value: Any = None
    ttl: Optional[int] = None
    memory_type: str = "short_term"


@dataclass(frozen=True)
class MemoryQueryEvent(BaseEvent):
    """
    Event to query memory.
    
    Attributes:
        query: Search query or key
        memory_type: Type of memory to search
        limit: Maximum results to return
    """
    
    query: str = ""
    memory_type: str = "all"
    limit: int = 10


@dataclass(frozen=True)
class MemoryQueryResultEvent(BaseEvent):
    """
    Event containing memory query results.
    
    Attributes:
        results: List of matching memory items
        query: Original query
        total_matches: Total number of matches (may be more than returned)
    """
    
    results: List[Dict[str, Any]] = field(default_factory=list)
    query: str = ""
    total_matches: int = 0
    source: str = field(default="memory_agent")


@dataclass(frozen=True)
class ConversationContextEvent(BaseEvent):
    """
    Event containing conversation context for other agents.
    
    Provides recent conversation history for context-aware responses.
    """
    
    messages: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""
    session_id: UUID = field(default_factory=uuid4)
    source: str = field(default="memory_agent")


# =============================================================================
# Agent Lifecycle Events
# =============================================================================

@dataclass(frozen=True)
class AgentStartedEvent(BaseEvent):
    """Event emitted when an agent starts."""
    
    agent_name: str = ""
    agent_type: str = ""
    capabilities: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentStoppedEvent(BaseEvent):
    """Event emitted when an agent stops."""
    
    agent_name: str = ""
    reason: str = ""
    clean_shutdown: bool = True


@dataclass(frozen=True)
class AgentErrorEvent(BaseEvent):
    """Event emitted when an agent encounters an error."""
    
    agent_name: str = ""
    error_type: str = ""
    error_message: str = ""
    is_recoverable: bool = True
    priority: EventPriority = field(default=EventPriority.HIGH)


@dataclass(frozen=True)
class AgentHealthCheckEvent(BaseEvent):
    """
    Event for agent health monitoring.
    
    TODO: Add detailed health metrics
    """
    
    agent_name: str = ""
    is_healthy: bool = True
    last_activity: Optional[datetime] = None
    pending_tasks: int = 0


# =============================================================================
# Orchestrator Events
# =============================================================================

@dataclass(frozen=True)
class TaskAssignedEvent(BaseEvent):
    """
    Event emitted when a task is assigned to an agent.
    
    Used by the orchestrator to delegate work.
    """
    
    task_id: UUID = field(default_factory=uuid4)
    target_agent: str = ""
    task_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    source: str = field(default="orchestrator")


@dataclass(frozen=True)
class TaskCompletedEvent(BaseEvent):
    """Event emitted when a task is completed."""
    
    task_id: UUID = field(default_factory=uuid4)
    success: bool = False
    result: Any = None
    error: Optional[str] = None


@dataclass(frozen=True)
class ShutdownRequestedEvent(BaseEvent):
    """
    Event to request system shutdown.
    
    All agents should cleanup when receiving this event.
    """
    
    reason: str = ""
    force: bool = False
    priority: EventPriority = field(default=EventPriority.CRITICAL)
    source: str = field(default="orchestrator")


# =============================================================================
# Response Events - For agent responses to user
# =============================================================================

@dataclass(frozen=True)
class ResponseGeneratedEvent(BaseEvent):
    """
    Event containing a generated response for the user.
    
    Attributes:
        text: The response text
        should_speak: Whether to speak the response
        display_text: Optional formatted text for display (may differ from spoken)
        actions: List of follow-up actions available
    """
    
    text: str = ""
    should_speak: bool = True
    display_text: Optional[str] = None
    actions: List[Dict[str, Any]] = field(default_factory=list)
    source: str = field(default="orchestrator")


@dataclass(frozen=True)
class HUDUpdateEvent(BaseEvent):
    """
    Event for HUD-specific updates such as screenshot thumbnails.
    """

    image_path: str = ""
    status_text: str = ""
    source: str = field(default="orchestrator")


@dataclass(frozen=True)
class HUDSearchResultsEvent(BaseEvent):
    """
    Event for HUD search panel updates.

    Attributes:
        query: Search query string
        summary: Natural-language summary for the query
        sources: Top sources as list of {"title": str, "url": str}
    """

    query: str = ""
    summary: str = ""
    sources: List[Dict[str, str]] = field(default_factory=list)
    source: str = field(default="WebSearchAgent")


@dataclass(frozen=True)
class HUDGraphStateEvent(BaseEvent):
    """
    Event for live reasoning graph progress in HUD.

    Attributes:
        current_node: Active node name
        plan_steps: Planned steps with checkmark prefixes
        tool_results: Collected tool outputs so far
    """

    current_node: str = ""
    plan_steps: List[str] = field(default_factory=list)
    tool_results: List[str] = field(default_factory=list)
    source: str = field(default="ReasoningEngine")


# =============================================================================
# Event Registry - For dynamic event handling
# =============================================================================

EVENT_REGISTRY: Dict[str, type] = {
    "VoiceInputEvent": VoiceInputEvent,
    "VoiceOutputEvent": VoiceOutputEvent,
    "WakeWordDetectedEvent": WakeWordDetectedEvent,
    "ListeningStateChangedEvent": ListeningStateChangedEvent,
    "IntentRecognizedEvent": IntentRecognizedEvent,
    "IntentUnknownEvent": IntentUnknownEvent,
    "MultiIntentEvent": MultiIntentEvent,
    "ParsedIntent": ParsedIntent,
    "SystemCommandEvent": SystemCommandEvent,
    "MacOSCommandEvent": MacOSCommandEvent,
    "SystemCommandResultEvent": SystemCommandResultEvent,
    "ApplicationLaunchedEvent": ApplicationLaunchedEvent,
    "SystemNotificationEvent": SystemNotificationEvent,
    # Vision events
    "VisionStartEvent": VisionStartEvent,
    "VisionStopEvent": VisionStopEvent,
    "GestureDetectedEvent": GestureDetectedEvent,
    "FaceDetectedEvent": FaceDetectedEvent,
    "FaceLostEvent": FaceLostEvent,
    "VisionCommandEvent": VisionCommandEvent,
    "VisionCommandResultEvent": VisionCommandResultEvent,
    "ScreenshotEvent": ScreenshotEvent,
    "PresenceChangedEvent": PresenceChangedEvent,
    # Memory events
    "MemoryStoreEvent": MemoryStoreEvent,
    "MemoryQueryEvent": MemoryQueryEvent,
    "MemoryQueryResultEvent": MemoryQueryResultEvent,
    "ConversationContextEvent": ConversationContextEvent,
    "AgentStartedEvent": AgentStartedEvent,
    "AgentStoppedEvent": AgentStoppedEvent,
    "AgentErrorEvent": AgentErrorEvent,
    "AgentHealthCheckEvent": AgentHealthCheckEvent,
    "TaskAssignedEvent": TaskAssignedEvent,
    "TaskCompletedEvent": TaskCompletedEvent,
    "ShutdownRequestedEvent": ShutdownRequestedEvent,
    "ResponseGeneratedEvent": ResponseGeneratedEvent,
    "HUDUpdateEvent": HUDUpdateEvent,
    "HUDSearchResultsEvent": HUDSearchResultsEvent,
    "HUDGraphStateEvent": HUDGraphStateEvent,
    # Orchestrator events
    "ActionRequestEvent": ActionRequestEvent,
    "ActionResultEvent": ActionResultEvent,
    "PlanCreatedEvent": PlanCreatedEvent,
    "ToolCallStartedEvent": ToolCallStartedEvent,
    "ToolCallFinishedEvent": ToolCallFinishedEvent,
    "PlanStepCompletedEvent": PlanStepCompletedEvent,
    "PlanCompletedEvent": PlanCompletedEvent,
    "ContextUpdatedEvent": ContextUpdatedEvent,
    "ClarificationNeededEvent": ClarificationNeededEvent,
    # Sensor & proactive events
    "AppFocusChangedEvent": AppFocusChangedEvent,
    "UpcomingMeetingEvent": UpcomingMeetingEvent,
    "ObservationEvent": ObservationEvent,
    "BriefingRequestedEvent": BriefingRequestedEvent,
}


def get_event_class(event_name: str) -> Optional[type]:
    """
    Get event class by name.
    
    Useful for deserializing events from storage/network.
    """
    return EVENT_REGISTRY.get(event_name)
