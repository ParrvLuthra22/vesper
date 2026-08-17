"""Single source of truth for event -> wire-JSON serialization.

Every event type the gateway forwards to clients is exactly one entry in
``WIRE_SPEC``: its wire ``type`` string plus an extractor for its payload
fields. The gateway derives BOTH what it subscribes to
(``FORWARDED_EVENT_TYPES``) and how it serializes (``to_wire``) from this one
mapping, so subscription and serialization can never drift. Exposing a new
event type to clients is a one-line addition here.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from schemas.events import (
    BaseEvent,
    BriefingRequestedEvent,
    ConfirmationRequestedEvent,
    LocalModelStateEvent,
    ObservationEvent,
    PlanCreatedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
    VoiceOutputEvent,
    WakeEvent,
)

Extractor = Callable[[Any], Dict[str, Any]]

#: event class -> (wire "type", payload extractor). ONE line per exposed event.
WIRE_SPEC: Dict[Type[BaseEvent], Tuple[str, Extractor]] = {
    VoiceOutputEvent: ("reply", lambda e: {"text": e.text}),
    PlanCreatedEvent: (
        "plan",
        lambda e: {
            "plan_id": str(e.plan_id),
            "description": e.description,
            "total_steps": e.total_steps,
            "steps": e.steps,
        },
    ),
    ToolCallStartedEvent: (
        "tool_started",
        lambda e: {"tool": e.tool_name, "arguments": e.arguments},
    ),
    ToolCallFinishedEvent: (
        "tool_finished",
        lambda e: {
            "tool": e.tool_name,
            "success": e.success,
            "latency_ms": e.latency_ms,
            "result": e.result,
            "error": e.error,
        },
    ),
    ObservationEvent: (
        "observation",
        lambda e: {"kind": e.kind, "detail": e.detail, "observation_id": e.observation_id},
    ),
    ConfirmationRequestedEvent: (
        "confirmation_requested",
        lambda e: {
            "request_id": e.request_id,
            "summary": e.summary,
            "tool_name": e.tool_name,
            "arguments": e.arguments,
        },
    ),
    BriefingRequestedEvent: (
        "briefing_requested",
        lambda e: {"schedule_name": e.schedule_name},
    ),
    WakeEvent: (
        "wake",
        lambda e: {"animate": e.animate},
    ),
    # 8GB guard (PF6): tells memory-hungry clients (voice output) that the
    # local LLM rescue is holding ~2.5GB, so they should queue rather than
    # load alongside it.
    LocalModelStateEvent: (
        "local_model",
        lambda e: {"active": e.active, "model": e.model},
    ),
}

#: The event classes the gateway subscribes to and forwards to clients.
#: Derived from WIRE_SPEC so the two are always in lockstep.
FORWARDED_EVENT_TYPES: List[Type[BaseEvent]] = list(WIRE_SPEC.keys())


def _ts(event: BaseEvent) -> Optional[str]:
    ts = getattr(event, "timestamp", None)
    if ts is None:
        return None
    try:
        return ts.isoformat()
    except AttributeError:
        return str(ts)


def to_wire(event: BaseEvent) -> Optional[Dict[str, Any]]:
    """Serialize a forwarded event to its compact ``{type, ...fields, ts}``
    wire shape, or ``None`` if this event type isn't exposed to clients."""
    spec = WIRE_SPEC.get(type(event))
    if spec is None:
        return None
    wire_type, extract = spec
    return {"type": wire_type, **extract(event), "ts": _ts(event)}
