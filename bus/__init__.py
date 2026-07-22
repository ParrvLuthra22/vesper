"""
VESPER Virtual Assistant - Event Bus Package.

This package provides the event-driven communication infrastructure.
"""

from bus.event_bus import EventBus, SubscriptionToken, get_event_bus

__all__ = [
    "EventBus",
    "SubscriptionToken",
    "get_event_bus",
]
