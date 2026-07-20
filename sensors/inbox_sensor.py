"""
InboxSensor — stub.

Email-based sensing arrives in P07 (inbox monitoring, triage cues, etc.).
This file gives that work a stable home; the sensor itself isn't
implemented yet. Disabled by default (no `sensors.inbox.enabled` config
key), so BaseSensor.start() no-ops and poll() is never actually invoked.
"""

from __future__ import annotations

from sensors.base_sensor import BaseSensor


class InboxSensor(BaseSensor):
    """Not implemented yet — see P07."""

    config_key = "sensors.inbox"
    default_poll_interval_seconds = 300.0

    async def poll(self) -> None:
        raise NotImplementedError("InboxSensor arrives in P07")
