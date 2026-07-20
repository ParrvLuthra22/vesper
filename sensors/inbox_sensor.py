"""
InboxSensor — polls the Gmail unread count and publishes an
ObservationEvent(kind="inbox_surge") when a lot of new mail arrives at
once (self-contained: threshold and cooldown are enforced here, not by a
separate ProactiveEngine rule).

Reuses the "list_unread" tool the Gmail MCP bridge registers (see
tools/mcp_bridge.py) rather than talking to Gmail directly — if the
Gmail server isn't connected (disabled, or still mid-OAuth-consent),
list_unread simply isn't in the registry yet and this sensor no-ops.

list_unread is sampled up to UNREAD_SAMPLE_SIZE per poll as a count
proxy (Gmail's own tool doesn't expose a separate lightweight count
endpoint here) — fine for surge detection, since we only care about a
jump of a few messages, not an exact count above that ceiling.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from schemas.events import ObservationEvent
from sensors.base_sensor import BaseSensor
from tools.registry import ToolRegistry, get_registry

DEFAULT_SURGE_THRESHOLD = 5
DEFAULT_SURGE_COOLDOWN_MINUTES = 60.0
UNREAD_SAMPLE_SIZE = 50

Clock = Callable[[], datetime]


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


class InboxSensor(BaseSensor):
    """Polls unread count; flags a burst of new mail as an observation."""

    config_key = "sensors.inbox"
    default_poll_interval_seconds = 600.0  # 10 minutes

    def __init__(
        self,
        event_bus=None,
        config=None,
        registry: Optional[ToolRegistry] = None,
        clock: Optional[Clock] = None,
    ):
        super().__init__(event_bus=event_bus, config=config)
        self._registry = registry or get_registry()
        self._clock: Clock = clock or _default_clock
        self._last_count: Optional[int] = None
        self._last_surge_at: Optional[datetime] = None

    @property
    def surge_threshold(self) -> int:
        return int(self._get_config(f"{self.config_key}.surge_threshold", DEFAULT_SURGE_THRESHOLD))

    @property
    def surge_cooldown_minutes(self) -> float:
        return float(self._get_config(f"{self.config_key}.surge_cooldown_min", DEFAULT_SURGE_COOLDOWN_MINUTES))

    async def poll(self) -> None:
        count = await self._fetch_unread_count()
        if count is None:
            return

        previous = self._last_count
        self._last_count = count
        if previous is None:
            return  # first observation — nothing to compare against yet

        jump = count - previous
        if jump < self.surge_threshold:
            return

        now = self._clock()
        if self._last_surge_at is not None and (now - self._last_surge_at) < timedelta(minutes=self.surge_cooldown_minutes):
            return

        self._last_surge_at = now
        await self._event_bus.emit(
            ObservationEvent(
                kind="inbox_surge",
                detail=(
                    f"Sir, {jump} new emails have arrived since I last checked — "
                    f"you now have at least {count} unread."
                ),
                source="InboxSensor",
            )
        )

    async def _fetch_unread_count(self) -> Optional[int]:
        tool_spec = self._registry.get("list_unread")
        if tool_spec is None or tool_spec.handler is None:
            return None
        try:
            raw = await tool_spec.handler({"max_n": UNREAD_SAMPLE_SIZE}, {})
            messages = json.loads(raw)
        except Exception as exc:
            self._logger.debug(f"InboxSensor: list_unread failed (non-fatal): {exc}")
            return None
        return len(messages)
