"""
BaseSensor — minimal async poll-loop base for local-only sensors.

Sensors observe local machine state (frontmost app, calendar, and later:
inbox) and publish typed events onto the shared EventBus. Nothing a
sensor observes ever leaves the machine except as short context lines
handed to the LLM planner (see orchestrator/planner.py's {context} block)
— sensors never call out to any network service themselves.

Each sensor is individually toggleable via its own `<config_key>.enabled`
config flag; start() no-ops if disabled, so callers can unconditionally
start every sensor and let config decide what actually runs.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from bus.event_bus import EventBus, get_event_bus
from utils.logger import get_logger


class BaseSensor(ABC):
    """Async poll-loop base class for local-only sensors."""

    #: Dot-path config key for this sensor's settings, e.g. "sensors.focus".
    config_key: str = ""
    #: Default polling interval in seconds; overridable via
    #: `<config_key>.poll_interval_seconds`.
    default_poll_interval_seconds: float = 5.0

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._event_bus = event_bus or get_event_bus()
        self._config = config or {}
        self._logger = get_logger(f"sensors.{self.__class__.__name__}")
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def _get_config(self, key: str, default: Any = None) -> Any:
        """Dot-path lookup against the full app config."""
        value: Any = self._config
        for part in key.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(part)
            if value is None:
                return default
        return value

    @property
    def enabled(self) -> bool:
        return bool(self._get_config(f"{self.config_key}.enabled", False))

    @property
    def poll_interval_seconds(self) -> float:
        return float(
            self._get_config(f"{self.config_key}.poll_interval_seconds", self.default_poll_interval_seconds)
        )

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start the poll loop, unless disabled by config."""
        if not self.enabled:
            self._logger.info(
                f"{self.__class__.__name__} disabled by config ({self.config_key}.enabled=false)"
            )
            return
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        self._logger.info(f"{self.__class__.__name__} started (poll every {self.poll_interval_seconds:.0f}s)")

    async def stop(self) -> None:
        """Stop the poll loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self.poll()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.error(f"{self.__class__.__name__} poll failed: {exc}", exc_info=True)
            await asyncio.sleep(self.poll_interval_seconds)

    @abstractmethod
    async def poll(self) -> None:
        """One polling iteration. Publish events onto the bus as needed."""
        raise NotImplementedError
