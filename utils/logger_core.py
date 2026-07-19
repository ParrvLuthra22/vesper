"""Required logging setup for JARVIS runtime.

This module provides the simplified, requirement-driven logger contract while
preserving compatibility helpers consumed by existing code.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Deque, Dict, Optional


_LOGGER_CONFIGURED = False
_LOGGER_LOCK = threading.RLock()
_LOG_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("jarvis_log_context", default={})


class AgentNameFormatter(logging.Formatter):
    """Formatter that injects a concise agent/logger name."""

    def format(self, record: logging.LogRecord) -> str:
        name = getattr(record, "agent_name", "") or record.name
        record.agent_name = str(name).split(".")[-1]
        return super().format(record)


class ContextFilter(logging.Filter):
    """Inject context fields into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _LOG_CONTEXT.get() or {}
        for key, value in ctx.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


def configure_logging(
    level: str = "INFO",
    *,
    log_dir: str = "logs",
    log_file: str = "jarvis.log",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> None:
    """Configure application-wide logging handlers."""
    global _LOGGER_CONFIGURED

    with _LOGGER_LOCK:
        root = logging.getLogger()

        for handler in list(root.handlers):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

        formatter_console = AgentNameFormatter(
            "[%(asctime)s] %(levelname)s %(agent_name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        formatter_file = AgentNameFormatter(
            "%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        context_filter = ContextFilter()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, str(console_level).upper(), logging.INFO))
        console_handler.setFormatter(formatter_console)
        console_handler.addFilter(context_filter)

        logs_path = Path(log_dir)
        logs_path.mkdir(parents=True, exist_ok=True)
        file_path = logs_path / log_file

        file_handler = RotatingFileHandler(
            filename=str(file_path),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, str(file_level).upper(), logging.DEBUG))
        file_handler.setFormatter(formatter_file)
        file_handler.addFilter(context_filter)

        root.addHandler(console_handler)
        root.addHandler(file_handler)
        _LOGGER_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _LOGGER_CONFIGURED:
        configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
    return logging.getLogger(name)


def set_log_context(**kwargs: Any) -> None:
    current = dict(_LOG_CONTEXT.get() or {})
    current.update(kwargs)
    _LOG_CONTEXT.set(current)


def clear_log_context() -> None:
    _LOG_CONTEXT.set({})


@dataclass
class _TraceRecord:
    timestamp: str
    phase: str
    event_type: str
    event_id: str
    actor: str
    duration_ms: float = 0.0
    error: str = ""


class EventTracer:
    """Lightweight in-memory event tracer used by event bus."""

    _instance: Optional["EventTracer"] = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        self.enabled: bool = True
        self._records: Deque[_TraceRecord] = deque(maxlen=5000)
        self._records_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "EventTracer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = EventTracer()
            return cls._instance

    def _append(self, record: _TraceRecord) -> None:
        if not self.enabled:
            return
        with self._records_lock:
            self._records.append(record)

    def record_emit(self, event_type: str, event_id: str, source: str) -> None:
        self._append(
            _TraceRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                phase="emit",
                event_type=event_type,
                event_id=event_id,
                actor=source,
            )
        )

    def record_receive(self, event_type: str, event_id: str, handler_name: str) -> None:
        self._append(
            _TraceRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                phase="receive",
                event_type=event_type,
                event_id=event_id,
                actor=handler_name,
            )
        )

    def record_handle(
        self,
        event_type: str,
        event_id: str,
        handler_name: str,
        duration_ms: float,
        error: str = "",
    ) -> None:
        self._append(
            _TraceRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                phase="handle",
                event_type=event_type,
                event_id=event_id,
                actor=handler_name,
                duration_ms=duration_ms,
                error=error,
            )
        )

    def dump_text(self, limit: int = 200) -> str:
        with self._records_lock:
            rows = list(self._records)[-limit:]
        return "\n".join(
            f"{r.timestamp} | {r.phase} | {r.event_type} | {r.event_id} | {r.actor} | {r.duration_ms:.2f}ms | {r.error}"
            for r in rows
        )


class AgentLogger:
    def __init__(self, agent_name: str):
        self._logger = get_logger(f"agents.{agent_name}")

    def debug(self, message: str) -> None:
        self._logger.debug(message)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str, exc_info: bool = False) -> None:
        self._logger.error(message, exc_info=exc_info)

    def event_received(self, event_type: str, event_id: str, source: str) -> None:
        self._logger.debug(f"EVENT_RECEIVED type={event_type} id={event_id} source={source}")

    def event_handled(
        self,
        event_type: str,
        event_id: str,
        duration_ms: float,
        success: bool = True,
        error: str = "",
    ) -> None:
        level = logging.INFO if success else logging.ERROR
        self._logger.log(
            level,
            f"EVENT_HANDLED type={event_type} id={event_id} duration={duration_ms:.2f}ms success={success} error={error}",
        )

    def event_failed(self, event_type: str, event_id: str, error: str) -> None:
        self._logger.error(f"EVENT_FAILED type={event_type} id={event_id} error={error}")


def get_agent_logger(agent_name: str) -> AgentLogger:
    return AgentLogger(agent_name)


def init_from_config(config: Dict[str, Any]) -> None:
    level = str(config.get("log_level", os.environ.get("LOG_LEVEL", "INFO"))).upper()
    log_file_path = Path(str(config.get("log_file", "logs/jarvis.log")))

    configure_logging(
        level=level,
        log_dir=str(log_file_path.parent) if str(log_file_path.parent) else "logs",
        log_file=log_file_path.name,
        console_level="INFO",
        file_level="DEBUG",
    )

    EventTracer.get_instance().enabled = bool(config.get("event_tracing", True))
