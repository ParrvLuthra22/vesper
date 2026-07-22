"""
Logger Utility Module - Centralized logging configuration with observability.

This module provides a production-ready logging system for the entire application.
It supports file and console output, log rotation, structured JSON logging,
per-agent context, and event tracing for observability.

Features:
    - Colored console output for human readability
    - JSON structured logging for machine parsing (ELK, Loki, Splunk)
    - File-based logging with rotation
    - Per-agent logger with automatic context
    - Event trace logging with correlation IDs
    - Performance metrics logging
    - Context-aware logging (correlation IDs, agent names)
    - Debug mode with verbose output

Usage:
    from utils.logger import get_logger, get_agent_logger
    
    # Standard logger
    logger = get_logger(__name__)
    logger.info("Application started")
    
    # Agent-specific logger with context
    agent_logger = get_agent_logger("VoiceAgent")
    agent_logger.info("Listening for wake word")
    
    # Event trace logging
    agent_logger.event("VoiceInputEvent", event_id="abc-123", payload={"text": "hello"})

Configuration via settings.yaml:
    general:
      log_level: INFO
      debug_mode: false
      json_logs: false
      event_tracing: true
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, TypeVar, Union
from uuid import uuid4

# Optional: colorama for Windows color support
try:
    import colorama
    colorama.init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


# =============================================================================
# Type Definitions
# =============================================================================

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Log Level Extensions
# =============================================================================

# Add custom log levels for observability
TRACE = 5  # More detailed than DEBUG
EVENT = 25  # Between INFO and WARNING - for event tracing

logging.addLevelName(TRACE, "TRACE")
logging.addLevelName(EVENT, "EVENT")


def trace(self, message: str, *args, **kwargs) -> None:
    """Log a TRACE level message (more detailed than DEBUG)."""
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


def event(self, message: str, *args, **kwargs) -> None:
    """Log an EVENT level message for event tracing."""
    if self.isEnabledFor(EVENT):
        self._log(EVENT, message, args, **kwargs)


# Add custom methods to Logger class
logging.Logger.trace = trace
logging.Logger.event = event


# =============================================================================
# Color Codes for Console Output
# =============================================================================

class LogColors:
    """ANSI color codes for log levels."""
    
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Log level colors
    TRACE = "\033[90m"      # Dark gray
    DEBUG = "\033[36m"      # Cyan
    INFO = "\033[32m"       # Green
    EVENT = "\033[34m"      # Blue
    WARNING = "\033[33m"    # Yellow
    ERROR = "\033[31m"      # Red
    CRITICAL = "\033[41m"   # Red background
    
    # Component colors
    TIMESTAMP = "\033[90m"  # Gray
    NAME = "\033[35m"       # Magenta
    AGENT = "\033[96m"      # Light cyan
    EVENT_ID = "\033[94m"   # Light blue
    MESSAGE = "\033[37m"    # White
    CORRELATION = "\033[93m"  # Light yellow


# =============================================================================
# JSON Structured Formatter
# =============================================================================

class JSONFormatter(logging.Formatter):
    """
    JSON structured formatter for machine-readable logs.
    
    Outputs each log record as a single JSON line for easy parsing
    by log aggregation systems (ELK, Loki, Splunk, CloudWatch).
    
    Output format:
        {"timestamp": "...", "level": "INFO", "logger": "...", "message": "...", ...}
    """
    
    # Fields to exclude from extra data
    RESERVED_FIELDS = {
        'name', 'msg', 'args', 'created', 'filename', 'funcName',
        'levelname', 'levelno', 'lineno', 'module', 'msecs',
        'pathname', 'process', 'processName', 'relativeCreated',
        'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
        'message', 'asctime',
    }
    
    def __init__(self, include_extra: bool = True, indent: Optional[int] = None):
        """
        Initialize JSON formatter.
        
        Args:
            include_extra: Include extra fields from log record
            indent: JSON indent level (None for compact, 2 for pretty)
        """
        super().__init__()
        self.include_extra = include_extra
        self.indent = indent
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        # Build base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add thread info
        log_entry["thread"] = {
            "id": record.thread,
            "name": record.threadName,
        }
        
        # Add process info
        log_entry["process"] = {
            "id": record.process,
            "name": record.processName,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }
        
        # Add extra fields from record
        if self.include_extra:
            extra = {}
            for key, value in record.__dict__.items():
                if key not in self.RESERVED_FIELDS:
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        extra[key] = value
                    except (TypeError, ValueError):
                        extra[key] = str(value)
            
            if extra:
                log_entry["extra"] = extra
        
        return json.dumps(log_entry, default=str, indent=self.indent)


# =============================================================================
# Enhanced Colored Formatter
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log output.
    
    Only applies colors when outputting to a TTY (terminal).
    Supports custom log levels (TRACE, EVENT) and agent context.
    """
    
    LEVEL_COLORS = {
        TRACE: LogColors.TRACE,
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        EVENT: LogColors.EVENT,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }
    
    def __init__(
        self, 
        fmt: str, 
        datefmt: Optional[str] = None, 
        use_colors: bool = True,
        show_agent: bool = True,
        show_correlation: bool = True,
    ):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()
        self.show_agent = show_agent
        self.show_correlation = show_correlation
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors and context."""
        # Store original values
        original_levelname = record.levelname
        original_name = record.name
        
        if self.use_colors:
            # Add color to level name
            level_color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
            record.levelname = f"{level_color}{record.levelname:8}{LogColors.RESET}"
            
            # Add color to logger name
            record.name = f"{LogColors.NAME}{record.name}{LogColors.RESET}"
        
        # Format the base message
        formatted = super().format(record)
        
        # Add agent context if present
        if self.show_agent and hasattr(record, 'agent_name') and record.agent_name:
            agent_str = f"[{record.agent_name}]"
            if self.use_colors:
                agent_str = f"{LogColors.AGENT}{agent_str}{LogColors.RESET}"
            formatted = formatted.replace(record.getMessage(), f"{agent_str} {record.getMessage()}", 1)
        
        # Add correlation ID if present
        if self.show_correlation and hasattr(record, 'correlation_id') and record.correlation_id != '-':
            corr_str = f"[corr:{record.correlation_id[:8]}]"
            if self.use_colors:
                corr_str = f"{LogColors.CORRELATION}{corr_str}{LogColors.RESET}"
            formatted = f"{formatted} {corr_str}"
        
        # Restore original values
        record.levelname = original_levelname
        record.name = original_name
        
        return formatted


class ContextFilter(logging.Filter):
    """
    Filter that adds context information to log records.
    
    Adds correlation_id, agent_name, and other context for request tracking.
    Thread-safe for concurrent operations.
    """
    
    def __init__(self):
        super().__init__()
        self._context: Dict[str, Any] = {}
        self._lock = threading.RLock()
        # Thread-local storage for per-request context
        self._thread_local = threading.local()
    
    def set_context(self, **kwargs) -> None:
        """Set global context values that will be added to all log records."""
        with self._lock:
            self._context.update(kwargs)
    
    def set_thread_context(self, **kwargs) -> None:
        """Set thread-local context (for per-request tracking)."""
        if not hasattr(self._thread_local, 'context'):
            self._thread_local.context = {}
        self._thread_local.context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all global context values."""
        with self._lock:
            self._context.clear()
    
    def clear_thread_context(self) -> None:
        """Clear thread-local context."""
        if hasattr(self._thread_local, 'context'):
            self._thread_local.context.clear()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to the log record."""
        # Add global context
        with self._lock:
            for key, value in self._context.items():
                if not hasattr(record, key):
                    setattr(record, key, value)
        
        # Add thread-local context (overrides global)
        if hasattr(self._thread_local, 'context'):
            for key, value in self._thread_local.context.items():
                setattr(record, key, value)
        
        # Set defaults for standard fields
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = '-'
        if not hasattr(record, 'agent_name'):
            record.agent_name = ''
        if not hasattr(record, 'event_type'):
            record.event_type = ''
        if not hasattr(record, 'event_id'):
            record.event_id = ''
        
        return True


# =============================================================================
# Logger Configuration
# =============================================================================

# Global context filter for correlation IDs
_context_filter = ContextFilter()

# Track configured loggers
_configured_loggers: set = set()

# Default format
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    rotation: str = "daily",
    max_size_mb: int = 10,
    backup_count: int = 7,
    json_format: bool = False,
) -> None:
    """
    Configure the root logger and all handlers.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE)
        log_file: Path to log file (None = console only)
        log_format: Log message format (ignored if json_format=True)
        date_format: Date format string
        rotation: Rotation strategy ('daily', 'size', 'none')
        max_size_mb: Max file size before rotation (for size rotation)
        backup_count: Number of backup files to keep
        json_format: Whether to use JSON structured logging for file output
    """
    log_format = log_format or DEFAULT_FORMAT
    date_format = date_format or DEFAULT_DATE_FORMAT
    
    # Handle custom log levels
    log_level = level.upper()
    if log_level == "TRACE":
        numeric_level = TRACE
    elif log_level == "EVENT":
        numeric_level = EVENT
    else:
        numeric_level = getattr(logging, log_level, logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add context filter
    root_logger.addFilter(_context_filter)
    
    # Console handler - always human-readable with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = ColoredFormatter(
        log_format, 
        date_format, 
        use_colors=True,
        show_agent=True,
        show_correlation=True,
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        _add_file_handler(
            root_logger,
            log_file,
            log_format,
            date_format,
            rotation,
            max_size_mb,
            backup_count,
            json_format,
        )
    
    # Mark as configured
    _configured_loggers.add('root')


def _add_file_handler(
    logger: logging.Logger,
    log_file: str,
    log_format: str,
    date_format: str,
    rotation: str,
    max_size_mb: int,
    backup_count: int,
    json_format: bool = False,
) -> None:
    """
    Add a file handler to the logger.
    
    Args:
        logger: Logger to add handler to
        log_file: Path to log file
        log_format: Format string (used if json_format=False)
        date_format: Date format string
        rotation: Rotation strategy ('daily', 'size', 'none')
        max_size_mb: Max file size for rotation
        backup_count: Number of backup files
        json_format: Use JSON formatter for machine-readable logs
    """
    # Create log directory if needed
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create appropriate handler based on rotation strategy
    if rotation == "daily":
        file_handler = TimedRotatingFileHandler(
            log_file,
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding="utf-8",
        )
    elif rotation == "size":
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
    else:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
    
    file_handler.setLevel(logging.DEBUG)
    
    # Choose formatter based on json_format flag
    if json_format:
        file_formatter = JSONFormatter(include_extra=True)
    else:
        file_formatter = logging.Formatter(log_format, date_format)
    
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Also create a JSON log file for machine parsing if not already JSON
    if not json_format:
        json_log_file = str(log_path.with_suffix('.json.log'))
        json_handler = RotatingFileHandler(
            json_log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter(include_extra=True))
        logger.addHandler(json_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for a module.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured Logger instance
    
    Example:
        logger = get_logger(__name__)
        logger.info("Module initialized")
    """
    # Auto-configure if not done yet
    if 'root' not in _configured_loggers:
        configure_logging()
    
    logger = logging.getLogger(name)
    
    # Add context filter if not already added
    if not any(isinstance(f, ContextFilter) for f in logger.filters):
        logger.addFilter(_context_filter)
    
    return logger


def set_log_context(**kwargs) -> None:
    """
    Set context values that will be added to all log records.
    
    Useful for adding correlation IDs, user IDs, etc.
    
    Args:
        **kwargs: Key-value pairs to add to log context
    
    Example:
        set_log_context(correlation_id="abc-123", user_id="user-456")
    """
    _context_filter.set_context(**kwargs)


def clear_log_context() -> None:
    """Clear all log context values."""
    _context_filter.clear_context()


def set_thread_context(**kwargs) -> None:
    """Set thread-local context for per-request tracking."""
    _context_filter.set_thread_context(**kwargs)


def clear_thread_context() -> None:
    """Clear thread-local context."""
    _context_filter.clear_thread_context()


# =============================================================================
# Agent Logger - Per-Agent Logging with Context
# =============================================================================

class AgentLogger:
    """
    Specialized logger for agents with automatic context injection.
    
    Provides structured logging with agent name, event tracking,
    and performance metrics built-in.
    
    Usage:
        logger = AgentLogger("VoiceAgent")
        logger.info("Listening for wake word")
        logger.event_received("VoiceInputEvent", event_id="abc-123")
        logger.event_emitted("IntentRecognizedEvent", event_id="def-456")
        
        with logger.operation("transcribe_audio") as op:
            # ... do work ...
            op.set_result({"text": "hello"})
    """
    
    def __init__(self, agent_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize agent logger.
        
        Args:
            agent_name: Name of the agent (e.g., "VoiceAgent")
            logger: Optional base logger (creates one if not provided)
        """
        self.agent_name = agent_name
        self._logger = logger or get_logger(f"agents.{agent_name}")
        self._operation_stack: List[str] = []
    
    def _log(self, level: int, message: str, **extra) -> None:
        """Internal log method with agent context."""
        # Add agent name to extra
        extra['agent_name'] = self.agent_name
        
        # Add operation context if in an operation
        if self._operation_stack:
            extra['operation'] = self._operation_stack[-1]
        
        self._logger.log(level, f"[{self.agent_name}] {message}", extra=extra)
    
    def trace(self, message: str, **extra) -> None:
        """Log at TRACE level (more detailed than DEBUG)."""
        self._log(TRACE, message, **extra)
    
    def debug(self, message: str, **extra) -> None:
        """Log at DEBUG level."""
        self._log(logging.DEBUG, message, **extra)
    
    def info(self, message: str, **extra) -> None:
        """Log at INFO level."""
        self._log(logging.INFO, message, **extra)
    
    def warning(self, message: str, **extra) -> None:
        """Log at WARNING level."""
        self._log(logging.WARNING, message, **extra)
    
    def error(self, message: str, **extra) -> None:
        """Log at ERROR level."""
        self._log(logging.ERROR, message, **extra)
    
    def critical(self, message: str, **extra) -> None:
        """Log at CRITICAL level."""
        self._log(logging.CRITICAL, message, **extra)
    
    def exception(self, message: str, exc: Optional[Exception] = None, **extra) -> None:
        """Log an exception with traceback."""
        extra['agent_name'] = self.agent_name
        if exc:
            extra['exception_type'] = type(exc).__name__
            extra['exception_message'] = str(exc)
        self._logger.exception(f"[{self.agent_name}] {message}", extra=extra)
    
    # -------------------------------------------------------------------------
    # Event Tracing
    # -------------------------------------------------------------------------
    
    def event_received(
        self, 
        event_type: str, 
        event_id: str, 
        source: str = "",
        payload_summary: str = "",
    ) -> None:
        """
        Log receipt of an event.
        
        Args:
            event_type: Type of event (e.g., "VoiceInputEvent")
            event_id: Unique event ID
            source: Source agent that emitted the event
            payload_summary: Brief description of payload
        """
        self._log(
            EVENT,
            f"EVENT_RECEIVED | type={event_type} | id={event_id[:8]}... | from={source}",
            event_type=event_type,
            event_id=event_id,
            event_source=source,
            payload_summary=payload_summary[:100] if payload_summary else "",
        )
    
    def event_emitted(
        self, 
        event_type: str, 
        event_id: str,
        payload_summary: str = "",
    ) -> None:
        """
        Log emission of an event.
        
        Args:
            event_type: Type of event (e.g., "IntentRecognizedEvent")
            event_id: Unique event ID
            payload_summary: Brief description of payload
        """
        self._log(
            EVENT,
            f"EVENT_EMITTED | type={event_type} | id={event_id[:8]}...",
            event_type=event_type,
            event_id=event_id,
            payload_summary=payload_summary[:100] if payload_summary else "",
        )
    
    def event_handled(
        self, 
        event_type: str, 
        event_id: str,
        duration_ms: float,
        success: bool = True,
        error: str = "",
    ) -> None:
        """
        Log completion of event handling.
        
        Args:
            event_type: Type of event handled
            event_id: Unique event ID
            duration_ms: Time taken to handle event
            success: Whether handling was successful
            error: Error message if not successful
        """
        status = "OK" if success else "FAILED"
        msg = f"EVENT_HANDLED | type={event_type} | id={event_id[:8]}... | duration={duration_ms:.2f}ms | status={status}"
        if error:
            msg += f" | error={error}"
        
        level = EVENT if success else logging.ERROR
        self._log(
            level,
            msg,
            event_type=event_type,
            event_id=event_id,
            duration_ms=duration_ms,
            success=success,
            error=error,
        )
    
    # -------------------------------------------------------------------------
    # Operation Tracking
    # -------------------------------------------------------------------------
    
    @contextmanager
    def operation(self, name: str):
        """
        Context manager for tracking operations with timing.
        
        Usage:
            with logger.operation("transcribe_audio") as op:
                result = transcribe(audio)
                op.set_result({"length": len(result)})
        
        Args:
            name: Name of the operation
        
        Yields:
            OperationContext for setting result/error
        """
        self._operation_stack.append(name)
        op = OperationContext(name)
        start_time = time.perf_counter()
        
        self.trace(f"OPERATION_START | {name}")
        
        try:
            yield op
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            if op.error:
                self.error(
                    f"OPERATION_FAILED | {name} | duration={duration_ms:.2f}ms | error={op.error}",
                    duration_ms=duration_ms,
                    operation_error=op.error,
                )
            else:
                self.debug(
                    f"OPERATION_COMPLETE | {name} | duration={duration_ms:.2f}ms",
                    duration_ms=duration_ms,
                    operation_result=str(op.result)[:100] if op.result else "",
                )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.exception(
                f"OPERATION_EXCEPTION | {name} | duration={duration_ms:.2f}ms",
                exc=e,
                duration_ms=duration_ms,
            )
            raise
        finally:
            self._operation_stack.pop()
    
    # -------------------------------------------------------------------------
    # State Changes
    # -------------------------------------------------------------------------
    
    def state_change(self, from_state: str, to_state: str, reason: str = "") -> None:
        """
        Log an agent state transition.
        
        Args:
            from_state: Previous state
            to_state: New state
            reason: Optional reason for transition
        """
        msg = f"STATE_CHANGE | {from_state} -> {to_state}"
        if reason:
            msg += f" | reason={reason}"
        
        self.info(msg, from_state=from_state, to_state=to_state, reason=reason)


@dataclass
class OperationContext:
    """Context object for tracking operation results."""
    
    name: str
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def set_result(self, result: Any) -> None:
        """Set successful result."""
        self.result = result
    
    def set_error(self, error: str) -> None:
        """Set error message."""
        self.error = error
    
    def add_metadata(self, **kwargs) -> None:
        """Add metadata to operation context."""
        self.metadata.update(kwargs)


def get_agent_logger(agent_name: str) -> AgentLogger:
    """
    Get an AgentLogger for a specific agent.
    
    Args:
        agent_name: Name of the agent
    
    Returns:
        AgentLogger instance with agent context
    """
    return AgentLogger(agent_name)


# =============================================================================
# Event Tracer - Cross-Agent Event Tracing
# =============================================================================

@dataclass
class EventTrace:
    """Record of a single event in the trace."""
    
    timestamp: datetime
    event_type: str
    event_id: str
    source: str
    handler: str
    action: str  # 'emitted', 'received', 'handled'
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "event_id": self.event_id,
            "source": self.source,
            "handler": self.handler,
            "action": self.action,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class EventTracer:
    """
    Global event tracer for cross-agent event flow visualization.
    
    Maintains a rolling buffer of recent events for debugging
    and can output traces in various formats.
    
    Usage:
        tracer = EventTracer.get_instance()
        tracer.record_emit("VoiceInputEvent", "abc-123", "VoiceAgent")
        tracer.record_receive("VoiceInputEvent", "abc-123", "IntentAgent")
        tracer.record_handle("VoiceInputEvent", "abc-123", "IntentAgent", 45.2)
        
        # Dump recent traces
        for trace in tracer.get_recent(100):
            print(trace)
    """
    
    _instance: Optional["EventTracer"] = None
    _lock = threading.Lock()
    
    MAX_TRACES = 10000
    
    def __init__(self):
        self._traces: Deque[EventTrace] = deque(maxlen=self.MAX_TRACES)
        self._trace_lock = threading.Lock()
        self._enabled = True
        self._logger = get_logger("event_tracer")
    
    @classmethod
    def get_instance(cls) -> "EventTracer":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable tracing."""
        self._enabled = value
    
    def record_emit(self, event_type: str, event_id: str, source: str) -> None:
        """Record an event emission."""
        if not self._enabled:
            return
        
        trace = EventTrace(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            event_id=event_id,
            source=source,
            handler="",
            action="emitted",
        )
        
        with self._trace_lock:
            self._traces.append(trace)
    
    def record_receive(self, event_type: str, event_id: str, handler: str) -> None:
        """Record an event receipt by a handler."""
        if not self._enabled:
            return
        
        trace = EventTrace(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            event_id=event_id,
            source="",
            handler=handler,
            action="received",
        )
        
        with self._trace_lock:
            self._traces.append(trace)
    
    def record_handle(
        self, 
        event_type: str, 
        event_id: str, 
        handler: str,
        duration_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Record completion of event handling."""
        if not self._enabled:
            return
        
        trace = EventTrace(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            event_id=event_id,
            source="",
            handler=handler,
            action="handled",
            duration_ms=duration_ms,
            error=error,
        )
        
        with self._trace_lock:
            self._traces.append(trace)
    
    def get_recent(self, count: int = 100) -> List[EventTrace]:
        """Get most recent traces."""
        with self._trace_lock:
            return list(self._traces)[-count:]
    
    def get_by_event_id(self, event_id: str) -> List[EventTrace]:
        """Get all traces for a specific event ID."""
        with self._trace_lock:
            return [t for t in self._traces if t.event_id == event_id]
    
    def get_by_type(self, event_type: str, count: int = 100) -> List[EventTrace]:
        """Get traces for a specific event type."""
        with self._trace_lock:
            matching = [t for t in self._traces if t.event_type == event_type]
            return matching[-count:]
    
    def clear(self) -> None:
        """Clear all traces."""
        with self._trace_lock:
            self._traces.clear()
    
    def dump_json(self, count: int = 100) -> str:
        """Dump recent traces as JSON."""
        traces = self.get_recent(count)
        return json.dumps([t.to_dict() for t in traces], indent=2)
    
    def dump_text(self, count: int = 100) -> str:
        """Dump recent traces as human-readable text."""
        traces = self.get_recent(count)
        lines = []
        for t in traces:
            ts = t.timestamp.strftime("%H:%M:%S.%f")[:-3]
            if t.action == "emitted":
                line = f"{ts} | EMIT  | {t.event_type:30} | {t.source:15} | id={t.event_id[:8]}"
            elif t.action == "received":
                line = f"{ts} | RECV  | {t.event_type:30} | {t.handler:15} | id={t.event_id[:8]}"
            else:
                status = "OK" if not t.error else f"ERR: {t.error[:20]}"
                line = f"{ts} | DONE  | {t.event_type:30} | {t.handler:15} | {t.duration_ms:.1f}ms | {status}"
            lines.append(line)
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def log_exception(
    logger: logging.Logger,
    message: str,
    exc: Exception,
    level: int = logging.ERROR,
) -> None:
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        message: Context message
        exc: The exception to log
        level: Log level to use
    """
    logger.log(level, f"{message}: {type(exc).__name__}: {exc}", exc_info=True)


def log_performance(
    logger: logging.Logger,
    operation: str,
    duration_ms: float,
    threshold_ms: float = 1000,
) -> None:
    """
    Log operation performance.
    
    Logs as DEBUG normally, WARNING if over threshold.
    
    Args:
        logger: Logger instance
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        threshold_ms: Warning threshold in milliseconds
    """
    level = logging.WARNING if duration_ms > threshold_ms else logging.DEBUG
    logger.log(level, f"Performance: {operation} took {duration_ms:.2f}ms")


def timed(logger: Optional[logging.Logger] = None, threshold_ms: float = 1000):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger to use (uses function's module if not provided)
        threshold_ms: Threshold for warning level
    
    Usage:
        @timed()
        def my_function():
            ...
        
        @timed(threshold_ms=500)
        async def my_async_function():
            ...
    """
    def decorator(func: F) -> F:
        func_logger = logger or get_logger(func.__module__)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                log_performance(func_logger, func.__name__, duration_ms, threshold_ms)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                log_performance(func_logger, func.__name__, duration_ms, threshold_ms)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Debug Mode Utilities
# =============================================================================

_debug_mode = False


def set_debug_mode(enabled: bool) -> None:
    """
    Enable or disable debug mode globally.
    
    Debug mode:
    - Sets log level to DEBUG
    - Enables TRACE level logging
    - Enables verbose event tracing
    - Logs full payloads instead of summaries
    """
    global _debug_mode
    _debug_mode = enabled
    
    if enabled:
        logging.getLogger().setLevel(logging.DEBUG)
        get_logger(__name__).info("Debug mode ENABLED")
    else:
        logging.getLogger().setLevel(logging.INFO)
        get_logger(__name__).info("Debug mode DISABLED")


def is_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return _debug_mode


# =============================================================================
# Module Initialization
# =============================================================================

def init_from_config(config: Dict[str, Any]) -> None:
    """
    Initialize logging from configuration dictionary.
    
    Expected config structure matches settings.yaml 'general' section:
        general:
          log_level: INFO
          log_file: logs/vesper.log
          log_format: "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
          log_rotation: daily
          log_max_size_mb: 10
          log_backup_count: 7
          debug_mode: false
          json_logs: false
          event_tracing: true
    
    Args:
        config: Configuration dictionary
    """
    # Configure base logging
    json_logs = config.get("json_logs", False)
    
    configure_logging(
        level=config.get("log_level", "INFO"),
        log_file=config.get("log_file"),
        log_format=config.get("log_format"),
        rotation=config.get("log_rotation", "daily"),
        max_size_mb=config.get("log_max_size_mb", 10),
        backup_count=config.get("log_backup_count", 7),
        json_format=json_logs,
    )
    
    # Enable debug mode if configured
    if config.get("debug_mode", False):
        set_debug_mode(True)
    
    # Configure event tracing
    event_tracing = config.get("event_tracing", True)
    EventTracer.get_instance().enabled = event_tracing


# Pre-configure with defaults when module is imported
# This ensures logging works even if configure_logging is never called
if 'root' not in _configured_loggers:
    configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

# -----------------------------------------------------------------------------
# Requirement-driven logger override (keeps legacy module path compatible)
# -----------------------------------------------------------------------------
from .logger_core import (  # noqa: E402
    AgentLogger as _CoreAgentLogger,
    EventTracer as _CoreEventTracer,
    clear_log_context as _core_clear_log_context,
    configure_logging as _core_configure_logging,
    get_agent_logger as _core_get_agent_logger,
    get_logger as _core_get_logger,
    init_from_config as _core_init_from_config,
    set_log_context as _core_set_log_context,
)

configure_logging = _core_configure_logging
get_logger = _core_get_logger
get_agent_logger = _core_get_agent_logger
set_log_context = _core_set_log_context
clear_log_context = _core_clear_log_context
EventTracer = _CoreEventTracer
AgentLogger = _CoreAgentLogger
init_from_config = _core_init_from_config
