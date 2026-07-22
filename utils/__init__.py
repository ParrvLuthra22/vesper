"""
VESPER Virtual Assistant - Utils Package.

This package contains utility modules for logging, configuration, etc.
"""

from utils.logger import (
    get_logger,
    configure_logging,
    set_log_context,
    clear_log_context,
)

__all__ = [
    "get_logger",
    "configure_logging",
    "set_log_context",
    "clear_log_context",
]
