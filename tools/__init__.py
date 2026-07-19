"""
Tools Module — the registry the planner (LLM router) plans over.

Importing this package registers all built-in bus-routed capability tools
(tools.builtin) as a side effect, so `get_registry()` is populated as soon
as `tools` is imported.
"""

from tools.registry import ToolRegistry, ToolSpec, get_registry, tool

import tools.builtin  # noqa: F401  (side effect: registers built-in tools)

__all__ = [
    "ToolSpec",
    "ToolRegistry",
    "tool",
    "get_registry",
]
