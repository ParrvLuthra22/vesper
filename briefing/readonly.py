"""
The collectors' only way to reach tools — and it can only read.

A collector gets a ReadOnlyTools, not the registry. `call()` refuses anything that
is not on READONLY_TOOLS, not registered, or not `safe` tier, so a typo or a future
config change cannot turn a collector into something that archives, drafts or sends.
(Tools hidden from the planner, like list_inbox, are reachable here by design.)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from tools.registry import ToolRegistry, get_registry

#: Every tool a collector may ever call. All are pure reads.
READONLY_TOOLS = frozenset({
    "list_inbox", "unread_ids", "sent_summary",   # Gmail MCP (briefing extras)
    "upcoming", "today_events",                   # apple_pim MCP (EventKit reads)
})


class ReadOnlyViolation(RuntimeError):
    """A collector tried to call something that is not an allow-listed safe read."""


class ReadOnlyTools:
    def __init__(self, registry: Optional[ToolRegistry] = None):
        self._registry = registry or get_registry()

    def available(self, name: str) -> bool:
        spec = self._registry.get(name)
        return name in READONLY_TOOLS and spec is not None and spec.handler is not None and spec.tier == "safe"

    async def call(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        if name not in READONLY_TOOLS:
            raise ReadOnlyViolation(f"'{name}' is not an allow-listed read-only briefing tool")
        spec = self._registry.get(name)
        if spec is None or spec.handler is None:
            raise ReadOnlyViolation(f"read-only tool '{name}' is not registered (is its MCP server connected?)")
        if spec.tier != "safe":
            raise ReadOnlyViolation(f"'{name}' is tier '{spec.tier}', not 'safe' — refusing to call it from a collector")
        raw = await spec.handler(arguments or {}, {})
        return json.loads(raw) if isinstance(raw, str) else raw
