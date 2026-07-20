#!/usr/bin/env python3
"""
Apple PIM MCP server — Calendar + Reminders, stdio transport, built on
the official `mcp` SDK (FastMCP). Runs under its own Python 3.10+
virtualenv (mcp_servers/.venv), separate from the main app's Python
3.9 environment; tools/mcp_bridge.py talks to it as a subprocess over
stdin/stdout using the plain MCP wire protocol.

Every tool description below is written for the calling LLM (the
Planner) — verbs, when-to-use, argument semantics — matching the
convention in tools/builtin.py.

Reads (today_events, upcoming, reminders_due) are safe tier. Writes
(add_reminder, create_event) are confirm tier — see eventkit_client.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eventkit_client  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("apple_pim")


@mcp.tool()
async def today_events() -> str:
    """List today's remaining calendar events (from now until midnight).

    Use for "what's on my calendar today" / day-planning requests.
    Returns a JSON array of {title, start, end, calendar, notes, all_day}.
    """
    events = await eventkit_client.today_events()
    return json.dumps(events)


@mcp.tool()
async def upcoming(days: int = 7) -> str:
    """List calendar events over the next `days` days (default 7).

    Use for "what do I have coming up" / "what's on this week" requests.
    Returns a JSON array of {title, start, end, calendar, notes, all_day}.
    """
    events = await eventkit_client.upcoming(days=days)
    return json.dumps(events)


@mcp.tool()
async def reminders_due(scope: str = "today") -> str:
    """List incomplete reminders.

    `scope` is one of: "today" (due today or already overdue),
    "overdue" (past due only), "no_date" (no due date set), or "all".
    Returns a JSON array of {id, title, due, notes, list}.
    """
    reminders = await eventkit_client.reminders_due(scope=scope)
    return json.dumps(reminders)


@mcp.tool()
async def add_reminder(title: str, due: Optional[str] = None) -> str:
    """Create a new reminder.

    `due`, if given, is an ISO-8601 datetime string (e.g.
    "2026-07-22T09:00:00"). Omit it for a reminder with no due date.
    """
    result = await eventkit_client.add_reminder(title=title, due=due)
    return json.dumps(result)


@mcp.tool()
async def create_event(
    title: str,
    start: str,
    end: str,
    calendar_name: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """Create a new calendar event.

    `start`/`end` are ISO-8601 datetime strings (e.g.
    "2026-07-22T09:00:00"). `calendar_name`, if given, must match an
    existing calendar's name exactly; omit it to use the default
    calendar for new events.
    """
    result = await eventkit_client.create_event(
        title=title, start=start, end=end, calendar_name=calendar_name, notes=notes,
    )
    return json.dumps(result)


if __name__ == "__main__":
    mcp.run(transport="stdio")
