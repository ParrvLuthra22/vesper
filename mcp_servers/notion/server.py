#!/usr/bin/env python3
"""
Notion MCP server — stdio transport, built on the official `mcp` SDK
(FastMCP). Runs under its own Python 3.10+ virtualenv (mcp_servers/.venv),
separate from the main app's Python 3.9 environment; tools/mcp_bridge.py
talks to it as a subprocess over stdin/stdout using the plain MCP wire
protocol.

Read-only in v1: notion_search, notion_get_page, notion_get_database_rows.
All safe tier — see notion_client.py for why (no create/update/delete
capability at all). Tools are prefixed "notion_" since Gmail already
has its own "search" tool — the shared ToolRegistry is keyed by name
across every MCP server, so names must be globally unique.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import notion_client  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("notion")

_configured_database_names = notion_client.configured_database_names()
_database_names_note = (
    f" Configured databases you can name directly: {', '.join(_configured_database_names)}."
    if _configured_database_names
    else " No named databases are configured yet — pass a raw database id."
)


@mcp.tool()
async def notion_search(query: str) -> str:
    """Search Notion for pages and databases matching `query` by title.

    Use for "find the Notion page about X" / "is there a doc on Y"
    requests. Returns a JSON array of {id, title, url, last_edited_time, archived}.
    """
    results = await notion_client.search(query=query)
    return json.dumps(results)


@mcp.tool()
async def notion_get_page(id: str) -> str:
    """Get one Notion page's title and properties by id.

    Use when you already have a page id (from notion_search or a
    database row) and need its full property values.
    """
    result = await notion_client.get_page(page_id=id)
    return json.dumps(result)


@mcp.tool(
    description=(
        "Query rows from a Notion database. `db_id` may be a configured "
        "friendly name or a raw database id." + _database_names_note
        + " `filter`, if given, is a Notion API filter object "
        '(e.g. {"property": "Status", "status": {"equals": "Done"}}); '
        "omit it to get every row. Returns a JSON array of {id, url, properties}."
    )
)
async def notion_get_database_rows(db_id: str, filter: Optional[Dict[str, Any]] = None) -> str:
    results = await notion_client.get_database_rows(db_id=db_id, filter=filter)
    return json.dumps(results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
