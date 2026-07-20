#!/usr/bin/env python3
"""
Gmail MCP server — stdio transport, built on the official `mcp` SDK
(FastMCP). Runs under its own Python 3.10+ virtualenv
(mcp_servers/.venv), separate from the main app's Python 3.9
environment; tools/mcp_bridge.py talks to it as a subprocess over
stdin/stdout using the plain MCP wire protocol.

Every tool description below is written for the calling LLM (the
Planner) — verbs, when-to-use, argument semantics — matching the
convention in tools/builtin.py.

Scope: gmail.modify only (see gmail_client.py). draft_reply creates a
Gmail DRAFT and nothing more; there is no send capability in v1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gmail_client  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("gmail")


@mcp.tool()
async def list_unread(max_n: int = 10) -> str:
    """List unread emails, most recent first.

    Use to answer "what's in my inbox" / "any new email" style questions.
    Returns a JSON array of {id, thread_id, sender, subject, snippet, date}.
    """
    messages = await gmail_client.list_unread(max_n=max_n)
    return json.dumps(messages)


@mcp.tool()
async def get_message(id: str) -> str:
    """Get the full content of one email by id: sender, subject, date, and body text.

    Use when you need to read one specific email in full (id comes from
    list_unread or search results).
    """
    message = await gmail_client.get_message(id)
    return json.dumps(message)


@mcp.tool()
async def summarize_thread(id: str) -> str:
    """Fetch every message in an email thread as formatted text (From/Date/Subject/body per message).

    Use for "summarize this email thread" requests. This returns the raw
    thread content for you to summarize in your own reply — it does not
    summarize itself. May take a few seconds for long threads.
    """
    return await gmail_client.thread_text(id)


@mcp.tool()
async def search(query: str) -> str:
    """Search email using Gmail's query syntax, e.g. "from:alice subject:invoice", "has:attachment newer_than:7d".

    Use for "find the email about X" / "emails from Y" requests. Returns
    a JSON array of {id, thread_id, sender, subject, snippet, date}.
    """
    messages = await gmail_client.search(query=query)
    return json.dumps(messages)


@mcp.tool()
async def draft_reply(id: str, instruction: str) -> str:
    """Create a Gmail DRAFT replying to the given message. Never sends anything.

    `instruction` must be the full reply body, composed and ready to
    send (not a paraphrase of what the user asked for) — write it in
    Vesper's voice on the user's behalf. The draft is left in Gmail for
    the user to review and send themselves.
    """
    draft_id = await gmail_client.draft_reply(message_id=id, instruction=instruction)
    return f"Draft created (id={draft_id}). It has not been sent — the user must send it from Gmail."


@mcp.tool()
async def archive(id: str) -> str:
    """Archive an email: remove it from the inbox (it remains in All Mail, not deleted)."""
    await gmail_client.archive(message_id=id)
    return "Archived."


@mcp.tool()
async def mark_read(id: str) -> str:
    """Mark an email as read."""
    await gmail_client.mark_read(message_id=id)
    return "Marked as read."


@mcp.tool()
async def unread_count() -> str:
    """Get just the number of unread emails, cheaply (no per-message detail fetches).

    Use for surge/volume checks where you only need a count, not the
    messages themselves — list_unread is the right tool once you need
    sender/subject/snippet.
    """
    count = await gmail_client.unread_count()
    return str(count)


if __name__ == "__main__":
    mcp.run(transport="stdio")
