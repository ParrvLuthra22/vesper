#!/usr/bin/env python3
"""
Live smoke test for the Gmail MCP server + bridge (P07 VERIFY).

Enables mcp.servers.gmail at runtime (checked-in settings.yaml default
stays `enabled: false`) and exercises the registered tools against real
Gmail. First call triggers the Google OAuth consent flow in a browser
(token cached afterward at data/google_token.json).

Requires:
    - GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET set in .env
    - mcp_servers/.venv (Python 3.10+) with mcp_servers/requirements.txt installed

Usage:
    python scripts/smoke_gmail_mcp.py [list|draft]
        list  (default) -> list_unread + search, no writes
        draft           -> also drafts a reply to the newest unread message
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from config.settings import load_config_dict
from tools.mcp_bridge import MCPBridge
from tools.registry import ToolRegistry


async def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "list"

    config = load_config_dict()
    config["mcp"]["servers"]["gmail"]["enabled"] = True

    registry = ToolRegistry()
    bridge = MCPBridge(config=config, registry=registry)

    print("Starting Gmail MCP server (browser will open for OAuth consent on first run)...")
    await bridge.start()
    try:
        names = {spec.name for spec in registry.list_all()}
        print(f"Registered tools: {sorted(names)}\n")
        if not names:
            print("No tools registered -- check server startup logs / OAuth credentials.")
            return 1

        print("--- list_unread(max_n=10) ---")
        raw = await registry.get("list_unread").handler({"max_n": 10}, {})
        messages = json.loads(raw)
        print(f"{len(messages)} unread message(s):")
        for m in messages:
            print(f"  [{m['id']}] {m['sender']!r} -- {m['subject']!r}")
        print()

        print("--- search(query='is:unread') ---")
        raw = await registry.get("search").handler({"query": "is:unread"}, {})
        print(raw)
        print()

        if mode == "draft" and messages:
            newest = messages[0]
            print(f"--- draft_reply(id={newest['id']!r}) ---")
            result = await registry.get("draft_reply").handler(
                {"id": newest["id"], "instruction": "I'll respond by Friday."}, {}
            )
            print(result)
    finally:
        await bridge.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
