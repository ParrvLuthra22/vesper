#!/usr/bin/env python3
"""
A minimal, hand-written MCP server used only by tests/test_mcp_bridge.py.

Deliberately dependency-free (stdlib only, no `mcp` SDK) so tests run
under the main app's Python 3.9 venv without needing the separate 3.10+
mcp_servers/.venv — this exercises tools/mcp_bridge.py's wire-protocol
handling against a real subprocess, independent of the Gmail server.

Tools:
    echo(text: str)   -> echoes the text back
    fail(reason: str) -> always returns an MCP tool error
    slow(ms: int)     -> sleeps `ms` milliseconds, then returns "done"
"""

from __future__ import annotations

import asyncio
import json
import sys


async def _handle(message: dict) -> None:
    method = message.get("method")
    msg_id = message.get("id")

    if method == "initialize":
        _write(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "serverInfo": {"name": "fake", "version": "0.0.1"},
                },
            }
        )
    elif method == "notifications/initialized":
        pass  # no response expected
    elif method == "tools/list":
        _write(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo the given text back.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                            },
                        },
                        {
                            "name": "fail",
                            "description": "Always fails, for error-path testing.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"reason": {"type": "string"}},
                            },
                        },
                        {
                            "name": "slow",
                            "description": "Sleeps `ms` milliseconds, then returns 'done'.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"ms": {"type": "integer"}},
                            },
                        },
                    ]
                },
            }
        )
    elif method == "tools/call":
        params = message.get("params", {})
        name = params.get("name")
        arguments = params.get("arguments") or {}

        if name == "echo":
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": arguments.get("text", "")}],
                        "isError": False,
                    },
                }
            )
        elif name == "fail":
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": arguments.get("reason", "failed")}],
                        "isError": True,
                    },
                }
            )
        elif name == "slow":
            await asyncio.sleep(float(arguments.get("ms", 0)) / 1000.0)
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": "done"}], "isError": False},
                }
            )
        else:
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True},
                }
            )


def _write(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


async def main() -> None:
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            message = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        await _handle(message)


if __name__ == "__main__":
    asyncio.run(main())
