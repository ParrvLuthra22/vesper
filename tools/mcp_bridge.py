"""
mcp_bridge — a generic MCP client that connects to configured MCP
servers at startup, introspects their tools, and auto-registers each one
into the ToolRegistry with a configured permission tier.

This is a hand-written JSON-RPC-2.0-over-stdio client rather than the
official `mcp` SDK: the SDK requires Python 3.10+, and the main app runs
on 3.9 (pyobjc/langgraph/chromadb are pinned against it). MCP servers
themselves (mcp_servers/*/) run the real SDK under their own dedicated
3.10+ virtualenv as separate subprocesses — this bridge only needs to
speak the wire protocol (newline-delimited JSON-RPC 2.0 messages over
stdin/stdout), which has no Python-version requirement of its own.

Adding a new MCP server requires zero new plumbing here: one config
entry under mcp.servers.<name> (command, args, tiers, optional
slow_tools), and MCPBridge.start() connects, lists its tools, and
registers them exactly like Gmail's.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import ToolRegistry, ToolSpec, get_registry
from utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
#: Permission tier applied when a server's config doesn't map a tool name.
#: Fails safe: an unmapped tool still requires confirmation, never silently allowed.
DEFAULT_UNMAPPED_TIER = "confirm"


class MCPServerConnection:
    """One subprocess connection to an MCP server via stdio JSON-RPC."""

    def __init__(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ):
        self._name = name
        self._command = command
        self._args = args or []
        self._cwd = cwd
        self._request_timeout_seconds = request_timeout_seconds

        self._process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._next_id = 1
        self._pending: Dict[int, "asyncio.Future[Dict[str, Any]]"] = {}
        self.tools: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        """Spawn the server subprocess and complete the MCP initialize handshake."""
        self._process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

        await self._request(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "vesper", "version": "0.1.0"},
            },
        )
        await self._notify("notifications/initialized", {})

        result = await self._request("tools/list", {})
        self.tools = result.get("tools", [])

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call one tool and return its text content, or raise on error."""
        result = await self._request("tools/call", {"name": tool_name, "arguments": arguments})
        text = "\n".join(
            item.get("text", "") for item in result.get("content", []) if item.get("type") == "text"
        )
        if result.get("isError"):
            raise RuntimeError(text or f"MCP tool '{tool_name}' failed")
        return text

    async def stop(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
        if self._stderr_task is not None:
            self._stderr_task.cancel()
        if self._process is not None:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.close()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                self._process.kill()
        # Fail any requests still awaiting a response.
        for future in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError(f"MCP server '{self._name}' stopped"))
        self._pending.clear()

    async def _request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        future: "asyncio.Future[Dict[str, Any]]" = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        try:
            response = await asyncio.wait_for(future, timeout=self._request_timeout_seconds)
        finally:
            self._pending.pop(request_id, None)
        if "error" in response:
            raise RuntimeError(f"MCP server '{self._name}' error on {method}: {response['error']}")
        return response.get("result", {})

    async def _notify(self, method: str, params: Dict[str, Any]) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def _write(self, message: Dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError(f"MCP server '{self._name}' is not running")
        line = json.dumps(message) + "\n"
        self._process.stdin.write(line.encode("utf-8"))
        await self._process.stdin.drain()

    async def _read_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                try:
                    message = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    logger.debug(f"[MCPBridge:{self._name}] non-JSON line on stdout: {line!r}")
                    continue
                msg_id = message.get("id")
                if msg_id is not None:
                    future = self._pending.get(msg_id)
                    if future is not None and not future.done():
                        future.set_result(message)
        except asyncio.CancelledError:
            pass

    async def _drain_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                logger.debug(f"[MCPBridge:{self._name} stderr] {line.decode('utf-8', errors='replace').rstrip()}")
        except asyncio.CancelledError:
            pass


class MCPBridge:
    """Connects to configured MCP servers at startup and auto-registers their tools."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, registry: Optional[ToolRegistry] = None):
        self._config = config or {}
        self._registry = registry or get_registry()
        self._connections: Dict[str, MCPServerConnection] = {}

    def _get_config(self, key: str, default: Any = None) -> Any:
        value: Any = self._config
        for part in key.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(part)
            if value is None:
                return default
        return value

    async def start(self) -> None:
        """Connect to every enabled configured server and register its tools."""
        servers_config = self._get_config("mcp.servers", {}) or {}
        for server_name, server_cfg in servers_config.items():
            if not server_cfg.get("enabled", False):
                continue
            await self._start_server(server_name, server_cfg)

    async def _start_server(self, server_name: str, server_cfg: Dict[str, Any]) -> None:
        # A relative cwd (default: the project root) makes relative
        # command/args paths in config resolve the same way regardless of
        # where the main app was launched from.
        cwd = server_cfg.get("cwd") or str(PROJECT_ROOT)
        connection = MCPServerConnection(
            name=server_name,
            command=server_cfg["command"],
            args=server_cfg.get("args", []),
            cwd=cwd,
        )
        try:
            await connection.start()
        except Exception as exc:
            logger.error(f"[MCPBridge] failed to start MCP server '{server_name}': {exc}")
            return

        self._connections[server_name] = connection
        self._register_tools(server_name, connection, server_cfg)
        logger.info(f"[MCPBridge] '{server_name}' connected: {len(connection.tools)} tool(s) registered")

    def _register_tools(self, server_name: str, connection: MCPServerConnection, server_cfg: Dict[str, Any]) -> None:
        tiers = server_cfg.get("tiers", {}) or {}
        slow_tools = set(server_cfg.get("slow_tools", []) or [])

        for tool_schema in connection.tools:
            tool_name = tool_schema["name"]
            tier = tiers.get(tool_name)
            if tier is None:
                logger.warning(
                    f"[MCPBridge] '{server_name}.{tool_name}' has no configured tier; "
                    f"defaulting to {DEFAULT_UNMAPPED_TIER!r}"
                )
                tier = DEFAULT_UNMAPPED_TIER

            self._registry.register(
                ToolSpec(
                    name=tool_name,
                    description=tool_schema.get("description", ""),
                    parameters=tool_schema.get("inputSchema") or {"type": "object", "properties": {}},
                    tier=tier,
                    handler=self._make_handler(connection, tool_name),
                    category=f"mcp:{server_name}",
                    slow=tool_name in slow_tools,
                )
            )

    @staticmethod
    def _make_handler(connection: MCPServerConnection, tool_name: str):
        async def handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
            return await connection.call_tool(tool_name, arguments)

        return handler

    async def stop(self) -> None:
        for connection in self._connections.values():
            await connection.stop()
        self._connections.clear()
