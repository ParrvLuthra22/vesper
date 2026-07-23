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

Resilience (P10): each connection is watched for an unexpected process
exit (a crash mid-session, not a clean MCPBridge.stop()). On crash, its
tools are immediately marked unavailable (ToolSpec.enabled = False, so
they drop out of the LLM's schema rather than the model repeatedly
trying and failing) and a bounded exponential-backoff reconnect loop
starts; on success, the same ToolSpecs are simply re-enabled — handlers
resolve the live connection by server name at call time, so nothing
about the registered tool needs to change across a reconnect.
"""

from __future__ import annotations

import asyncio
import json
import random
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

#: Reconnect backoff: 2s, 4s, 8s, 16s, 32s, then give up for the session.
RECONNECT_BASE_DELAY_SECONDS = 2.0
RECONNECT_MAX_DELAY_SECONDS = 32.0
RECONNECT_MAX_ATTEMPTS = 5


class MCPServerConnection:
    """One subprocess connection to an MCP server via stdio JSON-RPC."""

    def __init__(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        on_crash: Optional[Any] = None,
    ):
        self._name = name
        self._command = command
        self._args = args or []
        self._cwd = cwd
        self._request_timeout_seconds = request_timeout_seconds
        #: Called (no args, may be a coroutine function) if the subprocess
        #: exits unexpectedly — never on an explicit stop().
        self._on_crash = on_crash

        self._process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None
        self._stopping = False
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
        self._watchdog_task = asyncio.create_task(self._watch_process())

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
        self._stopping = True
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
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
        self._fail_pending(RuntimeError(f"MCP server '{self._name}' stopped"))

    def _fail_pending(self, exc: BaseException) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(exc)
        self._pending.clear()

    async def _watch_process(self) -> None:
        """Detect an unexpected subprocess exit (a crash) and notify the bridge."""
        if self._process is None:
            return
        try:
            await self._process.wait()
        except asyncio.CancelledError:
            return
        if self._stopping:
            return  # expected — MCPBridge.stop() already handled cleanup

        logger.warning(
            f"[MCPBridge:{self._name}] subprocess exited unexpectedly "
            f"(code={self._process.returncode})"
        )
        self._fail_pending(RuntimeError(f"MCP server '{self._name}' crashed"))
        if self._on_crash is not None:
            try:
                await self._on_crash()
            except Exception as exc:
                logger.error(f"[MCPBridge:{self._name}] on_crash handler failed: {exc}")

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
        self._server_configs: Dict[str, Dict[str, Any]] = {}
        self._reconnect_tasks: Dict[str, asyncio.Task] = {}
        self._stopping = False

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
            self._server_configs[server_name] = server_cfg
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
            on_crash=lambda sn=server_name: self._handle_crash(sn),
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
        # Optional allowlist: when set, ONLY these tools are exposed to the
        # planner — the rest of the server's tools (e.g. merge/close/force-push)
        # are never registered. Empty/absent means "expose everything".
        expose = server_cfg.get("expose")
        expose_set = set(expose) if expose else None

        for tool_schema in connection.tools:
            tool_name = tool_schema["name"]
            if expose_set is not None and tool_name not in expose_set:
                logger.debug(f"[MCPBridge] '{server_name}.{tool_name}' not in expose list; skipping")
                continue
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
                    handler=self._make_handler(server_name, tool_name),
                    category=f"mcp:{server_name}",
                    slow=tool_name in slow_tools,
                )
            )

    def _make_handler(self, server_name: str, tool_name: str):
        """
        Resolves the live connection by server name on every call (rather
        than closing over the connection object at registration time) so
        a reconnect can swap in a new MCPServerConnection without ever
        needing to touch the ToolRegistry entry itself.
        """

        async def handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
            connection = self._connections.get(server_name)
            if connection is None:
                raise RuntimeError(f"MCP server '{server_name}' is not connected right now")
            return await connection.call_tool(tool_name, arguments)

        return handler

    # =========================================================================
    # Crash recovery
    # =========================================================================

    async def _handle_crash(self, server_name: str) -> None:
        self._connections.pop(server_name, None)
        self._set_tools_enabled(server_name, enabled=False)
        logger.warning(f"[MCPBridge] '{server_name}' tools marked unavailable after crash")

        if self._stopping:
            return
        existing = self._reconnect_tasks.get(server_name)
        if existing is not None and not existing.done():
            return  # a reconnect attempt is already in flight
        self._reconnect_tasks[server_name] = asyncio.create_task(self._reconnect_with_backoff(server_name))

    def _set_tools_enabled(self, server_name: str, enabled: bool) -> None:
        category = f"mcp:{server_name}"
        for spec in self._registry.list_all(enabled_only=False):
            if spec.category == category:
                spec.enabled = enabled

    async def _reconnect_with_backoff(self, server_name: str) -> None:
        server_cfg = self._server_configs.get(server_name)
        if server_cfg is None:
            return

        delay = RECONNECT_BASE_DELAY_SECONDS
        for attempt in range(1, RECONNECT_MAX_ATTEMPTS + 1):
            # Full jitter avoids every crashed server (e.g. after a shared
            # dependency outage) hammering reconnects in lockstep.
            await asyncio.sleep(delay * (0.5 + random.random()))
            if self._stopping:
                return

            logger.info(f"[MCPBridge] reconnecting '{server_name}' (attempt {attempt}/{RECONNECT_MAX_ATTEMPTS})")
            cwd = server_cfg.get("cwd") or str(PROJECT_ROOT)
            connection = MCPServerConnection(
                name=server_name,
                command=server_cfg["command"],
                args=server_cfg.get("args", []),
                cwd=cwd,
                on_crash=lambda sn=server_name: self._handle_crash(sn),
            )
            try:
                await connection.start()
            except Exception as exc:
                logger.warning(f"[MCPBridge] reconnect attempt {attempt} for '{server_name}' failed: {exc}")
                delay = min(delay * 2, RECONNECT_MAX_DELAY_SECONDS)
                continue

            self._connections[server_name] = connection
            self._set_tools_enabled(server_name, enabled=True)
            logger.info(f"[MCPBridge] '{server_name}' reconnected successfully")
            return

        logger.error(
            f"[MCPBridge] '{server_name}' did not reconnect after {RECONNECT_MAX_ATTEMPTS} attempts; "
            "its tools remain unavailable for the rest of this session"
        )

    async def stop(self) -> None:
        self._stopping = True
        for task in self._reconnect_tasks.values():
            task.cancel()
        self._reconnect_tasks.clear()
        for connection in self._connections.values():
            await connection.stop()
        self._connections.clear()
