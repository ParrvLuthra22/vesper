"""
Tests for tools/mcp_bridge.py, against tests/fixtures/fake_mcp_server.py —
a minimal, dependency-free MCP server (no `mcp` SDK needed) used only for
these tests, independent of the real Gmail server / its separate 3.10+
venv.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from tools.mcp_bridge import DEFAULT_UNMAPPED_TIER, MCPBridge
from tools.registry import ToolRegistry

FAKE_SERVER_PATH = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"


def _config(**server_overrides: Any) -> Dict[str, Any]:
    server_cfg = {
        "enabled": True,
        "command": sys.executable,
        "args": [str(FAKE_SERVER_PATH)],
        "tiers": {"echo": "safe", "fail": "confirm"},
        "slow_tools": ["slow"],
    }
    server_cfg.update(server_overrides)
    return {"mcp": {"servers": {"fake": server_cfg}}}


@pytest.mark.asyncio
async def test_bridge_registers_tools_from_fake_server() -> None:
    registry = ToolRegistry()
    bridge = MCPBridge(config=_config(), registry=registry)
    await bridge.start()
    try:
        names = {spec.name for spec in registry.list_all()}
        assert names == {"echo", "fail", "slow"}

        echo_spec = registry.get("echo")
        assert echo_spec.description == "Echo the given text back."
        assert echo_spec.parameters["properties"]["text"]["type"] == "string"
        assert echo_spec.category == "mcp:fake"
    finally:
        await bridge.stop()


@pytest.mark.asyncio
async def test_tier_mapping_applied_per_tool() -> None:
    registry = ToolRegistry()
    bridge = MCPBridge(config=_config(), registry=registry)
    await bridge.start()
    try:
        assert registry.get("echo").tier == "safe"
        assert registry.get("fail").tier == "confirm"
    finally:
        await bridge.stop()


@pytest.mark.asyncio
async def test_unmapped_tool_defaults_to_confirm_tier() -> None:
    """'slow' is deliberately absent from tiers -- must fail safe, not silently allow."""
    registry = ToolRegistry()
    bridge = MCPBridge(config=_config(), registry=registry)
    await bridge.start()
    try:
        assert registry.get("slow").tier == DEFAULT_UNMAPPED_TIER == "confirm"
    finally:
        await bridge.stop()


@pytest.mark.asyncio
async def test_slow_tools_marked_on_the_tool_spec() -> None:
    registry = ToolRegistry()
    bridge = MCPBridge(config=_config(), registry=registry)
    await bridge.start()
    try:
        assert registry.get("slow").slow is True
        assert registry.get("echo").slow is False
    finally:
        await bridge.stop()


@pytest.mark.asyncio
async def test_registered_tool_handler_calls_through_to_server() -> None:
    registry = ToolRegistry()
    bridge = MCPBridge(config=_config(), registry=registry)
    await bridge.start()
    try:
        result = await registry.get("echo").handler({"text": "hello, Sir"}, {})
        assert result == "hello, Sir"
    finally:
        await bridge.stop()


@pytest.mark.asyncio
async def test_tool_error_result_raises() -> None:
    registry = ToolRegistry()
    bridge = MCPBridge(config=_config(), registry=registry)
    await bridge.start()
    try:
        with pytest.raises(RuntimeError, match="deliberate failure"):
            await registry.get("fail").handler({"reason": "deliberate failure"}, {})
    finally:
        await bridge.stop()


@pytest.mark.asyncio
async def test_disabled_server_is_not_started() -> None:
    registry = ToolRegistry()
    bridge = MCPBridge(config=_config(enabled=False), registry=registry)
    await bridge.start()
    try:
        assert registry.list_all() == []
    finally:
        await bridge.stop()


@pytest.mark.asyncio
async def test_bad_command_does_not_raise_and_registers_nothing() -> None:
    """A misconfigured server must not take down the whole bridge/app."""
    registry = ToolRegistry()
    bridge = MCPBridge(config=_config(command="/no/such/executable"), registry=registry)
    await bridge.start()  # must not raise
    assert registry.list_all() == []
    await bridge.stop()
