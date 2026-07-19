"""
Tests for tools/registry.py (ToolSpec, ToolRegistry, @tool decorator) and
tools/builtin.py (the wrapped SystemAgent/WebSearchAgent capability tools).
"""

from __future__ import annotations

import pytest

from tools.registry import ToolRegistry, ToolSpec, tool


async def _noop_handler(arguments, context):
    return arguments


# =============================================================================
# ToolSpec validation
# =============================================================================

def test_toolspec_requires_exactly_one_route():
    with pytest.raises(ValueError):
        ToolSpec(name="bad", description="bad")  # neither handler nor bus route

    with pytest.raises(ValueError):
        ToolSpec(
            name="bad2",
            description="bad2",
            handler=_noop_handler,
            target_agent="SystemAgent",
            action="open_app",
        )  # both


def test_toolspec_invalid_tier_raises():
    with pytest.raises(ValueError):
        ToolSpec(name="x", description="x", tier="yolo", handler=_noop_handler)


def test_bus_routed_tool_has_no_handler():
    spec = ToolSpec(
        name="open_app",
        description="Open an app",
        target_agent="SystemAgent",
        action="open_app",
    )
    assert spec.is_bus_routed
    assert spec.handler is None


def test_direct_handler_tool_is_not_bus_routed():
    spec = ToolSpec(name="echo", description="Echo", handler=_noop_handler)
    assert not spec.is_bus_routed
    assert spec.handler is _noop_handler


# =============================================================================
# ToolRegistry: register / get
# =============================================================================

def test_register_and_get():
    registry = ToolRegistry()
    spec = ToolSpec(name="ping", description="Ping", handler=_noop_handler)
    registry.register(spec)

    assert registry.get("ping") is spec
    assert registry.get("missing") is None


def test_register_duplicate_raises():
    registry = ToolRegistry()
    registry.register(ToolSpec(name="x", description="x", handler=_noop_handler))

    with pytest.raises(ValueError):
        registry.register(ToolSpec(name="x", description="x again", handler=_noop_handler))


# =============================================================================
# ToolRegistry: schema generation
# =============================================================================

def test_to_llm_schema_shape():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="get_time",
            description="Get the time",
            parameters={"type": "object", "properties": {}},
            target_agent="SystemAgent",
            action="get_time",
        )
    )

    assert registry.to_llm_schema() == [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the time",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_to_llm_schema_excludes_disabled():
    registry = ToolRegistry()
    registry.register(ToolSpec(name="a", description="a", target_agent="X", action="a", enabled=True))
    registry.register(ToolSpec(name="b", description="b", target_agent="X", action="b", enabled=False))

    names = {f["function"]["name"] for f in registry.to_llm_schema()}
    assert names == {"a"}


# =============================================================================
# ToolRegistry: tier assignment / listing
# =============================================================================

def test_list_by_tier():
    registry = ToolRegistry()
    registry.register(ToolSpec(name="safe1", description="d", target_agent="X", action="a", tier="safe"))
    registry.register(ToolSpec(name="confirm1", description="d", target_agent="X", action="b", tier="confirm"))
    registry.register(ToolSpec(name="dangerous1", description="d", target_agent="X", action="c", tier="dangerous"))

    assert [t.name for t in registry.list_by_tier("safe")] == ["safe1"]
    assert [t.name for t in registry.list_by_tier("confirm")] == ["confirm1"]
    assert [t.name for t in registry.list_by_tier("dangerous")] == ["dangerous1"]


def test_list_by_tier_rejects_invalid_tier():
    registry = ToolRegistry()
    with pytest.raises(ValueError):
        registry.list_by_tier("not_a_tier")


# =============================================================================
# @tool decorator
# =============================================================================

def test_tool_decorator_registers_into_custom_registry():
    registry = ToolRegistry()

    @tool(name="echo", description="Echo back", tier="safe", registry=registry)
    async def echo(arguments, context):
        return arguments

    spec = registry.get("echo")
    assert spec is not None
    assert spec.handler is echo
    assert spec.tier == "safe"
    assert not spec.is_bus_routed


def test_tool_decorator_defaults_parameters_to_empty_object_schema():
    registry = ToolRegistry()

    @tool(name="noop", description="Does nothing", registry=registry)
    async def noop(arguments, context):
        return None

    assert registry.get("noop").parameters == {"type": "object", "properties": {}}


# =============================================================================
# tools/builtin.py: wrapped SystemAgent/WebSearchAgent capabilities
# =============================================================================

def test_builtin_tools_registered_with_expected_tiers():
    import tools.builtin  # noqa: F401  (registers into the shared default registry)
    from tools.registry import get_registry

    registry = get_registry()

    expected_confirm = {"close_app", "lock_screen"}
    for name in expected_confirm:
        spec = registry.get(name)
        assert spec is not None, f"missing tool: {name}"
        assert spec.tier == "confirm"

    expected_safe = {
        "open_app", "focus_app", "list_apps", "set_volume", "get_volume", "mute",
        "set_brightness", "get_time", "get_date", "get_battery", "system_info",
        "take_screenshot", "show_notification", "open_url", "search_web",
    }
    for name in expected_safe:
        spec = registry.get(name)
        assert spec is not None, f"missing tool: {name}"
        assert spec.tier == "safe"


def test_builtin_search_web_routes_to_web_search_agent():
    # Previously mis-routed to SystemAgent, whose search_web handler just
    # opens a browser URL instead of running the real search+summarize
    # pipeline that lives on WebSearchAgent.
    import tools.builtin  # noqa: F401
    from tools.registry import get_registry

    spec = get_registry().get("search_web")
    assert spec.target_agent == "WebSearchAgent"
    assert spec.action == "search_web"


def test_builtin_tools_are_all_bus_routed():
    import tools.builtin  # noqa: F401
    from tools.registry import get_registry

    builtin_names = {
        "open_app", "close_app", "focus_app", "list_apps", "set_volume", "get_volume",
        "mute", "set_brightness", "get_time", "get_date", "get_battery", "system_info",
        "take_screenshot", "show_notification", "open_url", "search_web", "lock_screen",
    }
    registry = get_registry()
    for name in builtin_names:
        spec = registry.get(name)
        assert spec is not None, f"missing tool: {name}"
        assert spec.is_bus_routed, f"{name} should be bus-routed (no local handler)"
