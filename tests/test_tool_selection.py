"""
Tests for tools/selection.py — per-turn tool-schema filtering.

The filter's job is to cut the largest fixed cost in every planning prompt
without ever costing correctness, so these cover both halves: that it
actually trims, and that it declines to trim when it cannot tell what is
relevant.
"""

from __future__ import annotations

import pytest

from llm.token_meter import estimate_tokens
from tools.registry import ToolRegistry, ToolSpec
from tools.selection import CORE_TOOL_NAMES, select_tools, to_schema


async def _noop(arguments, context):  # pragma: no cover - never executed
    return "ok"


def _spec(name: str, category: str, description: str = "A tool.") -> ToolSpec:
    return ToolSpec(name=name, description=description, category=category, handler=_noop)


@pytest.fixture
def registry() -> ToolRegistry:
    """A miniature stand-in for the real registry, spanning several groups."""
    reg = ToolRegistry()
    for name, category in [
        ("open_app", "system"),
        ("close_app", "system"),
        ("focus_app", "system"),
        ("list_apps", "system"),
        ("set_volume", "system"),
        ("mute", "system"),
        ("set_brightness", "system"),
        ("get_time", "system"),
        ("get_date", "system"),
        ("git_commit", "dev"),
        ("git_status", "dev"),
        ("run_tests", "dev"),
        ("search_web", "web"),
        ("open_url", "web"),
        ("current_weather", "general"),
    ]:
        reg.register(_spec(name, category))
    return reg


def _names(registry: ToolRegistry, text: str):
    selected, reason = select_tools(registry, text)
    return {spec.name for spec in selected}, reason


# ---------------------------------------------------------------------------
# It selects the relevant group
# ---------------------------------------------------------------------------


def test_selects_the_matching_group_and_drops_the_rest(registry):
    names, reason = _names(registry, "commit this with a sensible message")

    assert reason == "filtered"
    assert {"git_commit", "git_status"} <= names
    # Unrelated groups are gone — that's where the tokens come back.
    assert "set_volume" not in names
    assert "current_weather" not in names


def test_a_match_promotes_its_whole_group(registry):
    """"commit" must bring git_status/git_diff along; a commit needs context."""
    names, _ = _names(registry, "commit this")
    assert {"git_commit", "git_status", "run_tests"} <= names


def test_groups_are_finer_than_categories(registry):
    """
    Volume and brightness are both category "system", but must not travel
    together — expanding the whole 15-tool system category gave back most
    of the savings.
    """
    names, _ = _names(registry, "turn the volume down")
    assert "set_volume" in names
    assert "set_brightness" not in names
    assert "open_app" not in names


def test_weather_request_selects_almost_nothing(registry):
    names, _ = _names(registry, "what's the weather like")
    assert "current_weather" in names
    assert "git_commit" not in names
    assert "open_app" not in names


# ---------------------------------------------------------------------------
# Core tools and the safety rules
# ---------------------------------------------------------------------------


def test_core_tools_are_always_included(registry):
    """The clock and the search escape hatch ride along on every turn."""
    names, _ = _names(registry, "commit this")
    assert CORE_TOOL_NAMES <= names


def test_unmatchable_text_sends_the_full_catalog(registry):
    """
    A vague turn is exactly the turn where guessing is most likely to be
    wrong, so it is never the turn we economize on.
    """
    names, reason = _names(registry, "tell me a joke")
    assert reason == "full:no-match"
    assert names == {spec.name for spec in registry.list_all()}


def test_empty_text_sends_the_full_catalog(registry):
    names, reason = _names(registry, "")
    assert reason == "full:no-text"
    assert names == {spec.name for spec in registry.list_all()}


def test_disabled_tools_are_never_selected(registry):
    registry.register(
        ToolSpec(name="git_push", description="Push.", category="dev", handler=_noop, enabled=False)
    )
    names, _ = _names(registry, "commit this")
    assert "git_push" not in names


# ---------------------------------------------------------------------------
# It actually saves tokens
# ---------------------------------------------------------------------------


def test_filtering_meaningfully_reduces_the_schema(registry):
    full = estimate_tokens(registry.to_llm_schema())
    selected, _ = select_tools(registry, "what's the weather like")
    trimmed = estimate_tokens(to_schema(selected))

    assert trimmed < full * 0.5


def test_plurals_still_match(registry):
    """"apps" has to reach an "app" trigger."""
    names, reason = _names(registry, "what apps are running")
    assert reason == "filtered"
    assert "list_apps" in names


def test_mcp_tools_group_by_server_without_hand_maintained_entries(registry):
    """
    MCP-provided tools carry category "mcp:<server>" and no TOOL_GROUPS
    entry, so their index has to come from the name and category alone.
    """
    registry.register(_spec("list_unread", "mcp:gmail"))
    registry.register(_spec("send_email", "mcp:gmail"))

    names, reason = _names(registry, "check my gmail")
    assert reason == "filtered"
    assert {"list_unread", "send_email"} <= names
    assert "set_volume" not in names
