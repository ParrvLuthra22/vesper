"""Category-based tool selection: a typical planner call carries <= 12 tools instead of the whole catalog (45 tools / ~4,500
tokens blew Groq's 8k tokens-per-minute budget). Dangerous / outbound / confirm tools are never added by a category; unclear
requests fall back to the full set."""
from __future__ import annotations

import pytest

import tools  # noqa: F401  (registers the real built-in tools)
from llm.token_meter import estimate_tokens
from tools.registry import ToolRegistry, ToolSpec, get_registry
from tools.selection import categories_for, select_tools, to_schema


async def _noop(a, c):  # pragma: no cover
    return "ok"


DESC = "Does a thing for the user and returns a short result. Use when the user asks for it; arguments are described below."
PARAMS = {"type": "object", "properties": {"query": {"type": "string", "description": "What to look for."},
                                           "limit": {"type": "integer", "description": "Maximum number of items."}}}


def spec(name, category, tier="safe", outbound=False):
    return ToolSpec(name=name, description=DESC, parameters=PARAMS, tier=tier, category=category, handler=_noop, outbound=outbound)


@pytest.fixture
def full() -> ToolRegistry:
    """The real built-in tools + the MCP tools config/settings.yaml enables or documents (names/tiers copied from it)."""
    reg = ToolRegistry()
    for t in get_registry().list_all(enabled_only=True):
        reg.register(t)
    for n, tier in [("list_unread", "safe"), ("get_message", "safe"), ("search", "safe"), ("summarize_thread", "safe"),
                    ("unread_count", "safe"), ("draft_reply", "confirm"), ("archive", "confirm"), ("mark_read", "confirm")]:
        reg.register(spec(f"gmail_{n}", "mcp:gmail", tier))
    reg.register(spec("gmail_create_draft", "mcp:gmail", "draft"))
    for n, tier in [("today_events", "safe"), ("upcoming", "safe"), ("reminders_due", "safe"), ("add_reminder", "confirm"),
                    ("create_event", "confirm")]:
        reg.register(spec(n, "mcp:apple_pim", tier))
    for n, tier, out in [("list_unreads", "safe", False), ("get_mentions", "safe", False), ("search", "safe", False),
                         ("post_message", "confirm", True), ("reply_thread", "confirm", True)]:
        reg.register(spec(f"slack_{n}", "mcp:slack", tier, out))
    for n in ("get_daily_briefing", "plan_my_day", "forget_memory"):
        reg.register(spec(n, "general"))
    return reg


def names(reg, text):
    sel, why = select_tools(reg, text)
    return {t.name for t in sel}, why


@pytest.mark.parametrize("text,cats", [
    ("what's on my calendar", ["calendar"]), ("any meetings tomorrow", ["calendar"]), ("check my email", ["mail"]),
    ("any unread in my inbox", ["mail"]), ("search for rust async", ["web"]), ("what's the weather", ["web"]),
    ("draft a reply to Priya", ["drafts"]), ("any slack dms", ["messages"]), ("open the file in the editor", ["files"]),
    ("turn the volume up", ["system"]), ("tell me a joke", []), ("hello", []),
])
def test_categories_map_correctly(text, cats):
    assert categories_for(text) == cats


@pytest.mark.parametrize("text", ["what's on my calendar today", "any new email in my inbox", "search for the latest on rust",
                                  "draft a reply to Priya", "any slack mentions", "show me the git diff of this repo"])
def test_a_typical_call_carries_12_tools_or_fewer(full, text):
    n, why = names(full, text)
    assert len(n) <= 12 and why.startswith("category:"), (text, why, len(n))
    assert {"get_time", "get_date", "web_search"} <= n                        # the core three always travel


def test_tokens_before_and_after_on_three_typical_requests(full):
    before = estimate_tokens(full.to_llm_schema())
    rows = []
    for text in ("what's on my calendar", "check my email", "search for rust async"):
        sel, why = select_tools(full, text)
        rows.append((text, len(sel), estimate_tokens(to_schema(sel))))
    print(f"\nschema tokens: full set {len(full.list_all())} tools = {before}")
    for r in rows:
        print("  %-26s %2d tools %5d tokens" % r)
    assert all(t < before * 0.45 for _, _, t in rows)
    assert all(n <= 12 for _, n, _ in rows)


@pytest.mark.parametrize("text", ["what's on my calendar", "check my email", "any slack messages", "search for rust",
                                  "draft a reply", "show me the repo diff", "what is the weather"])
def test_dangerous_confirm_and_outbound_tools_are_never_auto_added(full, text):
    n, _ = names(full, text)
    risky = {t.name for t in full.list_all() if t.tier in ("dangerous", "confirm") or t.outbound}
    assert n & risky == set(), (text, n & risky)


def test_a_risky_tool_still_appears_when_the_request_names_it(full):
    n, why = names(full, "close the Notes app")                 # explicit verb -> the old keyword index, not a category
    assert "close_app" in n
    assert "run_shell" not in n


def test_unclear_requests_fall_back_to_the_full_set(full):
    n, why = names(full, "tell me something interesting about octopuses")
    assert why.startswith("full:") and len(n) == len(full.list_all())


def test_a_category_with_no_registered_tools_falls_back(full):
    reg = ToolRegistry()
    for t in full.list_all():
        if t.category != "mcp:gmail" and t.name != "get_daily_briefing":
            reg.register(t)
    n, why = names(reg, "check my email")                       # Gmail not connected: do not send a near-empty set
    assert not why.startswith("category:")


def test_mixed_system_and_category_words_use_the_group_logic(full):
    n, why = names(full, "turn the volume down and check my calendar")
    assert not why.startswith("category:") and "set_volume" in n


def test_category_selection_is_deterministic(full):
    assert names(full, "what's on my calendar") == names(full, "what's on my calendar")
