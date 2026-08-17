"""
Tests for llm/token_meter.py — the estimation and rolling-window pacing
that keep planning calls under Groq's 8k tokens/minute free-tier ceiling.
"""

from __future__ import annotations

import time

from llm.token_meter import TokenBudget, estimate_request_tokens, estimate_tokens


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_handles_empty_and_none():
    assert estimate_tokens(None) == 0
    assert estimate_tokens("") == 0
    assert estimate_tokens([]) == 0


def test_estimate_tokens_grows_with_content():
    small = estimate_tokens("hello")
    large = estimate_tokens("hello " * 500)
    assert 0 < small < large


def test_estimate_tokens_accepts_structured_payloads():
    """Messages and tool schemas are lists of dicts, not strings."""
    messages = [{"role": "user", "content": "what time is it"}]
    assert estimate_tokens(messages) > 0


def test_estimate_request_tokens_sums_messages_and_tools():
    messages = [{"role": "user", "content": "open safari"}]
    tools = [{"type": "function", "function": {"name": "open_app", "description": "Open an app."}}]

    combined = estimate_request_tokens(messages, tools)
    assert combined == estimate_tokens(messages) + estimate_tokens(tools)
    assert combined > estimate_request_tokens(messages, None)


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------


def test_budget_disabled_when_no_limit_configured():
    budget = TokenBudget(tokens_per_minute=0)
    assert not budget.enabled
    budget.record(5000)
    assert budget.delay_for(5000) == 0.0


def test_budget_allows_calls_under_the_ceiling():
    budget = TokenBudget(tokens_per_minute=8000)
    budget.record(1000)
    assert budget.delay_for(1000) == 0.0


def test_budget_delays_a_call_that_would_cross_the_ceiling():
    """The whole point: pace instead of firing into a guaranteed 429."""
    budget = TokenBudget(tokens_per_minute=8000)  # effective limit 7200 at 0.9 headroom
    budget.record(7000)

    delay = budget.delay_for(2000)
    assert delay > 0
    assert delay <= budget.max_wait_seconds


def test_budget_delay_is_capped_at_max_wait():
    budget = TokenBudget(tokens_per_minute=8000, max_wait_seconds=4.0)
    budget.record(7200)
    assert budget.delay_for(5000) <= 4.0


def test_budget_never_stalls_on_a_request_bigger_than_the_whole_ceiling():
    """
    A single call larger than the limit can never fit no matter how long we
    wait, so waiting is pure latency. Send it and let the provider rule.
    """
    budget = TokenBudget(tokens_per_minute=8000)
    assert budget.delay_for(50_000) == 0.0


def test_budget_forgets_spend_outside_the_window():
    budget = TokenBudget(tokens_per_minute=8000, window_seconds=60.0)
    # Backdate a big spend to just outside the rolling window.
    budget._spend.append((time.monotonic() - 61.0, 8000))

    assert budget.used() == 0
    assert budget.delay_for(4000) == 0.0


def test_budget_counts_only_spend_inside_the_window():
    budget = TokenBudget(tokens_per_minute=8000, window_seconds=60.0)
    budget._spend.append((time.monotonic() - 61.0, 5000))  # expired
    budget.record(1000)  # current

    assert budget.used() == 1000


def test_headroom_keeps_us_below_the_advertised_limit():
    """Estimation is approximate, so we deliberately aim under the real line."""
    budget = TokenBudget(tokens_per_minute=8000, headroom=0.9)
    assert budget.effective_limit == 7200
