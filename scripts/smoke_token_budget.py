#!/usr/bin/env python3
"""
Measure what PF3's prompt trimming actually saved, against real Groq calls.

Two passes over the same commands — filtering off (the old behavior) then on
— reporting the provider's own `usage.prompt_tokens` for the first planning
call of each. That first call is the one that matters: it carries the full
persona + tool schema, and it is re-paid on every iteration of a turn.

Deliberately stops after one completion per command and never executes a
tool, so running this has no side effects on the machine.

Requires GROQ_API_KEY (environment or .env).

Usage:
    .venv/bin/python scripts/smoke_token_budget.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from config.settings import load_config_dict
from llm.router import ModelRouter
from llm.types import RouterError
from orchestrator.planner import Planner
from tools.registry import get_registry
import tools.builtin  # noqa: F401  (registers the builtin tools)

COMMANDS: List[str] = [
    "what time is it",
    "how much battery do I have left",
    "what's the weather like",
    "give me the git status",
    "set the volume to 40",
    "take a screenshot",
    "what apps are running",
    "search the web for the M3 memory bandwidth",
]


async def _first_call_tokens(planner: Planner, router: ModelRouter, text: str) -> Tuple[int, int]:
    """
    Send exactly the prompt the planner would send for `text`, and report
    (prompt_tokens, tool_count) from the provider's own usage.
    """
    system_prompt = planner._render_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    schema, _ = planner._select_tools_for(text)

    response = await router.complete(messages=messages, tools=schema, purpose="planning")
    if isinstance(response, RouterError):
        return -1, len(schema)
    return int(response.usage.get("prompt_tokens", 0)), len(schema)


async def _run_pass(label: str, filtering: bool, config: dict) -> List[int]:
    config = {**config, "llm": {**config["llm"], "tool_selection": {"enabled": filtering}}}
    router = ModelRouter(config=config)
    planner = Planner(router=router, registry=get_registry(), config=config)

    print(f"\n{'=' * 72}")
    print(f"{label}  (llm.tool_selection.enabled = {filtering})")
    print("=" * 72)

    totals: List[int] = []
    for text in COMMANDS:
        tokens, tool_count = await _first_call_tokens(planner, router, text)
        if tokens < 0:
            print(f"  {'ERR':>6}                    | {text}")
            continue
        totals.append(tokens)
        print(f"  {tokens:>6} tokens_in  {tool_count:>2} tools | {text}")
        # Space the calls so this measurement doesn't itself trip the limit.
        await asyncio.sleep(1.0)

    if totals:
        print(f"\n  average tokens_in: {sum(totals) / len(totals):.0f}")
    return totals


async def main() -> int:
    config = load_config_dict()

    before = await _run_pass("BEFORE — full catalog every call", False, config)

    # The BEFORE pass deliberately spends most of an 8k minute. Without a
    # full window drain the AFTER pass starts already throttled and measures
    # contention rather than its own cost.
    print("\n  draining the rolling 60s rate-limit window before the second pass...")
    await asyncio.sleep(62)

    after = await _run_pass("AFTER  — per-turn tool filtering", True, config)

    if not before or not after:
        print("\nNo usable samples (check GROQ_API_KEY).")
        return 1

    avg_before = sum(before) / len(before)
    avg_after = sum(after) / len(after)
    print(f"\n{'=' * 72}")
    print("RESULT")
    print("=" * 72)
    print(f"  average tokens_in before : {avg_before:>7.0f}")
    print(f"  average tokens_in after  : {avg_after:>7.0f}")
    print(f"  reduction                : {100 * (1 - avg_after / avg_before):>6.1f}%")
    print(f"  calls per minute under an 8k ceiling: "
          f"{8000 / avg_before:.1f} -> {8000 / avg_after:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
