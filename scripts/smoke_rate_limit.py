#!/usr/bin/env python3
"""
Verify that rapid multi-step turns COMPLETE on Groq rather than failing or
falling to the local model.

Fires four turns back to back, each making three planning calls — the call
pattern of a real multi-step request, and previously the exact shape that
walked into Groq's 8k tokens/minute ceiling and 429'd. Reports which
provider each call landed on, so a turn that survived via pacing or a
retry-after wait is visible as such.

Uses the real ModelRouter and the real trimmed prompt, but never executes a
tool, so running it has no side effects on the machine.

Requires GROQ_API_KEY (environment or .env).

Usage:
    .venv/bin/python scripts/smoke_rate_limit.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import List

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
import tools.builtin  # noqa: F401

TURNS: List[str] = [
    "what time is it and how much battery do I have",
    "give me the git status and then the time",
    "what apps are running, and set the volume to 40",
    "take a screenshot and tell me the date",
]

CALLS_PER_TURN = 3


async def main() -> int:
    config = load_config_dict()
    router = ModelRouter(config=config)
    planner = Planner(router=router, registry=get_registry(), config=config)

    notices: List[str] = []

    print("=" * 78)
    print("4 rapid multi-step turns x 3 planning calls — 8k TPM ceiling")
    print("=" * 78)

    completed = 0
    on_groq = 0
    on_local = 0
    failed = 0
    started = time.monotonic()

    for turn_number, text in enumerate(TURNS, start=1):
        print(f"\nTurn {turn_number}: {text!r}")
        schema, _ = planner._select_tools_for(text)
        system_prompt = planner._render_system_prompt()

        turn_ok = True
        for call in range(1, CALLS_PER_TURN + 1):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
            call_started = time.monotonic()
            response = await router.complete(
                messages=messages,
                tools=schema,
                purpose="planning",
                on_status=notices.append,
            )
            elapsed = time.monotonic() - call_started

            if isinstance(response, RouterError):
                turn_ok = False
                failed += 1
                print(f"  call {call}: FAILED  ({elapsed:.1f}s) {response.primary_error[:90]}")
                continue

            if response.provider == "groq":
                on_groq += 1
            else:
                on_local += 1

            print(
                f"  call {call}: {response.provider:<7} "
                f"tokens_in={response.usage.get('prompt_tokens', 0):<5} "
                f"({elapsed:.1f}s)"
            )

        if turn_ok:
            completed += 1

    total = time.monotonic() - started
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  turns completed      : {completed}/{len(TURNS)}")
    print(f"  calls on groq        : {on_groq}")
    print(f"  calls on local       : {on_local}")
    print(f"  calls failed         : {failed}")
    print(f"  wall clock           : {total:.1f}s")
    if notices:
        print(f"  user-facing notices  : {notices}")
    else:
        print("  user-facing notices  : none (no wait was long enough to mention)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
