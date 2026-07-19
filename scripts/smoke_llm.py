#!/usr/bin/env python3
"""
Smoke test for llm/router.py: makes one real Groq call and one real Ollama
call and prints the normalized LLMResponse for each.

Requires:
    - GROQ_API_KEY set in the environment or .env
    - A local Ollama server running (`ollama serve`) with the configured
      fallback model pulled (see config/settings.yaml -> llm.fallback.model,
      `ollama pull <model>`)

Usage:
    python scripts/smoke_llm.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

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
from llm.types import LLMResponse, RouterError

MESSAGES = [{"role": "user", "content": "In one short sentence, who are you?"}]


def _print_result(label: str, result: "LLMResponse | RouterError") -> None:
    print(f"\n--- {label} ---")
    if isinstance(result, RouterError):
        print(f"ROUTER ERROR: {result.message}")
        print(f"user_message: {result.user_message!r}")
        return

    print(f"provider:   {result.provider}")
    print(f"model:      {result.model}")
    print(f"latency_ms: {result.latency_ms:.1f}")
    print(f"usage:      {result.usage}")
    print(f"tool_calls: {result.tool_calls}")
    print(f"text:       {result.text}")


async def main() -> int:
    config = load_config_dict(None)
    llm_cfg = config.get("llm", {})
    primary_tier = llm_cfg.get("primary", {})
    fallback_tier = llm_cfg.get("fallback", {})

    # Route directly at each provider (primary == fallback) so a real failure
    # on one backend can't be silently masked by falling over to the other.
    groq_router = ModelRouter(
        config={"llm": {**llm_cfg, "primary": primary_tier, "fallback": primary_tier}}
    )
    ollama_router = ModelRouter(
        config={"llm": {**llm_cfg, "primary": fallback_tier, "fallback": fallback_tier}}
    )

    print(f"Calling Groq ({primary_tier.get('model')}) ...")
    groq_result = await groq_router.complete(messages=MESSAGES, purpose="planning")
    _print_result("Groq (real call)", groq_result)

    print(f"\nCalling Ollama ({fallback_tier.get('model')}) ...")
    ollama_result = await ollama_router.complete(messages=MESSAGES, purpose="planning")
    _print_result("Ollama (real call)", ollama_result)

    failures = [r for r in (groq_result, ollama_result) if isinstance(r, RouterError)]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
