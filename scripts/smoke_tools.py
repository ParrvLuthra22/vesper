#!/usr/bin/env python3
"""
Smoke test for the tool registry: imports tools.builtin (registering all
built-in bus-routed capability tools), then prints every registered tool's
name, tier, and route, followed by the full OpenAI-style LLM tool schema.

Usage:
    python scripts/smoke_tools.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.builtin  # noqa: F401  (side effect: registers built-in tools)
from tools.registry import get_registry


def main() -> int:
    registry = get_registry()
    all_tools = registry.list_all()

    print(f"Registered tools: {len(all_tools)}\n")
    for spec in sorted(all_tools, key=lambda t: (t.tier, t.name)):
        route = f"handler:{spec.handler.__name__}" if spec.handler else f"{spec.target_agent}.{spec.action}"
        print(
            f"  [{spec.tier:9s}] {spec.name:20s} -> {route:35s} "
            f"(category={spec.category}, slow={spec.slow})"
        )

    print("\n--- LLM tool schema (OpenAI-style) ---")
    print(json.dumps(registry.to_llm_schema(), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
