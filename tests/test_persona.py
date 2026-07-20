"""
Persona regression tests, driven by tests/persona_cases.yaml.

Two kinds of case:
    - mode: prompt — checked against the system prompt the Planner actually
      builds (real config/persona.md + rendered {context} block), using a
      mocked ModelRouter that just captures the messages it was called
      with. No live model call; always runs.
    - mode: live — checked against a real model reply (real ModelRouter).
      Marked @pytest.mark.integration and skipped unless GROQ_API_KEY is
      configured (via the environment or .env).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest
import yaml

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from bus.event_bus import EventBus
from guardian.gate import Guardian
from llm.router import ModelRouter
from llm.types import LLMResponse
from orchestrator.planner import Planner
from tools.registry import ToolRegistry
from tracing.tracer import Tracer
from config.settings import load_config_dict

CASES_PATH = Path(__file__).parent / "persona_cases.yaml"


def _load_cases(mode: str) -> List[Dict[str, Any]]:
    with CASES_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [case for case in data["cases"] if case["mode"] == mode]


def _check_patterns(text: str, case: Dict[str, Any]) -> None:
    for pattern in case.get("required_patterns", []):
        assert pattern in text, (
            f"[{case['name']}] expected to find {pattern!r} in:\n{text}"
        )
    for pattern in case.get("forbidden_patterns", []):
        assert pattern not in text, (
            f"[{case['name']}] did not expect to find {pattern!r} in:\n{text}"
        )


def _fresh_planner(router: Any) -> Planner:
    EventBus.reset_instance()
    bus = EventBus()
    guardian = Guardian(event_bus=bus)
    registry = ToolRegistry()
    # This mode checks the constructed prompt, not tracing itself -- disable
    # it so the (fast, no-network) default test run never writes to the
    # real data/traces/traces.jsonl.
    tracer = Tracer(config={"tracing": {"enabled": False}})
    return Planner(router=router, registry=registry, guardian=guardian, event_bus=bus, tracer=tracer)


# =============================================================================
# mode: prompt — structural checks (mocked router, no network)
# =============================================================================

@pytest.mark.parametrize("case", _load_cases("prompt"), ids=lambda c: c["name"])
@pytest.mark.asyncio
async def test_persona_prompt_case(case: Dict[str, Any]) -> None:
    router = AsyncMock()
    router.complete = AsyncMock(return_value=LLMResponse(text="ok", tool_calls=[]))

    planner = _fresh_planner(router)

    if case.get("kind") == "greeting":
        await planner.greet()
    else:
        await planner.run(user_text=case.get("input", "hello"))

    system_prompt = router.complete.call_args.kwargs["messages"][0]["content"]
    _check_patterns(system_prompt, case)


# =============================================================================
# mode: live — behavioral checks (real ModelRouter, real Groq call)
# =============================================================================

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="requires GROQ_API_KEY for a live LLM check")
@pytest.mark.parametrize("case", _load_cases("live"), ids=lambda c: c["name"])
@pytest.mark.asyncio
async def test_persona_live_case(case: Dict[str, Any]) -> None:
    EventBus.reset_instance()
    bus = EventBus()
    router = ModelRouter(config=load_config_dict(None))
    guardian = Guardian(event_bus=bus)
    registry = ToolRegistry()
    planner = Planner(router=router, registry=registry, guardian=guardian, event_bus=bus)

    if case.get("kind") == "greeting":
        result = await planner.greet()
    else:
        result = await planner.run(user_text=case["input"])

    _check_patterns(result.text, case)
