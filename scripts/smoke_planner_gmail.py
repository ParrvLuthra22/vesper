#!/usr/bin/env python3
"""
Live, full-stack smoke test for P07's Planner-facing VERIFY scenarios --
drives the real Planner (real Groq router, real Guardian, real Gmail MCP
tools, real calendar sensor, real memory agent) exactly the way Brain.
handle_user_text() does, without needing microphone/voice hardware.

Auto-approves any confirmation request it sees (simulating the user
saying "yes"), so draft_reply's confirm-tier gate is exercised for real.

Usage:
    python scripts/smoke_planner_gmail.py
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

from bus.event_bus import EventBus
from config.settings import load_config_dict
from guardian.gate import Guardian
from llm.router import ModelRouter
from orchestrator.planner import Planner
from proactive.briefing import make_get_daily_briefing_handler
from schemas.events import ConfirmationRequestedEvent, ConfirmationResponseEvent
from sensors.calendar_sensor import CalendarSensor
from agents.memory_agent import MemoryAgent
from tools.mcp_bridge import MCPBridge
from tools.registry import ToolRegistry, ToolSpec
from tracing.tracer import Tracer


async def _auto_approve(bus: EventBus, event: ConfirmationRequestedEvent) -> None:
    print(f"    [confirmation requested] {event.summary} -- auto-approving")
    await bus.emit(ConfirmationResponseEvent(request_id=event.request_id, approved=True, source="smoke_test"))


async def main() -> int:
    EventBus.reset_instance()
    bus = EventBus()

    config = load_config_dict()
    config["mcp"]["servers"]["gmail"]["enabled"] = True

    registry = ToolRegistry()
    guardian = Guardian(event_bus=bus)
    tracer = Tracer(config=config)
    router = ModelRouter(config=config)

    bus.subscribe(ConfirmationRequestedEvent, lambda e: _auto_approve(bus, e))

    bridge = MCPBridge(config=config, registry=registry)
    await bridge.start()

    calendar_sensor = CalendarSensor(event_bus=bus, config=config)
    memory_agent = MemoryAgent(event_bus=bus, config=config)
    await memory_agent.start()

    registry.register(
        ToolSpec(
            name="get_daily_briefing",
            description=(
                "Gather today's briefing data: unread email triage, today's "
                "remaining calendar events, and any carried-over items from "
                "yesterday's briefing. Use when the user asks to be briefed "
                "-- \"brief me\", \"what's my day look like\", a status update."
            ),
            parameters={"type": "object", "properties": {}},
            tier="safe",
            handler=make_get_daily_briefing_handler(
                registry=registry, calendar_sensor=calendar_sensor, memory_agent=memory_agent,
            ),
            category="briefing",
        )
    )

    planner = Planner(router=router, registry=registry, guardian=guardian, event_bus=bus, tracer=tracer)

    try:
        for prompt in [
            "what's in my inbox?",
            "draft a reply to the newest one saying I'll respond by Friday",
            "brief me",
        ]:
            print(f"\n{'=' * 70}\nUSER: {prompt}\n{'=' * 70}")
            result = await planner.run(user_text=prompt)
            print(f"VESPER: {result.text}")
    finally:
        await memory_agent.stop()
        await bridge.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
