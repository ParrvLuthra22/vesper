#!/usr/bin/env python3
"""
Live smoke test for P07's scheduled-briefing VERIFY scenario: starts a real
ProactiveEngine with proactive.schedule.morning_briefing.time overridden to
fire ~2 minutes from now (rather than waiting for a real 08:30), and
confirms BriefingRequestedEvent -> deliver_scheduled_briefing -> a rendered
VoiceOutputEvent-equivalent actually happens end to end.

Usage:
    python scripts/smoke_scheduled_briefing.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
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
from proactive.briefing import deliver_scheduled_briefing
from proactive.engine import ProactiveEngine
from schemas.events import BriefingRequestedEvent
from sensors.calendar_sensor import CalendarSensor
from agents.memory_agent import MemoryAgent
from tools.mcp_bridge import MCPBridge
from tools.registry import ToolRegistry
from tracing.tracer import Tracer

FIRE_IN_SECONDS = 90


async def main() -> int:
    EventBus.reset_instance()
    bus = EventBus()

    fire_at = datetime.now() + timedelta(seconds=FIRE_IN_SECONDS)
    time_str = fire_at.strftime("%H:%M")
    print(f"Scheduling morning_briefing for {time_str} local time (~{FIRE_IN_SECONDS}s from now)...")

    config = load_config_dict()
    config["mcp"]["servers"]["gmail"]["enabled"] = True
    config.setdefault("proactive", {}).setdefault("schedule", {})["morning_briefing"] = {
        "enabled": True,
        "time": time_str,
    }

    registry = ToolRegistry()
    guardian = Guardian(event_bus=bus)
    tracer = Tracer(config=config)
    router = ModelRouter(config=config)

    bridge = MCPBridge(config=config, registry=registry)
    await bridge.start()

    calendar_sensor = CalendarSensor(event_bus=bus, config=config)
    memory_agent = MemoryAgent(event_bus=bus, config=config)
    await memory_agent.start()

    planner = Planner(router=router, registry=registry, guardian=guardian, event_bus=bus, tracer=tracer)

    fired = asyncio.Event()
    delivered_text = None

    async def on_briefing_requested(event: BriefingRequestedEvent) -> None:
        nonlocal delivered_text
        print(f"BriefingRequestedEvent received (schedule_name={event.schedule_name!r}) -- delivering...")
        result = await deliver_scheduled_briefing(
            planner=planner, registry=registry, calendar_sensor=calendar_sensor, memory_agent=memory_agent,
        )
        delivered_text = result.text
        fired.set()

    bus.subscribe(BriefingRequestedEvent, on_briefing_requested)

    engine = ProactiveEngine(event_bus=bus, config=config)
    await engine.start()

    try:
        await asyncio.wait_for(fired.wait(), timeout=FIRE_IN_SECONDS + 30)
        print(f"\nVESPER (scheduled briefing): {delivered_text}\n")
        print("Scheduled briefing fired successfully.")
    except asyncio.TimeoutError:
        print("TIMEOUT: scheduled briefing did not fire in time.")
        return 1
    finally:
        await engine.stop()
        await memory_agent.stop()
        await bridge.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
