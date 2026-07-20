"""
proactive/briefing.py — assembles Vesper's morning briefing.

Two invocation paths share the same data assembly (assemble_briefing_text):
    - Scheduled: Brain's BriefingRequestedEvent handler (fired by
      ProactiveEngine's morning_briefing cron job) calls
      deliver_scheduled_briefing() directly — there's no ongoing Planner
      turn to attach to, so it starts a fresh one, exactly like the
      session-start greeting.
    - On demand ("brief me"): the get_daily_briefing tool (registered by
      Brain) is just assemble_briefing_text() wrapped as a ToolSpec
      handler. The user's turn is already mid-Planner-loop, so the tool
      simply returns the raw data as a tool result; the model produces
      the actual structured reply on its next iteration, same as any
      other tool call.

Inbox triage lets the model pick the top 3 by apparent importance itself
— the assembled text hands over enough raw signal (sender/subject/snippet)
for that judgment call, rather than a hand-rolled importance heuristic.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional

from agents.memory_agent import MemoryAgent
from orchestrator.planner import Planner, PlannerResult
from sensors.calendar_sensor import CalendarSensor
from tools.registry import ToolRegistry, get_registry
from utils.logger import get_logger

logger = get_logger(__name__)

BRIEFING_MEMORY_CATEGORY = "briefing"
BRIEFING_MEMORY_KEY = "last_top_unread_ids"
INBOX_TRIAGE_MAX_MESSAGES = 20

BRIEFING_STYLE_NOTE = (
    "(Deliver the requested briefing now. This is a status briefing, not idle "
    "chat — structure it clearly: inbox, then today's calendar, then any "
    "carried-over items — but stay as concise as the content honestly allows.)"
)


async def gather_inbox_triage(registry: ToolRegistry, max_n: int = INBOX_TRIAGE_MAX_MESSAGES) -> Dict[str, Any]:
    """Fetch unread mail metadata for the model to triage in the briefing prompt."""
    tool_spec = registry.get("list_unread")
    if tool_spec is None or tool_spec.handler is None:
        return {"available": False, "count": 0, "messages": []}

    try:
        raw = await tool_spec.handler({"max_n": max_n}, {})
        messages = json.loads(raw)
    except Exception as exc:
        logger.warning(f"[Briefing] inbox triage failed: {exc}")
        return {"available": False, "count": 0, "messages": []}

    return {"available": True, "count": len(messages), "messages": messages}


async def gather_calendar(calendar_sensor: Optional[CalendarSensor]) -> List[Dict[str, str]]:
    """Fetch today's remaining calendar events, if the sensor is available."""
    if calendar_sensor is None:
        return []
    try:
        events = await calendar_sensor.fetch_todays_events()
    except Exception as exc:
        logger.warning(f"[Briefing] calendar fetch failed: {exc}")
        return []
    return [{"title": title, "start_time": start.isoformat()} for title, start in events]


def _carried_over_ids(memory_agent: Optional[MemoryAgent], today_ids: List[str]) -> List[str]:
    """Which of today's top unread ids were also flagged in yesterday's briefing."""
    if memory_agent is None or memory_agent.store is None:
        return []
    previous = memory_agent.store.get(
        memory_type="long_term", category=BRIEFING_MEMORY_CATEGORY, key=BRIEFING_MEMORY_KEY, limit=1
    )
    if not previous:
        return []
    previous_ids = set(previous[0].get("value") or [])
    return [i for i in today_ids if i in previous_ids]


def _save_today_ids(memory_agent: Optional[MemoryAgent], today_ids: List[str]) -> None:
    if memory_agent is None or memory_agent.store is None:
        return
    memory_agent.store.store(
        memory_type="long_term",
        category=BRIEFING_MEMORY_CATEGORY,
        key=BRIEFING_MEMORY_KEY,
        value=today_ids,
    )


def render_briefing_data(inbox: Dict[str, Any], calendar_events: List[Dict[str, str]], carried_over_count: int) -> str:
    """Render the assembled raw data as text for the Planner's model to turn into a briefing."""
    lines: List[str] = []

    if inbox["available"]:
        lines.append(f"Unread email: {inbox['count']} total.")
        for m in inbox["messages"][:INBOX_TRIAGE_MAX_MESSAGES]:
            lines.append(f"  - id={m.get('id','')} From: {m.get('sender','')} | Subject: {m.get('subject','')} | {m.get('snippet','')}")
        lines.append("(Pick the 3 most important above by apparent importance and summarize each in one line.)")
    else:
        lines.append("Unread email: unavailable right now.")

    lines.append("")
    if calendar_events:
        lines.append("Today's calendar:")
        for e in calendar_events:
            lines.append(f"  - {e['title']} at {e['start_time']}")
    else:
        lines.append("Today's calendar: nothing scheduled (or calendar unavailable).")

    if carried_over_count:
        lines.append("")
        lines.append(
            f"Note: {carried_over_count} of today's top unread items were also in "
            "yesterday's briefing and still haven't been dealt with — mention this."
        )

    return "\n".join(lines)


async def assemble_briefing_text(
    registry: Optional[ToolRegistry] = None,
    calendar_sensor: Optional[CalendarSensor] = None,
    memory_agent: Optional[MemoryAgent] = None,
) -> str:
    """Gather inbox triage + today's calendar + carried-over note, as raw text."""
    registry = registry or get_registry()

    inbox = await gather_inbox_triage(registry)
    calendar_events = await gather_calendar(calendar_sensor)

    today_ids = [m.get("id", "") for m in inbox["messages"][:3] if m.get("id")]
    carried_over = _carried_over_ids(memory_agent, today_ids)
    _save_today_ids(memory_agent, today_ids)

    return render_briefing_data(inbox, calendar_events, len(carried_over))


def make_get_daily_briefing_handler(
    registry: Optional[ToolRegistry],
    calendar_sensor: Optional[CalendarSensor],
    memory_agent: Optional[MemoryAgent],
):
    """ToolSpec handler for the on-demand "brief me" tool: just the raw data,
    the calling Planner turn's model renders the actual reply."""

    async def handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        return await assemble_briefing_text(registry, calendar_sensor, memory_agent)

    return handler


async def deliver_scheduled_briefing(
    planner: Planner,
    registry: Optional[ToolRegistry] = None,
    calendar_sensor: Optional[CalendarSensor] = None,
    memory_agent: Optional[MemoryAgent] = None,
) -> PlannerResult:
    """Scheduled path: no ongoing turn to attach to, so start a fresh one."""
    data_text = await assemble_briefing_text(registry, calendar_sensor, memory_agent)
    prompt = f"{BRIEFING_STYLE_NOTE}\n\n{data_text}"
    return await planner.run(user_text=prompt, purpose="planning")
