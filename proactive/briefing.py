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
from proactive.day_planner import DEFAULT_NOTION_TASKS_DB, assemble_day_plan_text
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
    "carried-over items, then a time-blocked day plan if day-plan inputs are "
    "present below — but stay as concise as the content honestly allows.)"
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


def _as_list(raw: Any) -> List[Any]:
    """Best-effort: MCP tool results are JSON — pull out the list of items."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "notifications", "pull_requests", "results", "data"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


def _looks_ci_failure(item: Any) -> bool:
    s = (item if isinstance(item, str) else json.dumps(item)).lower()
    return ("ci" in s or "workflow" in s or "check" in s) and ("fail" in s or "error" in s)


async def gather_dev(registry: ToolRegistry) -> Optional[str]:
    """Optional development section — present ONLY when the GitHub MCP tools are
    connected (mcp.servers.github.enabled). Unread notifications, PRs awaiting
    review, and failing CI if visible from notifications. Returns raw data for
    the model to voice in 2-3 lines, or None if GitHub isn't wired up."""
    notif_tool = registry.get("list_notifications")
    pr_tool = registry.get("list_pull_requests")
    have_notif = notif_tool is not None and notif_tool.handler is not None
    have_pr = pr_tool is not None and pr_tool.handler is not None
    if not have_notif and not have_pr:
        return None

    lines: List[str] = []
    if have_notif:
        try:
            items = _as_list(await notif_tool.handler({}, {}))
            ci_fail = sum(1 for it in items if _looks_ci_failure(it))
            note = f"GitHub notifications: {len(items)} unread"
            if ci_fail:
                note += f" ({ci_fail} about failing CI)"
            lines.append(note)
        except Exception as exc:
            logger.warning(f"[Briefing] github notifications failed: {exc}")
    if have_pr:
        try:
            prs = _as_list(await pr_tool.handler({}, {}))
            lines.append(f"Pull requests awaiting your review: {len(prs)}")
        except Exception as exc:
            logger.warning(f"[Briefing] github PRs failed: {exc}")

    if not lines:
        return None
    return "Development (keep to 2-3 lines in your voice):\n" + "\n".join(f"  - {ln}" for ln in lines)


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
    include_day_plan: bool = False,
    notion_db: str = DEFAULT_NOTION_TASKS_DB,
) -> str:
    """Gather inbox triage + today's calendar + carried-over note, as raw
    text, plus an optional day-plan section (calendar + reminders + Notion
    tasks + email pressure + protected-time memories, from day_planner.py)."""
    registry = registry or get_registry()

    inbox = await gather_inbox_triage(registry)
    calendar_events = await gather_calendar(calendar_sensor)

    today_ids = [m.get("id", "") for m in inbox["messages"][:3] if m.get("id")]
    carried_over = _carried_over_ids(memory_agent, today_ids)
    _save_today_ids(memory_agent, today_ids)

    text = render_briefing_data(inbox, calendar_events, len(carried_over))

    if include_day_plan:
        day_plan_text = await assemble_day_plan_text(registry, memory_agent, notion_db)
        text = f"{text}\n\n{day_plan_text}"

    dev_text = await gather_dev(registry)
    if dev_text:
        text = f"{text}\n\n{dev_text}"

    return text


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
    include_day_plan: bool = False,
    notion_db: str = DEFAULT_NOTION_TASKS_DB,
) -> PlannerResult:
    """Scheduled path: no ongoing turn to attach to, so start a fresh one."""
    data_text = await assemble_briefing_text(
        registry, calendar_sensor, memory_agent, include_day_plan=include_day_plan, notion_db=notion_db
    )
    prompt = f"{BRIEFING_STYLE_NOTE}\n\n{data_text}"
    return await planner.run(user_text=prompt, purpose="planning")
