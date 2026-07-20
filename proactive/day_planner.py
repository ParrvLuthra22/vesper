"""
proactive/day_planner.py — assembles data for Vesper's day-planning
capability: the plan_my_day tool ("plan my day") and an optional
section wired into the morning briefing (see proactive/briefing.py).

Same two-path design as briefing.py: gather + render produce raw text;
the calling Planner turn's model — whichever one is already running,
the on-demand plan_my_day tool call or the scheduled briefing's fresh
planner.run() — renders the actual structured, time-blocked plan. No
nested LLM call happens here.

Every source degrades gracefully when unavailable (Notion not
configured, Calendar/Reminders access not yet granted, Gmail
disabled) — a day plan with fewer inputs is still useful; a missing
one should never break the others.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agents.memory_agent import MemoryAgent
from tools.registry import ToolRegistry, get_registry
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_NOTION_TASKS_DB = "tasks"
#: Retrieves whatever the user has told Vesper to always protect (e.g. "the
#: gym slot is non-negotiable") -- written by the reflection job, read here
#: by semantic similarity rather than an exact keyword match.
PROTECTED_BLOCKS_MEMORY_QUERY = "protected sacred non-negotiable time block schedule preference"
PROTECTED_BLOCKS_TOP_K = 5


async def _call_tool(registry: ToolRegistry, name: str, arguments: Dict[str, Any]) -> Optional[str]:
    tool_spec = registry.get(name)
    if tool_spec is None or tool_spec.handler is None:
        return None
    try:
        return await tool_spec.handler(arguments, {})
    except Exception as exc:
        logger.warning(f"[DayPlanner] {name} failed: {exc}")
        return None


async def gather_calendar_events(registry: ToolRegistry) -> List[Dict[str, Any]]:
    raw = await _call_tool(registry, "today_events", {})
    if raw is None:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


async def gather_reminders(registry: ToolRegistry, scope: str = "today") -> List[Dict[str, Any]]:
    raw = await _call_tool(registry, "reminders_due", {"scope": scope})
    if raw is None:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


async def gather_notion_tasks(registry: ToolRegistry, db_id: str = DEFAULT_NOTION_TASKS_DB) -> List[Dict[str, Any]]:
    raw = await _call_tool(registry, "notion_get_database_rows", {"db_id": db_id})
    if raw is None:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


async def gather_email_pressure(registry: ToolRegistry) -> Optional[int]:
    raw = await _call_tool(registry, "unread_count", {})
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


async def gather_protected_blocks(memory_agent: Optional[MemoryAgent]) -> List[str]:
    """Semantic memories describing sacred/protected time (e.g. "the gym
    slot is non-negotiable") — so the rendering model routes around them."""
    if memory_agent is None:
        return []
    try:
        matches = await memory_agent.semantic_retrieve(
            query=PROTECTED_BLOCKS_MEMORY_QUERY, top_k=PROTECTED_BLOCKS_TOP_K
        )
    except Exception as exc:
        logger.warning(f"[DayPlanner] memory retrieval failed: {exc}")
        return []
    return [m["text"] for m in matches]


def render_day_plan_data(
    calendar_events: List[Dict[str, Any]],
    reminders: List[Dict[str, Any]],
    notion_tasks: List[Dict[str, Any]],
    unread_count: Optional[int],
    protected_blocks: List[str],
) -> str:
    """Render the assembled raw data as text for the Planner's model to turn
    into a time-blocked plan."""
    lines: List[str] = ["Day plan inputs:", ""]

    if calendar_events:
        lines.append("Calendar (fixed commitments):")
        for e in calendar_events:
            span = f"{e.get('start', '?')} - {e.get('end')}" if e.get("end") else str(e.get("start", "?"))
            lines.append(f"  - {span}: {e.get('title', 'Untitled')}")
    else:
        lines.append("Calendar: nothing scheduled today.")
    lines.append("")

    if reminders:
        lines.append("Reminders due:")
        for r in reminders:
            due = f" (due {r['due']})" if r.get("due") else ""
            lines.append(f"  - {r.get('title', 'Untitled')}{due}")
    else:
        lines.append("Reminders due: none.")
    lines.append("")

    if notion_tasks:
        lines.append("Notion tasks:")
        for t in notion_tasks:
            props = t.get("properties", {}) or {}
            summary = ", ".join(f"{k}={v}" for k, v in props.items() if v not in (None, "", []))
            lines.append(f"  - {summary or t.get('id', 'untitled task')}")
    else:
        lines.append("Notion tasks: none available.")
    lines.append("")

    if unread_count is not None:
        lines.append(f"Unread email: {unread_count} — factor in a block to process it if that's a lot.")
    else:
        lines.append("Unread email: unavailable.")

    if protected_blocks:
        lines.append("")
        lines.append("Relevant memories (protected time — the plan must route around these, never schedule over them):")
        for m in protected_blocks:
            lines.append(f"  - {m}")

    lines.append("")
    lines.append(
        "(Produce a realistic, time-blocked plan from the above — account for "
        "actual gaps between fixed commitments, don't overpack it. Flag any "
        "scheduling conflicts you notice.)"
    )

    return "\n".join(lines)


async def assemble_day_plan_text(
    registry: Optional[ToolRegistry] = None,
    memory_agent: Optional[MemoryAgent] = None,
    notion_db: str = DEFAULT_NOTION_TASKS_DB,
) -> str:
    """Gather calendar + reminders + Notion tasks + email pressure + any
    protected-time memories, as raw text."""
    registry = registry or get_registry()

    calendar_events = await gather_calendar_events(registry)
    reminders = await gather_reminders(registry)
    notion_tasks = await gather_notion_tasks(registry, db_id=notion_db)
    unread = await gather_email_pressure(registry)
    protected_blocks = await gather_protected_blocks(memory_agent)

    return render_day_plan_data(calendar_events, reminders, notion_tasks, unread, protected_blocks)


def make_plan_my_day_handler(
    registry: Optional[ToolRegistry],
    memory_agent: Optional[MemoryAgent],
    notion_db: str = DEFAULT_NOTION_TASKS_DB,
):
    """ToolSpec handler for the on-demand "plan my day" tool: just the raw
    data, the calling Planner turn's model renders the actual plan."""

    async def handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        return await assemble_day_plan_text(registry, memory_agent, notion_db)

    return handler
