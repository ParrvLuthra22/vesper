"""
proactive/reflection.py — extracts durable, long-term-relevant memory
from a day's conversation and writes it via MemoryAgent into semantic
(Chroma) memory.

Runs two ways, both calling run_reflection():
    - Session end: Brain.stop() calls it directly, right before agents
      (including MemoryAgent) are torn down.
    - Daily at midnight: ProactiveEngine's midnight_reflection cron job
      emits ReflectionRequestedEvent; Brain's handler calls it too.

A strict extraction prompt asks the model for at most 5 one-line items,
each tagged with a kind (preference|fact|pattern) — anything not in
that exact shape is skipped rather than guessed at, since this feeds
memory unsupervised and a malformed line silently becoming a garbled
memory would be worse than losing it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

from agents.memory_agent import MemoryAgent
from llm.router import ModelRouter
from llm.types import RouterError
from utils.logger import get_logger

logger = get_logger(__name__)

VALID_KINDS = ("preference", "fact", "pattern")
MAX_MEMORIES_PER_REFLECTION = 5

REFLECTION_PROMPT_TEMPLATE = """You are extracting durable, long-term-relevant memory from a conversation transcript, for a personal assistant's memory store.

Extract at most {max_items} items worth remembering long-term: durable facts about the user, explicitly stated preferences, or clearly observed patterns. Do NOT include one-off requests, transient task details, or anything unlikely to matter again.

Music preferences count and should be captured as `preference` — e.g. "coding playlist = Deep Focus", "likes lo-fi while working", "dislikes vocals when concentrating", "gym music = high-energy". Capture the user's reaction to a suggested track/playlist too, if they gave one.

Output exactly one item per line, formatted as:
kind: text

where kind is one of: preference, fact, pattern. If nothing in the transcript is worth remembering long-term, output nothing at all — an empty response is correct and expected.

Transcript:
{transcript}
"""

_LINE_RE = re.compile(r"^\s*(preference|fact|pattern)\s*:\s*(.+?)\s*$", re.IGNORECASE)


@dataclass
class ReflectionItem:
    kind: str
    text: str


def _render_transcript(turns: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for turn in turns:
        user_text = turn.get("user", "")
        response = turn.get("response", "")
        if user_text:
            lines.append(f"User: {user_text}")
        if response:
            lines.append(f"Vesper: {response}")
    return "\n".join(lines)


def _parse_reflection_output(text: str) -> List[ReflectionItem]:
    items: List[ReflectionItem] = []
    for line in text.splitlines():
        match = _LINE_RE.match(line)
        if not match:
            continue
        kind, item_text = match.group(1).lower(), match.group(2).strip()
        if not item_text:
            continue
        items.append(ReflectionItem(kind=kind, text=item_text))
        if len(items) >= MAX_MEMORIES_PER_REFLECTION:
            break
    return items


async def run_reflection(
    turns: List[Dict[str, Any]],
    router: ModelRouter,
    memory_agent: Optional[MemoryAgent],
) -> List[ReflectionItem]:
    """
    Extract and store durable memories from `turns` (the shape
    ConversationContext.get_recent_context() produces).

    Returns what was stored, for logging/testing — an empty list means
    either there was nothing worth remembering, or memory isn't
    available (never raises for either reason: reflection is a
    background nicety, not something that should ever break a session
    ending or a scheduled job).
    """
    if not turns or memory_agent is None:
        return []

    transcript = _render_transcript(turns)
    if not transcript.strip():
        return []

    prompt = REFLECTION_PROMPT_TEMPLATE.format(max_items=MAX_MEMORIES_PER_REFLECTION, transcript=transcript)

    try:
        response = await router.complete(
            messages=[{"role": "user", "content": prompt}], purpose="reflection",
        )
    except Exception as exc:
        logger.warning(f"[Reflection] extraction call raised: {exc}")
        return []

    if isinstance(response, RouterError):
        logger.warning(f"[Reflection] extraction call failed: {response.message}")
        return []

    items = _parse_reflection_output(response.text)
    if not items:
        return []

    today = date.today().isoformat()
    for item in items:
        try:
            await memory_agent.index_memory_text(
                text=item.text,
                memory_type="long_term",
                intent=f"reflection_{item.kind}",
                metadata={"kind": item.kind, "source": "reflection", "date": today},
                salience=0.8,
            )
        except Exception as exc:
            logger.warning(f"[Reflection] failed to store item {item!r}: {exc}")

    logger.info(f"[Reflection] stored {len(items)} memory item(s)")
    return items
