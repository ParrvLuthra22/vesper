"""Glue between the briefing engine and the rest of Vesper: the get_daily_briefing tool
handler and the "good morning" fast path."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Optional

from briefing.config import BriefingConfig
from briefing.service import BriefingService

_STRIP = re.compile(r"[^a-z0-9' ]+")
_TRAILING_NAMES = ("vesper", "jarvis", "sir", "friday")


def make_cached_briefing_handler(service: BriefingService):
    """get_daily_briefing handler: read the cache; refresh synchronously only if some
    source is older than briefing max_cache_age_minutes. Returns the capped,
    sanitized, spoken-friendly block (the tool is registered untrusted_output)."""

    async def handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        try:
            await service.ensure_fresh()
        except Exception:  # a failed refresh still leaves a (stale, labelled) briefing
            pass
        return service.tool_result()

    return handler


def normalize_utterance(text: str) -> str:
    s = (text or "").replace("\u2019", "'").lower()
    words = _STRIP.sub(" ", s).split()
    while words and words[-1] in _TRAILING_NAMES:
        words.pop()
    return " ".join(words)


def is_briefing_request(text: str, cfg: BriefingConfig, now: Optional[datetime] = None) -> bool:
    """True for a short, unambiguous request for the briefing ("good morning", "brief me").
    Long utterances ("good morning, remind me to ...") never match — they go to the planner.
    A bare greeting only counts before `morning_until_hour`."""
    if not cfg.fast_path_enabled:
        return False
    norm = normalize_utterance(text).replace("'", "")
    if len(norm.split()) > 8:
        return False
    phrases = {normalize_utterance(p).replace("'", "") for p in cfg.fast_path_phrases}
    if norm not in phrases:
        return False
    if norm in ("good morning", "morning"):
        return (now or datetime.now()).hour < cfg.morning_until_hour
    return True
