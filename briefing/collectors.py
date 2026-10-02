"""
Collectors: read-only producers of `Item`s.

CONTRACT (enforced by construction, tested): a collector never sends, modifies,
archives, drafts or labels anything. It only reaches tools through ReadOnlyTools,
which allows nothing but the pure-read tools in READONLY_TOOLS.

    fetch_since(cursor)  -> the items that are new since `cursor` (cursor None = first run)
    next_cursor(...)     -> the cursor to persist after a successful fetch
    live_ids()           -> ids still "live" at the source (unread / still on the calendar),
                            so the cache can drop read mail and cancelled meetings
    aux()                -> optional slow-changing side data (Gmail: what I've sent)

A failing collector raises; BriefingService isolates it so the others still run.
"""

from __future__ import annotations

import hashlib
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from briefing.config import BriefingConfig
from briefing.items import Item
from briefing.readonly import ReadOnlyTools
from briefing.sanitize import clean_text
from utils.logger import get_logger

logger = get_logger(__name__)


class Collector(ABC):
    name: str = ""
    #: Marker for the contract above; the service refuses collectors that clear it.
    read_only: bool = True

    @abstractmethod
    async def fetch_since(self, cursor: Optional[str]) -> List[Item]:
        """Items new since `cursor`. Read-only."""

    def available(self) -> bool:
        return True

    def next_cursor(self, cursor: Optional[str], items: List[Item], now: float) -> Optional[str]:
        return str(int(now))

    async def live_ids(self) -> Optional[Set[str]]:
        return None

    async def aux(self) -> Optional[Dict[str, Any]]:
        return None


# --------------------------------------------------------------------------- Gmail

def _epoch_from_ms(ms: Any) -> float:
    try:
        return float(ms) / 1000.0
    except (TypeError, ValueError):
        return 0.0


class GmailCollector(Collector):
    name = "gmail"
    #: seconds of overlap when advancing the cursor, so a message that lands during a
    #: fetch is not missed (upsert makes the re-read harmless)
    OVERLAP_SECONDS = 120

    def __init__(self, tools: ReadOnlyTools, cfg: BriefingConfig, clock=time.time):
        self._tools = tools
        self._cfg = cfg
        self._clock = clock

    def available(self) -> bool:
        return self._tools.available("list_inbox") and self._tools.available("unread_ids")

    async def fetch_since(self, cursor: Optional[str]) -> List[Item]:
        now = self._clock()
        if cursor:
            try:
                since = max(0.0, float(cursor) - self.OVERLAP_SECONDS)
            except ValueError:
                since = now - self._cfg.gmail_lookback_days * 86400
        else:
            since = now - self._cfg.gmail_lookback_days * 86400
        rows = await self._tools.call("list_inbox", {
            "max_n": self._cfg.gmail_max_new,
            "query": f"in:inbox is:unread after:{int(since)}",
        })
        return [self._to_item(r) for r in (rows if isinstance(rows, list) else []) if r.get("id")]

    def _to_item(self, r: Dict[str, Any]) -> Item:
        subject = str(r.get("subject") or "")
        return Item(
            id=f"gmail:{r['id']}",
            source="gmail",
            sender=str(r.get("sender") or ""),
            title=subject,
            snippet=clean_text(r.get("snippet"), self._cfg.snippet_chars),
            timestamp=_epoch_from_ms(r.get("internal_date_ms")),
            thread_id=str(r.get("thread_id") or ""),
            is_reply=bool(r.get("in_reply_to")) or bool(re.match(r"\s*(re|aw|fwd?)\s*:", subject, re.IGNORECASE)),
            unread=bool(r.get("unread", True)),
            signals={
                "labels": list(r.get("labels") or []),
                "list_id": str(r.get("list_id") or ""),
                "has_list_unsubscribe": bool(r.get("has_list_unsubscribe")),
                "precedence": str(r.get("precedence") or ""),
                "auto_submitted": str(r.get("auto_submitted") or ""),
            },
        )

    async def live_ids(self) -> Optional[Set[str]]:
        ids = await self._tools.call("unread_ids", {"max_n": 500})
        return {f"gmail:{i}" for i in ids} if isinstance(ids, list) else None

    async def aux(self) -> Optional[Dict[str, Any]]:
        if not self._tools.available("sent_summary"):
            return None
        data = await self._tools.call("sent_summary", {"days": self._cfg.sent_days, "max_n": 200})
        return data if isinstance(data, dict) else None


# ------------------------------------------------------------------------ Calendar

def _parse_iso(value: Any) -> Optional[float]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class CalendarCollector(Collector):
    """Calendar is state, not a stream: every run re-reads the [now, now+window] window
    and the cache drops events that vanished (cancelled/moved). The cursor only records
    when we last looked."""

    name = "calendar"

    def __init__(self, tools: ReadOnlyTools, cfg: BriefingConfig):
        self._tools = tools
        self._cfg = cfg
        self._last_ids: Optional[Set[str]] = None

    def available(self) -> bool:
        return self._tools.available("upcoming")

    async def fetch_since(self, cursor: Optional[str]) -> List[Item]:
        rows = await self._tools.call("upcoming", {"days": self._cfg.calendar_window_days})
        items = [self._to_item(r) for r in (rows if isinstance(rows, list) else [])]
        self._last_ids = {i.id for i in items}
        return items

    async def live_ids(self) -> Optional[Set[str]]:
        return self._last_ids

    @staticmethod
    def _to_item(r: Dict[str, Any]) -> Item:
        title = str(r.get("title") or "Untitled")
        start = _parse_iso(r.get("start"))
        end = _parse_iso(r.get("end"))
        digest = hashlib.sha1(f"{title}|{r.get('start')}".encode("utf-8")).hexdigest()[:16]
        return Item(
            id=f"calendar:{digest}",
            source="calendar",
            sender=str(r.get("calendar") or ""),
            title=title,
            # Event notes are third-party text and are deliberately NOT collected.
            snippet="",
            timestamp=start or 0.0,
            start_at=start,
            end_at=end,
            all_day=bool(r.get("all_day")),
            signals={"calendar": str(r.get("calendar") or "")},
        )
