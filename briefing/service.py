"""
BriefingService — runs the collectors on a jittered timer and serves briefings from the cache.

    refresh()        every available collector, concurrently and ISOLATED: one that raises or
                     times out is recorded as a failure (its data goes stale and says so) and
                     never blocks or fails the others
    ensure_fresh()   refresh synchronously only if some source is older than max_cache_age
    start()/stop()   the background loop: first refresh right away, then every
                     interval_minutes +/- jitter
    briefing()       rescore + select from the cache (no network)
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from briefing.builder import Briefing, build_briefing, render_tool_result, spoken_script
from briefing.cache import BriefingCache
from briefing.collectors import CalendarCollector, Collector, GmailCollector
from briefing.config import BriefingConfig, load_briefing_config
from briefing.known import META_KEY as KNOWN_META_KEY, derive_known
from briefing.readonly import ReadOnlyTools
from tools.registry import ToolRegistry, get_registry
from utils.logger import get_logger

logger = get_logger(__name__)

SENT_META_KEY = "gmail_sent"


@dataclass
class CollectorReport:
    name: str
    ok: bool
    new_items: int = 0
    error: Optional[str] = None
    skipped: bool = False


class BriefingService:
    def __init__(
        self,
        collectors: List[Collector],
        cfg: Optional[BriefingConfig] = None,
        cache: Optional[BriefingCache] = None,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ):
        for c in collectors:
            if not getattr(c, "read_only", False):
                raise ValueError(f"collector {c.name!r} does not declare read_only=True")
        self.cfg = cfg or load_briefing_config()
        self.cache = cache or BriefingCache(self.cfg.cache_path)
        self._collectors = list(collectors)
        self._clock, self._sleep, self._rng = clock, sleep, rng
        self._lock = asyncio.Lock()
        self._task: Optional["asyncio.Task[None]"] = None

    @classmethod
    def from_registry(cls, registry: Optional[ToolRegistry] = None, cfg: Optional[BriefingConfig] = None,
                      cache: Optional[BriefingCache] = None) -> "BriefingService":
        cfg = cfg or load_briefing_config()
        tools = ReadOnlyTools(registry or get_registry())
        return cls([GmailCollector(tools, cfg), CalendarCollector(tools, cfg)], cfg, cache)

    # ------------------------------------------------------------- refresh
    async def refresh(self) -> Dict[str, CollectorReport]:
        async with self._lock:
            results = await asyncio.gather(*(self._run_one(c) for c in self._collectors))
        for r in results:
            if r.error:
                logger.warning(f"[Briefing] {r.name} refresh failed: {r.error}")
        self.cache.prune(now=self._clock())
        return {r.name: r for r in results}

    async def _run_one(self, c: Collector) -> CollectorReport:
        timeout = self.cfg.collector_timeout_seconds
        started = self._clock()
        if not c.available():
            self.cache.record_failure(c.name, "tools unavailable (is its MCP server connected?)", now=started)
            return CollectorReport(c.name, ok=False, skipped=True, error="tools unavailable")
        try:
            cursor = self.cache.get_cursor(c.name)
            items = await asyncio.wait_for(c.fetch_since(cursor), timeout)
            self.cache.upsert_items(items, now=started)
            live = await asyncio.wait_for(c.live_ids(), timeout)
            if live is not None:
                self.cache.deactivate_missing(c.name, set(live) | {i.id for i in items})
            await self._refresh_aux(c, timeout)
            self.cache.record_success(c.name, c.next_cursor(cursor, items, started), now=self._clock())
            return CollectorReport(c.name, ok=True, new_items=len(items))
        except asyncio.TimeoutError:
            msg = f"timed out after {timeout:g}s"
            self.cache.record_failure(c.name, msg, now=self._clock())
            return CollectorReport(c.name, ok=False, error=msg)
        except Exception as exc:  # isolation: nothing a collector raises escapes
            msg = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            self.cache.record_failure(c.name, msg, now=self._clock())
            return CollectorReport(c.name, ok=False, error=msg)

    async def _refresh_aux(self, c: Collector, timeout: float) -> None:
        if c.name != "gmail":
            return
        stored, updated = self.cache.get_meta(SENT_META_KEY)
        same_window = bool(stored) and stored.get("window_days") == self.cfg.sent_days
        if updated is not None and same_window and (self._clock() - updated) < self.cfg.sent_refresh_hours * 3600:
            return   # fresh enough, and derived from the window the config asks for
        try:
            aux = await asyncio.wait_for(c.aux(), timeout)
        except Exception as exc:   # the sent summary only sharpens scoring; never fail the run over it
            logger.warning(f"[Briefing] sent-mail summary unavailable: {exc}")
            return
        if aux:
            aux = {**aux, "window_days": self.cfg.sent_days}
            self.cache.set_meta(SENT_META_KEY, aux, now=self._clock())
            # Derived, header-only: addresses/domains I have written to (never bodies).
            self.cache.set_meta(KNOWN_META_KEY, derive_known(aux, self.cfg).to_dict(), now=self._clock())

    def is_stale(self, max_age_minutes: Optional[float] = None) -> bool:
        limit = (self.cfg.max_cache_age_minutes if max_age_minutes is None else max_age_minutes) * 60
        now = self._clock()
        for c in self._collectors:
            if not c.available():
                continue
            age = self.cache.health(c.name).age_seconds(now)
            if age is None or age > limit:
                return True
        return False

    async def ensure_fresh(self, max_age_minutes: Optional[float] = None) -> bool:
        """Refresh synchronously only if the cache is older than the limit. True if it refreshed."""
        if not self.is_stale(max_age_minutes):
            return False
        await self.refresh()
        return True

    # ----------------------------------------------------------- background
    def start(self) -> None:
        if self._task is None or self._task.done():
            if self.cfg.enabled:
                self._task = asyncio.ensure_future(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    def next_delay(self) -> float:
        base = self.cfg.interval_minutes * 60.0
        return max(30.0, base * (1.0 + self.cfg.jitter_fraction * (2.0 * self._rng() - 1.0)))

    async def _loop(self) -> None:
        while True:
            try:
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception as exc:   # refresh() already isolates collectors; belt and braces
                logger.warning(f"[Briefing] refresh loop error: {exc}")
            await self._sleep(self.next_delay())

    # --------------------------------------------------------------- output
    def briefing(self, now: Optional[float] = None) -> Briefing:
        avail = {c.name: c.available() for c in self._collectors}
        return build_briefing(self.cache, self.cfg, self._clock() if now is None else now, avail)

    def tool_result(self, now: Optional[float] = None) -> str:
        text, _ = render_tool_result(self.briefing(now), self.cfg)
        return text

    def spoken(self, now: Optional[float] = None) -> str:
        return spoken_script(self.briefing(now), self.cfg)
