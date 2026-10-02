"""Rate limiting, log throttling and reply splitting — provider-independent."""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Deque, Dict, List, Tuple


class RateLimiter:
    """Sliding window: at most `max_events` per `window_seconds` for each key."""

    def __init__(self, max_events: int, window_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.max_events = max(1, int(max_events))
        self.window = float(window_seconds)
        self._clock = clock
        self._events: Dict[str, Deque[float]] = {}

    def allow(self, key: str) -> bool:
        now = self._clock()
        q = self._events.setdefault(str(key), deque())
        while q and now - q[0] >= self.window:
            q.popleft()
        if len(q) >= self.max_events:
            return False
        q.append(now)
        return True

    def retry_after(self, key: str) -> float:
        q = self._events.get(str(key))
        if not q or len(q) < self.max_events:
            return 0.0
        return max(0.0, self.window - (self._clock() - q[0]))


class LogThrottle:
    """At most one log line per `interval` seconds for a key; reports how many were suppressed
    in between, so a flood of dropped updates costs one line, not one per update."""

    def __init__(self, interval: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.interval = float(interval)
        self._clock = clock
        self._last: Dict[str, float] = {}
        self._suppressed: Dict[str, int] = {}

    def ready(self, key: str) -> Tuple[bool, int]:
        """(should_log_now, suppressed_since_last_logged)."""
        now = self._clock()
        last = self._last.get(key)
        if last is None or now - last >= self.interval:
            self._last[key] = now
            n = self._suppressed.pop(key, 0)
            return True, n
        self._suppressed[key] = self._suppressed.get(key, 0) + 1
        return False, 0


_BREAKS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def split_reply(text: str, chunk_chars: int = 4000, max_total: int = 12000) -> List[str]:
    """Cut `text` into messages of at most `chunk_chars`, preferring paragraph, line, sentence and word
    boundaries, never mid-word unless a single word is longer than a chunk. If the whole reply is
    longer than `max_total` it is truncated with an explicit note. Multi-part replies are numbered
    "(1/3)" so the order survives, and every part including its number stays within `chunk_chars`."""
    text = (text or "").strip() or "(no reply)"
    chunk_chars = max(50, int(chunk_chars))
    note = ""
    if len(text) > max_total:
        text = text[:max_total].rstrip()
        note = "\n\n[reply truncated — it was longer than the channel limit]"
    if len(text) + len(note) <= chunk_chars:
        return [text + note]

    budget = chunk_chars - 8          # room for the "(12/12) " prefix
    pieces: List[str] = []
    rest = text + note
    while rest:
        if len(rest) <= budget:
            pieces.append(rest)
            break
        cut = -1
        window = rest[: budget + 1]
        for sep in _BREAKS:
            idx = window.rfind(sep)
            if idx >= budget // 3:                 # do not make a uselessly short chunk
                cut = idx + len(sep)
                break
        if cut <= 0:
            cut = budget                           # one giant unbroken word
        pieces.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    pieces = [p for p in pieces if p]
    n = len(pieces)
    return [f"({i}/{n}) {p}" for i, p in enumerate(pieces, 1)] if n > 1 else pieces
