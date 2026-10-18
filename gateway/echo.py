"""Echo filter and loop breaker (gateway side). The built-in mic hears the speaker; these keep Vesper's own voice from becoming a command.

EchoFilter  A voice transcript that is mostly one of the last N assistant utterances is the assistant hearing itself: dropped before the
            planner, logged as a COUNT and a REASON CODE only (never the text). "Mostly" = the fraction of the transcript's words covered, in
            order, by one reply (difflib matching blocks) >= threshold, for transcripts of at least `min_words` words. A real command that merely
            shares a few words with a reply ("open safari for me" after "Opening Safari for you now, Sir.") is far below the bar.
LoopBreaker If `failures` turns fail within `window_s`, or `quick_turns` consecutive turns start within `quick_gap_s` of speech ending, voice
            input is paused for `pause_s` and ONE short HUD notice is shown (nothing is spoken).
"""
from __future__ import annotations

import difflib
import re
import time
from collections import deque
from typing import Callable, Deque, List, Optional


def normalize(text: str) -> List[str]:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower().replace("'", "").replace("’", "")).split()


def coverage(transcript: List[str], reply: List[str]) -> float:
    """Fraction of the transcript's words that match, in order, words of the reply."""
    if not transcript or not reply:
        return 0.0
    blocks = difflib.SequenceMatcher(None, transcript, reply, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) / len(transcript)


class EchoFilter:
    REASON = "echo_match"

    def __init__(self, history: int = 5, threshold: float = 0.7, min_words: int = 4):
        self._replies: Deque[List[str]] = deque(maxlen=max(1, int(history)))
        self.threshold, self.min_words = float(threshold), int(min_words)
        self.dropped = 0

    def add_reply(self, text: str) -> None:
        words = normalize(text)
        if words:
            self._replies.append(words)

    def check(self, transcript: str) -> Optional[str]:
        """A reason code if `transcript` is the assistant's own voice, else None."""
        words = normalize(transcript)
        if len(words) < self.min_words:
            return None
        for reply in self._replies:
            if coverage(words, reply) >= self.threshold:
                self.dropped += 1
                return self.REASON
        return None


class LoopBreaker:
    def __init__(self, clock: Callable[[], float] = time.monotonic, failures: int = 3, window_s: float = 60.0, quick_turns: int = 3,
                 quick_gap_s: float = 3.0, pause_s: float = 30.0, on_trip: Callable[[str], None] = lambda reason: None):
        self._clock, self.failures, self.window_s = clock, int(failures), float(window_s)
        self.quick_turns, self.quick_gap_s, self.pause_s = int(quick_turns), float(quick_gap_s), float(pause_s)
        self._on_trip = on_trip
        self._fails: Deque[float] = deque()
        self._quick = 0
        self._paused_until = 0.0
        self.trips = 0

    def paused_for(self) -> float:
        return max(0.0, self._paused_until - self._clock())

    def _trip(self, reason: str) -> None:
        self._paused_until = self._clock() + self.pause_s
        self._fails.clear()
        self._quick = 0
        self.trips += 1
        self._on_trip(reason)

    def record_failure(self) -> None:
        now = self._clock()
        self._fails.append(now)
        while self._fails and now - self._fails[0] > self.window_s:
            self._fails.popleft()
        if len(self._fails) >= self.failures and self.paused_for() == 0:
            self._trip("failed_turns")

    def record_turn_start(self, since_speech_end: Optional[float]) -> None:
        """`since_speech_end` = seconds since the assistant stopped speaking (0 while it is speaking), None if it has not spoken."""
        if since_speech_end is not None and since_speech_end <= self.quick_gap_s:
            self._quick += 1
        else:
            self._quick = 0
        if self._quick >= self.quick_turns and self.paused_for() == 0:
            self._trip("quick_turns")
