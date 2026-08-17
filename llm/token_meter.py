"""
Token accounting for the LLM layer.

Two jobs, both in service of staying under a free tier's tokens-per-minute
ceiling (Groq's free tier is 8k TPM, which a single tool-heavy planning
turn used to blow through in three calls):

    estimate_tokens()  — how big is this payload, before we send it?
    TokenBudget        — how much have we spent in the rolling last 60s,
                         and how long must we wait before spending more?

Estimation prefers `tiktoken` when it is installed (o200k_base matches the
OpenAI-family tokenizers Groq serves) and degrades to a chars/4 heuristic
otherwise, so nothing here becomes a hard dependency on tiktoken.
"""

from __future__ import annotations

import json
import time
from collections import deque
from typing import Any, Deque, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

#: Fallback ratio when tiktoken is unavailable. Deliberately conservative
#: (real English+JSON runs ~3.6-4.2 chars/token) — overestimating spend is
#: safe here, underestimating it is what causes a 429.
CHARS_PER_TOKEN = 3.6

_encoder: Any = None
_encoder_resolved = False


def _get_encoder() -> Any:
    """Load tiktoken's o200k_base encoder once; None if unavailable."""
    global _encoder, _encoder_resolved
    if not _encoder_resolved:
        _encoder_resolved = True
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("o200k_base")
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug(f"[TokenMeter] tiktoken unavailable, using chars/{CHARS_PER_TOKEN} estimate: {exc}")
            _encoder = None
    return _encoder


def estimate_tokens(payload: Any) -> int:
    """
    Approximate the token cost of `payload` (a string, or anything
    JSON-serializable such as a messages list or a tools schema).

    This is an estimate used for pacing and for before/after reporting —
    the provider's own reported `usage` remains the source of truth after
    a call completes.
    """
    # An empty payload costs nothing — but `json.dumps([])` is "[]", which
    # would otherwise be charged a token and make "no tools" look non-free.
    if not payload:
        return 0
    text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    if not text:
        return 0

    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:  # pragma: no cover - defensive
            pass
    return int(len(text) / CHARS_PER_TOKEN) + 1


def estimate_request_tokens(
    messages: Optional[List[dict]] = None, tools: Optional[List[dict]] = None
) -> int:
    """Estimated prompt-token cost of one chat completion request."""
    return estimate_tokens(messages) + estimate_tokens(tools)


class TokenBudget:
    """
    Rolling-window token-spend tracker, used to pace calls so a request
    that *would* exceed the provider's tokens-per-minute ceiling waits a
    moment instead of firing into a guaranteed 429.

    Not thread-safe by design: it is driven from the single asyncio loop
    the router runs on.
    """

    def __init__(
        self,
        tokens_per_minute: int,
        window_seconds: float = 60.0,
        headroom: float = 0.9,
        max_wait_seconds: float = 4.0,
    ):
        """
        Args:
            tokens_per_minute: The provider's advertised ceiling.
            window_seconds: Width of the rolling window (60s for a TPM limit).
            headroom: Fraction of the ceiling we allow ourselves to use, so
                estimation error doesn't put us over the real limit.
            max_wait_seconds: Never delay a call longer than this — beyond
                it, firing and handling the 429 beats stalling the user.
        """
        self.tokens_per_minute = max(0, int(tokens_per_minute))
        self.window_seconds = window_seconds
        self.headroom = headroom
        self.max_wait_seconds = max_wait_seconds
        self._spend: Deque[Tuple[float, int]] = deque()

    @property
    def enabled(self) -> bool:
        """Pacing is off when no positive ceiling is configured."""
        return self.tokens_per_minute > 0

    @property
    def effective_limit(self) -> int:
        return int(self.tokens_per_minute * self.headroom)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._spend and self._spend[0][0] <= cutoff:
            self._spend.popleft()

    def used(self, now: Optional[float] = None) -> int:
        """Tokens spent inside the current rolling window."""
        now = time.monotonic() if now is None else now
        self._prune(now)
        return sum(tokens for _, tokens in self._spend)

    def record(self, tokens: int) -> None:
        """Record a completed call's token spend."""
        if not self.enabled or tokens <= 0:
            return
        self._spend.append((time.monotonic(), int(tokens)))

    def delay_for(self, estimated_tokens: int) -> float:
        """
        Seconds to wait before spending `estimated_tokens` so the rolling
        window stays under the effective limit. 0.0 when it already fits.

        A single request larger than the whole ceiling can never fit, so
        waiting for it is pointless — those return 0.0 and are simply
        sent (the provider is the one who gets to reject them).
        """
        if not self.enabled or estimated_tokens <= 0:
            return 0.0

        now = time.monotonic()
        self._prune(now)
        limit = self.effective_limit

        if estimated_tokens >= limit:
            return 0.0

        used = sum(tokens for _, tokens in self._spend)
        if used + estimated_tokens <= limit:
            return 0.0

        # Wait only long enough for the oldest entries to age out of the
        # window — freeing exactly the overage, not the whole window.
        must_free = used + estimated_tokens - limit
        freed = 0
        for timestamp, tokens in self._spend:
            freed += tokens
            if freed >= must_free:
                wait = (timestamp + self.window_seconds) - now
                return max(0.0, min(wait, self.max_wait_seconds))
        return 0.0
