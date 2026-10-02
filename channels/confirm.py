"""Per-action confirmation nonces for button-based approval (provider-independent).

A Guardian confirmation card is rendered as Approve / Deny buttons. Each button press carries a
NONCE that is bound to exactly one pending action and the one user it was shown to:

    issue(request_id, user_id) -> nonce       one nonce per action, 2-minute lifetime (configurable)
    resolve(nonce, user_id, approved)         -> APPROVED | DENIED, or a rejection

Rejections (all audited by the caller, none of them reach the Guardian as an approval):
    STALE     the nonce is older than its lifetime (the Guardian has already expired the action)
    REUSED    the nonce was already used — a double tap, a replay, a forwarded button
    FOREIGN   the callback came from a different user than the one the card was issued to; the nonce is
              NOT consumed, so the real owner can still answer
    UNKNOWN   no such nonce (never issued, or forgotten after a restart)
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional

DEFAULT_TTL_SECONDS = 120.0


class Outcome(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    STALE = "stale"
    REUSED = "reused"
    FOREIGN = "foreign_user"
    UNKNOWN = "unknown"

    @property
    def accepted(self) -> bool:
        return self in (Outcome.APPROVED, Outcome.DENIED)


@dataclass(frozen=True)
class Resolution:
    outcome: Outcome
    request_id: str = ""


@dataclass
class _Entry:
    request_id: str
    user_id: str
    expires_at: float
    used: bool = False


class ConfirmationBroker:
    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS, clock: Callable[[], float] = time.monotonic,
                 nonce_factory: Optional[Callable[[], str]] = None):
        self.ttl = float(ttl_seconds)
        self._clock = clock
        self._nonce = nonce_factory or (lambda: secrets.token_urlsafe(9))
        self._entries: Dict[str, _Entry] = {}

    def issue(self, request_id: str, user_id: str) -> str:
        self._sweep()
        nonce = self._nonce()
        while nonce in self._entries:        # a collision must never alias two actions
            nonce = self._nonce()
        self._entries[nonce] = _Entry(str(request_id), str(user_id), self._clock() + self.ttl)
        return nonce

    def resolve(self, nonce: str, user_id: str, approved: bool) -> Resolution:
        entry = self._entries.get(str(nonce))
        if entry is None:
            return Resolution(Outcome.UNKNOWN)
        if str(user_id) != entry.user_id:
            return Resolution(Outcome.FOREIGN, entry.request_id)      # not consumed
        if entry.used:
            return Resolution(Outcome.REUSED, entry.request_id)
        entry.used = True
        if self._clock() > entry.expires_at:
            return Resolution(Outcome.STALE, entry.request_id)
        return Resolution(Outcome.APPROVED if approved else Outcome.DENIED, entry.request_id)

    def _sweep(self) -> None:
        """Forget entries a while after they expired (kept briefly so a late tap reads as STALE/REUSED
        rather than UNKNOWN)."""
        cutoff = self._clock() - 10 * self.ttl
        for nonce in [n for n, e in self._entries.items() if e.expires_at < cutoff]:
            del self._entries[nonce]

    def __len__(self) -> int:
        return len(self._entries)
