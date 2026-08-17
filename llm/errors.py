"""
Internal exception types raised by LLM providers.

ModelRouter only ever catches these two types (never provider-SDK-specific
exceptions), which is what lets new providers be added without touching
router logic: a provider just needs to translate its own SDK's exceptions
into one of these before they leave `complete()`.
"""

from __future__ import annotations


from typing import Optional


class ProviderError(Exception):
    """Generic provider failure: network error, bad response, misconfiguration."""


class RateLimitError(ProviderError):
    """
    Raised by a provider when the backend reports HTTP 429 / rate limiting.

    `retry_after` carries the wait the provider itself asked for, in
    seconds, when it sent one (Groq returns `retry-after` on 429). The
    router prefers it over blind exponential backoff: on a tokens-per-minute
    limit the reset is usually a second or two, so waiting the stated window
    and retrying keeps the turn on the fast provider instead of dropping it
    to a slow local fallback.
    """

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after
