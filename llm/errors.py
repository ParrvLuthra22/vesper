"""
Internal exception types raised by LLM providers.

ModelRouter only ever catches these two types (never provider-SDK-specific
exceptions), which is what lets new providers be added without touching
router logic: a provider just needs to translate its own SDK's exceptions
into one of these before they leave `complete()`.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Generic provider failure: network error, bad response, misconfiguration."""


class RateLimitError(ProviderError):
    """Raised by a provider when the backend reports HTTP 429 / rate limiting."""
