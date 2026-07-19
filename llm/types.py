"""
Shared data types for the LLM routing layer.

These are plain dataclasses (not pydantic models) so provider modules can
construct them without importing config/pydantic machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ToolCall:
    """A single tool/function call requested by the model."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class LLMResponse:
    """
    Normalized completion response — identical shape regardless of which
    provider produced it.

    Attributes:
        text: Assistant text content (may be empty if the model only emitted
            tool calls).
        tool_calls: Parsed, normalized tool calls (arguments always a dict).
        raw: The provider SDK's original response object, kept for debugging.
        provider: Name of the provider that produced this response (e.g. "groq").
        model: Model id used for the call.
        latency_ms: Wall-clock time for the call, in milliseconds.
        usage: Token counts, if the provider reported them
            (keys: prompt_tokens, completion_tokens, total_tokens).
    """

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Any = None
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class RouterError:
    """
    Returned (not raised) by ModelRouter.complete() when every configured
    provider for a purpose has failed.

    Callers should surface `user_message` to the user and may log
    `primary_error`/`fallback_error` for diagnostics.
    """

    user_message: str = "Sir, I'm having trouble thinking right now."
    primary_error: str = ""
    fallback_error: str = ""
    purpose: str = ""

    @property
    def message(self) -> str:
        """Technical summary suitable for logs (not for the end user)."""
        return (
            f"LLM routing failed for purpose={self.purpose!r}: "
            f"primary_error={self.primary_error!r} fallback_error={self.fallback_error!r}"
        )
