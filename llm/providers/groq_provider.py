"""Groq provider — chat completions with tool calling via the `groq` Python SDK."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from llm.errors import ProviderError, RateLimitError
from llm.json_repair import parse_tool_arguments
from llm.providers import register_provider
from llm.providers.base import LLMProvider
from llm.types import LLMResponse, ToolCall
from utils.api_keys import get_groq_api_key
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from groq import (
        APIConnectionError as _GroqAPIConnectionError,
        APIStatusError as _GroqAPIStatusError,
        APITimeoutError as _GroqAPITimeoutError,
        AsyncGroq,
        RateLimitError as _GroqRateLimitError,
    )

    GROQ_SDK_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    AsyncGroq = None  # type: ignore[assignment]
    _GroqAPIConnectionError = _GroqAPIStatusError = _GroqAPITimeoutError = Exception  # type: ignore[assignment]
    _GroqRateLimitError = Exception  # type: ignore[assignment]
    GROQ_SDK_AVAILABLE = False


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """
    Pull the wait Groq asked for out of a 429 response.

    Groq sends `retry-after` (seconds) and the more precise
    `x-ratelimit-reset-tokens` (e.g. "2.5s") on a token-per-minute trip.
    Either is far better than guessing: the reset on an 8k TPM limit is
    typically a second or two, so the turn can stay on Groq.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    for name in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        raw = headers.get(name)
        if not raw:
            continue
        try:
            return float(str(raw).strip().rstrip("s"))
        except (TypeError, ValueError):
            continue
    return None


@register_provider("groq")
class GroqProvider(LLMProvider):
    """Routes completions through Groq's OpenAI-compatible chat API."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._api_key = get_groq_api_key(self._get_config)
        self._timeout = float(self._get_config("llm.groq.timeout_seconds", 30.0))
        self._client: Optional["AsyncGroq"] = None

    async def is_available(self) -> bool:
        return bool(GROQ_SDK_AVAILABLE and self._api_key)

    def _get_client(self) -> "AsyncGroq":
        if not GROQ_SDK_AVAILABLE:
            raise ProviderError("groq package is not installed (pip install groq)")
        if not self._api_key:
            raise ProviderError("GROQ_API_KEY is not configured")
        if self._client is None:
            # max_retries=0: ModelRouter owns retry/backoff decisions, so the
            # SDK's own transparent retries would double up and hide latency.
            self._client = AsyncGroq(api_key=self._api_key, timeout=self._timeout, max_retries=0)
        return self._client

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        client = self._get_client()

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        start = time.monotonic()
        try:
            if on_token is not None:
                return await self._complete_streamed(client, kwargs, model, on_token, start)
            response = await client.chat.completions.create(**kwargs)
        except _GroqRateLimitError as exc:
            raise RateLimitError(
                f"Groq rate limit: {exc}", retry_after=_retry_after_seconds(exc)
            ) from exc
        except _GroqAPIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if status == 429:
                raise RateLimitError(
                    f"Groq rate limit ({status}): {exc}", retry_after=_retry_after_seconds(exc)
                ) from exc
            raise ProviderError(f"Groq API error ({status}): {exc}") from exc
        except (_GroqAPIConnectionError, _GroqAPITimeoutError) as exc:
            raise ProviderError(f"Groq network error: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive catch-all
            raise ProviderError(f"Groq request failed: {exc}") from exc
        latency_ms = (time.monotonic() - start) * 1000

        message = response.choices[0].message
        text = message.content or ""

        tool_calls: List[ToolCall] = []
        for tc in message.tool_calls or []:
            arguments = parse_tool_arguments(tc.function.arguments)
            tool_calls.append(ToolCall(name=tc.function.name, arguments=arguments, id=tc.id or ""))

        usage: Dict[str, int] = {}
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = {
                "prompt_tokens": getattr(raw_usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(raw_usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(raw_usage, "total_tokens", 0) or 0,
            }

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            raw=response,
            provider="groq",
            model=model,
            latency_ms=latency_ms,
            usage=usage,
        )

    async def _complete_streamed(
        self,
        client: "AsyncGroq",
        kwargs: Dict[str, Any],
        model: str,
        on_token: Callable[[str], None],
        start: float,
    ) -> LLMResponse:
        """
        Stream the completion, calling `on_token` per text delta as it
        arrives, while accumulating any tool_calls deltas in parallel —
        models occasionally emit a short preamble before deciding to call a
        tool, so this can't assume "streaming means no tool calls."
        Exceptions during the request or mid-stream propagate to the
        caller's except clauses unchanged (raised inside this same await).
        """
        # Unlike OpenAI's SDK, Groq's create() has no stream_options param —
        # usage simply isn't reported for streamed responses.
        stream = await client.chat.completions.create(**kwargs, stream=True)

        text_parts: List[str] = []
        tool_call_parts: Dict[int, Dict[str, str]] = {}
        usage: Dict[str, int] = {}
        last_response: Any = None

        async for chunk in stream:
            last_response = chunk
            raw_usage = getattr(chunk, "usage", None)
            if raw_usage is not None:
                usage = {
                    "prompt_tokens": getattr(raw_usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(raw_usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(raw_usage, "total_tokens", 0) or 0,
                }
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text_parts.append(delta.content)
                on_token(delta.content)
            for tc_delta in delta.tool_calls or []:
                slot = tool_call_parts.setdefault(tc_delta.index, {"id": "", "name": "", "arguments": ""})
                if tc_delta.id:
                    slot["id"] = tc_delta.id
                if tc_delta.function is not None:
                    if tc_delta.function.name:
                        slot["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        slot["arguments"] += tc_delta.function.arguments

        latency_ms = (time.monotonic() - start) * 1000

        tool_calls: List[ToolCall] = []
        for _, slot in sorted(tool_call_parts.items()):
            arguments = parse_tool_arguments(slot["arguments"]) if slot["arguments"] else {}
            tool_calls.append(ToolCall(name=slot["name"], arguments=arguments, id=slot["id"] or ""))

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            raw=last_response,
            provider="groq",
            model=model,
            latency_ms=latency_ms,
            usage=usage,
        )
