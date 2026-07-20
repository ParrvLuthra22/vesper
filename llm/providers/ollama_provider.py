"""Ollama provider — local chat completions with tool calling via the `ollama` Python lib."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from llm.errors import ProviderError, RateLimitError
from llm.json_repair import parse_tool_arguments
from llm.providers import register_provider
from llm.providers.base import LLMProvider
from llm.types import LLMResponse, ToolCall
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import ollama as _ollama_sdk
    from ollama import AsyncClient as OllamaAsyncClient

    OLLAMA_SDK_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _ollama_sdk = None  # type: ignore[assignment]
    OllamaAsyncClient = None  # type: ignore[assignment]
    OLLAMA_SDK_AVAILABLE = False


@register_provider("ollama")
class OllamaProvider(LLMProvider):
    """Routes completions through a local (or remote) Ollama server."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._host = self._get_config("llm.ollama.endpoint", "http://127.0.0.1:11434")
        self._timeout = float(self._get_config("llm.ollama.timeout_seconds", 60.0))
        self._client: Optional["OllamaAsyncClient"] = None

    async def is_available(self) -> bool:
        return bool(OLLAMA_SDK_AVAILABLE)

    def _get_client(self) -> "OllamaAsyncClient":
        if not OLLAMA_SDK_AVAILABLE:
            raise ProviderError("ollama package is not installed (pip install ollama)")
        if self._client is None:
            self._client = OllamaAsyncClient(host=self._host, timeout=self._timeout)
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
        # Streaming isn't implemented for this provider yet; on_token is
        # accepted for interface parity with GroqProvider and simply unused
        # — the router/caller still gets a complete LLMResponse as normal.
        client = self._get_client()

        if tools and tool_choice not in ("auto", None):
            # Ollama's chat API has no concept of forcing/forbidding a specific
            # tool — it always lets the model decide from the provided list.
            logger.debug(
                f"Ollama provider ignoring tool_choice={tool_choice!r} "
                "(unsupported by the Ollama tools API; behaves as 'auto')"
            )

        start = time.monotonic()
        try:
            response = await client.chat(
                model=model,
                messages=messages,
                tools=tools or None,
                options={"temperature": temperature},
            )
        except _ollama_sdk.ResponseError as exc:  # type: ignore[union-attr]
            status = getattr(exc, "status_code", None)
            if status == 429:
                raise RateLimitError(f"Ollama rate limit ({status}): {exc}") from exc
            raise ProviderError(f"Ollama API error ({status}): {exc}") from exc
        except _ollama_sdk.RequestError as exc:  # type: ignore[union-attr]
            raise ProviderError(f"Ollama request error: {exc}") from exc
        except Exception as exc:
            # Covers connection failures (server not running, DNS, etc.) from
            # the underlying httpx client.
            raise ProviderError(f"Ollama network error: {exc}") from exc
        latency_ms = (time.monotonic() - start) * 1000

        message = response.message
        text = message.content or ""

        tool_calls: List[ToolCall] = []
        for tc in message.tool_calls or []:
            arguments = parse_tool_arguments(tc.function.arguments)
            # Ollama does not assign tool-call ids; synthesize one so callers
            # have a stable handle to match tool results back later.
            tool_calls.append(ToolCall(name=tc.function.name, arguments=arguments, id=str(uuid4())))

        usage = {
            "prompt_tokens": response.prompt_eval_count or 0,
            "completion_tokens": response.eval_count or 0,
            "total_tokens": (response.prompt_eval_count or 0) + (response.eval_count or 0),
        }

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            raw=response,
            provider="ollama",
            model=model,
            latency_ms=latency_ms,
            usage=usage,
        )
