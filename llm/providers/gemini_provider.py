"""Gemini provider — Google's generous free-tier LLM, wired into the same
ModelRouter contract as Groq/Ollama.

The one interesting part is translation: the rest of VESPER speaks the
OpenAI/Groq message+tool dialect (roles system/user/assistant/tool, tools as
``{"type":"function","function":{...}}``), while Gemini wants ``contents`` with
only user/model roles, a separate ``system_instruction``, ``function_call`` /
``function_response`` parts, and ``FunctionDeclaration`` tool schemas. This
module does that conversion in both directions and normalizes the reply back
into a plain ``LLMResponse``.

Streaming (``on_token``) is intentionally not implemented — the base contract
allows a provider to ignore it and return the full response, which keeps the
tool-call handling here simple and robust.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm.errors import ProviderError, RateLimitError
from llm.providers import register_provider
from llm.providers.base import LLMProvider
from llm.types import LLMResponse, ToolCall
from utils.api_keys import get_gemini_api_key
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types

    GENAI_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    genai = None  # type: ignore[assignment]
    genai_errors = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]
    GENAI_AVAILABLE = False

#: JSON-Schema keys Gemini's FunctionDeclaration accepts. Anything else
#: (additionalProperties, $schema, title, default, format, examples, ...) makes
#: the API reject the whole tool, so we strip them.
_ALLOWED_SCHEMA_KEYS = frozenset(
    {"type", "description", "properties", "items", "enum", "required", "nullable"}
)


@register_provider("gemini")
class GeminiProvider(LLMProvider):
    """Routes completions through Google's Gemini API (google-genai SDK)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._api_key = get_gemini_api_key(self._get_config)
        self._client = None

    async def is_available(self) -> bool:
        return bool(GENAI_AVAILABLE and self._api_key)

    def _get_client(self):
        if not GENAI_AVAILABLE:
            raise ProviderError("google-genai is not installed (pip install google-genai)")
        if not self._api_key:
            raise ProviderError("GEMINI_API_KEY is not configured")
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    # =========================================================================
    # OpenAI/Groq dialect  ->  Gemini
    # =========================================================================

    @staticmethod
    def _sanitize_schema(schema: Any) -> Any:
        """Recursively drop JSON-Schema keys Gemini rejects."""
        if not isinstance(schema, dict):
            return schema
        out: Dict[str, Any] = {}
        for key, value in schema.items():
            if key not in _ALLOWED_SCHEMA_KEYS:
                continue
            if key == "properties" and isinstance(value, dict):
                out[key] = {k: GeminiProvider._sanitize_schema(v) for k, v in value.items()}
            elif key == "items":
                out[key] = GeminiProvider._sanitize_schema(value)
            else:
                out[key] = value
        return out

    @classmethod
    def _to_contents(cls, messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Any]]:
        """Split OpenAI-style messages into (system_instruction, gemini_contents).

        Gemini only knows user/model roles: system text becomes a separate
        system_instruction, assistant tool_calls become model ``function_call``
        parts, and ``role:"tool"`` results become user ``function_response``
        parts (matched back to their function name by tool_call_id).
        """
        system_parts: List[str] = []
        contents: List[Any] = []
        id_to_name: Dict[str, str] = {}

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content") or ""

            if role == "system":
                if content:
                    system_parts.append(str(content))

            elif role == "user":
                contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=str(content))]))

            elif role == "assistant":
                parts: List[Any] = []
                if content:
                    parts.append(genai_types.Part(text=str(content)))
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {}) or {}
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments")
                    args = cls._coerce_args(raw_args)
                    parts.append(
                        genai_types.Part(function_call=genai_types.FunctionCall(name=name, args=args))
                    )
                    if tc.get("id"):
                        id_to_name[str(tc["id"])] = name
                if parts:
                    contents.append(genai_types.Content(role="model", parts=parts))

            elif role == "tool":
                name = id_to_name.get(str(msg.get("tool_call_id", "")), "") or msg.get("name", "") or "tool"
                contents.append(
                    genai_types.Content(
                        role="user",
                        parts=[genai_types.Part.from_function_response(name=name, response={"result": str(content)})],
                    )
                )

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return system_instruction, contents

    @staticmethod
    def _coerce_args(raw_args: Any) -> Dict[str, Any]:
        """Tool-call arguments arrive as a JSON string (OpenAI shape) or a dict."""
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed = json.loads(raw_args)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    @classmethod
    def _to_tools(cls, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Any]]:
        """OpenAI ``tools`` array -> a single Gemini Tool of FunctionDeclarations."""
        if not tools:
            return None
        declarations: List[Any] = []
        for tool in tools:
            fn = tool.get("function", {}) if tool.get("type") == "function" else tool
            name = fn.get("name")
            if not name:
                continue
            kwargs: Dict[str, Any] = {"name": name, "description": fn.get("description", "")}
            params = cls._sanitize_schema(fn.get("parameters"))
            # Only attach parameters when there is at least one real property;
            # Gemini rejects an empty object schema.
            if isinstance(params, dict) and params.get("properties"):
                kwargs["parameters"] = params
            declarations.append(genai_types.FunctionDeclaration(**kwargs))
        if not declarations:
            return None
        return [genai_types.Tool(function_declarations=declarations)]

    # =========================================================================
    # Gemini response  ->  normalized LLMResponse
    # =========================================================================

    @staticmethod
    def _parse_response(response: Any, model: str, latency_ms: float) -> LLMResponse:
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            for i, part in enumerate(getattr(content, "parts", None) or []):
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc is not None and getattr(fc, "name", None):
                    args = dict(fc.args) if getattr(fc, "args", None) else {}
                    tool_calls.append(ToolCall(name=fc.name, arguments=args, id=f"{fc.name}_{i}"))

        usage: Dict[str, int] = {}
        um = getattr(response, "usage_metadata", None)
        if um is not None:
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(um, "total_token_count", 0) or 0,
            }

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            raw=response,
            provider="gemini",
            model=model,
            latency_ms=latency_ms,
            usage=usage,
        )

    # =========================================================================
    # complete()
    # =========================================================================

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

        system_instruction, contents = self._to_contents(messages)
        gemini_tools = self._to_tools(tools)

        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            tools=gemini_tools,
        )

        start = time.monotonic()
        try:
            response = await client.aio.models.generate_content(
                model=model, contents=contents, config=config
            )
        except genai_errors.ClientError as exc:
            code = getattr(exc, "code", None)
            text = str(exc)
            if code == 429 or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower():
                raise RateLimitError(f"Gemini rate limit: {exc}") from exc
            raise ProviderError(f"Gemini API error ({code}): {exc}") from exc
        except genai_errors.ServerError as exc:
            raise ProviderError(f"Gemini server error: {exc}") from exc
        except (RateLimitError, ProviderError):
            raise
        except Exception as exc:  # pragma: no cover - defensive catch-all
            raise ProviderError(f"Gemini request failed: {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000
        return self._parse_response(response, model, latency_ms)
