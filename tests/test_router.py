"""
Tests for llm/router.py (ModelRouter) and the provider registry.

Providers are mocked at the SDK-client boundary (groq.AsyncGroq /
ollama.AsyncClient) so the real GroqProvider/OllamaProvider translation
logic (exception mapping, tool-call normalization, JSON repair) is actually
exercised, without ever touching the network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from llm.errors import ProviderError, RateLimitError
from llm.json_repair import parse_tool_arguments
from llm.providers import register_provider, registered_providers
from llm.providers.base import LLMProvider
from llm.router import ModelRouter
from llm.types import LLMResponse, RouterError, ToolCall


# =============================================================================
# Shared fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def _groq_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """GroqProvider treats a missing key as ProviderError; keep it configured."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")


BASE_CONFIG: Dict[str, Any] = {
    "llm": {
        "primary": {"provider": "groq", "model": "openai/gpt-oss-120b"},
        "fallback": {"provider": "ollama", "model": "qwen3.5:latest"},
        "purposes": {
            "chat": {"primary": {"model": "openai/gpt-oss-20b"}},
        },
        "groq": {"timeout_seconds": 5},
        "ollama": {"endpoint": "http://127.0.0.1:11434", "timeout_seconds": 5},
    }
}


def _make_router(monkeypatch: pytest.MonkeyPatch, config: Optional[Dict[str, Any]] = None) -> ModelRouter:
    # Backoff sleeps would otherwise add real wall-clock delay to the 429 test.
    monkeypatch.setattr("llm.router.asyncio.sleep", AsyncMock())
    return ModelRouter(config=config or BASE_CONFIG)


# =============================================================================
# SDK response/error builders
# =============================================================================

def _groq_response(
    content: str = "",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    usage: Optional[Dict[str, int]] = None,
) -> SimpleNamespace:
    tc_objs = [
        SimpleNamespace(
            id=tc.get("id", "call_1"),
            function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]),
        )
        for tc in (tool_calls or [])
    ]
    message = SimpleNamespace(content=content, tool_calls=tc_objs or None)
    choice = SimpleNamespace(message=message)
    usage_obj = SimpleNamespace(**usage) if usage else None
    return SimpleNamespace(choices=[choice], usage=usage_obj)


def _ollama_response(
    content: str = "",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    prompt_eval_count: int = 0,
    eval_count: int = 0,
) -> SimpleNamespace:
    tc_objs = [
        SimpleNamespace(function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]))
        for tc in (tool_calls or [])
    ]
    message = SimpleNamespace(content=content, tool_calls=tc_objs or None)
    return SimpleNamespace(
        message=message,
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
    )


def _groq_rate_limit_error(message: str = "rate limited") -> Exception:
    import groq

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    return groq.RateLimitError(message, response=response, body=None)


def _groq_connection_error(message: str = "connection failed") -> Exception:
    import groq

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.APIConnectionError(message=message, request=request)


def _patch_groq_client(monkeypatch: pytest.MonkeyPatch, result: Any) -> AsyncMock:
    """`result`: a response to return, an exception to raise, or a list mixing both
    (consumed in order across successive calls, via AsyncMock's side_effect)."""
    fake_create = AsyncMock()
    if isinstance(result, BaseException) or isinstance(result, list):
        fake_create.side_effect = result
    else:
        fake_create.return_value = result

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr("llm.providers.groq_provider.AsyncGroq", MagicMock(return_value=fake_client))
    return fake_create


def _patch_ollama_client(monkeypatch: pytest.MonkeyPatch, result: Any) -> AsyncMock:
    fake_chat = AsyncMock()
    if isinstance(result, BaseException) or isinstance(result, list):
        fake_chat.side_effect = result
    else:
        fake_chat.return_value = result

    fake_client = SimpleNamespace(chat=fake_chat)
    monkeypatch.setattr("llm.providers.ollama_provider.OllamaAsyncClient", MagicMock(return_value=fake_client))
    return fake_chat


# =============================================================================
# Normal completion
# =============================================================================

@pytest.mark.asyncio
async def test_normal_completion_uses_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_groq_client(monkeypatch, _groq_response(content="Hello, Sir.", usage={
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
    }))
    router = _make_router(monkeypatch)

    result = await router.complete(messages=[{"role": "user", "content": "hi"}], purpose="planning")

    assert isinstance(result, LLMResponse)
    assert result.text == "Hello, Sir."
    assert result.provider == "groq"
    assert result.model == "openai/gpt-oss-120b"
    assert result.tool_calls == []
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert result.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_purpose_override_selects_smaller_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_create = _patch_groq_client(monkeypatch, _groq_response(content="ok"))
    router = _make_router(monkeypatch)

    result = await router.complete(messages=[{"role": "user", "content": "hi"}], purpose="chat")

    assert isinstance(result, LLMResponse)
    assert result.model == "openai/gpt-oss-20b"
    assert fake_create.call_args.kwargs["model"] == "openai/gpt-oss-20b"


# =============================================================================
# Tool-call parsing
# =============================================================================

@pytest.mark.asyncio
async def test_tool_call_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_groq_client(
        monkeypatch,
        _groq_response(
            content="",
            tool_calls=[{"id": "call_abc", "name": "get_weather", "arguments": '{"city": "SF"}'}],
        ),
    )
    router = _make_router(monkeypatch)

    result = await router.complete(
        messages=[{"role": "user", "content": "weather in SF?"}],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )

    assert isinstance(result, LLMResponse)
    assert result.tool_calls == [ToolCall(name="get_weather", arguments={"city": "SF"}, id="call_abc")]


# =============================================================================
# Malformed-JSON repair
# =============================================================================

def test_json_repair_trailing_comma() -> None:
    assert parse_tool_arguments('{"city": "SF", "unit": "F",}') == {"city": "SF", "unit": "F"}


def test_json_repair_single_quotes() -> None:
    assert parse_tool_arguments("{'city': 'SF'}") == {"city": "SF"}


def test_json_repair_passthrough_dict() -> None:
    assert parse_tool_arguments({"city": "SF"}) == {"city": "SF"}


def test_json_repair_gives_up_gracefully() -> None:
    assert parse_tool_arguments("this is not json at all") == {}
    assert parse_tool_arguments(None) == {}
    assert parse_tool_arguments("") == {}


@pytest.mark.asyncio
async def test_tool_call_with_malformed_arguments_is_repaired(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_groq_client(
        monkeypatch,
        _groq_response(
            content="",
            tool_calls=[{"id": "call_1", "name": "get_weather", "arguments": "{'city': 'SF',}"}],
        ),
    )
    router = _make_router(monkeypatch)

    result = await router.complete(messages=[{"role": "user", "content": "weather?"}])

    assert isinstance(result, LLMResponse)
    assert result.tool_calls[0].arguments == {"city": "SF"}


# =============================================================================
# 429 -> retry -> fallback
# =============================================================================

@pytest.mark.asyncio
async def test_rate_limit_retries_then_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_create = _patch_groq_client(
        monkeypatch,
        [_groq_rate_limit_error(), _groq_rate_limit_error()],
    )
    fake_chat = _patch_ollama_client(monkeypatch, _ollama_response(content="fallback answer"))
    sleep_mock = AsyncMock()
    monkeypatch.setattr("llm.router.asyncio.sleep", sleep_mock)

    router = ModelRouter(config=BASE_CONFIG)
    result = await router.complete(messages=[{"role": "user", "content": "hi"}])

    assert isinstance(result, LLMResponse)
    assert result.provider == "ollama"
    assert result.text == "fallback answer"
    # One retry: two attempts against Groq, one backoff sleep in between.
    assert fake_create.call_count == 2
    assert sleep_mock.call_count == 1
    assert fake_chat.call_count == 1


@pytest.mark.asyncio
async def test_network_failure_falls_back_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_create = _patch_groq_client(monkeypatch, _groq_connection_error())
    _patch_ollama_client(monkeypatch, _ollama_response(content="fallback answer"))
    router = _make_router(monkeypatch)

    result = await router.complete(messages=[{"role": "user", "content": "hi"}])

    assert isinstance(result, LLMResponse)
    assert result.provider == "ollama"
    # No retries for a plain network/API error -- only one attempt against Groq.
    assert fake_create.call_count == 1


# =============================================================================
# Both fail -> structured RouterError
# =============================================================================

@pytest.mark.asyncio
async def test_both_providers_fail_returns_router_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_groq_client(monkeypatch, _groq_connection_error("groq down"))
    _patch_ollama_client(monkeypatch, ConnectionError("ollama not running"))
    router = _make_router(monkeypatch)

    result = await router.complete(messages=[{"role": "user", "content": "hi"}])

    assert isinstance(result, RouterError)
    assert result.user_message == "Sir, I'm having trouble thinking right now."
    assert "groq down" in result.primary_error
    assert "ollama not running" in result.fallback_error
    assert result.purpose == "planning"


# =============================================================================
# Provider extensibility: registering a third provider requires no router changes
# =============================================================================

@pytest.mark.asyncio
async def test_dummy_third_provider_requires_no_router_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    @register_provider("dummy_test")
    class DummyProvider(LLMProvider):
        """A minimal third-party provider, registered entirely outside router.py."""

        async def complete(
            self,
            messages: List[Dict[str, str]],
            model: str,
            tools: Optional[List[Dict[str, Any]]] = None,
            tool_choice: str = "auto",
            temperature: float = 0.3,
        ) -> LLMResponse:
            return LLMResponse(
                text="dummy response",
                provider="dummy_test",
                model=model,
                latency_ms=1.0,
            )

    assert "dummy_test" in registered_providers()

    config = {
        "llm": {
            "primary": {"provider": "dummy_test", "model": "dummy-model-1"},
            "fallback": {"provider": "ollama", "model": "qwen3.5:latest"},
        }
    }
    router = _make_router(monkeypatch, config=config)

    result = await router.complete(messages=[{"role": "user", "content": "hi"}])

    assert isinstance(result, LLMResponse)
    assert result.provider == "dummy_test"
    assert result.text == "dummy response"


# =============================================================================
# Error-type sanity (providers must translate SDK errors correctly)
# =============================================================================

@pytest.mark.asyncio
async def test_groq_provider_translates_rate_limit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm.providers.groq_provider import GroqProvider

    _patch_groq_client(monkeypatch, _groq_rate_limit_error())
    provider = GroqProvider(config=BASE_CONFIG)

    with pytest.raises(RateLimitError):
        await provider.complete(messages=[{"role": "user", "content": "hi"}], model="openai/gpt-oss-120b")


@pytest.mark.asyncio
async def test_ollama_provider_translates_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm.providers.ollama_provider import OllamaProvider

    _patch_ollama_client(monkeypatch, ConnectionError("could not connect to ollama server"))
    provider = OllamaProvider(config=BASE_CONFIG)

    with pytest.raises(ProviderError):
        await provider.complete(messages=[{"role": "user", "content": "hi"}], model="qwen3.5:latest")


# =============================================================================
# Token streaming (on_token) — P08
# =============================================================================

async def _async_stream(chunks: List[Any]):
    for chunk in chunks:
        yield chunk


def _groq_stream_chunk(
    content: Optional[str] = None,
    tool_call_deltas: Optional[List[Any]] = None,
    usage: Optional[Dict[str, int]] = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, tool_calls=tool_call_deltas)
    choice = SimpleNamespace(delta=delta)
    usage_obj = SimpleNamespace(**usage) if usage else None
    return SimpleNamespace(choices=[choice], usage=usage_obj)


def _tool_call_delta(
    index: int, id_: Optional[str] = None, name: Optional[str] = None, arguments: Optional[str] = None
) -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments=arguments) if (name or arguments) else None
    return SimpleNamespace(index=index, id=id_, function=function)


def _patch_groq_stream(monkeypatch: pytest.MonkeyPatch, chunks: List[Any]) -> AsyncMock:
    fake_create = AsyncMock(return_value=_async_stream(chunks))
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr("llm.providers.groq_provider.AsyncGroq", MagicMock(return_value=fake_client))
    return fake_create


@pytest.mark.asyncio
async def test_streaming_calls_on_token_and_returns_full_text(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_create = _patch_groq_stream(
        monkeypatch,
        [
            _groq_stream_chunk(content="Hello"),
            _groq_stream_chunk(content=", Sir."),
            _groq_stream_chunk(usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}),
        ],
    )
    router = _make_router(monkeypatch)

    received: List[str] = []
    result = await router.complete(
        messages=[{"role": "user", "content": "hi"}], purpose="planning", on_token=received.append
    )

    assert isinstance(result, LLMResponse)
    assert result.text == "Hello, Sir."
    assert result.provider == "groq"
    assert received == ["Hello", ", Sir."]
    assert result.usage == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    assert fake_create.call_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_streaming_accumulates_tool_call_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_groq_stream(
        monkeypatch,
        [
            _groq_stream_chunk(
                tool_call_deltas=[_tool_call_delta(0, id_="call_1", name="get_weather", arguments='{"ci')]
            ),
            _groq_stream_chunk(tool_call_deltas=[_tool_call_delta(0, arguments='ty": "SF"}')]),
        ],
    )
    router = _make_router(monkeypatch)

    result = await router.complete(
        messages=[{"role": "user", "content": "weather?"}],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
        on_token=lambda _t: None,
    )

    assert isinstance(result, LLMResponse)
    assert result.tool_calls == [ToolCall(name="get_weather", arguments={"city": "SF"}, id="call_1")]


@pytest.mark.asyncio
async def test_on_token_ignored_for_provider_without_streaming_support(monkeypatch: pytest.MonkeyPatch) -> None:
    """A third-party provider predating on_token has no obligation to accept it
    (see test_dummy_third_provider_requires_no_router_changes) — the router
    must silently skip streaming for it rather than raising a TypeError."""

    @register_provider("dummy_no_stream")
    class DummyProvider(LLMProvider):
        async def complete(
            self,
            messages: List[Dict[str, str]],
            model: str,
            tools: Optional[List[Dict[str, Any]]] = None,
            tool_choice: str = "auto",
            temperature: float = 0.3,
        ) -> LLMResponse:
            return LLMResponse(text="no stream here", provider="dummy_no_stream", model=model, latency_ms=1.0)

    config = {
        "llm": {
            "primary": {"provider": "dummy_no_stream", "model": "m1"},
            "fallback": {"provider": "ollama", "model": "qwen3.5:latest"},
        }
    }
    router = _make_router(monkeypatch, config=config)

    received: List[str] = []
    result = await router.complete(messages=[{"role": "user", "content": "hi"}], on_token=received.append)

    assert isinstance(result, LLMResponse)
    assert result.text == "no stream here"
    assert received == []


# =============================================================================
# Provider health snapshot (/status) — P08
# =============================================================================

@pytest.mark.asyncio
async def test_provider_status_reports_tier_and_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _make_router(monkeypatch)  # groq primary (API key set via fixture), ollama fallback

    statuses = await router.provider_status(purpose="planning")

    assert [s["tier"] for s in statuses] == ["primary", "fallback"]
    assert statuses[0]["provider"] == "groq"
    assert statuses[0]["model"] == "openai/gpt-oss-120b"
    assert statuses[0]["available"] is True
    assert statuses[1]["provider"] == "ollama"
    assert statuses[1]["model"] == "qwen3.5:latest"


@pytest.mark.asyncio
async def test_provider_status_unconfigured_tier() -> None:
    router = ModelRouter(config={"llm": {"primary": {"provider": "groq", "model": "m1"}}})

    statuses = await router.provider_status(purpose="planning")

    fallback = statuses[1]
    assert fallback["tier"] == "fallback"
    assert fallback["available"] is False
    assert fallback["provider"] == "(not configured)"


# =============================================================================
# Ollama message normalization (P10) — a prior assistant tool-call message
# (built by Planner._assistant_tool_call_message in OpenAI/Groq's wire
# format, arguments as a JSON string) must not reach Ollama's SDK as-is:
# it validates function.arguments as a dict and raises otherwise. This is
# the exact bug that repeatedly broke the "Groq down -> falls back to
# Ollama" failure drill whenever the turn already involved a tool call.
# =============================================================================

def test_normalize_messages_for_ollama_converts_stringified_arguments() -> None:
    from llm.providers.ollama_provider import _normalize_messages_for_ollama

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_time", "arguments": '{"tz": "UTC"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "12:00 UTC"},
    ]

    normalized = _normalize_messages_for_ollama(messages)

    assert normalized[0] == messages[0]
    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == {"tz": "UTC"}
    assert normalized[2] == messages[2]
    # The original list/dicts must not be mutated in place.
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"tz": "UTC"}'


def test_normalize_messages_for_ollama_leaves_dict_arguments_untouched() -> None:
    from llm.providers.ollama_provider import _normalize_messages_for_ollama

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_time", "arguments": {"tz": "UTC"}}}
            ],
        },
    ]

    normalized = _normalize_messages_for_ollama(messages)

    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {"tz": "UTC"}


def test_normalize_messages_for_ollama_leaves_plain_messages_untouched() -> None:
    from llm.providers.ollama_provider import _normalize_messages_for_ollama

    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello, Sir."}]
    assert _normalize_messages_for_ollama(messages) == messages


@pytest.mark.asyncio
async def test_ollama_provider_normalizes_stringified_tool_call_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm.providers.ollama_provider import OllamaProvider

    fake_chat = _patch_ollama_client(monkeypatch, _ollama_response(content="ok"))
    provider = OllamaProvider(config=BASE_CONFIG)

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_time", "arguments": '{"tz": "UTC"}'}}
            ],
        },
    ]

    await provider.complete(messages=messages, model="qwen3.5:latest")

    sent_messages = fake_chat.call_args.kwargs["messages"]
    assert sent_messages[1]["tool_calls"][0]["function"]["arguments"] == {"tz": "UTC"}
