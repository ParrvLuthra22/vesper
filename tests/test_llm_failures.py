"""LLM failures are explained by CLASS in one sentence (rate limit, timeout, network, auth); Retry-After is honoured with one retry up to
20 s; a reply that was half-streamed when the call failed is discarded. Mocked providers; no network, no real sleeping."""
from __future__ import annotations

import asyncio
from typing import List, Optional

import pytest

from llm import router as router_mod
from llm.errors import ProviderError, RateLimitError
from llm.router import STREAM_RESET, ModelRouter, RouterError, explain_failure
from llm.types import LLMResponse


class Provider:
    def __init__(self, script):
        self.script, self.calls = list(script), 0

    async def complete(self, messages, model, tools=None, tool_choice="auto", temperature=0.3, on_token=None):
        self.calls += 1
        step = self.script.pop(0)
        if callable(step):
            return step(on_token)
        if isinstance(step, BaseException):
            raise step
        return step


def ok(text="fine"):
    return LLMResponse(text=text, provider="p", model="m", usage={"total_tokens": 10})


def make(primary: Provider, fallback: Optional[Provider] = None, **llm):
    r = ModelRouter(config={"llm": {"primary": {"provider": "p", "model": "m"},
                                    "fallback": {"provider": "f", "model": "m", "purposes": ["reflection"]}, **llm}})
    r._provider_instances = {"p": primary, "f": fallback or Provider([ProviderError("x")])}

    async def no_pace(*a, **k):
        return None

    r._pace = no_pace
    return r


@pytest.fixture
def sleeps(monkeypatch):
    waited: List[float] = []

    async def fake_sleep(s):
        waited.append(s)

    monkeypatch.setattr(router_mod.asyncio, "sleep", fake_sleep)
    return waited


async def run(r, on_token=None, on_status=None):
    return await r.complete(messages=[{"role": "user", "content": "hi"}], on_token=on_token, on_status=on_status)


@pytest.mark.asyncio
@pytest.mark.parametrize("exc,expect", [
    (ProviderError("Groq timeout: x", kind="timeout"), "took too long"),
    (ProviderError("Groq network error: x", kind="network"), "can't reach the language model"),
    (ProviderError("GROQ_API_KEY is not configured", kind="auth"), "rejected my API key"),
    (ProviderError("Groq API error (401): bad", kind="auth"), "rejected my API key"),
    (ProviderError("Groq API error (503)", kind="server"), "server error"),
    (ProviderError("???"), "trouble thinking"),
])
async def test_each_failure_class_gets_its_own_one_sentence(exc, expect, sleeps):
    out = await run(make(Provider([exc])))
    assert isinstance(out, RouterError) and expect in out.user_message
    assert out.user_message.count(".") <= 2 and "\n" not in out.user_message


@pytest.mark.asyncio
async def test_rate_limit_waits_the_retry_after_once_and_succeeds(sleeps):
    p = Provider([RateLimitError("429", retry_after=19.0), ok("hello")])
    out = await run(make(p))
    assert out.text == "hello" and p.calls == 2 and sleeps == [19.0]


@pytest.mark.asyncio
async def test_retry_after_beyond_20_seconds_is_not_waited_out_and_says_how_long(sleeps):
    p = Provider([RateLimitError("429", retry_after=45.0)])
    out = await run(make(p))
    assert isinstance(out, RouterError) and sleeps == [] and p.calls == 1
    assert "rate limit" in out.user_message and "about 45 seconds" in out.user_message


@pytest.mark.asyncio
async def test_only_one_retry_even_if_the_second_attempt_is_limited_again(sleeps):
    p = Provider([RateLimitError("429", retry_after=2.0), RateLimitError("429", retry_after=2.0), ok()])
    out = await run(make(p))
    assert isinstance(out, RouterError) and p.calls == 2 and sleeps == [2.0] and "rate limit" in out.user_message


@pytest.mark.asyncio
async def test_without_retry_after_the_short_backoff_is_used(sleeps):
    p = Provider([RateLimitError("429"), ok()])
    assert (await run(make(p))).text == "fine" and sleeps == [ModelRouter.BACKOFF_BASE_SECONDS]


@pytest.mark.asyncio
async def test_the_20s_ceiling_is_configurable(sleeps):
    p = Provider([RateLimitError("429", retry_after=30.0), ok()])
    assert (await run(make(p, rate_limit={"retry_after_max_seconds": 40}))).text == "fine"


@pytest.mark.asyncio
async def test_a_status_notice_is_sent_while_waiting(sleeps):
    seen: List[str] = []
    await run(make(Provider([RateLimitError("429", retry_after=3.0), ok()])), on_status=seen.append)
    assert seen == [router_mod.RATE_LIMIT_WAIT_MESSAGE]


@pytest.mark.asyncio
async def test_timeout_mid_stream_discards_the_partial_reply(sleeps):
    shown: List[str] = []

    def half(on_token):
        on_token("I was saying that"); on_token(" the")
        raise ProviderError("Groq timeout", kind="timeout")

    out = await run(make(Provider([half])), on_token=shown.append)
    assert isinstance(out, RouterError) and "took too long" in out.user_message
    assert shown == ["I was saying that", " the", STREAM_RESET]                    # the consumer is told to throw the text away


@pytest.mark.asyncio
async def test_rate_limit_mid_stream_discards_then_the_retry_streams_a_whole_reply(sleeps):
    shown: List[str] = []

    def half(on_token):
        on_token("Partial")
        raise RateLimitError("429", retry_after=1.0)

    def whole(on_token):
        on_token("Complete answer.")
        return ok("Complete answer.")

    out = await run(make(Provider([half, whole])), on_token=shown.append)
    assert out.text == "Complete answer." and shown == ["Partial", STREAM_RESET, "Complete answer."]


@pytest.mark.asyncio
async def test_no_reset_is_sent_when_nothing_was_streamed(sleeps):
    shown: List[str] = []
    out = await run(make(Provider([ProviderError("x", kind="network")])), on_token=shown.append)
    assert isinstance(out, RouterError) and shown == []


def test_explain_failure_wording():
    assert explain_failure("rate_limit", 7.4) == "I've hit the language model's rate limit, Sir. Try again in about 7 seconds."
    assert "Try again in a minute" in explain_failure("rate_limit", None)


def test_groq_provider_classifies_its_errors():
    import inspect
    from llm.providers import groq_provider
    src = inspect.getsource(groq_provider)
    assert 'kind="timeout"' in src and 'kind="network"' in src and 'kind="auth"' in src


def test_the_cli_erases_a_discarded_stream(capsys):
    from rich.console import Console
    from unittest.mock import AsyncMock, MagicMock
    from cli.app import VesperCLI

    class Brain:
        async def handle_user_text(self, text, on_token=None, on_status=None):
            on_token("Half a sen"); on_token(STREAM_RESET); on_token("Whole.")
            return type("R", (), {"text": "Whole."})()

    console = Console(record=True, width=100)
    cli = VesperCLI.__new__(VesperCLI)
    cli._brain, cli._console, cli._vesper_style, cli._suppress_voice_output = Brain(), console, "", False
    asyncio.run(cli._run_turn("x"))
    out = console.export_text()
    assert "reply interrupted" in out and "Whole." in out
