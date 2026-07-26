"""Gemini provider: registration + the OpenAI<->Gemini translation logic.

These are offline (no network / no API key) — they exercise the message, tool,
and response conversion that is the whole point of the adapter.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm.providers import get_provider_class
from llm.providers.gemini_provider import GENAI_AVAILABLE, GeminiProvider
from llm.types import LLMResponse

pytestmark = pytest.mark.skipif(not GENAI_AVAILABLE, reason="google-genai not installed")


def test_provider_is_registered():
    assert get_provider_class("gemini") is GeminiProvider


def test_sanitize_schema_strips_unsupported_keys():
    dirty = {
        "type": "object",
        "$schema": "http://json-schema.org/draft-07/schema#",
        "additionalProperties": False,
        "title": "X",
        "properties": {
            "q": {"type": "string", "description": "query", "default": "", "format": "uri"},
        },
        "required": ["q"],
    }
    clean = GeminiProvider._sanitize_schema(dirty)
    assert clean["type"] == "object"
    assert "$schema" not in clean and "additionalProperties" not in clean and "title" not in clean
    assert clean["properties"]["q"] == {"type": "string", "description": "query"}
    assert clean["required"] == ["q"]


def test_to_contents_splits_system_and_maps_roles():
    messages = [
        {"role": "system", "content": "You are Vesper."},
        {"role": "user", "content": "search my mail"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "list_unread", "arguments": '{"max_n": 5}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "[]"},
    ]
    system, contents = GeminiProvider._to_contents(messages)

    assert system == "You are Vesper."
    # user, model(function_call), user(function_response) — system is NOT a content
    assert [c.role for c in contents] == ["user", "model", "user"]

    # assistant tool call became a function_call part with parsed dict args
    fc = contents[1].parts[0].function_call
    assert fc.name == "list_unread"
    assert dict(fc.args) == {"max_n": 5}

    # tool result became a function_response matched back to the function name
    fr = contents[2].parts[0].function_response
    assert fr.name == "list_unread"


def test_to_tools_builds_function_declarations():
    tools = [
        {"type": "function", "function": {
            "name": "open_app",
            "description": "Open a macOS app",
            "parameters": {"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]},
        }},
        {"type": "function", "function": {
            "name": "get_time", "description": "Current time",
            "parameters": {"type": "object", "properties": {}},  # no params -> omitted
        }},
    ]
    gemini_tools = GeminiProvider._to_tools(tools)
    assert gemini_tools is not None and len(gemini_tools) == 1
    decls = gemini_tools[0].function_declarations
    names = {d.name for d in decls}
    assert names == {"open_app", "get_time"}


def test_parse_response_extracts_text_tool_calls_and_usage():
    fake = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[
            SimpleNamespace(text="On it, Sir.", function_call=None),
            SimpleNamespace(text=None, function_call=SimpleNamespace(name="open_app", args={"app_name": "Safari"})),
        ]))],
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=4, total_token_count=14),
    )
    result = GeminiProvider._parse_response(fake, model="gemini-2.0-flash", latency_ms=12.3)

    assert isinstance(result, LLMResponse)
    assert result.text == "On it, Sir."
    assert result.provider == "gemini"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "open_app"
    assert result.tool_calls[0].arguments == {"app_name": "Safari"}
    assert result.usage["total_tokens"] == 14
