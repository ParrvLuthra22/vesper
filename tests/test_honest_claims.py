"""Claim grounding: a reply may only describe what the tool results support. open_app alone never supports "a conversation window for
<person>, ready for you to type". Fakes only."""
from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import AsyncMock

import pytest

from agents.system_agent import AppleScriptExecutor
from bus.event_bus import EventBus
from guardian.gate import Guardian
from llm.types import LLMResponse, ToolCall
from orchestrator import claim_verifier as cv
from orchestrator.planner import Planner
from tools.registry import ToolRegistry, ToolSpec
from tracing.tracer import Tracer
from utils.grounding import grounded, parse, say

LIE = "iMessage is now open with a conversation window for Priya, ready for you to type."
OPEN_OK = grounded("Opening Messages", can=["app_opened", "app_frontmost"], cannot=["conversation", "window", "page"], frontmost="yes")
OPEN_BACK = grounded("Opening Messages", can=["app_opened"], cannot=["conversation", "window", "page"], frontmost="no")


@pytest.fixture(autouse=True)
def _bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


# ------------------------------------------------------------------ grounding format

def test_grounded_results_round_trip():
    g = parse(OPEN_OK)
    assert g.present and g.frontmost == "yes" and g.can == {"app_opened", "app_frontmost"} and "conversation" in g.cannot
    assert g.app == "Messages"
    assert not parse("Opened Safari").present


# ------------------------------------------------------------------ frontmost check

class Exec(AppleScriptExecutor):
    def __init__(self, fronts):
        self.fronts = list(fronts)
        self.slept = []

    def frontmost_application(self, timeout=5.0):
        return self.fronts.pop(0) if self.fronts else self.last

    last = None


def test_frontmost_yes_after_activation_settles():
    ex = Exec(["Terminal", "Terminal", "Messages"])
    assert ex.verify_frontmost("Messages", sleep=ex.slept.append) == "yes" and len(ex.slept) == 2


def test_frontmost_no_when_another_app_stays_in_front():
    ex = Exec(["Terminal"] * 4)
    ex.last = "Terminal"
    assert ex.verify_frontmost("Safari", sleep=lambda s: None) == "no"


def test_frontmost_unknown_when_system_events_cannot_be_read():
    assert Exec([None]).verify_frontmost("Safari", sleep=lambda s: None) == "unknown"


def test_frontmost_matches_app_bundle_names_case_insensitively():
    assert Exec(["zoom.us"]).verify_frontmost("Zoom.us.app", sleep=lambda s: None) == "yes"


@pytest.mark.parametrize("front,expect_can,expect_line", [("yes", "app_frontmost", "frontmost: yes"), ("no", None, "frontmost: no")])
def test_open_app_handler_reports_frontmost_in_the_result(front, expect_can, expect_line):
    from agents.system_agent import SystemAgent

    agent = SystemAgent.__new__(SystemAgent)
    agent._logger = type("L", (), {"info": lambda *a, **k: None})()
    agent._event_bus = type("B", (), {"emit": AsyncMock()})()
    agent._name = "SystemAgent"
    agent.executor = type("E", (), {"open_application": lambda self, n: (True, "ok"),
                                    "verify_frontmost": lambda self, n: front})()

    async def go():
        return agent._handle_open_app({"app_name": "Messages"}, "")

    out = asyncio.run(go())
    g = parse(out)
    assert out.startswith("Opening Messages") and expect_line in out and "app_opened" in g.can
    assert ("app_frontmost" in g.can) is (front == "yes")
    assert "conversation" in g.cannot


# ------------------------------------------------------------------ verifier

def test_open_app_alone_does_not_back_a_conversation_window_claim():
    claim = cv.find_unbacked_claim(LIE, "message Priya", ["open_app"], results=[("open_app", OPEN_OK)])
    assert claim is not None and claim.kind == "specific"


@pytest.mark.parametrize("sentence", [
    "iMessage is now open with a conversation window for Priya.", "The conversation with Priya is open.",
    "I've started a chat with Priya.", "It's ready for you to type, Sir.", "Messages is open, ready for you to type your message.",
    "I've typed your message into the message box.",
])
def test_specific_claims_without_support_are_flagged(sentence):
    assert cv.find_unbacked_claim(sentence, "message Priya", ["open_app"], results=[("open_app", OPEN_OK)]).kind == "specific"


@pytest.mark.parametrize("sentence", [
    "I opened Messages.", "Opening Messages, Sir.", "Shall I open a conversation with Priya?", "I can't open a specific conversation.",
    "I could open a conversation for you if you like.", "Your conversation with Priya has three unread messages.",
])
def test_plain_or_hedged_sentences_pass(sentence):
    assert cv.find_unbacked_claim(sentence, "open messages", ["open_app"], results=[("open_app", OPEN_OK)]) is None


def test_backed_claims_pass():
    compose = grounded("Opened a Messages conversation (nothing sent)", can=["app_opened", "conversation_opened", "compose_ready"])
    assert cv.find_unbacked_claim(LIE, "message Priya", ["messages_compose"], results=[("messages_compose", compose)]) is None
    assert cv.find_unbacked_claim("Messages is in front.", "open messages", ["open_app"], results=[("open_app", OPEN_OK)]) is None


def test_in_front_claim_is_flagged_when_frontmost_was_no():
    c = cv.find_unbacked_claim("Messages is now in front.", "open messages", ["open_app"], results=[("open_app", OPEN_BACK)])
    assert c is not None and c.kind == "specific"


def test_honest_reply_is_built_from_the_tool_results():
    claim = cv.find_unbacked_claim(LIE, "message Priya", ["open_app"], results=[("open_app", OPEN_OK)])
    assert cv.honest_reply(claim, [("open_app", OPEN_OK)]) == "I opened Messages, but I can't open a specific conversation."
    claim2 = cv.find_unbacked_claim(LIE, "message Priya", ["open_app"], results=[("open_app", OPEN_BACK)])
    assert cv.honest_reply(claim2, [("open_app", OPEN_BACK)]) == \
        "I opened Messages, though it isn't in front, but I can't open a specific conversation."
    assert cv.honest_reply(claim, []) == cv.HONEST_REPLY
    assert "Priya" not in cv.honest_reply(claim, [("open_app", OPEN_OK)])


def test_say_reports_an_unknown_frontmost_state():
    unk = grounded("Opening Notes", can=["app_opened"], frontmost="unknown")
    assert "couldn't confirm" in say(parse(unk))


# ------------------------------------------------------------------ in the planner

def planner(responses, results):
    reg = ToolRegistry()

    async def open_app(a, c):
        return results

    reg.register(ToolSpec(name="open_app", description="d", tier="safe", handler=open_app))
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=responses)
    bus = EventBus()
    return Planner(router=router, registry=reg, guardian=Guardian(event_bus=bus), event_bus=bus,
                   tracer=Tracer(config={"tracing": {"enabled": False}})), router


@pytest.mark.asyncio
async def test_planner_rewrites_a_reply_claiming_a_conversation_after_open_app_only():
    p, router = planner([LLMResponse(tool_calls=[ToolCall(name="open_app", arguments={"app_name": "Messages"}, id="1")]),
                         LLMResponse(text=LIE)], OPEN_OK)
    streamed: List[str] = []
    out = await p.run(user_text="message Priya", on_token=streamed.append)
    assert out.text == "I opened Messages, but I can't open a specific conversation."
    assert streamed[-1] == out.text and "claim_check:specific:conversation" in out.tool_trace
    assert router.complete.call_count == 2                                  # no retry spent on it


@pytest.mark.asyncio
async def test_planner_lets_an_honest_reply_through_untouched():
    p, _ = planner([LLMResponse(tool_calls=[ToolCall(name="open_app", arguments={"app_name": "Messages"}, id="1")]),
                    LLMResponse(text="I opened Messages, Sir, but I can't open a conversation with Priya.")], OPEN_OK)
    out = await p.run(user_text="message Priya")
    assert out.text.startswith("I opened Messages, Sir")
