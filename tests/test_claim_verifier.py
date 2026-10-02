"""
Action-claim verifier (orchestrator/claim_verifier.py + Planner integration).

Origin: 2026-10-02 audit — "Remind me to call the dentist tomorrow at 9am" got
"Consider it noted, Sir" with no add_reminder call, and the phantom reminder then
resurfaced in a morning briefing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from bus.event_bus import EventBus
from guardian.gate import Guardian
from llm.types import LLMResponse, ToolCall
from orchestrator import claim_verifier as cv
from orchestrator.planner import Planner
from tools.registry import ToolRegistry, ToolSpec
from tracing.tracer import Tracer

DENTIST = "Remind me to call the dentist tomorrow at 9am"


@pytest.fixture(autouse=True)
def _fresh_bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


# ------------------------------------------------------------ detector units

@pytest.mark.parametrize("reply,user,category", [
    # the audit case: vague ack + user asked for a reminder
    ("Consider it noted, Sir: call the dentist tomorrow at 9 am.", DENTIST, "reminder"),
    ("Noted, Sir.", "Remind me to buy milk", "reminder"),
    ("Done, Sir.", "Set a reminder to stretch at 3", "reminder"),
    # explicit claims, user text irrelevant
    ("I've set a reminder for 9 am, Sir.", "what's up", "reminder"),
    ("Reminder set for tomorrow at nine.", "hm", "reminder"),
    ("I'll remind you at nine, Sir.", "hm", "reminder"),
    ("I have scheduled the meeting for Friday.", "hm", "calendar event"),
    ("The event has been created, Sir.", "hm", "calendar event"),
    ("I've drafted a reply to Alice, Sir.", "hm", "email draft"),
    ("Your draft is ready, Sir.", "hm", "email draft"),
    ("I've archived that email.", "hm", "email archive"),
    ("Archived, Sir.", "hm", "email archive"),
    ("I've marked it as read.", "hm", "mark read"),
    ("I've sent the email to Alice, Sir.", "hm", "message sent"),
    ("The message was sent.", "hm", "message sent"),
    ("I've saved a note about that.", "hm", "note saved"),
    ("Opened Safari, Sir.", "hm", "app opened"),
    ("I've launched Notes for you.", "hm", "app opened"),
    ("Closed Slack.", "hm", "app closed"),
    ("I've muted the volume.", "hm", "volume / brightness"),
    ("I've locked the screen, Sir.", "hm", "screen locked"),
    ("Committed a1b2c3d on main.", "hm", "git commit"),
    ("Consider it done, Sir.", "Please archive the newest email", "email archive"),
    ("All set, Sir.", "Open Safari for me", "app opened"),
])
def test_unbacked_claims_are_detected(reply, user, category):
    claim = cv.find_unbacked_claim(reply, user, succeeded_tools=[])
    assert claim is not None, reply
    assert claim.category == category


@pytest.mark.parametrize("reply,user,tool", [
    ("Consider it noted, Sir.", DENTIST, "add_reminder"),
    ("Consider it noted, Sir.", DENTIST, "create_event"),
    ("I've drafted a reply, Sir.", "reply to Bob", "draft_reply"),
    ("Archived, Sir.", "archive it", "archive"),
    ("Archived, Sir.", "archive it", "gmail_archive"),  # MCP collision rename
    ("Opened Safari, Sir.", "open safari", "open_app"),
    ("Closed Slack.", "close slack", "close_app"),
    ("I've muted it, Sir.", "mute", "mute"),
    ("Committed a1b2c3d on main.", "commit", "git_commit"),
])
def test_claims_backed_by_a_successful_tool_pass(reply, user, tool):
    assert cv.find_unbacked_claim(reply, user, succeeded_tools=[tool]) is None


def test_a_different_tool_does_not_back_the_claim():
    assert cv.find_unbacked_claim("Reminder set, Sir.", DENTIST, ["get_time"]) is not None
    assert cv.find_unbacked_claim("I've sent the email.", "hm", ["draft_reply"]) is not None  # draft != send


@pytest.mark.parametrize("reply,user", [
    # ordinary answers — must pass untouched
    ("It's 02:01 AM, Sir.", "what time is it"),
    ("Good morning, Sir. You have three meetings today; the first is at nine.", "brief me"),
    ("Your reminder to call the dentist is at 9 am tomorrow.", "what reminders do I have"),
    ("You have 4 unread emails, Sir; two look urgent.", "check mail"),
    ("The weather in Pune is 31 degrees and clear.", "weather"),
    # offers / questions / hedges are not claims
    ("Shall I set a reminder for that, Sir?", "the dentist is tomorrow"),
    ("I can open Safari if you like.", "hm"),
    ("Would you like me to archive it?", "hm"),
    ("The command you wish to run is: echo hi. Shall I proceed, Sir?", "run echo hi"),
    # failures and refusals are honest
    ("I couldn't open Safari, Sir.", "open safari"),
    ("I'm unable to send email, Sir — I have no send tool.", "email bob"),
    ("I didn't set the reminder; the tool failed.", DENTIST),
    ("Denied, Sir. Nothing was archived.", "archive it"),
    # vague ack with NO action request from the user is fine
    ("Noted, Sir.", "My name is Parr"),
    ("Understood, Sir. Done.", "thanks that's all"),
    ("Noted.", "I prefer jazz while coding"),
    ("", "hm"),
])
def test_normal_replies_pass_untouched(reply, user):
    assert cv.find_unbacked_claim(reply, user, succeeded_tools=[]) is None, reply


# ------------------------------------------------------- planner integration

def _planner(registry: ToolRegistry, responses: List[Any], tmp_path: Path, filtering: bool = True):
    bus = EventBus()
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=responses)
    guardian = Guardian(event_bus=bus, audit_log_path=tmp_path / "audit.jsonl")
    config = None if filtering else {"llm": {"tool_selection": {"enabled": False}}}
    planner = Planner(router=router, registry=registry, guardian=guardian, event_bus=bus, config=config,
                      tracer=Tracer(config={"tracing": {"enabled": False}}))
    return planner, router


def _registry(calls: List[Dict[str, Any]]) -> ToolRegistry:
    reg = ToolRegistry()

    async def add_reminder(arguments, context):
        calls.append(arguments)
        return "Reminder added"

    reg.register(ToolSpec(
        name="add_reminder", description="Add a reminder.",
        parameters={"type": "object", "properties": {"title": {"type": "string"}, "due": {"type": "string"}}},
        tier="safe", handler=add_reminder,
    ))
    reg.register(ToolSpec(name="get_time", description="time", tier="safe",
                          handler=AsyncMock(return_value="02:00")))
    return reg


@pytest.mark.asyncio
async def test_dentist_case_bare_confirmation_triggers_real_tool_call(tmp_path):
    calls: List[Dict[str, Any]] = []
    planner, router = _planner(_registry(calls), [
        LLMResponse(text="Consider it noted, Sir: call the dentist tomorrow at 9 am."),   # the bug
        LLMResponse(tool_calls=[ToolCall(name="add_reminder",
                                         arguments={"title": "call the dentist"}, id="1")]),
        LLMResponse(text="Reminder set, Sir."),
    ], tmp_path)
    statuses: List[str] = []
    result = await planner.run(user_text=DENTIST, on_status=statuses.append)

    assert calls == [{"title": "call the dentist"}]                       # a REAL call happened
    assert result.text == "Reminder set, Sir."
    assert result.tool_trace[0].startswith("claim_check:retry")
    assert result.tool_trace[-1] == "add_reminder:ok"
    assert router.complete.await_count == 3
    # the retry carried the corrective nudge and the FULL catalog
    retry_kwargs = router.complete.await_args_list[1].kwargs
    assert any(m["role"] == "system" and "no matching tool" in m["content"] for m in retry_kwargs["messages"])
    assert {t["function"]["name"] for t in retry_kwargs["tools"]} == {"add_reminder", "get_time"}
    assert cv.CORRECTION_NOTICE in statuses


@pytest.mark.asyncio
async def test_dentist_case_honest_refusal_is_accepted(tmp_path):
    calls: List[Dict[str, Any]] = []
    planner, router = _planner(_registry(calls), [
        LLMResponse(text="Consider it noted, Sir."),
        LLMResponse(text="I'm afraid I can't set reminders right now, Sir."),
    ], tmp_path)
    result = await planner.run(user_text=DENTIST)
    assert calls == []
    assert result.text == "I'm afraid I can't set reminders right now, Sir."
    assert router.complete.await_count == 2


@pytest.mark.asyncio
async def test_persistent_false_claim_is_replaced_never_spoken(tmp_path):
    calls: List[Dict[str, Any]] = []
    planner, router = _planner(_registry(calls), [
        LLMResponse(text="Consider it noted, Sir."),
        LLMResponse(text="Done, Sir. Your reminder is set."),
    ], tmp_path)
    streamed: List[str] = []
    result = await planner.run(user_text=DENTIST, on_token=streamed.append)
    assert result.text == cv.HONEST_REPLY
    assert "noted" not in result.text.lower() and "set" not in result.text.lower().replace("tell you", "")
    assert result.aborted is False
    assert result.tool_trace[-1].startswith("claim_check:blocked")
    assert streamed[-1] == cv.HONEST_REPLY
    assert router.complete.await_count == 2          # bounded: one retry only
    assert calls == []


@pytest.mark.asyncio
async def test_normal_answer_costs_no_extra_call(tmp_path):
    planner, router = _planner(_registry([]), [LLMResponse(text="It's two in the morning, Sir.")], tmp_path)
    result = await planner.run(user_text="what time is it")
    assert result.text == "It's two in the morning, Sir."
    assert router.complete.await_count == 1
    assert result.tool_trace == []


@pytest.mark.asyncio
async def test_claim_after_successful_tool_passes_through(tmp_path):
    calls: List[Dict[str, Any]] = []
    planner, router = _planner(_registry(calls), [
        LLMResponse(tool_calls=[ToolCall(name="add_reminder", arguments={"title": "x"}, id="1")]),
        LLMResponse(text="Reminder set for nine, Sir."),
    ], tmp_path, filtering=False)
    result = await planner.run(user_text=DENTIST)
    assert result.text == "Reminder set for nine, Sir."
    assert router.complete.await_count == 2 and len(calls) == 1
    assert not any(t.startswith("claim_check") for t in result.tool_trace)


@pytest.mark.asyncio
async def test_claim_after_failed_tool_is_still_unbacked(tmp_path):
    reg = ToolRegistry()

    async def broken(arguments, context):
        raise RuntimeError("EventKit denied")

    reg.register(ToolSpec(name="add_reminder", description="d", tier="safe", handler=broken))
    planner, router = _planner(reg, [
        LLMResponse(tool_calls=[ToolCall(name="add_reminder", arguments={"title": "x"}, id="1")]),
        LLMResponse(text="Reminder set, Sir."),                       # lie: the tool errored
        LLMResponse(text="That failed, Sir — the reminder was not created."),
    ], tmp_path, filtering=False)
    result = await planner.run(user_text=DENTIST)
    assert result.text == "That failed, Sir — the reminder was not created."
    assert "add_reminder:error" in result.tool_trace
