"""gmail_create_draft: DRAFT only. Every test uses a mocked backend / fake Gmail service — no real draft is created,
no network is touched, nothing is delivered."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from bus.event_bus import EventBus
from guardian.gate import Guardian, SessionPolicy, VerdictType, channel_context, session_policy
from llm.types import LLMResponse, ToolCall
from orchestrator.planner import Planner
from schemas.events import ConfirmationRequestedEvent, ConfirmationResponseEvent
from tools import gmail_draft
from tools.gmail_draft import (
    MAX_BODY_CHARS, MAX_RECIPIENTS, MAX_SUBJECT_CHARS, Draft, DraftBackend, DraftRejected, DraftUnavailable,
    RegistryDraftBackend, confirm_summary, make_tool, render_confirmation, validate_draft)
from tools.registry import ToolRegistry, ToolSpec
from tracing.tracer import Tracer

ROOT = Path(__file__).resolve().parents[1]
GOOD = {"to": ["Priya <Priya@Example.com>"], "subject": "Re: launch plan", "body": "Thanks Priya, Thursday works.\nBest"}


# ------------------------------------------------------------------ validation

def test_valid_draft_is_normalised():
    d = validate_draft({**GOOD, "cc": "a@x.io; b@y.io", "thread_id": "18f3a_b-C"})
    assert d.to == ["priya@example.com"] and d.cc == ["a@x.io", "b@y.io"] and d.thread_id == "18f3a_b-C"
    assert d.subject == "Re: launch plan" and d.body.endswith("Best")


def test_recipients_are_deduplicated_and_counted_across_to_and_cc():
    d = validate_draft({**GOOD, "to": ["a@x.io", "A@x.io"], "cc": ["a@x.io", "c@x.io"]})
    assert d.to == ["a@x.io"] and d.cc == ["c@x.io"]
    many = [f"u{i}@x.io" for i in range(MAX_RECIPIENTS + 1)]
    with pytest.raises(DraftRejected, match="too many recipients"):
        validate_draft({**GOOD, "to": many})


@pytest.mark.parametrize("args,match", [
    ({**GOOD, "to": []}, "at least one recipient"),
    ({**GOOD, "to": ["not-an-address"]}, "not a valid email"),
    ({**GOOD, "to": ["a@x.io\r\nBcc: evil@x.io"]}, "line break"),
    ({**GOOD, "to": ["a@x.io, evil@x.io\nBcc: z@z.io"]}, "line break"),
    ({**GOOD, "subject": "hi\r\nBcc: evil@x.io"}, "line break"),
    ({**GOOD, "subject": ""}, "needs a subject"),
    ({**GOOD, "subject": "s" * (MAX_SUBJECT_CHARS + 1)}, "subject is too long"),
    ({**GOOD, "body": ""}, "needs a body"),
    ({**GOOD, "body": "b" * (MAX_BODY_CHARS + 1)}, "body is too long"),
    ({**GOOD, "thread_id": "../../etc/passwd"}, "thread id"),
    ({**GOOD, "to": 5}, "list of email addresses"),
])
def test_invalid_drafts_are_rejected_with_a_reason(args, match):
    with pytest.raises(DraftRejected, match=match):
        validate_draft(args)


def test_the_caps_are_inclusive():
    validate_draft({**GOOD, "subject": "s" * MAX_SUBJECT_CHARS, "body": "b" * MAX_BODY_CHARS})


def test_control_characters_are_stripped_from_the_body():
    assert validate_draft({**GOOD, "body": "hello\x00 wor\x07ld"}).body == "hello world"


# ------------------------------------------------------------------ the approval card

@pytest.mark.asyncio
async def test_the_card_shows_every_recipient_the_subject_and_the_whole_body():
    args = {"to": ["a@x.io", "b@y.io"], "cc": ["c@z.io"], "subject": "Plan", "body": "line one\nline two\n" + "w " * 200}
    card = await confirm_summary(args)
    for part in ("To: a@x.io, b@y.io", "Cc: c@z.io", "Subject: Plan", "line one\nline two", "Want me to save this as a draft?",
                 "nothing is delivered to anyone"):
        assert part in card
    assert card.split("Body:")[1].split().count("w") == 200         # the body is not truncated
    assert args["to"] == ["a@x.io", "b@y.io"]                       # arguments normalised in place


@pytest.mark.asyncio
async def test_an_invalid_draft_says_it_will_be_refused():
    args = {**GOOD, "to": ["nope"]}
    assert "will be refused" in await confirm_summary(args)


@pytest.mark.asyncio
async def test_what_is_approved_is_exactly_what_is_saved():
    backend = MagicMock(spec=DraftBackend)
    backend.create_draft = AsyncMock(return_value="d1")
    tool = make_tool(backend)
    args = {"to": "Priya <PRIYA@example.com>", "subject": "  Re:   plan ", "body": " ok\r\nthanks "}
    card = await tool.confirm_summary(args)
    await tool.handler(args, {})
    saved: Draft = backend.create_draft.call_args.args[0]
    assert saved.to == ["priya@example.com"] and saved.subject == "Re: plan" and saved.body == "ok\nthanks"
    assert "To: priya@example.com" in card and "Subject: Re: plan" in card


# ------------------------------------------------------------------ the tool

def test_the_tool_is_draft_tier_with_recipient_args():
    tool = make_tool(MagicMock(spec=DraftBackend))
    assert tool.tier == "draft" and tool.recipient_args == ("to", "cc") and tool.name == "gmail_create_draft"
    assert tool.untrusted_output is False and tool.outbound is False


@pytest.mark.asyncio
async def test_saving_reports_that_nothing_was_delivered():
    backend = MagicMock(spec=DraftBackend)
    backend.create_draft = AsyncMock(return_value="r-123")
    out = await make_tool(backend).handler(dict(GOOD), {})
    assert "Draft saved" in out and "r-123" in out and "NOT been delivered" in out
    backend.create_draft.assert_awaited_once()


@pytest.mark.asyncio
async def test_invalid_input_never_reaches_the_backend():
    backend = MagicMock(spec=DraftBackend)
    backend.create_draft = AsyncMock()
    out = await make_tool(backend).handler({**GOOD, "to": ["bad"]}, {})
    assert out.startswith("Error:") and not backend.create_draft.called


@pytest.mark.asyncio
async def test_gmail_not_connected_is_a_clear_error():
    reg = ToolRegistry()
    out = await make_tool(RegistryDraftBackend(reg)).handler(dict(GOOD), {})
    assert out.startswith("Error:") and "isn't connected" in out


@pytest.mark.asyncio
async def test_the_registry_backend_calls_only_the_hidden_create_draft_tool():
    calls = []

    async def hidden(arguments, context):
        calls.append(arguments)
        return "draft-42"

    reg = ToolRegistry()
    reg.register(ToolSpec(name="create_draft", description="d", tier="confirm", handler=hidden, enabled=False))
    out = await make_tool(RegistryDraftBackend(reg)).handler(dict(GOOD), {})
    assert "draft-42" in out and calls == [{"to": ["priya@example.com"], "subject": "Re: launch plan",
                                            "body": GOOD["body"]}]


def test_register_is_idempotent_and_the_hidden_tool_is_not_offered_to_the_planner():
    reg = ToolRegistry()
    a = gmail_draft.register(reg, MagicMock(spec=DraftBackend))
    assert gmail_draft.register(reg, MagicMock(spec=DraftBackend)) is a
    import yaml
    gm = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())["mcp"]["servers"]["gmail"]
    assert "create_draft" in gm["hidden_tools"] and gm["tiers"]["create_draft"] == "confirm"


# ------------------------------------------------------------------ hard guards: it cannot deliver mail

def test_the_backend_interface_has_exactly_one_method():
    methods = [m for m in dir(DraftBackend) if not m.startswith("_")]
    assert methods == ["create_draft"]
    mock = MagicMock(spec=DraftBackend)
    for verb in ("send", "send_message", "forward", "deliver"):
        with pytest.raises(AttributeError):
            getattr(mock, verb)


def test_the_tool_module_never_mentions_delivering_mail():
    """Fails the build if the verbs for delivering mail ever appear anywhere in the module — code, comments or docs."""
    text = (ROOT / "tools" / "gmail_draft.py").read_text()
    assert re.findall(r"\b(?:send|sends|sending|sent|forward\w*)\b", text, flags=re.IGNORECASE) == []


def _calls_and_attrs(path: Path):
    tree = ast.parse(path.read_text())
    return {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
           {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


@pytest.mark.parametrize("rel", ["mcp_servers/gmail/gmail_client.py", "mcp_servers/gmail/server.py", "tools/gmail_draft.py"])
def test_no_code_path_can_deliver_mail(rel):
    used = _calls_and_attrs(ROOT / rel)
    assert not {"send", "forward", "batchSend", "sendAs"} & used, rel


def _load_gmail_client(monkeypatch):
    """The Google client libraries live only in mcp_servers/.venv, so load the module against stand-ins for them."""
    import sys
    import types
    for name in ("google", "google.auth", "google.auth.exceptions", "google.auth.transport", "google.auth.transport.requests",
                 "google.oauth2", "google.oauth2.credentials", "google_auth_oauthlib", "google_auth_oauthlib.flow",
                 "googleapiclient", "googleapiclient.discovery"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["google.auth.exceptions"].RefreshError = type("RefreshError", (Exception,), {})
    sys.modules["google.auth.transport.requests"].Request = object
    sys.modules["google.oauth2.credentials"].Credentials = object
    sys.modules["google_auth_oauthlib.flow"].InstalledAppFlow = object
    sys.modules["googleapiclient.discovery"].build = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network"))
    spec = importlib.util.spec_from_file_location("gmail_client_under_test", ROOT / "mcp_servers" / "gmail" / "gmail_client.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeGmail:
    """A Gmail service that records the chain and refuses to know any delivery call."""

    def __init__(self):
        self.created = []

    def users(self):
        return self

    def drafts(self):
        return self

    def create(self, userId, body):
        self.created.append((userId, body))
        return SimpleNamespace(execute=lambda: {"id": "draft-777"})

    def __getattr__(self, name):
        raise AssertionError(f"unexpected Gmail call: {name}")


def test_gmail_client_create_draft_stores_a_draft_and_nothing_else(monkeypatch):
    client = _load_gmail_client(monkeypatch)
    fake = FakeGmail()
    monkeypatch.setattr(client, "_get_service_sync", lambda: fake)
    draft_id = client._create_draft_sync(["a@x.io"], ["c@x.io"], "Hello", "Body text", "thr1")
    assert draft_id == "draft-777"
    (user, body), = fake.created
    assert user == "me" and body["message"]["threadId"] == "thr1"
    import base64
    raw = base64.urlsafe_b64decode(body["message"]["raw"]).decode()
    assert "To: a@x.io" in raw and "Cc: c@x.io" in raw and "Subject: Hello" in raw and "Body text" in raw


# ------------------------------------------------------------------ planner + Guardian integration

def _planner(registry, bus, responses, tmp_path):
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=responses)
    guardian = Guardian(event_bus=bus, audit_log_path=tmp_path / "p.jsonl", confirmation_timeout_seconds=30)
    return Planner(router=router, registry=registry, guardian=guardian, event_bus=bus,
                   tracer=Tracer(config={"tracing": {"enabled": False}})), guardian


async def _until(pred, n=400):
    for _ in range(n):
        if pred():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached")


def _setup(tmp_path, call_args, backend):
    EventBus.reset_instance()
    bus = EventBus()
    reg = ToolRegistry()
    reg.register(make_tool(backend))
    planner, guardian = _planner(reg, bus, [
        LLMResponse(tool_calls=[ToolCall(name="gmail_create_draft", arguments=call_args, id="1")]),
        LLMResponse(text="Saved, Sir."),
    ], tmp_path)
    requested: List[ConfirmationRequestedEvent] = []

    async def on_req(e):
        requested.append(e)

    bus.subscribe(ConfirmationRequestedEvent, on_req)
    return bus, planner, guardian, requested


@pytest.mark.asyncio
async def test_a_recipient_the_user_never_said_raises_the_tier_and_nothing_is_saved_until_approved(tmp_path):
    backend = MagicMock(spec=DraftBackend)
    backend.create_draft = AsyncMock(return_value="d9")
    bus, planner, guardian, requested = _setup(tmp_path, dict(GOOD), backend)
    task = asyncio.create_task(planner.run(user_text="draft a reply to the one from Priya"))
    await _until(lambda: len(requested) == 1)
    assert not backend.create_draft.called
    assert "Raised from draft to confirm" in requested[0].summary and "To: priya@example.com" in requested[0].summary
    assert "Want me to save this as a draft?" in requested[0].summary
    await bus.emit(ConfirmationResponseEvent(request_id=requested[0].request_id, approved=True, source="voice_agent"))
    await task
    backend.create_draft.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_recipient_the_user_said_is_still_a_draft_tier_confirmation(tmp_path):
    backend = MagicMock(spec=DraftBackend)
    backend.create_draft = AsyncMock(return_value="d9")
    bus, planner, guardian, requested = _setup(tmp_path, dict(GOOD, to=["priya@example.com"]), backend)
    task = asyncio.create_task(planner.run(user_text="draft an email to priya@example.com about the launch plan"))
    await _until(lambda: len(requested) == 1)
    assert "Raised from" not in requested[0].summary and not backend.create_draft.called    # still needs a yes
    await bus.emit(ConfirmationResponseEvent(request_id=requested[0].request_id, approved=False, source="voice_agent"))
    await task
    assert not backend.create_draft.called


@pytest.mark.asyncio
async def test_a_denied_draft_is_never_saved(tmp_path):
    backend = MagicMock(spec=DraftBackend)
    backend.create_draft = AsyncMock()
    bus, planner, guardian, requested = _setup(tmp_path, dict(GOOD), backend)
    task = asyncio.create_task(planner.run(user_text="draft a reply to the one from Priya"))
    await _until(lambda: len(requested) == 1)
    await bus.emit(ConfirmationResponseEvent(request_id=requested[0].request_id, approved=False, source="cli"))
    await task
    assert not backend.create_draft.called


@pytest.mark.asyncio
async def test_saving_a_draft_requested_over_telegram_needs_the_mac(tmp_path):
    backend = MagicMock(spec=DraftBackend)
    backend.create_draft = AsyncMock(return_value="d9")
    bus, planner, guardian, requested = _setup(tmp_path, dict(GOOD), backend)

    async def turn():
        with channel_context("telegram", "111"), session_policy(SessionPolicy(name="telegram", allow_dangerous=False)):
            return await planner.run(user_text="draft a reply to the one from Priya")

    task = asyncio.create_task(turn())
    await _until(lambda: len(requested) == 1)
    assert requested[0].mac_only is True and requested[0].channel == "telegram"
    await bus.emit(ConfirmationResponseEvent(request_id=requested[0].request_id, approved=True, source="telegram:111"))
    await asyncio.sleep(0.05)
    assert not backend.create_draft.called and not task.done()                  # waiting for the Mac
    await bus.emit(ConfirmationResponseEvent(request_id=requested[0].request_id, approved=True, source="gateway"))
    await task
    backend.create_draft.assert_awaited_once()
    import json
    entries = [json.loads(l) for l in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert any(e["tool"] == "channel:approval_needs_mac" and e["channel"] == "telegram" for e in entries)
    saved = [e for e in entries if e["tool"] == "gmail_create_draft"][-1]
    assert saved["who_approved"] == "gateway" and saved["channel"] == "telegram" and saved["verdict"] == "allow"


@pytest.mark.asyncio
async def test_a_draft_cannot_be_raised_to_dangerous_by_a_remote_taint_it_becomes_confirm(tmp_path):
    EventBus.reset_instance()
    bus = EventBus()
    g = Guardian(event_bus=bus, audit_log_path=tmp_path / "a.jsonl")
    tool = make_tool(MagicMock(spec=DraftBackend))
    with channel_context("telegram", "111"), session_policy(SessionPolicy(name="telegram", allow_dangerous=False)):
        v = await g.check(tool, dict(GOOD), context={"tainted_input": True, "taint_reason": "email"})
    assert v.outcome == VerdictType.NEEDS_CONFIRMATION            # not DENY: draft -> confirm, never dangerous


def test_planner_recipient_check_is_exact_about_what_the_user_said():
    tool = make_tool(MagicMock(spec=DraftBackend))
    said = Planner._recipient_not_said
    assert said(tool, {"to": ["priya@example.com"]}, "draft something to priya@example.com") is False
    assert said(tool, {"to": ["priya@example.com"]}, "draft a reply to the one from priya") is True
    assert said(tool, {"to": ["a@x.io"], "cc": ["b@x.io"]}, "email a@x.io") is True        # cc not said
    assert said(ToolSpec(name="x", description="d", handler=AsyncMock()), {"to": ["z@z.io"]}, "") is False


# ------------------------------------------------------------------ the voice-friendly flow

def test_spoken_readback_is_short_and_ends_with_the_question():
    d = validate_draft({**GOOD, "cc": ["b@y.io"]})
    text = gmail_draft.spoken_readback(d)
    assert text.startswith("A draft to priya@example.com and 1 more, subject Re: launch plan,")
    assert text.endswith("Want me to save this as a draft?") and len(text.split()) < 30
    assert d.body not in text                                          # a readback, not the whole body


def test_the_persona_teaches_the_draft_flow_without_ever_claiming_a_mail_was_sent():
    persona = (ROOT / "config" / "persona.md").read_text()
    assert "gmail_create_draft" in persona and "Want me to save this as a draft?" in persona
    assert "approval on the Mac" in persona and "never claim a mail was sent" in persona


def test_a_drafted_reply_is_backed_by_the_draft_tool_for_the_claim_verifier():
    from orchestrator import claim_verifier as cv
    assert cv.find_unbacked_claim("I've drafted a reply, Sir.", "reply to Bob", ["gmail_create_draft"]) is None
    assert cv.find_unbacked_claim("I've sent the email.", "hm", ["gmail_create_draft"]) is not None     # draft != delivered
