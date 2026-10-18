"""contacts_lookup (read-only, fuzzy, asks when ambiguous) and messages_compose (confirm tier, opens the conversation, NEVER sends).
Fakes only: no Contacts, no Messages, nothing opened."""
from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path
from typing import List, Optional

import pytest

from launcher import doctor
from tools import contacts_tools as ct
from tools.contacts_tools import Contact, ContactRefs, Handle
from utils.grounding import parse

PEOPLE = [
    Contact("Priya Sharma", "Priya", "Sharma", "", "", (Handle("phone", "+91 98765 43210", "mobile"), Handle("email", "priya@corp.example"))),
    Contact("Priya Nair", "Priya", "Nair", "", "", (Handle("phone", "(555) 010-2000", "work"),)),
    Contact("Rahul Verma", "Rahul", "Verma", "Rahi", "", (Handle("email", "rahul@corp.example"),)),
    Contact("Katherine Moss", "Katherine", "Moss", "", "", (Handle("phone", "+44 20 7946 0958", "mobile"),)),
    Contact("Dmitri Volkov", "Dmitri", "Volkov", "", "", (Handle("phone", "+7 495 123 45 67", ""),)),
]


class Fake:
    def __init__(self, people=PEOPLE, auth=3, boom=None):
        self.people, self.auth, self.boom = people, auth, boom

    def authorization(self):
        return self.auth

    def contacts(self):
        if self.boom:
            raise self.boom
        return self.people


def run(coro):
    return asyncio.run(coro)


def lookup(name, refs=None, **kw):
    refs = refs or ContactRefs()
    return run(ct.make_lookup(Fake(**kw), refs)({"name": name}, {})), refs


# ------------------------------------------------------------------ matching

@pytest.mark.parametrize("spoken,expect", [
    ("Rahul", "Rahul Verma"), ("rahul verma", "Rahul Verma"), ("Rahi", "Rahul Verma"), ("Katharine", "Katherine Moss"),
    ("Catherine", "Katherine Moss"), ("Dmitry", "Dmitri Volkov"), ("Rahool", "Rahul Verma"),
])
def test_single_match_including_misheard_names(spoken, expect):
    out, refs = lookup(spoken)
    assert out.startswith("1 contact matches.") and expect in out


def test_several_matches_ask_the_user_to_choose():
    out, _ = lookup("Priya")
    assert out.startswith("2 contacts match.") and "Priya Sharma" in out and "Priya Nair" in out and "Ask the user which one" in out


def test_no_match_says_so_and_does_not_guess():
    out, _ = lookup("Zebediah")
    assert out.startswith("No contact matches") and "do not guess" in out


def test_the_model_never_sees_a_phone_number_or_address():
    out, _ = lookup("Priya")
    assert not re.search(r"\d{4,}", out) and "@" not in out and "98765" not in out and "corp.example" not in out


def test_permission_not_granted_is_reported_not_faked():
    out, _ = lookup("Priya", auth=2)
    assert "not granted" in out and "Privacy & Security > Contacts" in out and "Priya Sharma" not in out


def test_backend_failure_is_a_sentence_not_a_guess():
    out, _ = lookup("Priya", boom=PermissionError("denied"))
    assert "could not be read (PermissionError)" in out


def test_lookup_is_marked_private_and_compose_is_confirm_tier():
    import tools  # noqa: F401
    from tools.registry import get_registry
    reg = get_registry()
    assert reg.get("contacts_lookup").private and reg.get("contacts_lookup").tier == "safe" and not reg.get("contacts_lookup").outbound
    assert reg.get("messages_compose").private and reg.get("messages_compose").tier == "confirm"


# ------------------------------------------------------------------ compose

def compose(arguments, refs, front="yes", ok=True):
    opened: List[str] = []
    fn = ct.make_compose(Fake(), refs, opener=lambda url: (opened.append(url) or (ok, "")), frontmost=lambda: front)
    return run(fn(arguments, {})), opened


def test_compose_opens_an_imessage_url_to_the_resolved_handle_and_never_sends():
    out, refs = lookup("Rahul")
    ref = re.search(r"(c\d+):", out).group(1)
    res, opened = compose({"contact_ref": ref}, refs)
    assert opened == ["imessage://rahul@corp.example"]
    g = parse(res)
    assert {"app_opened", "conversation_opened", "compose_ready", "app_frontmost"} <= g.can and "sent" in g.cannot
    assert "NOTHING WAS SENT" in res


def test_phone_is_preferred_and_mobile_first_and_normalised():
    out, refs = lookup("rahul")
    out2, refs2 = lookup("Priya Sharma")
    ref = re.search(r"(c\d+):", out2).group(1)
    _res, opened = compose({"contact_ref": ref}, refs2)
    assert opened == ["imessage://+919876543210"]


def test_body_is_prefilled_url_encoded_and_sms_service_works():
    _o, refs = lookup("Dmitri")
    ref = "c1"
    res, opened = compose({"contact_ref": ref, "body": "Running 10 min late & sorry!"}, refs)
    assert opened == ["imessage://+74951234567?body=Running%2010%20min%20late%20%26%20sorry%21"]
    assert "body_prefilled" in parse(res).can
    _res, opened = compose({"contact_ref": ref, "body": "hi", "service": "sms"}, refs)
    assert opened == ["sms:+74951234567&body=hi"]


def test_not_in_front_means_no_conversation_claim():
    _o, refs = lookup("Dmitri")
    res, _ = compose({"contact_ref": "c1"}, refs, front="no")
    g = parse(res)
    assert "conversation_opened" not in g.can and "compose_ready" not in g.can and {"conversation", "sent"} <= g.cannot


def test_open_failure_claims_nothing():
    _o, refs = lookup("Dmitri")
    res, _ = compose({"contact_ref": "c1"}, refs, ok=False)
    assert parse(res).can == frozenset() and "could not be opened" in res


def test_unknown_or_expired_ref_is_refused():
    t = [0.0]
    refs = ContactRefs(clock=lambda: t[0])
    ref = refs.add(PEOPLE[0])
    t[0] = ct.REF_TTL_S + 1
    res, opened = compose({"contact_ref": ref}, refs)
    assert opened == [] and "unknown or expired" in res
    assert compose({"contact_ref": "c99"}, ContactRefs())[1] == []


def test_a_contact_without_a_usable_handle_is_refused():
    refs = ContactRefs()
    ref = refs.add(Contact("No Handle", "No", "Handle", handles=(Handle("phone", "12"),)))
    res, opened = compose({"contact_ref": ref}, refs)
    assert opened == [] and "no usable" in res


@pytest.mark.parametrize("bad", ["+1 555; rm -rf /", "a b", "x$(id)", "../../etc"])
def test_url_building_rejects_anything_that_is_not_a_phone_or_address(bad):
    with pytest.raises(ValueError):
        ct.build_url(Handle("email", bad))


def test_the_confirmation_summary_carries_no_name_and_no_number():
    s = run(ct._compose_summary({"contact_ref": "c1", "body": "hello"}))
    assert "Nothing will be sent" in s and "c1" not in s and "pre-filled" in s


# ------------------------------------------------------------------ there is no send path

SRC = Path(ct.__file__)


def _string_constants(tree):
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                docstrings.add(id(body[0].value))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


def test_source_scan_there_is_no_send_path():
    tree = ast.parse(SRC.read_text())
    strings = " \n".join(_string_constants(tree)).lower()
    for forbidden in ("keystroke", "key code", "key down", "osascript", "tell application", "system events", "click", "return key",
                      "\\r", "applescript", "to send", "send(", "send_message"):
        assert forbidden not in strings, forbidden
    code = SRC.read_text()
    assert "osascript" not in code.lower() and "run_applescript" not in code and "run_script_with_args" not in code
    assert "pyautogui" not in code and "CGEvent" not in code and "keystroke" not in re.sub(r'""".*?"""', "", code, flags=re.S).lower()
    # the only thing it executes is `open <url>`, via argv (no shell)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") in ("run_argv", "run", "Popen", "system")]
    assert len(calls) == 1 and calls[0].func.attr == "run_argv"
    assert [c.value for c in ast.walk(calls[0]) if isinstance(c, ast.Constant) and isinstance(c.value, str)][:1] == ["open"]


def test_no_tool_in_the_registry_can_send_a_message_or_press_return_through_this_module():
    import tools  # noqa: F401
    from tools.registry import get_registry
    spec = get_registry().get("messages_compose")
    assert spec.outbound is False and "never sends" in spec.description.lower() or "NEVER sends" in spec.description


# ------------------------------------------------------------------ audit/logs carry counts only

@pytest.mark.asyncio
async def test_private_tools_withhold_arguments_and_results_from_events_and_audit(tmp_path):
    import json
    from unittest.mock import AsyncMock
    from bus.event_bus import EventBus
    from guardian.gate import Guardian
    from llm.types import LLMResponse, ToolCall
    from orchestrator.planner import Planner
    from schemas.events import ConfirmationRequestedEvent, ConfirmationResponseEvent, ToolCallFinishedEvent, ToolCallStartedEvent
    from tools.registry import ToolRegistry, ToolSpec
    from tracing.tracer import Tracer

    EventBus.reset_instance()
    bus = EventBus()
    reg = ToolRegistry()

    async def secret(a, c):
        return "c1: Priya Sharma (phone on file)"

    reg.register(ToolSpec(name="contacts_lookup", description="d", tier="safe", handler=secret, private=True))
    reg.register(ToolSpec(name="messages_compose", description="d", tier="confirm", handler=secret, private=True))
    events = []

    async def grab(e):
        events.append(e)

    for t in (ToolCallStartedEvent, ToolCallFinishedEvent, ConfirmationRequestedEvent):
        bus.subscribe(t, grab)

    async def approve(e):
        await bus.emit(ConfirmationResponseEvent(request_id=e.request_id, approved=True, source="test"))

    bus.subscribe(ConfirmationRequestedEvent, approve)
    audit = tmp_path / "audit.jsonl"
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=[
        LLMResponse(tool_calls=[ToolCall(name="contacts_lookup", arguments={"name": "Priya Sharma"}, id="1")]),
        LLMResponse(tool_calls=[ToolCall(name="messages_compose", arguments={"contact_ref": "c1", "body": "hello Priya"}, id="2")]),
        LLMResponse(text="Done.")])
    p = Planner(router=router, registry=reg, guardian=Guardian(event_bus=bus, audit_log_path=audit), event_bus=bus,
                tracer=Tracer(config={"tracing": {"enabled": False}}))
    await p.run(user_text="message Priya")
    blob = " ".join(str(vars(e)) + str(getattr(e, "payload", "")) for e in events)
    assert "Priya" not in blob and "hello" not in blob
    assert "withheld" in blob
    text = audit.read_text()
    assert "Priya" not in text and "hello" not in text and '"withheld": 2' in text
    # the model still got the result (the tool message), the logs did not
    assert any("Priya Sharma" in str(m) for call in router.complete.call_args_list[1:] for m in call.kwargs["messages"])


# ------------------------------------------------------------------ doctor

def test_doctor_contacts_permission_names_the_app():
    [c] = doctor.check_contacts_access({"TERM_PROGRAM": "iTerm.app"}, status=lambda: 2)
    assert c.status == doctor.FAIL and "iTerm" in c.detail and "Privacy & Security > Contacts" in c.fix
    assert doctor.check_contacts_access({}, status=lambda: 3)[0].status == doctor.PASS
    assert doctor.check_contacts_access({"TERM_PROGRAM": "Apple_Terminal"}, status=lambda: 0)[0].status == doctor.WARN
    assert doctor.check_contacts_access({}, status=lambda: None)[0].status == doctor.WARN


def test_doctor_check_never_requests_access():
    import inspect
    src = inspect.getsource(ct.SystemContacts.authorization)
    assert "authorizationStatusForEntityType" in src and "requestAccess" not in src
    # doctor uses exactly that call, so it and the real lookup cannot disagree
    assert "SystemContacts().authorization()" in inspect.getsource(doctor.contacts_authorization)


def test_message_requests_offer_the_contacts_tools():
    import tools  # noqa: F401
    from tools.registry import get_registry
    from tools.selection import select_tools
    for text in ("message Priya that I'm late", "text Rahul", "send Priya an iMessage"):
        names = {t.name for t in select_tools(get_registry(), text)[0]}
        assert {"contacts_lookup", "messages_compose"} <= names, text
