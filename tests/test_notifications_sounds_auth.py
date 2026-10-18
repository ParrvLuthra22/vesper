"""Honest notifications, the play_sound tool, and `vesper contacts-auth`. Fakes only: no notification is displayed, no sound is played, no
macOS permission prompt is triggered."""
from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from launcher import doctor
from orchestrator import claim_verifier as cv
from tools import sounds
from utils import contacts_auth as ca
from utils import notifications as nt
from utils.grounding import parse

TERM = {"TERM_PROGRAM": "Apple_Terminal"}


@pytest.fixture
def state(tmp_path):
    return nt.NotifyState(tmp_path / "notifications.json")


# ------------------------------------------------------------------ the state file

def test_state_defaults_to_unconfirmed_and_is_per_app(state):
    assert state.status("Terminal") == nt.UNCONFIRMED
    state.save(nt.CONFIRMED, "Terminal", datetime(2026, 10, 18, 9, 0, tzinfo=timezone.utc))
    assert state.status("Terminal") == nt.CONFIRMED and state.status("iTerm") == nt.UNCONFIRMED
    rec = state.load()
    assert set(rec) == {"status", "app", "at"} and rec["app"] == "Terminal" and rec["at"].startswith("2026-10-18T09:00")


def test_state_file_never_holds_message_text(state):
    state.save(nt.FAILED, "Terminal")
    assert set(state.load()) == {"status", "app", "at"}


def test_a_corrupt_state_file_is_just_unconfirmed(tmp_path):
    p = tmp_path / "n.json"
    p.write_text("{not json")
    assert nt.NotifyState(p).status("Terminal") == nt.UNCONFIRMED


# ------------------------------------------------------------------ notify-test

def run_test(state, post_ok=True, answer="y", err=""):
    lines: List[str] = []
    rc = nt.run_notify_test(lambda t, m: (post_ok, err), lambda prompt: (_ for _ in ()).throw(EOFError()) if answer is None else answer,
                            state, "Terminal", lines.append)
    return rc, lines


def test_notify_test_yes_confirms(state):
    rc, lines = run_test(state, answer="y")
    assert rc == 0 and state.status("Terminal") == nt.CONFIRMED and any("Confirmed" in l for l in lines)


def test_notify_test_no_records_failed_and_prints_the_settings_path_and_app(state):
    rc, lines = run_test(state, answer="n")
    text = "\n".join(lines)
    assert rc == 1 and state.status("Terminal") == nt.FAILED
    assert "System Settings > Notifications > Terminal > Allow Notifications" in text and "Focus" in text and "Script Editor" in text


def test_notify_test_osascript_error_is_failed_without_asking(state):
    rc, lines = run_test(state, post_ok=False, err="not allowed", answer="y")
    assert rc == 1 and state.status("Terminal") == nt.FAILED and not any("Did you see" in l for l in lines)


@pytest.mark.parametrize("answer", ["", "maybe", None])
def test_notify_test_without_a_clear_answer_changes_nothing(state, answer):
    rc, lines = run_test(state, answer=answer)
    assert rc == 2 and state.load() == {} and any("unconfirmed" in l for l in lines)


def test_notify_test_asks_the_exact_question(state):
    asked = []
    nt.run_notify_test(lambda t, m: (True, ""), lambda p: asked.append(p) or "y", state, "Terminal", lambda l: None)
    assert asked == ["A test notification was requested. Did you see it? (y/n) "]


# ------------------------------------------------------------------ the tool result + the verifier

def agent_with(show_ok=True, status=None, tmp=None, monkeypatch=None):
    from agents.system_agent import SystemAgent

    a = SystemAgent.__new__(SystemAgent)
    a._name, a._config = "SystemAgent", {}
    a._event_bus = type("B", (), {"emit": AsyncMock()})()
    a.executor = type("E", (), {"show_notification": lambda self, **k: (show_ok, "Notification shown" if show_ok else "denied")})()
    return a


def notify(monkeypatch, tmp_path, status, show_ok=True, config=None):
    st = nt.NotifyState(tmp_path / "n.json")
    if status != nt.UNCONFIRMED:
        st.save(status, "Terminal")
    monkeypatch.setattr(nt, "STATE_PATH", tmp_path / "n.json")
    monkeypatch.setattr(nt, "terminal_app", lambda env=None: "Terminal")
    a = agent_with(show_ok)
    if config is not None:
        a._config = config

    async def go():
        return a._handle_notify({"title": "Vesper", "message": "Tea is ready"}, "")
    return asyncio.run(go()), a


def test_unconfirmed_notification_result_never_claims_display(monkeypatch, tmp_path):
    out, _ = notify(monkeypatch, tmp_path, nt.UNCONFIRMED)
    g = parse(out)
    assert g.can == {"notification_requested"} and "notification_displayed" in g.cannot
    assert "unconfirmed" in out and "cannot say it appeared on screen" in out and "notify-test" in out
    assert "Notification shown" not in out


def test_osascript_exiting_zero_without_a_banner_is_still_not_a_display_claim(monkeypatch, tmp_path):
    from agents.system_agent import AppleScriptExecutor

    ex = AppleScriptExecutor.__new__(AppleScriptExecutor)
    ex.run_script_with_args = lambda *a, **k: (True, "", "")                  # exit 0, nothing on screen
    assert ex.show_notification("t", "m")[0] is True                          # the executor's own bool is only "asked"
    out, _ = notify(monkeypatch, tmp_path, nt.UNCONFIRMED)
    assert "notification_displayed" not in parse(out).can


def test_failed_state_also_refuses_to_claim(monkeypatch, tmp_path):
    out, _ = notify(monkeypatch, tmp_path, nt.FAILED)
    assert "notification_displayed" not in parse(out).can and "failed" in out


def test_confirmed_state_allows_the_display_claim(monkeypatch, tmp_path):
    out, _ = notify(monkeypatch, tmp_path, nt.CONFIRMED)
    g = parse(out)
    assert {"notification_requested", "notification_displayed"} <= g.can and "notification_displayed" not in g.cannot


def test_a_macos_refusal_claims_nothing(monkeypatch, tmp_path):
    out, _ = notify(monkeypatch, tmp_path, nt.CONFIRMED, show_ok=False)
    assert parse(out).can == frozenset() and "could not be requested" in out


@pytest.mark.parametrize("reply", ["Notification displayed, Sir.", "The notification has been shown.", "You should see it now.",
                                   "I've posted a notification to your screen.", "The banner popped up."])
def test_verifier_flags_display_claims_without_confirmation(reply, monkeypatch, tmp_path):
    out, _ = notify(monkeypatch, tmp_path, nt.UNCONFIRMED)
    c = cv.find_unbacked_claim(reply, "notify me", ["show_notification"], results=[("show_notification", out)])
    assert c is not None and c.kind == "specific" and c.category == "notification displayed"
    assert "can't confirm it appeared" in cv.honest_reply(c, [("show_notification", out)])


def test_verifier_lets_confirmed_claims_and_honest_sentences_through(monkeypatch, tmp_path):
    confirmed, _ = notify(monkeypatch, tmp_path, nt.CONFIRMED)
    assert cv.find_unbacked_claim("Notification displayed, Sir.", "notify me", ["show_notification"],
                                  results=[("show_notification", confirmed)]) is None
    unconfirmed, _ = notify(monkeypatch, tmp_path, nt.UNCONFIRMED)
    for ok in ("I asked macOS to show a notification, Sir, but I can't confirm it appeared.", "Shall I send a notification?"):
        assert cv.find_unbacked_claim(ok, "notify me", ["show_notification"], results=[("show_notification", unconfirmed)]) is None


def test_the_notify_tool_delivers_to_the_hud_always_and_speaks_only_in_voice_mode(monkeypatch, tmp_path):
    from schemas.events import ProactiveNotificationEvent, VoiceOutputEvent

    voice_cfg = {"voice": {"output": {"enabled": True}, "input": {"enabled": True}}}
    quiet = datetime(2026, 10, 18, 2, 0, tzinfo=timezone.utc)     # 07:30 IST: not quiet
    monkeypatch.setattr(nt, "in_quiet_hours", lambda *a, **k: False)
    out, a = notify(monkeypatch, tmp_path, nt.UNCONFIRMED, config=voice_cfg)
    kinds = [type(c.args[0]).__name__ for c in a._event_bus.emit.call_args_list]
    assert kinds == ["ProactiveNotificationEvent", "VoiceOutputEvent"] and "HUD and spoken" in out
    out, a = notify(monkeypatch, tmp_path, nt.UNCONFIRMED, config={})
    assert [type(c.args[0]).__name__ for c in a._event_bus.emit.call_args_list] == ["ProactiveNotificationEvent"]
    monkeypatch.setattr(nt, "in_quiet_hours", lambda *a, **k: True)
    out, a = notify(monkeypatch, tmp_path, nt.UNCONFIRMED, config=voice_cfg)
    assert [type(c.args[0]).__name__ for c in a._event_bus.emit.call_args_list] == ["ProactiveNotificationEvent"]


def test_delivery_plan_truth_table():
    assert nt.delivery_plan(voice_on=False) == nt.Plan(True, False, True)
    assert nt.delivery_plan(voice_on=True) == nt.Plan(True, True, True)
    assert nt.delivery_plan(voice_on=True, quiet=True) == nt.Plan(True, False, True)
    assert nt.delivery_plan(voice_on=True, speak_enabled=False).voice is False
    assert nt.in_quiet_hours("23:00", "08:00", "Asia/Kolkata", datetime(2026, 10, 18, 18, 0, tzinfo=timezone.utc)) is True    # 23:30 IST
    assert nt.in_quiet_hours("23:00", "08:00", "Asia/Kolkata", datetime(2026, 10, 18, 6, 0, tzinfo=timezone.utc)) is False     # 11:30 IST


# ------------------------------------------------------------------ doctor: notifications

def test_doctor_notifications_three_states(state):
    [c] = doctor.check_notifications(TERM, state)
    assert c.status == doctor.WARN and "unconfirmed for Terminal" in c.detail and "vesper notify-test" in c.fix
    assert "System Settings > Notifications > Terminal > Allow Notifications" in c.fix
    state.save(nt.CONFIRMED, "Terminal")
    [c] = doctor.check_notifications(TERM, state)
    assert c.status == doctor.PASS and "confirmed" in c.detail and c.detail.startswith("path: osascript fallback")
    state.save(nt.FAILED, "Terminal")
    [c] = doctor.check_notifications(TERM, state)
    assert c.status == doctor.WARN and "failed" in c.detail and "Terminal" in c.fix
    state.save(nt.CONFIRMED, "iTerm")
    [c] = doctor.check_notifications(TERM, state)
    assert "unconfirmed for Terminal" in c.detail and "iTerm" in c.detail


# ------------------------------------------------------------------ play_sound

def run_sound(args, exists=True, ok=True, tmp=None):
    calls = []
    d = tmp
    if exists:
        for n in sounds.ALLOWED:
            (d / f"{n}.aiff").write_text("x")
    h = sounds.make_handler(lambda argv: (calls.append(list(argv)) or (ok, "" if ok else "boom")), d)
    return asyncio.run(h(args, {})), calls


def test_play_sound_runs_afplay_with_an_argv_list_and_reports_the_command_not_the_hearing(tmp_path):
    out, calls = run_sound({"name": "Tink"}, tmp=tmp_path)
    assert calls == [["afplay", str(tmp_path / "Tink.aiff")]]
    g = parse(out)
    assert out.startswith("played Tink (exit 0)") and g.can == {"sound_played"} and "sound_audible" in g.cannot


@pytest.mark.parametrize("bad", ["/etc/passwd", "../../etc/passwd", "Tink; rm -rf ~", "/System/Library/Sounds/../../../etc/hosts", "Basso.aiff",
                                 "$(id)", "", "unknown"])
def test_arbitrary_paths_and_names_are_refused(bad, tmp_path):
    out, calls = run_sound({"name": bad}, tmp=tmp_path) if bad else run_sound({"name": None}, tmp=tmp_path)
    if bad == "":                                   # an empty name means "the default beep", never a path
        assert calls and calls[0][1].endswith("Tink.aiff")
    else:
        assert calls == [] and "only play these system sounds" in out


def test_spoken_aliases_map_to_allow_listed_files(tmp_path):
    assert sounds.resolve("beep") == "Tink" and sounds.resolve("PING") == "Ping" and sounds.resolve("chime") == "Glass"
    assert sounds.resolve("glass") == "Glass" and sounds.resolve("nope") is None and sounds.resolve("/x/y") is None


def test_a_failing_player_claims_nothing(tmp_path):
    out, _ = run_sound({"name": "Pop"}, ok=False, tmp=tmp_path)
    assert parse(out).can == frozenset() and "Could not play Pop" in out


def test_a_missing_sound_file_is_reported(tmp_path):
    out, calls = run_sound({"name": "Pop"}, exists=False, tmp=tmp_path)
    assert calls == [] and "not on this Mac" in out


def test_play_sound_source_has_no_shell_and_only_afplay():
    src = inspect.getsource(sounds)
    assert "shell=True" not in src and "os.system" not in src and src.count('"afplay"') == 1


def test_play_sound_is_a_safe_tier_registered_tool():
    import tools  # noqa: F401
    from tools.registry import get_registry
    spec = get_registry().get("play_sound")
    assert spec.tier == "safe" and spec.handler is not None and not spec.outbound


@pytest.mark.parametrize("reply", ["You should hear it now, Sir.", "The sound was audible.", "You can hear the ping."])
def test_verifier_flags_audibility_claims(reply):
    out = "played Tink (exit 0)\ncan_claim: sound_played\ncannot_claim: sound_audible"
    c = cv.find_unbacked_claim(reply, "make a beep", ["play_sound"], results=[("play_sound", out)])
    assert c is not None and c.category == "sound audible"
    assert cv.find_unbacked_claim("I played Tink, Sir.", "make a beep", ["play_sound"], results=[("play_sound", out)]) is None


@pytest.mark.parametrize("text,sound", [("make a beep", "Tink"), ("beep", "Tink"), ("Ping", "Ping"), ("play a sound", "Tink"),
                                        ("play the glass sound", "Glass"), ("give me a chime", "Glass"), ("ping please", "Ping")])
def test_router_skill_claims_beep_requests(text, sound):
    from router import build_router
    m = build_router(installed=[], clock=lambda: datetime(2026, 10, 18, 9, 0)).match(text)
    assert m is not None and m.skill.name == "play_sound" and m.args["sound"] == sound


@pytest.mark.parametrize("text", ["play the evil sound", "play some music", "make a note", "play a game", "sound check my mic settings"])
def test_router_skill_ignores_other_requests(text):
    from router import build_router
    m = build_router(installed=[], clock=lambda: datetime(2026, 10, 18, 9, 0)).match(text)
    assert m is None or m.skill.name != "play_sound"


def test_router_beep_runs_the_tool_and_never_says_it_was_heard():
    from router import build_router
    from router.core import SkillRuntime
    router = build_router(installed=[], clock=lambda: datetime(2026, 10, 18, 9, 0))
    called = []

    async def call(tool, args):
        called.append((tool, args))
        return "played Tink (exit 0)"

    m = router.match("make a beep")
    res = asyncio.run(m.skill.handler(m, SkillRuntime(call=call)))
    assert called == [("play_sound", {"name": "Tink"})] and "hear" not in res.text.lower() and res.text.startswith("I played Tink")


# ------------------------------------------------------------------ contacts-auth

class Store:
    def __init__(self, before, after_request=None):
        self.s, self.after_request, self.requests = before, after_request, 0

    def status(self):
        return self.s

    def request(self):
        self.requests += 1
        if self.after_request is not None:
            self.s = self.after_request


def auth(store):
    lines: List[str] = []
    rc = ca.run_contacts_auth(store, "Terminal", lines.append)
    return rc, lines


def test_first_line_says_what_will_happen():
    _, lines = auth(Store(3))
    assert lines[0] == "This will ask macOS for Contacts access for Terminal (the app you are running this from)."


def test_not_determined_makes_exactly_one_request_and_reports_granted():
    s = Store(0, after_request=3)
    rc, lines = auth(s)
    assert rc == 0 and s.requests == 1 and lines[-1] == "Contacts access: granted (held by Terminal)."


def test_not_determined_then_denied_in_the_prompt():
    rc, lines = auth(Store(0, after_request=2))
    text = "\n".join(lines)
    assert rc == 1 and "denied (held by Terminal)" in text and "System Settings > Privacy & Security > Contacts > Terminal" in text


def test_already_granted_asks_nothing():
    s = Store(3)
    rc, lines = auth(s)
    assert rc == 0 and s.requests == 0


def test_previously_denied_does_not_re_ask_and_warns_about_tccutil():
    s = Store(2)
    rc, lines = auth(s)
    text = "\n".join(lines)
    assert rc == 1 and s.requests == 0 and "tccutil reset Contacts" in text and "EVERY app" in text
    assert "System Settings > Privacy & Security > Contacts" in text


def test_restricted_is_reported():
    s = Store(1)
    rc, lines = auth(s)
    assert rc == 1 and s.requests == 0 and "restricted" in "\n".join(lines)


def test_no_answer_recorded_stays_not_determined():
    s = Store(0)                                                     # the prompt was dismissed: status unchanged
    rc, lines = auth(s)
    assert rc == 2 and s.requests == 1 and "not determined (held by Terminal)" in "\n".join(lines)


def test_unknown_status_when_the_framework_is_missing():
    rc, lines = auth(Store(None))
    assert rc == 3 and "pyobjc-framework-Contacts" in "\n".join(lines)


def test_contacts_auth_prints_only_the_status_never_a_contact():
    src = inspect.getsource(ca)
    assert "enumerateContacts" not in src and "givenName" not in src and "phoneNumbers" not in src


def test_the_real_store_uses_the_same_status_call_as_contacts_lookup():
    from tools.contacts_tools import SystemContacts
    assert "SystemContacts().authorization()" in inspect.getsource(ca.SystemStore.status)
    assert "authorizationStatusForEntityType" in inspect.getsource(SystemContacts.authorization)


def test_lookup_tells_the_user_to_run_contacts_auth_when_never_asked():
    from tools import contacts_tools as ct

    class B:
        def authorization(self):
            return 0

        def contacts(self):
            raise AssertionError("must not read before access is granted")

    out = asyncio.run(ct.make_lookup(B(), ct.ContactRefs())({"name": "Priya"}, {}))
    assert "vesper contacts-auth" in out and "never been requested" in out


@pytest.mark.parametrize("st,status,needle", [(3, doctor.PASS, "granted to Terminal"), (0, doctor.WARN, "NOT DETERMINED for Terminal"),
                                              (2, doctor.FAIL, "denied for Terminal"), (1, doctor.FAIL, "restricted")])
def test_doctor_contacts_status_lines(st, status, needle):
    [c] = doctor.check_contacts_access(TERM, status=lambda: st)
    assert c.status == status and needle in c.detail
    if st == 0:
        assert "run `vesper contacts-auth`" in c.fix


def test_doctor_warns_when_the_running_stack_was_started_from_another_app():
    [c] = doctor.check_contacts_access(TERM, status=lambda: 0, stack_term="iTerm.app")
    assert "started from iTerm" in c.detail and "may not apply" in c.detail
    [c] = doctor.check_contacts_access(TERM, status=lambda: 0, stack_term="Apple_Terminal")
    assert "started from" not in c.detail
