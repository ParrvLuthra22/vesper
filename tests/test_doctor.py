"""`vesper doctor`: read-only, prints no secret, never opens the mic unless asked."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from launcher import doctor
from launcher.doctor import FAIL, PASS, WARN, Check

SECRET = "123456789:AAFakeTokenForTestsOnly_abcdefghijklmnopq"


def by_label(checks, label):
    return [c for c in checks if c.label.startswith(label)]


def test_microphone_default_never_opens_the_stream():
    def boom(seconds):
        raise AssertionError("the microphone must not be opened without --probe-mic")

    out = doctor.check_microphone(False, devices=lambda: [{"name": "x"}], frames=boom)
    assert [c.status for c in out] == [PASS, WARN] and "--probe-mic" in out[1].fix


def test_microphone_probe_outcomes():
    ok = doctor.check_microphone(True, devices=lambda: [{}], frames=lambda s: 32000)
    assert ok[-1].status == PASS
    silent = doctor.check_microphone(True, devices=lambda: [{}], frames=lambda s: 0)
    assert silent[-1].status == FAIL and "Microphone" in silent[-1].fix
    def denied(s):
        raise OSError("denied")
    assert doctor.check_microphone(True, devices=lambda: [{}], frames=denied)[-1].status == FAIL
    assert doctor.check_microphone(False, devices=lambda: [])[0].status == FAIL
    def nolib():
        raise ImportError("sounddevice")
    assert doctor.check_microphone(False, devices=nolib)[0].status == FAIL


CFG = {"channels": {"telegram": {"enabled": True, "allowed_user_ids": [1], "bot_token_env": "TOK",
                                 "keychain_service": "svc"}}}


def test_telegram_token_presence_only_never_the_value():
    seen = []
    out = doctor.check_telegram(CFG, {"TOK": SECRET}, keychain=lambda s: seen.append(s) or True)
    text = doctor.render(out)
    assert out[0].status == PASS and SECRET not in text and "AAFake" not in text
    assert "environment ($TOK)" in text and "Keychain (svc)" in text


def test_telegram_token_missing_and_config_only_and_disabled():
    assert doctor.check_telegram(CFG, {}, keychain=lambda s: False)[0].status == FAIL
    cfg = {"channels": {"telegram": {**CFG["channels"]["telegram"], "bot_token": SECRET}}}
    out = doctor.check_telegram(cfg, {}, keychain=lambda s: False)
    assert out[0].status == WARN and SECRET not in doctor.render(out)
    off = doctor.check_telegram({"channels": {"telegram": {"enabled": False}}}, {})
    assert [c.status for c in off] == [PASS]
    empty = {"channels": {"telegram": {**CFG["channels"]["telegram"], "allowed_user_ids": []}}}
    assert doctor.check_telegram(empty, {"TOK": SECRET})[1].status == FAIL


def test_keychain_probe_does_not_request_the_secret(monkeypatch):
    calls = []

    class R:
        returncode = 0

    monkeypatch.setattr(doctor.subprocess, "run", lambda argv, **k: calls.append(argv) or R())
    assert doctor.keychain_has("svc") is True
    assert "-w" not in calls[0] and "-g" not in calls[0]          # -w / -g would return the password


def test_models(tmp_path):
    cache = tmp_path / "hf"
    cfg = {"voice": {"output": {"enabled": True}, "input": {"enabled": True, "whisper_model": "base.en"}}}
    out = doctor.check_models(cfg, root=tmp_path, hf_cache=cache)
    assert [c.status for c in out if "Kokoro" in c.label] == [FAIL, FAIL]
    assert by_label(out, "Whisper")[0].status == FAIL and "offline_models" in by_label(out, "Whisper")[0].fix
    (tmp_path / "voice" / "models").mkdir(parents=True)
    (tmp_path / "voice/models/kokoro-v1.0.onnx").write_text("x")
    (tmp_path / "voice/models/voices-v1.0.bin").write_text("x")
    (cache / "models--Systran--faster-whisper-base.en").mkdir(parents=True)
    (cache / "models--sentence-transformers--all-MiniLM-L6-v2").mkdir()
    out = doctor.check_models(cfg, root=tmp_path, hf_cache=cache)
    assert all(c.status == PASS for c in out)
    custom = {"voice": {"input": {"enabled": True, "wake_model": "x/custom.onnx", "wake_model_fallback": "hey_jarvis"},
                        "output": {"enabled": False}}}
    assert by_label(doctor.check_models(custom, root=tmp_path, hf_cache=cache), "Wake")[0].status == WARN


def test_ports():
    assert doctor.check_ports({}, None, busy=lambda h, p: False)[0].status == PASS
    assert doctor.check_ports({}, 4242, busy=lambda h, p: True)[0].status == PASS
    c = doctor.check_ports({"gateway": {"port": 9999}}, None, busy=lambda h, p: True)[0]
    assert c.status == FAIL and "9999" in c.fix and "lsof" in c.fix


def make_db(path: Path, sources, unread=3):
    db = sqlite3.connect(path)
    db.executescript("""CREATE TABLE items (id TEXT, source TEXT, item TEXT, active INTEGER);
        CREATE TABLE sources (source TEXT PRIMARY KEY, cursor TEXT, last_attempt REAL, last_success REAL,
                              last_error TEXT, consecutive_failures INTEGER);""")
    for s in sources:
        db.execute("INSERT INTO sources VALUES (?,?,?,?,?,?)", (s[0], None, s[1], s[1], None, s[2]))
    for i in range(unread):
        db.execute("INSERT INTO items VALUES (?,?,?,1)", (f"gmail:{i}", "gmail", "{\"title\": \"SECRET SUBJECT\"}"))
    db.commit(); db.close()


def test_briefing_freshness_levels_and_no_content(tmp_path):
    now = time.time()
    assert doctor.check_briefing(str(tmp_path / "none.db"))[0].status == WARN
    p = tmp_path / "b.db"
    make_db(p, [("gmail", now - 120, 0), ("calendar", now - 5 * 3600, 0)])
    out = doctor.check_briefing(str(p), 10, now)
    assert by_label(out, "Briefing cache: gmail")[0].status == PASS
    assert by_label(out, "Briefing cache: calendar")[0].status == WARN
    assert "SECRET" not in doctor.render(out) and "3 cached item(s)" in doctor.render(out)
    q = tmp_path / "f.db"
    make_db(q, [("gmail", now - 60, 4)])
    assert doctor.check_briefing(str(q), 10, now)[0].status == WARN
    assert doctor.check_briefing(str(q), 10, now)[0].detail.endswith("4 consecutive failure(s)")
    bad = tmp_path / "bad.db"; bad.write_text("not sqlite")
    assert doctor.check_briefing(str(bad))[0].status == FAIL
    assert not (tmp_path / "none.db").exists()                    # read-only: it never creates the file


def test_audit_last_entry_channel(tmp_path):
    p = tmp_path / "a.jsonl"
    assert doctor.check_audit(p)[0].status == WARN
    p.write_text(json.dumps({"timestamp": "2026-10-02T10:00:00+00:00", "tool": "t"}) + "\n")
    c = doctor.check_audit(p)[0]
    assert c.status == WARN and "no channel field" in c.detail
    p.write_text(p.read_text() + json.dumps({"timestamp": "2026-10-02T11:00:00+00:00", "tool": "t",
                                             "channel": "telegram", "args": {"secret": "x"}}) + "\n")
    c = doctor.check_audit(p)[0]
    assert c.status == PASS and "'telegram'" in c.detail and "secret" not in doctor.render([c])
    p.write_text("garbage\n")
    assert doctor.check_audit(p)[0].status == FAIL
    big = tmp_path / "big.jsonl"
    big.write_text("x" * 50000 + "\n" + json.dumps({"channel": "local"}) + "\n")
    assert "'local'" in doctor.check_audit(big)[0].detail


def test_audit_file_is_not_modified(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps({"channel": "local"}) + "\n")
    before = (p.read_bytes(), p.stat().st_mtime_ns)
    doctor.check_audit(p)
    assert (p.read_bytes(), p.stat().st_mtime_ns) == before


def test_render_and_exit_status(tmp_path, monkeypatch):
    out = doctor.render([Check(PASS, "a"), Check(WARN, "b", "d", "do x"), Check(FAIL, "c", "", "do y")])
    assert "[PASS] a" in out and "fix: do x" in out and "fix: do y" in out and "1 pass, 1 warn, 1 fail" in out
    assert "fix:" not in doctor.render([Check(PASS, "a", "", "ignored")])
    monkeypatch.setattr(doctor, "run_checks", lambda *a, **k: [Check(FAIL, "x")])
    monkeypatch.setattr("launcher.stack.load_app_config", lambda: {})
    assert doctor.main([]) == 1
    monkeypatch.setattr(doctor, "run_checks", lambda *a, **k: [Check(WARN, "x")])
    assert doctor.main([]) == 0


def test_doctor_is_a_registered_subcommand():
    from launcher import cli
    assert "doctor" in cli.COMMANDS
    assert cli.build_parser().parse_args(["doctor", "--probe-mic"]).probe_mic is True


# ------------------------------------------------------------------ messages / Full Disk Access (no real chat.db)

def test_messages_check_reports_the_three_states_without_reading_anything():
    ok = doctor.check_messages(True, "/x/chat.db", state=lambda p: "ok")[0]
    assert ok.status == PASS and "nothing was read" in ok.detail and ok.fix == ""
    off = doctor.check_messages(False, "/x/chat.db", state=lambda p: "ok")[0]
    assert off.status == PASS and "messages.enabled: true" in off.fix
    denied = doctor.check_messages(False, "/x/chat.db", state=lambda p: "denied")[0]
    assert denied.status == WARN and "Full Disk Access" in denied.fix and "System Settings" in denied.fix
    missing_on = doctor.check_messages(True, "/x/chat.db", state=lambda p: "missing")[0]
    missing_off = doctor.check_messages(False, "/x/chat.db", state=lambda p: "missing")[0]
    assert missing_on.status == WARN and missing_off.status == PASS


def test_messages_check_only_opens_the_file_and_never_reads_it(tmp_path, monkeypatch):
    f = tmp_path / "chat.db"
    f.write_bytes(b"SQLite format 3\x00 not really a database")
    calls = []
    real_open = __import__("os").open
    monkeypatch.setattr("os.open", lambda path, flags, *a: calls.append((str(path), flags)) or real_open(path, flags, *a))
    monkeypatch.setattr("os.read", lambda *a: (_ for _ in ()).throw(AssertionError("read!")))
    out = doctor.check_messages(True, str(f))
    assert out[0].status == PASS and calls == [(str(f), __import__("os").O_RDONLY)]


def test_run_checks_includes_messages_only_when_asked(tmp_path):
    base = dict(env={}, root=tmp_path, audit_path=tmp_path / "a.jsonl", briefing_db=str(tmp_path / "b.db"))
    none = doctor.run_checks({}, **base)
    assert not any(c.label.startswith("Messages") for c in none)


# ------------------------------------------------------------------ which microphone would voice input use? (fake lists only)

def _dev(i, name, ch=1, rate=48000.0):
    return {"index": i, "name": name, "max_input_channels": ch, "default_samplerate": rate}


FAKE_LIST = lambda *d: (lambda: list(d))        # noqa: E731


def _by_label(checks):
    return {c.label: c for c in checks}


def test_doctor_shows_the_device_voice_input_would_use_by_name_and_passes_when_the_default_is_the_builtin_mic():
    out = _by_label(doctor.check_input_device(
        "builtin", FAKE_LIST(_dev(0, "MacBook Pro Microphone"), _dev(1, "Parrv's AirPods Pro")), default_index=0,
        transports={}, use_real_default=False))
    assert out["Voice input device"].status == PASS and "'MacBook Pro Microphone'" in out["Voice input device"].detail
    assert out["Built-in microphone"].status == PASS
    assert out["System default input"].status == PASS


def test_doctor_warns_when_the_system_default_is_airpods_but_voice_input_is_still_pinned_to_the_builtin_mic():
    out = _by_label(doctor.check_input_device(
        "builtin", FAKE_LIST(_dev(0, "Parrv's AirPods Pro"), _dev(2, "MacBook Pro Microphone")), default_index=0,
        transports={}, use_real_default=False))
    assert out["Voice input device"].status == PASS and "MacBook Pro Microphone" in out["Voice input device"].detail
    warn = out["System default input"]
    assert warn.status == WARN and "AirPods" in warn.detail and "pinned" in warn.fix


def test_doctor_warns_for_a_virtual_default_too():
    out = _by_label(doctor.check_input_device(
        "builtin", FAKE_LIST(_dev(1, "BlackHole 2ch"), _dev(2, "MacBook Pro Microphone")), default_index=1,
        transports={}, use_real_default=False))
    assert out["System default input"].status == WARN and "BlackHole" in out["System default input"].detail


def test_doctor_fails_when_the_builtin_mic_is_not_found_and_lists_what_is():
    checks = doctor.check_input_device("builtin", FAKE_LIST(_dev(0, "Parrv's AirPods Pro"), _dev(1, "BlackHole 2ch")),
                                       default_index=0, transports={}, use_real_default=False)
    out = _by_label(checks)
    assert out["Voice input device"].status == FAIL and "will not fall back" in out["Voice input device"].detail
    assert out["Built-in microphone"].status == FAIL
    assert "AirPods" in out["Built-in microphone"].detail and "BlackHole 2ch" in out["Built-in microphone"].detail
    assert any(c.status == FAIL for c in checks)


def test_doctor_with_an_explicit_device_and_no_builtin_only_warns_about_the_builtin_mic():
    out = _by_label(doctor.check_input_device(
        "Yeti Stereo Microphone", FAKE_LIST(_dev(3, "Yeti Stereo Microphone")), default_index=3, transports={},
        use_real_default=False))
    assert out["Voice input device"].status == PASS and "Yeti Stereo Microphone" in out["Voice input device"].detail
    assert out["Built-in microphone"].status == WARN


def test_doctor_fails_for_an_explicit_device_that_does_not_exist():
    out = _by_label(doctor.check_input_device("Studio Display Microphone", FAKE_LIST(_dev(2, "MacBook Pro Microphone")),
                                              default_index=2, transports={}, use_real_default=False))
    assert out["Voice input device"].status == FAIL and "MacBook Pro Microphone" in out["Voice input device"].detail


def test_doctor_reports_an_unlistable_audio_stack_as_a_failure():
    def boom():
        raise OSError("PortAudio not initialised")
    out = doctor.check_input_device("builtin", boom, use_real_default=False)
    assert out[0].status == FAIL and "cannot list input devices" in out[0].detail


def test_doctor_output_carries_device_names_only():
    checks = doctor.check_input_device(
        "builtin", FAKE_LIST(_dev(0, "Parrv's AirPods Pro", rate=16000.0), _dev(2, "MacBook Pro Microphone")),
        default_index=0, transports={}, use_real_default=False)
    text = doctor.render(checks)
    assert "MacBook Pro Microphone" in text
    assert "16000" not in text and "48000" not in text                      # no rates, ids or other device properties


def test_run_checks_includes_the_input_device_checks_only_when_asked(tmp_path, monkeypatch):
    base = dict(env={}, root=tmp_path, audit_path=tmp_path / "a.jsonl", briefing_db=str(tmp_path / "b.db"))
    monkeypatch.setattr(doctor, "check_microphone", lambda probe, device=None: [])
    none = doctor.run_checks({}, **base)
    assert not any(c.label.startswith(("Voice input device", "Built-in microphone")) for c in none)
    monkeypatch.setattr(doctor, "check_input_device", lambda setting, **k: [Check(PASS, "Voice input device", setting)])
    some = doctor.run_checks({}, voice_input_device=("builtin",), **base)
    assert any(c.label == "Voice input device" for c in some)


def test_the_mic_probe_opens_the_pinned_device_not_the_default(monkeypatch):
    seen = {}

    class FakeSD:
        @staticmethod
        def query_devices():
            return [_dev(2, "MacBook Pro Microphone")]

        class InputStream:
            def __init__(self, **kw):
                seen.update(kw)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", FakeSD)
    monkeypatch.setattr(doctor.time, "sleep", lambda s: None)
    doctor.check_microphone(True, device=2)
    assert seen.get("device") == 2 and seen.get("samplerate") == 16000


def test_doctor_says_so_when_the_system_default_input_cannot_be_determined():
    out = _by_label(doctor.check_input_device(
        "builtin", FAKE_LIST(_dev(2, "MacBook Pro Microphone")), default_index=None, transports={}, use_real_default=False))
    d = out["System default input"]
    assert d.status == WARN and "could not be determined" in d.detail and "pinned" in d.fix
    assert out["Voice input device"].status == PASS                       # and the pinned mic is unaffected
    gone = _by_label(doctor.check_input_device(
        "builtin", FAKE_LIST(_dev(2, "MacBook Pro Microphone")), default_index=99, transports={}, use_real_default=False))
    assert gone["System default input"].detail.startswith("could not be determined")      # a default that isn't an input device


# ---- calendar permission (names the exact app) and the custom wake model

@pytest.mark.parametrize("term,app", [("Apple_Terminal", "Terminal"), ("iTerm.app", "iTerm"), ("vscode", "Visual Studio Code"),
                                      ("ghostty", "Ghostty")])
def test_calendar_denied_names_the_terminal_app_and_the_settings_path(term, app):
    [c] = doctor.check_calendar_access({"TERM_PROGRAM": term}, status=lambda: 2)
    assert c.status == FAIL and app in c.detail
    assert "System Settings > Privacy & Security > Calendars" in c.fix and app in c.fix


def test_calendar_granted_not_asked_and_unknown():
    assert doctor.check_calendar_access({}, status=lambda: 3)[0].status == PASS
    c = doctor.check_calendar_access({"TERM_PROGRAM": "Apple_Terminal"}, status=lambda: 0)[0]
    assert c.status == WARN and "Terminal" in c.fix
    assert doctor.check_calendar_access({}, status=lambda: None)[0].status == WARN


def test_calendar_check_never_requests_access():
    import inspect
    src = inspect.getsource(doctor.calendar_authorization)
    assert "authorizationStatusForEntityType" in src and "requestAccess" not in src and "requestFullAccess" not in src


def test_wake_model_check(tmp_path):
    assert doctor.check_wake_model({}, tmp_path)[0].status == PASS
    cfg = {"voice": {"wake": {"model_path": "voice/models/hey_vesper.onnx"}}}
    c = doctor.check_wake_model(cfg, tmp_path)[0]
    assert c.status == WARN and "hey_vesper.onnx" in c.detail and "docs/WAKEWORD.md" in c.fix
    (tmp_path / "voice" / "models").mkdir(parents=True)
    (tmp_path / "voice" / "models" / "hey_vesper.onnx").write_bytes(b"x" * 4096)
    assert doctor.check_wake_model(cfg, tmp_path)[0].status == PASS
