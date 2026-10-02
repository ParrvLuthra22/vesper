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
