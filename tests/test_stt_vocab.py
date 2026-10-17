"""Name vocabulary for speech-to-text: built from known correspondents + Contacts, capped, passed to the recogniser as
initial_prompt, and NEVER logged. `vesper stt-test` uses only names given on the command line. Fake recogniser; no audio."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from voice.input import vocab
from voice.input.config import VoiceInputConfig
from voice.input.stages import Transcriber
from voice.input.stt_test import run_stt_test

ADDR = {"priya.sharma@corp.example": 9, "rahul@corp.example": 5, "info@shop.example": 40, "noreply@x.example": 90,
        "will@x.example": 3, "dmitri_volkov@y.example": 2, "j@z.example": 8, "alex99@w.example": 1}


def test_first_names_come_from_sent_addresses_most_written_to_first_without_roles_or_ordinary_words():
    assert vocab.names_from_addresses(ADDR) == ["Priya", "Rahul", "Dmitri", "Alex"]


def test_build_dedupes_case_insensitively_known_first_and_caps_at_the_limit():
    out = vocab.build_vocab(["Priya", "Rahul"], ["priya", "Katherine", "Zed", "Q"], cap=3)
    assert out == ["Priya", "Rahul", "Katherine"]
    many = [f"Name{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(200)]
    assert len(vocab.build_vocab(many, many, cap=60)) == 60
    assert vocab.build_vocab([], [], cap=60) == []


def test_prompt_is_bounded_and_empty_without_names():
    assert vocab.prompt_from([]) == ""
    names = [f"Longname{i:03d}" for i in range(60)]
    p = vocab.prompt_from(names)
    assert len(p) <= vocab.PROMPT_CHARS + 1 and p.startswith("Names that may be spoken: ") and p.endswith(".")


def test_known_addresses_reads_the_cache_read_only(tmp_path):
    import json, sqlite3
    db = tmp_path / "b.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated REAL NOT NULL)")
    con.execute("INSERT INTO meta VALUES (?,?,0)", ("known_correspondents", json.dumps({"addresses": {"priya@c.example": 3}})))
    con.commit(); con.close()
    before = db.stat().st_mtime_ns
    assert vocab.known_addresses(str(db)) == {"priya@c.example": 3}
    assert db.stat().st_mtime_ns == before and vocab.known_addresses(str(tmp_path / "missing.db")) == {}


def test_contacts_are_used_only_when_already_authorised():
    class B:
        def __init__(self, auth):
            self.auth = auth

        def authorization(self):
            return self.auth

        def contacts(self):
            from tools.contacts_tools import Contact
            return [Contact("Priya S", "Priya", "S"), Contact("No Given", "", "X")]

    assert vocab.contact_names(B(3)) == ["Priya"] and vocab.contact_names(B(2)) == [] and vocab.contact_names(B(0)) == []


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)

        class Seg:
            text = " hello "
        return [Seg()], None


def transcriber(cfg, names):
    t = Transcriber(cfg, vocab_provider=lambda: names)
    t._model = FakeModel()
    return t


def test_the_prompt_reaches_the_recogniser_as_initial_prompt():
    t = transcriber(VoiceInputConfig(), ["Priya", "Rahul"])
    assert t.transcribe(np.zeros(1600, dtype=np.int16)) == "hello"
    assert t._model.calls[0]["initial_prompt"] == "Names that may be spoken: Priya, Rahul."


def test_vocab_off_or_empty_sends_no_prompt():
    t = transcriber(VoiceInputConfig(stt_vocab=False), ["Priya"])
    t.transcribe(np.zeros(1600, dtype=np.int16))
    assert "initial_prompt" not in t._model.calls[0]
    t = transcriber(VoiceInputConfig(), [])
    t.transcribe(np.zeros(1600, dtype=np.int16))
    assert "initial_prompt" not in t._model.calls[0]


def test_names_are_never_logged_only_a_count(caplog):
    caplog.set_level(logging.DEBUG)
    t = transcriber(VoiceInputConfig(), ["Zanzibar", "Quillon"])
    t.transcribe(np.zeros(1600, dtype=np.int16))
    t.transcribe(np.zeros(1600, dtype=np.int16))
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "Zanzibar" not in text and "Quillon" not in text and "2 name(s)" in text
    assert text.count("stt vocabulary") == 1                        # built once


def test_a_failing_vocabulary_never_stops_transcription():
    def boom():
        raise RuntimeError("contacts exploded")

    t = Transcriber(VoiceInputConfig(), vocab_provider=boom)
    t._model = FakeModel()
    assert t.transcribe(np.zeros(1600, dtype=np.int16)) == "hello"


def test_config_defaults_and_keys():
    c = VoiceInputConfig.from_app_config({"voice": {"stt": {"vocab": False, "max_names": 10, "model": "small.en"}}})
    assert (c.stt_vocab, c.stt_max_names, c.whisper_model) == (False, 10, "small.en")
    d = VoiceInputConfig.from_app_config({})
    assert d.stt_vocab is True and d.stt_max_names == 60 and d.whisper_model == "base.en"
    from config.settings import load_config_dict
    assert load_config_dict()["voice"]["stt"] == {"vocab": True, "max_names": 60, "model": ""}


def test_stt_test_uses_only_the_names_given_and_prints_both_transcripts():
    calls, lines = [], []

    def transcribe(pcm, prompt):
        calls.append(prompt)
        return "message Rahool" if not prompt else "message Rahul"

    plain, primed = run_stt_test("message Rahul", ["Rahul"], lambda p: (np.sin(np.arange(24000) / 7.0) * 0.3, 24000), transcribe, lines.append)
    assert calls == ["", "Names that may be spoken: Rahul."] and (plain, primed) == ("message Rahool", "message Rahul")
    assert any("without     : message Rahool" in l for l in lines) and any("with vocab  : message Rahul" in l for l in lines)
    assert any(l.startswith("DIFFERENT") for l in lines)


def test_stt_test_without_names_runs_unprimed_and_never_touches_contacts_or_mail(monkeypatch):
    called = []
    monkeypatch.setattr(vocab, "contact_names", lambda *a, **k: called.append("contacts") or ["Leak"])
    monkeypatch.setattr(vocab, "known_addresses", lambda *a, **k: called.append("mail") or {"leak@x.example": 1})
    lines = []
    run_stt_test("what time is it", [], lambda p: (np.zeros(2400), 24000), lambda pcm, prompt: "what time is it", lines.append)
    assert called == [] and any("none given" in l for l in lines) and "Leak" not in " ".join(lines)
    src = (Path(__file__).resolve().parents[1] / "voice" / "input" / "stt_test.py").read_text()
    assert "load_vocab" not in src and "contact_names" not in src and "known_addresses" not in src
