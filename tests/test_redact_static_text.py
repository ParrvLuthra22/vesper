"""`--redact` must not block output just because a stored value happens to equal one of the command's own static
words, and must still catch the same word in a dynamic field. Invented fixtures only."""

from __future__ import annotations

import time

import pytest

from briefing import cli
from briefing.cache import BriefingCache
from briefing.items import Item
from briefing.known import derive_known
from briefing.redact import Redactor
from tests.test_briefing_tuning import cfg

# "gets" and "providers" sit in --known's own header lines; "recorded" / "undo" in --mark's. None of them is a
# SAFE_WORD, so a sender called this used to make the guard withhold perfectly harmless output.
WORDY = "Gets Providers <hello@quuxco.test>"
SENT = {"recipient_counts": {WORDY: 3, "ops@zorp-and-sons.test": 2}, "message_count": 5, "recipients": [], "thread_ids": []}


@pytest.fixture
def env(tmp_path, monkeypatch):
    db, rules = tmp_path / "b.db", tmp_path / "rules.json"
    monkeypatch.setenv("VESPER_BRIEFING_DB", str(db))
    (tmp_path / "briefing.yaml").write_text(f'collect:\n  rules_path: "{rules}"\n')
    monkeypatch.setenv("VESPER_BRIEFING_CONFIG", str(tmp_path / "briefing.yaml"))
    monkeypatch.setattr("tools.mcp_bridge.MCPBridge", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    now = time.time()
    cache = BriefingCache(db)
    cache.upsert_items([
        Item(id="gmail:fakewordy01", source="gmail", sender=WORDY, title="Undo Recorded Addresses",
             snippet="nothing sensitive", timestamp=now - 600),
    ], now=now)
    cache.record_success("gmail", "1", now=now)
    cache.record_success("calendar", "1", now=now)
    cache.set_meta("known_correspondents", derive_known(SENT, cfg(collect={"sent_days": 365})).to_dict(), now=now)
    cache.close()
    return {"rules": rules}


def run(capsys, *argv):
    code = cli.main(list(argv))
    cap = capsys.readouterr()
    return code, cap.out + "\n" + cap.err


# (a) harmless output is no longer blocked by a value that equals a static word -----------------------------

@pytest.mark.parametrize("argv", [["--known", "--redact"],
                                  ["--mark", "fakewordy01", "ignore", "--cache-only", "--redact"],
                                  ["--mark", "fakewordy01", "priority", "--domain", "--redact"],
                                  ["--rules", "--redact"], ["--explain", "--cache-only", "--redact"]])
def test_static_words_that_match_a_stored_value_do_not_block_harmless_output(env, capsys, argv):
    if argv[0] == "--rules":
        run(capsys, "--mark", "fakewordy01", "ignore", "--cache-only", "--redact")
    code, text = run(capsys, *argv)
    assert code == 0, text
    assert "GUARD TRIPPED" not in text
    low = text.lower()
    assert "hello@quuxco.test" not in low and "quuxco" not in low and "zorp-and-sons" not in low


# (b) the same word in a DYNAMIC field is still caught -----------------------------------------------------

def test_the_same_word_in_a_dynamic_field_is_still_caught(env, capsys, monkeypatch):
    monkeypatch.setattr(Redactor, "address", lambda self, addr: "Providers")        # a broken redaction: raw word
    code, text = run(capsys, "--known", "--redact")
    assert code == 3 and "GUARD TRIPPED" in text and "Providers" not in text and "providers\n" not in text.lower().split("tripped")[0]


def test_a_static_looking_line_with_a_smuggled_word_is_not_exempt():
    red = Redactor()
    cli.register_static(red, cfg())
    red.seed_identity(WORDY)
    ok = "Addresses (2) — mail from these gets +25:"
    assert red.findings(red.strip_static(ok + " (public mailbox providers such as gmail.com are never listed)")) >= 0
    assert red.findings(red.strip_static("\n")) == 0
    for smuggled in (ok + " Providers", "Addresses (Gets) — mail from these gets +25:", "  3×  Gets", "  3×  providers"):
        assert red.findings(red.strip_static(smuggled)) > 0, smuggled


def test_placeholders_are_stripped_but_nothing_else_inside_angle_brackets():
    red = Redactor()
    red.seed_identity(WORDY)
    assert red.findings(red.strip_static("  3×  <sender:1>")) == 0
    assert red.findings(red.strip_static("  3×  <providers:1>")) > 0


# (c) the leaky-formatter probe still trips the guard ------------------------------------------------------

def test_a_leaky_formatter_still_trips_the_guard(env, capsys, monkeypatch):
    def naive(s, b, red=None):
        it = s.item
        return {"id": "x", "score": s.score, "source": it.source, "when": "now", "who": f"{it.sender} - {it.title}",
                "why": "x", "flag": ""}
    monkeypatch.setattr(cli, "row_fields", naive)
    code, text = run(capsys, "--explain", "--cache-only", "--redact")
    assert code == 3 and "GUARD TRIPPED" in text
    assert "quuxco" not in text.lower() and "Undo Recorded" not in text
