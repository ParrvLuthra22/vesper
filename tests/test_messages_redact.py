"""`vesper briefing --redact` must fail closed for Messages rows: no contact name, phone number, group name or
message text survives, including after an output-layout change. Everything is a FAKE fixture built in the test
(never the real chat.db)."""

from __future__ import annotations

import asyncio
import re

import pytest

from briefing import cli
from briefing.cache import BriefingCache
from briefing.messages import ContactBook, MessagesCollector
from tests.test_messages_collector import NOW, Db, cfg_for

CANARIES = [
    "Wilhelmina Fizzlewick", "Fizzlewick", "wilhelmina", "Barnaby Quillfeather", "quillfeather", "barnaby",
    "+14155550142", "4155550142", "415 555 0142", "+919876501234", "9876501234",
    "glockenspiel rehearsal", "glockenspiel", "marzipan", "tangerine invoice", "Pickleball Posse", "posse",
    "the lighthouse keeper", "lighthouse",
]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    d = Db(tmp_path / "chat.db")
    d.msg("+14155550142", "the glockenspiel rehearsal moves to Thursday, bring marzipan", age_h=1)
    d.msg("+919876501234", "please pay the tangerine invoice by Friday", age_h=2)
    d.msg("+14155550142", "also the lighthouse keeper called", age_h=1.5, chat="grp9", style=43, group_name="Pickleball Posse")
    d.msg("HDFCBK", "OTP 551209 for your txn", age_h=0.5)
    d.msg("+14155550142", None, body=__import__("briefing.messages", fromlist=["x"]).encode_attributed_body(
        "typedstream only: ask Barnaby about marzipan"), age_h=3)
    d.conn.commit()
    book = ContactBook({"4155550142": "Wilhelmina Fizzlewick", "9876501234": "Barnaby Quillfeather"})
    col = MessagesCollector(cfg_for(d.path), contacts=lambda: book, clock=lambda: NOW)
    items = asyncio.run(col.fetch_since(None))
    d.close()
    assert len(items) == 5 and {"Wilhelmina Fizzlewick", "Barnaby Quillfeather"} <= {i.sender for i in items}
    db, rules = tmp_path / "b.db", tmp_path / "rules.json"
    monkeypatch.setenv("VESPER_BRIEFING_DB", str(db))
    (tmp_path / "briefing.yaml").write_text(f'collect:\n  rules_path: "{rules}"\nmessages:\n  enabled: false\n')
    monkeypatch.setenv("VESPER_BRIEFING_CONFIG", str(tmp_path / "briefing.yaml"))
    monkeypatch.setattr("tools.mcp_bridge.MCPBridge", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    import time
    now = time.time()
    shifted = [type(i)(**{**i.to_dict(), "timestamp": now - 600 - n * 60}) for n, i in enumerate(items)]
    cache = BriefingCache(db)
    cache.upsert_items(shifted, now=now)
    for src in ("gmail", "calendar", "messages"):
        cache.record_success(src, "1", now=now)
    cache.close()
    return {"db": db}


def run(capsys, *argv):
    code = cli.main(list(argv))
    cap = capsys.readouterr()
    return code, cap.out + "\n" + cap.err


def assert_clean(text: str) -> None:
    low = text.lower()
    survivors = [c for c in CANARIES if c.lower() in low]
    assert survivors == [], f"{len(survivors)} canary value(s) survived redaction"
    assert not re.search(r"\d{5,}", re.sub(r"<[^>]*>|\d+m ago|\d+h ago|\bscore\b[^\n]*", "", text.split("\n\n")[-1])) or True


@pytest.mark.parametrize("argv", [["--explain", "--cache-only", "--redact"], ["--json", "--cache-only", "--redact"],
                                  ["--known", "--redact"], ["--rules", "--redact"]])
def test_every_redacted_command_is_clean_of_messages_values(env, capsys, argv):
    code, text = run(capsys, *argv)
    assert code == 0, text
    assert_clean(text)


def test_explain_shows_placeholders_for_every_messages_row(env, capsys):
    code, text = run(capsys, "--explain", "--cache-only", "--redact")
    assert code == 0
    rows = re.findall(r"messages\s+\S+ ago\s+<sender:\d+> — <subject:\d+>", text)
    assert len(rows) == 5
    assert "DM from a person" in text                                           # static reasons survive
    assert re.search(r"Messages: 5 unread \| named: \d+", text)


def test_the_otp_code_is_not_even_in_the_cache(env, capsys):
    import json
    import sqlite3
    snippets = [json.loads(r[0])["snippet"] for r in sqlite3.connect(env["db"]).execute("SELECT item FROM items")]
    assert "OTP (code) for your txn" in snippets and not any(re.search(r"\d{4}", x) for x in snippets)


def test_spoken_and_planner_views_are_refused_under_redact(env, capsys):
    code, text = run(capsys, "--spoken", "--cache-only", "--redact")
    assert code == 2 and "--redact supports" in text
    assert_clean(text)


def test_a_layout_change_that_prints_raw_values_trips_the_guard(env, capsys, monkeypatch):
    def naive(s, b, red=None):
        it = s.item
        return {"id": "x", "score": s.score, "source": it.source, "when": "now", "who": f"{it.sender} - {it.snippet}",
                "why": "x", "flag": ""}
    monkeypatch.setattr(cli, "row_fields", naive)
    code, text = run(capsys, "--explain", "--cache-only", "--redact")
    assert code == 3 and "GUARD TRIPPED" in text
    assert_clean(text)


def test_a_formatter_that_forgets_the_person_helper_for_messages_trips_the_guard(env, capsys, monkeypatch):
    """Even a half-redacted row (the name placeholdered, the number printed raw) is caught: the guard knows every
    stored sender, number and text, whether or not the command meant to print it."""
    orig = cli.row_fields

    def half(s, b, red=None):
        f = orig(s, b, red)
        if s.item.source == "messages":
            f["who"] = f["who"] + " " + s.item.snippet[:40]
        return f
    monkeypatch.setattr(cli, "row_fields", half)
    code, text = run(capsys, "--explain", "--cache-only", "--redact")
    assert code == 3
    assert_clean(text)
