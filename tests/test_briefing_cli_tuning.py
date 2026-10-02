"""`vesper briefing --known / --mark / --rules / --explain` (ID column): local-only commands.
Nothing here may start the MCP bridge or touch the network."""

from __future__ import annotations

import json
import re
from datetime import datetime

import pytest

from briefing import cli
from briefing.cache import BriefingCache
from briefing.items import Item
from briefing.known import derive_known
from tests.test_briefing_tuning import NOW, SENT, cfg

H = 3600.0

@pytest.fixture
def env(tmp_path, monkeypatch):
    db, rules = tmp_path / "b.db", tmp_path / "rules.json"
    monkeypatch.setenv("VESPER_BRIEFING_DB", str(db))
    (tmp_path / "briefing.yaml").write_text(f'collect:\n  rules_path: "{rules}"\n')
    monkeypatch.setenv("VESPER_BRIEFING_CONFIG", str(tmp_path / "briefing.yaml"))

    def boom(*a, **k):  # --mark/--rules/--known must never start MCP servers
        raise AssertionError("the network/MCP bridge was started")

    monkeypatch.setattr("tools.mcp_bridge.MCPBridge", boom)
    cache = BriefingCache(db)
    now = __import__("time").time()
    items = [
        Item(id="gmail:aaaaaa1111", source="gmail", sender="Alice Example <alice@example.com>", title="Secret subject 1",
             snippet="Secret body text", timestamp=now - 600, thread_id="t1"),
        Item(id="gmail:aaaaaa2222", source="gmail", sender="News <news@list.example.org>", title="Weekly digest",
             timestamp=now - 600, signals={"has_list_unsubscribe": True}),
        Item(id="gmail:bbbbbb3333", source="gmail", sender="Bob <bob@other.com>", title="Hi", timestamp=now - 600),
        Item(id="calendar:cal0000000", source="calendar", title="Standup", start_at=now + H, end_at=now + 2 * H, timestamp=now + H),
    ]
    cache.upsert_items(items, now=now)
    cache.record_success("gmail", "1", now=now); cache.record_success("calendar", "1", now=now)
    cache.close()
    return {"rules": rules, "db": db}


def test_cli_mark_records_a_sender_rule_and_explains_it(env, capsys):
    assert cli.main(["--mark", "bbbbbb3333", "priority"]) == 0
    out = capsys.readouterr().out
    assert "Recorded rule #1: mail from bob@other.com -> PRIORITY" in out
    assert "+40 points" in out and "penalties are waived" in out
    assert "no subject, no text" in out and "no learning model" in out
    assert "--domain" in out and "--rules --remove 1" in out and "now scores" in out
    data = json.loads(env["rules"].read_text())
    assert [(r["scope"], r["value"], r["action"]) for r in data["rules"]] == [("sender", "bob@other.com", "priority")]
    assert "Hi" not in env["rules"].read_text() and "Secret" not in env["rules"].read_text()


def test_cli_mark_with_domain_and_ignore(env, capsys):
    assert cli.main(["--mark", "aaaaaa2222", "ignore", "--domain"]) == 0
    out = capsys.readouterr().out
    assert "Recorded rule #1: all mail from @list.example.org (and its subdomains) -> IGNORE" in out
    assert "-60 points, so it never reaches the priority list" in out


@pytest.mark.parametrize("ref", ["aaaaaa1111", "gmail:aaaaaa1111", "aaaaaa1", "aaaaaa1111xyz"[:10]])
def test_cli_mark_accepts_full_id_prefix_and_gmail_prefix(env, capsys, ref):
    assert cli.main(["--mark", ref, "priority"]) == 0
    assert "alice@example.com" in capsys.readouterr().out


@pytest.mark.parametrize("ref,message", [
    ("aaa", "too short"), ("aaaaaa", "matches 2 items"), ("zzzzzzzz", "No cached mail item"), ("cal0000000", "No cached mail item"),
])
def test_cli_mark_rejects_bad_references(env, ref, message):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--mark", ref, "priority"])
    assert message in str(exc.value)
    assert not env["rules"].exists()


def test_cli_mark_rejects_an_unknown_action(env):
    with pytest.raises(SystemExit):
        cli.main(["--mark", "bbbbbb3333", "boost"])


def test_cli_rules_lists_and_removes(env, capsys):
    cli.main(["--mark", "bbbbbb3333", "priority"]); cli.main(["--mark", "aaaaaa2222", "ignore", "--domain"])
    capsys.readouterr()
    assert cli.main(["--rules"]) == 0
    out = capsys.readouterr().out
    assert "#1: mail from bob@other.com -> PRIORITY" in out and "#2: all mail from @list.example.org" in out
    assert cli.main(["--rules", "--remove", "1"]) == 0
    assert "Removed rule #1" in capsys.readouterr().out
    assert cli.main(["--rules", "--remove", "42"]) == 1
    assert "No rule #42" in capsys.readouterr().out
    cli.main(["--rules"])
    assert "#1:" not in capsys.readouterr().out


def test_cli_rules_when_empty(env, capsys):
    assert cli.main(["--rules"]) == 0
    assert "No rules yet" in capsys.readouterr().out


def test_a_marked_rule_changes_the_next_explain(env, capsys):
    cli.main(["--explain", "--cache-only"])
    before = capsys.readouterr().out
    assert "your rule" not in before
    cli.main(["--mark", "aaaaaa2222", "priority"]); capsys.readouterr()
    cli.main(["--explain", "--cache-only"])
    after = capsys.readouterr().out
    assert "your rule #1: mail from news@list.example.org -> PRIORITY" in after and "bulk penalties waived" in after


def test_cli_explain_has_an_id_column_usable_by_mark(env, capsys):
    cli.main(["--explain", "--cache-only"])
    out = capsys.readouterr().out
    assert re.search(r"\n\s*\d+\s+aaaaaa1111\s+\d+\s+gmail", out) and " ID " in out.splitlines()[0]


def test_cli_known_empty_and_populated(env, capsys):
    assert cli.main(["--known"]) == 0
    assert "No known correspondents yet" in capsys.readouterr().out
    cache = BriefingCache(env["db"])
    cache.set_meta("known_correspondents", derive_known(SENT, cfg(collect={"sent_days": 365})).to_dict(), now=NOW)
    cache.close()
    assert cli.main(["--known"]) == 0
    out = capsys.readouterr().out
    assert "header addresses only" in out and "no message text is read or stored" in out
    assert "prof@college.edu" in out and "college.edu" in out and "+25" in out and "+10" in out
    assert "noreply@service.com" not in out and "gmail.com\n" not in out.split("Domains")[1]
