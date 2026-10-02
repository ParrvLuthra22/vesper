"""`vesper briefing --redact` must fail closed: no sender, subject, snippet or domain survives.

Everything here is a FAKE fixture (invented names/subjects/domains chosen to be unmistakable).
The canary list is checked against the full captured stdout+stderr of every redacted command.
"""

from __future__ import annotations

import json
import re
import time

import pytest

from briefing import cli
from briefing.cache import BriefingCache
from briefing.items import Item
from briefing.known import derive_known
from briefing.redact import RedactionLeak, Redactor, safe_reason
from tests.test_briefing_tuning import cfg

H = 3600.0

# Distinctive fakes: none of these words occur in any static CLI text.
CANARIES = [
    "Zelda Quimbleton", "Quimbleton", "zelda.quimbleton", "frobnitz-widgets", "frobnitz",
    "Orville Pennyfeather", "Pennyfeather", "orville", "gloop-industries", "gloop",
    "Quarterly zebra audit overdue", "zebra audit", "xylophone invoice", "wire the xylophone",
    "Gazebo review", "Project Gazebo", "Wombat Weekly", "wombat-news", "wombatnews",
    "Hortensia Blatherwick", "blatherwick", "snorklewhistle", "Snorkle", "plinth-and-sons",
    "secret banana plan",
]

SENT = {
    "recipient_counts": {"Hortensia Blatherwick <hortensia@snorklewhistle.test>": 4,
                         "ops@plinth-and-sons.test": 2},
    "message_count": 6, "recipients": [], "thread_ids": [],
}


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
        Item(id="gmail:fakeaaaa11", source="gmail", sender="Zelda Quimbleton <zelda.quimbleton@frobnitz-widgets.test>",
             title="Quarterly zebra audit overdue", snippet="Please wire the xylophone invoice today",
             timestamp=now - 600, thread_id="t1"),
        Item(id="gmail:fakeaaaa22", source="gmail", sender="Orville Pennyfeather <orville@gloop-industries.test>",
             title="Re: Pennyfeather secret banana plan", snippet="secret banana plan details", timestamp=now - 900,
             is_reply=True),
        Item(id="gmail:fakeaaaa33", source="gmail", sender="Wombat Weekly <news@wombat-news.test>",
             title="Wombat Weekly digest", timestamp=now - 1200, signals={"has_list_unsubscribe": True}),
        Item(id="gmail:fakeaaaa44", source="gmail", sender="Hortensia Blatherwick <hortensia@snorklewhistle.test>",
             title="Lunch?", timestamp=now - 300),
        Item(id="calendar:calfake000", source="calendar", title="Project Gazebo review",
             start_at=now + H, end_at=now + 2 * H, timestamp=now + H),
    ], now=now)
    cache.record_success("gmail", "1", now=now)
    cache.record_success("calendar", "1", now=now)
    cache.set_meta("known_correspondents", derive_known(SENT, cfg(collect={"sent_days": 365})).to_dict(), now=now)
    cache.close()
    return {"rules": rules, "db": db}


def run(capsys, *argv):
    code = cli.main(list(argv))
    cap = capsys.readouterr()
    return code, cap.out + "\n" + cap.err


def assert_clean(text: str) -> None:
    low = text.lower()
    survivors = [c for c in CANARIES if c.lower() in low]
    assert survivors == [], f"{len(survivors)} canary value(s) survived redaction"


# ------------------------------------------------------------------ the commands

def test_explain_redacted_has_placeholders_and_no_real_values(env, capsys):
    code, text = run(capsys, "--explain", "--cache-only", "--redact")
    assert code == 0
    assert_clean(text)
    assert re.search(r"<sender:\d+> — <subject:\d+>", text)
    assert re.search(r"<subject:\d+>", text) and "fakeaaaa11" in text   # ID column kept (needed for --mark)


def test_placeholders_are_stable_and_numbered_by_first_appearance(env, capsys):
    _, one = run(capsys, "--explain", "--cache-only", "--redact")
    _, two = run(capsys, "--explain", "--cache-only", "--redact")
    assert one == two
    nums = [int(n) for n in re.findall(r"<sender:(\d+)>", one)]
    assert nums == sorted(set(nums), key=nums.index) or max(nums) == len(set(nums))


def test_known_redacted(env, capsys):
    code, text = run(capsys, "--known", "--redact")
    assert code == 0 and "Known correspondents" in text
    assert_clean(text)
    assert re.search(r"<sender:\d+>", text) and re.search(r"<domain:\d+>", text)


def test_json_redacted(env, capsys):
    code, text = run(capsys, "--json", "--cache-only", "--redact")
    assert code == 0
    assert_clean(text)
    body = text.split("\n\n")[0] if text.strip().startswith("{") else text
    json.loads(text[text.index("{"): text.rindex("}") + 1])


def test_mark_and_rules_redacted(env, capsys):
    code, text = run(capsys, "--mark", "fakeaaaa11", "priority", "--redact")
    assert code == 0 and "Recorded rule #1" in text
    assert_clean(text)
    # the rule file itself is real data (that is its job); only the printed output is redacted
    assert "zelda.quimbleton@frobnitz-widgets.test" in env["rules"].read_text()
    code, text = run(capsys, "--mark", "fakeaaaa33", "ignore", "--domain", "--redact")
    assert_clean(text)
    code, text = run(capsys, "--rules", "--redact")
    assert code == 0 and "#1:" in text and "#2:" in text
    assert_clean(text)
    code, text = run(capsys, "--explain", "--cache-only", "--redact")
    assert "your rule" in text
    assert_clean(text)
    code, text = run(capsys, "--rules", "--remove", "1", "--redact")
    assert code == 0 and "Removed rule #1" in text
    assert_clean(text)


def test_error_messages_do_not_leak_either(env, capsys):
    code, text = run(capsys, "--mark", "zzzzzzzz", "priority", "--redact")
    assert code == 1 and "No cached mail item" in text
    assert_clean(text)


def test_unsupported_views_are_refused_not_guessed(env, capsys):
    for argv in (["--spoken", "--cache-only", "--redact"], ["--cache-only", "--redact"]):
        code, text = run(capsys, *argv)
        assert code == 2 and "--redact supports" in text
        assert_clean(text)


def test_unexpected_exception_text_is_not_printed(env, capsys, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("Zelda Quimbleton exploded the xylophone invoice")

    monkeypatch.setattr(cli, "explain_table", boom)
    code, text = run(capsys, "--explain", "--cache-only", "--redact")
    assert code == 1 and "RuntimeError" in text
    assert_clean(text)


# ------------------------------------------------------------------ fail-closed behaviour

def test_layout_change_cannot_leak(env, capsys, monkeypatch):
    """The layout code only receives placeholders, so ANY layout is clean — here: a different column
    order, separators and a JSON-ish line format, formatted from the same row fields."""

    def other_layout(b, red=None):
        rows = []
        for s in b.all_scored:
            f = cli.row_fields(s, b, red)
            rows.append(f"{f['who']}|{f['why']}|{f['id']}|{f['score']}|{f['when']}{f['flag']}")
        return "\n".join(rows)

    monkeypatch.setattr(cli, "explain_table", other_layout)
    code, text = run(capsys, "--explain", "--cache-only", "--redact")
    assert code == 0
    assert_clean(text)
    assert "<sender:" in text


def test_a_layout_that_formats_raw_fields_is_caught_by_the_guard(env, capsys, monkeypatch):
    """Second layer: even if someone later writes a layout that bypasses the redactor and prints a
    real field, the guard withholds ALL output and exits non-zero."""

    def leaky(b, red=None):
        return "\n".join(f"{s.item.sender} :: {s.item.title}" for s in b.all_scored)

    monkeypatch.setattr(cli, "explain_table", leaky)
    code, text = run(capsys, "--explain", "--cache-only", "--redact")
    assert code == 3
    assert "REDACTION GUARD TRIPPED" in text
    assert_clean(text)
    assert "<sender:" not in text            # nothing at all was released


@pytest.mark.parametrize("fragment", [
    "Quimbleton", "zelda.quimbleton@frobnitz-widgets.test", "frobnitz-widgets.test", "Quarterly zebra audit",
    "quarterly zebra audit overdue", "xylophone invoice today", "Project Gazebo", "@gloop-industries.test",
    "HORTENSIA", "snorklewhistle.test", "ops@plinth-and-sons.test",
])
def test_guard_catches_values_anywhere_in_the_cache_even_if_never_requested(env, capsys, monkeypatch, fragment):
    """The guard is seeded from the WHOLE cache + known set, not just what a command meant to print."""
    monkeypatch.setattr(cli, "status_lines", lambda b, red=None: [f"sources: ok ({fragment})"])
    code, text = run(capsys, "--explain", "--cache-only", "--redact")
    assert code == 3 and "REDACTION GUARD TRIPPED" in text
    assert fragment.lower() not in text.lower()


def test_guard_catches_a_truncated_subject(env, capsys, monkeypatch):
    monkeypatch.setattr(cli, "status_lines", lambda b, red=None: ["sources: ok (Quarterly zebra audit ov…)"])
    code, text = run(capsys, "--explain", "--cache-only", "--redact")
    assert code == 3


def test_stderr_is_guarded_too(env, capsys, monkeypatch):
    import sys

    monkeypatch.setattr(cli, "status_lines",
                        lambda b, red=None: (print("note: Zelda Quimbleton", file=sys.stderr) or ["sources: ok"]))
    code, text = run(capsys, "--explain", "--cache-only", "--redact")
    assert code == 3 and "Quimbleton" not in text


def test_non_redacted_mode_is_unchanged(env, capsys):
    code, text = run(capsys, "--explain", "--cache-only")
    assert code == 0 and "Zelda Quimbleton" in text and "<sender:" not in text


# ------------------------------------------------------------------ unit

def test_unknown_reason_labels_become_placeholders():
    red = Redactor()
    assert safe_reason("unread mail", red) == "unread mail"
    assert safe_reason("fresh (3h old)", red) == "fresh (3h old)"
    assert safe_reason("VIP sender (boss@secret-corp.test)", red) == "VIP sender (<sender>)"
    assert safe_reason("your rule #4: mail from a@b.test -> PRIORITY", red) == "your rule"
    assert safe_reason("some brand new label with Quimbleton in it", red).startswith("<reason:")


def test_redactor_guard_raises_without_echoing_the_value():
    red = Redactor()
    red.sender("Zelda Quimbleton <zelda@frobnitz-widgets.test>")
    with pytest.raises(RedactionLeak) as exc:
        red.guard("row: frobnitz-widgets.test")
    assert "frobnitz" not in str(exc.value)
    assert red.guard("row: <sender:1>") == "row: <sender:1>"


def test_same_sender_different_formats_share_a_placeholder():
    red = Redactor()
    a = red.sender("Zelda Quimbleton <Zelda.Quimbleton@Frobnitz-Widgets.test>")
    b = red.sender("zelda.quimbleton@frobnitz-widgets.test")
    c = red.sender("Someone Else <else@elsewhere.test>")
    assert a == b and a != c
