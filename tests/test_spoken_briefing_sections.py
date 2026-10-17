"""The spoken briefing: greeting, next timed event (or 'no meetings'), all-day items, priority mail, other unread, question.
All-day events are never meetings and never get a time ("12 AM" bug). Fake data only."""
from __future__ import annotations

from datetime import datetime

import pytest

from briefing.builder import build_briefing, spoken_script
from briefing.cache import BriefingCache
from briefing.config import BriefingConfig
from briefing.items import Item

NOW = datetime(2026, 10, 17, 20, 30).timestamp()
MIDNIGHT = datetime(2026, 10, 17, 0, 0).timestamp()
H = 3600.0


def all_day(title, day_offset=0):
    s = MIDNIGHT + day_offset * 86400
    return Item(id=f"calendar:ad-{title}-{day_offset}", source="calendar", title=title, start_at=s, end_at=s + 86400,
                timestamp=s, all_day=True)


def timed(title, in_h):
    s = NOW + in_h * H
    return Item(id=f"calendar:t-{title}", source="calendar", title=title, start_at=s, end_at=s + H, timestamp=s)


def mail(i, subject="Launch plan", sender="Priya Sharma <priya@corp.example>", age_h=1.0):
    return Item(id=f"gmail:m{i}", source="gmail", sender=sender, title=subject, snippet="please see", timestamp=NOW - age_h * H,
                thread_id=f"t{i}")


def script(items):
    cache = BriefingCache(":memory:")
    cache.upsert_items(items, now=NOW)
    cache.record_success("gmail", "1", now=NOW)
    cache.record_success("calendar", "1", now=NOW)
    cfg = BriefingConfig()
    return spoken_script(build_briefing(cache, cfg, NOW), cfg)


def test_all_day_only_is_spoken_as_all_day_today_never_a_meeting_never_a_time():
    t = script([all_day("Diwali"), all_day("Team offsite week")])
    assert "All day today: Diwali and Team offsite week." in t
    assert "No meetings today." in t
    for bad in ("12 AM", "12:00", "meeting at", "Your next meeting", "Nothing more today", "today you have"):
        assert bad not in t, bad


def test_timed_plus_all_day():
    t = script([timed("Board call", 2), all_day("Diwali")])
    assert t.startswith("Good evening, Sir.")
    assert "Your next meeting is Board call at 10:30 PM, in 2 hours." in t
    assert "All day today: Diwali." in t
    assert t.index("Your next meeting") < t.index("All day today")
    assert "Diwali at" not in t and "12 AM" not in t
    assert "Want to start with your Board call meeting?" in t


def test_an_all_day_event_never_crowds_out_a_real_meeting():
    items = [all_day(f"Holiday {i}") for i in range(4)] + [timed("Standup", 1), timed("Review", 3)]
    t = script(items)
    assert "Your next meeting is Standup" in t and "Then Review" in t
    assert "Holiday 0" in t


def test_no_calendar_items_at_all():
    t = script([])
    assert "No meetings today." in t and "All day" not in t and t.endswith("?")
    assert "Anything you'd like me to look into?" in t


def test_all_day_tomorrow_is_named_as_tomorrow():
    t = script([all_day("Conference", 1)])
    assert "All day tomorrow: Conference." in t


def test_meeting_tomorrow_reads_naturally():
    items = [Item(id="calendar:tm", source="calendar", title="Planning", start_at=NOW + 15 * H, end_at=NOW + 16 * H,
                  timestamp=NOW + 15 * H)]
    t = script(items)
    assert "No more meetings today. Tomorrow, Planning at 11:30 AM." in t and "Nothing more today" not in t


@pytest.mark.parametrize("items,expected", [
    ([timed("Board call", 2), mail(1, "Urgent contract")], "Want to start with your Board call meeting?"),
    ([all_day("Diwali"), mail(1, "Urgent contract")], "Want to start with the email from Priya Sharma?"),
    ([all_day("Diwali")], "Anything you'd like me to look into?"),
    ([mail(2, "Newsletter issue 5", "News <noreply@news.example>")], "Want me to read through your unread items?"),
    ([], "Anything you'd like me to look into?"),
])
def test_the_closing_question_names_a_timed_meeting_only_if_one_exists(items, expected):
    t = script(items)
    assert t.endswith(expected), t
    if not any(i.source == "calendar" and not i.all_day for i in items):
        assert " meeting?" not in t


def test_sections_come_in_the_stated_order():
    t = script([timed("Board call", 2), all_day("Diwali"), mail(1, "Urgent contract"), mail(2, "Weekly digest", "N <noreply@n.example>")])
    order = [t.index(x) for x in ("Good evening", "Your next meeting", "All day today", "priority message", "other unread", "Want to start")]
    assert order == sorted(order)
