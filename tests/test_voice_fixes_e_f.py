"""Live-test fixes (2026-10-17): (e) "what's on my calendar" in the forms the recogniser produces; (f) app names that arrive as
near-misses ("Sofari", "Message") resolve like exact ones, but closing never acts on a guess."""
from __future__ import annotations

from datetime import datetime

import pytest

from router.skills import AppResolver, build_router


def router():
    return build_router(installed=[], clock=lambda: datetime(2026, 10, 17, 20, 40))


@pytest.mark.parametrize("text", ["What's on my calendar?", "what is on my calendar", "What's on my calender", "what's on my schedule",
                                  "whats on the calendar", "what's on today", "anything on my calendar", "do I have any meetings",
                                  "What do I have", "what's my day", "what's on my calendar today"])
def test_calendar_questions_in_recogniser_forms_reach_the_calendar_skill(text):
    m = router().match(text)
    assert m is not None and m.skill.name == "calendar_today", text


@pytest.mark.parametrize("text", ["what's on my mind", "what's on tv", "what's on my desktop", "what's my name",
                                  "when is my next meeting", "what's the weather"])
def test_calendar_skill_does_not_overreach(text):
    m = router().match(text)
    assert m is None or m.skill.name != "calendar_today", text


@pytest.mark.parametrize("app", ["Safari", "Messages"])
def test_safari_and_messages_resolve_identically(app):
    m = router().match(f"open {app}")
    assert m.skill.name == "open_app" and m.args["app_name"] == app


@pytest.mark.parametrize("heard,app", [("Sofari", "Safari"), ("safarri", "Safari"), ("Message", "Messages"), ("messeges", "Messages"),
                                       ("Note", "Notes"), ("Spotifi", "Spotify")])
def test_recogniser_near_misses_open_the_right_app(heard, app):
    m = router().match(f"open {heard}")
    assert m is not None and m.args["app_name"] == app, heard


@pytest.mark.parametrize("heard", ["open sap", "open the door", "open safari and search for cats", "open mess", "open x"])
def test_unclear_names_still_fall_through(heard):
    m = router().match(heard)
    assert m is None


def test_closing_never_acts_on_a_guess():
    assert router().match("close Sofari") is None and router().match("quit message") is None
    assert router().match("close Safari").args["app_name"] == "Safari"


def test_ambiguous_near_miss_resolves_to_nothing():
    r = AppResolver(["Boxes", "Foxes"])
    assert r("voxes") is None and r("boxes") == "Boxes"
