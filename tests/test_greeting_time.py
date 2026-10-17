"""The spoken greeting follows the REAL local time of day (Asia/Kolkata), never the phrase that was said, and every greeting
/ "brief me" / "what's new" phrase triggers the briefing at any hour. A fake clock drives everything."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from briefing.builder import _tod, build_briefing, spoken_script
from briefing.cache import BriefingCache
from briefing.config import BriefingConfig, from_dict
from briefing.items import Item
from briefing.tool import is_briefing_request

IST = "Asia/Kolkata"


def utc(h, m=0, day=17):
    return datetime(2026, 10, day, h, m, tzinfo=timezone.utc).timestamp()


def test_time_of_day_is_computed_in_asia_kolkata_not_utc():
    assert _tod(utc(3, 0), IST) == "morning"            # 08:30 IST ...
    assert _tod(utc(3, 0), "UTC") == "evening"          # ... but 03:00 in UTC: the zone is what decides
    assert _tod(utc(15, 30), IST) == "evening"          # 21:00 IST
    assert _tod(utc(6, 30), IST) == "afternoon"         # 12:00 IST


@pytest.mark.parametrize("hour,expected", [(5, "morning"), (11, "morning"), (12, "afternoon"), (16, "afternoon"),
                                           (17, "evening"), (21, "evening"), (23, "evening"), (2, "evening"), (4, "evening")])
def test_time_of_day_boundaries(hour, expected):
    # hour in IST -> UTC = hour - 5:30
    ts = datetime(2026, 10, 17, hour, 0, tzinfo=__import__("zoneinfo").ZoneInfo(IST)).timestamp()
    assert _tod(ts, IST) == expected


def test_an_unknown_zone_falls_back_to_local_time_instead_of_crashing():
    assert _tod(utc(6), "Not/AZone") in ("morning", "afternoon", "evening")


def _script_at(hour: int) -> str:
    from zoneinfo import ZoneInfo
    now = datetime(2026, 10, 17, hour, 0, tzinfo=ZoneInfo(IST)).timestamp()
    cache = BriefingCache(":memory:")
    cache.upsert_items([Item(id="calendar:x", source="calendar", title="Board call", start_at=now + 3600, end_at=now + 7200,
                             timestamp=now + 3600)], now=now)
    cache.record_success("gmail", "1", now=now)
    cache.record_success("calendar", "1", now=now)
    cfg = BriefingConfig()
    return spoken_script(build_briefing(cache, cfg, now), cfg)


@pytest.mark.parametrize("hour,word", [(8, "morning"), (14, "afternoon"), (21, "evening")])
def test_the_spoken_greeting_is_the_real_time_of_day(hour, word):
    text = _script_at(hour)
    assert text.startswith(f"Good {word}, Sir.")
    others = {"morning", "afternoon", "evening"} - {word}
    assert not any(f"Good {o}" in text for o in others)               # exactly one greeting, never a second contradicting one


@pytest.mark.parametrize("phrase", ["Good morning", "good afternoon", "Good evening", "good night", "brief me", "What's new?"])
@pytest.mark.parametrize("hour", [0, 3, 8, 13, 19, 23])
def test_every_greeting_phrase_triggers_the_briefing_at_any_hour(phrase, hour):
    assert is_briefing_request(phrase, BriefingConfig(), datetime(2026, 10, 17, hour, 0)) is True


def test_morning_until_hour_no_longer_gates_anything():
    cfg = from_dict({"speech": {"morning_until_hour": 6}})
    assert is_briefing_request("good morning", cfg, datetime(2026, 10, 17, 21, 0)) is True


def test_the_timezone_is_configurable():
    assert BriefingConfig().timezone == IST
    assert from_dict({"speech": {"timezone": "Europe/London"}}).timezone == "Europe/London"


@pytest.mark.asyncio
async def test_good_morning_said_at_9pm_gets_one_good_evening_from_the_briefing_and_no_planner_turn():
    """The reported bug: 'good evening' (wake greeting) then 'good morning' (planner). Now: the briefing, greeted by the clock."""
    from orchestrator.brain import Brain
    from schemas.events import VoiceOutputEvent
    from zoneinfo import ZoneInfo

    brain = Brain(config={})
    now = datetime(2026, 10, 17, 21, 0, tzinfo=ZoneInfo(IST)).timestamp()
    cfg = BriefingConfig()
    cache = BriefingCache(":memory:")
    cache.record_success("gmail", "1", now=now)
    cache.record_success("calendar", "1", now=now)
    svc = MagicMock()
    svc.cfg = cfg
    svc.ensure_fresh = AsyncMock(return_value=False)
    svc.spoken = MagicMock(side_effect=lambda: spoken_script(build_briefing(cache, cfg, now), cfg))
    brain._briefing = svc
    brain._planner.run = AsyncMock(side_effect=AssertionError("planner must not run"))
    brain._planner.greet = AsyncMock(side_effect=AssertionError("no LLM greeting on this path"))
    spoken: List[str] = []

    async def on_voice(e):
        spoken.append(e.text)

    brain._event_bus.subscribe(VoiceOutputEvent, on_voice)
    result = await brain.handle_user_text("Good morning")
    assert result.tool_trace == ["briefing:fast_path"]
    assert result.text.startswith("Good evening, Sir.") and "Good morning" not in result.text
    for _ in range(50):
        if spoken:
            break
        await __import__("asyncio").sleep(0)
    assert spoken == [result.text]                                      # exactly ONE reply was spoken


def test_the_routers_default_briefing_phrases_come_from_the_briefing_config():
    from router.skills import build_router
    r = build_router(installed=[], clock=lambda: datetime(2026, 10, 17, 21, 0))
    for phrase in BriefingConfig().fast_path_phrases:
        assert r.match(phrase).skill.name == "briefing", phrase
