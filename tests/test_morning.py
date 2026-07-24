"""Morning routine (PC4) — the composed no-prompt routine.

Covers the required contracts:
  - composition: the routine gathers existing tools and hands the planner ONE
    structured briefing (weather → what matters → top-3 → day plan → offer);
  - partial-failure degradation: a failing source becomes a named gap and never
    blocks the rest of the briefing;
  - once-per-day: a second trigger the same day is a no-op;
  - defer/cancel: a "not now" defers by defer_minutes, a second cancels the day;
  - mid-conversation defer; and the single-confirmation environment-setup offer
    (offered by default, run directly only when pre_approved).
Plus a unit test for the weather tool.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

import proactive.morning_routine as mr
from bus.event_bus import EventBus
from guardian.gate import Guardian
from proactive.morning_routine import MorningRoutine
from schemas.events import ConfirmationRequestedEvent, ConfirmationResponseEvent, VoiceOutputEvent
from tools.registry import ToolRegistry, ToolSpec


@pytest.fixture(autouse=True)
def _fresh_bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


# --------------------------------- doubles ---------------------------------
class _FakePlanner:
    def __init__(self):
        self.prompts: List[str] = []

    async def run(self, user_text: str, purpose: str = "planning", **_: Any):
        self.prompts.append(user_text)
        return SimpleNamespace(text="Good morning, Sir. Here is your day.")


class _FakeCalendar:
    def __init__(self, events=None, fail=False):
        self._events = events or []
        self._fail = fail

    async def fetch_todays_events(self):
        if self._fail:
            raise RuntimeError("calendar backend down")
        return self._events


def _tool(name, handler, category="general", tier="safe"):
    return ToolSpec(name=name, description=name, tier=tier, handler=handler, category=category)


def _json_handler(payload):
    async def _h(arguments, context):
        return json.dumps(payload)
    return _h


def _raising_handler(exc_msg):
    async def _h(arguments, context):
        raise RuntimeError(exc_msg)
    return _h


def _full_registry() -> ToolRegistry:
    reg = ToolRegistry()

    async def _weather(arguments, context):
        return "London: clear, 15°C now, 9–18°C today."

    reg.register(_tool("current_weather", _weather))
    reg.register(_tool("list_unread", _json_handler([
        {"id": "1", "sender": "ceo@x", "subject": "Q3 numbers"},
        {"id": "2", "sender": "pm@x", "subject": "launch checklist"},
    ])))
    reg.register(_tool("list_notifications", _json_handler([{}, {}, {}])))
    reg.register(_tool("list_pull_requests", _json_handler([{}])))
    reg.register(_tool("get_mentions", _json_handler([{}, {}]), category="mcp:slack"))
    reg.register(_tool("reminders_due", _json_handler([{"title": "Renew domain"}])))
    return reg


def _config(**morning) -> Dict[str, Any]:
    base = {"playlist": "Deep Focus", "work_apps": ["Visual Studio Code", "Terminal"]}
    base.update(morning)
    return {"proactive": {"morning_routine": base}}


def _routine(reg, planner, *, guardian=None, config=None, calendar=None, bus=None) -> MorningRoutine:
    return MorningRoutine(
        planner=planner,
        guardian=guardian,
        registry=reg,
        event_bus=bus or EventBus(),
        calendar_sensor=calendar,
        config=config or _config(),
    )


@pytest.fixture(autouse=True)
def _stub_day_plan(monkeypatch):
    async def _fake_day_plan(registry, memory_agent, notion_db):
        return "Day plan:\n  09:00–11:00 deep work"
    monkeypatch.setattr(mr, "assemble_day_plan_text", _fake_day_plan)


# ============================= 1. composition ==============================
@pytest.mark.asyncio
async def test_routine_composes_one_structured_briefing():
    bus = EventBus()
    voiced: List[str] = []

    async def on_voice(e: VoiceOutputEvent):
        voiced.append(e.text)

    bus.subscribe(VoiceOutputEvent, on_voice)

    planner = _FakePlanner()
    reg = _full_registry()
    calendar = _FakeCalendar([("Standup", datetime(2026, 7, 24, 9, 30)), ("1:1", datetime(2026, 7, 24, 14, 0))])
    routine = _routine(reg, planner, config=_config(), calendar=calendar, bus=bus)  # guardian None → offer skipped

    result = await routine.run(now=datetime(2026, 7, 24, 7, 30))
    assert result is not None and not result.gaps

    prompt = planner.prompts[-1]
    assert "London" in prompt                       # weather line
    assert "THREE items" in prompt                  # top-3 structure instruction
    assert "Standup" in prompt and "09:30" in prompt  # calendar
    assert "Unread email: 2" in prompt              # email triage
    assert "GitHub notifications" in prompt         # github
    assert "Slack mentions" in prompt               # slack
    assert "Day plan" in prompt                     # day plan
    assert "set up his workspace" in prompt.lower() # the offer
    assert "Deep Focus" in prompt                   # the specific playlist offered

    # Delivered as one spoken line (TTS + HUD).
    assert voiced == ["Good morning, Sir. Here is your day."]


# ======================= 2. partial-failure degradation ====================
@pytest.mark.asyncio
async def test_failing_source_becomes_a_gap_and_does_not_block():
    planner = _FakePlanner()
    reg = _full_registry()
    reg._tools["current_weather"] = _tool("current_weather", _raising_handler("open-meteo timeout"))
    routine = _routine(reg, planner, calendar=_FakeCalendar(fail=True))  # weather AND calendar fail

    result = await routine.run(now=datetime(2026, 7, 24, 7, 30))

    assert result is not None
    assert "weather" in result.gaps and "calendar" in result.gaps
    prompt = planner.prompts[-1]
    # It still briefed on the sources that worked...
    assert "Unread email: 2" in prompt
    assert "GitHub notifications" in prompt
    # ...and named the gap in one clause.
    assert "unavailable" in prompt.lower()
    assert "weather" in prompt and "calendar" in prompt


# ========================= 3. once-per-day enforcement =====================
@pytest.mark.asyncio
async def test_runs_at_most_once_per_calendar_day():
    planner = _FakePlanner()
    routine = _routine(_full_registry(), planner, calendar=_FakeCalendar([]))

    first = await routine.run(now=datetime(2026, 7, 24, 7, 30))
    second = await routine.run(now=datetime(2026, 7, 24, 9, 0))  # same day, later
    assert first is not None
    assert second is None                       # suppressed
    assert len(planner.prompts) == 1            # delivered exactly once

    # A new day runs again.
    third = await routine.run(now=datetime(2026, 7, 25, 7, 30))
    assert third is not None
    assert len(planner.prompts) == 2


# ============================ 4. defer / cancel ============================
def test_not_now_defers_then_cancels():
    routine = _routine(_full_registry(), _FakePlanner())
    t0 = datetime(2026, 7, 24, 7, 30)

    assert routine.should_run(t0) is True

    assert routine.not_now(t0) == "deferred"
    assert routine.should_run(t0) is False                       # within the defer window
    assert routine.should_run(t0 + timedelta(minutes=30)) is False
    assert routine.should_run(t0 + timedelta(minutes=61)) is True  # window elapsed (defer_minutes=60)

    # A second "not now" the same day cancels it entirely.
    assert routine.not_now(t0 + timedelta(minutes=61)) == "cancelled"
    assert routine.should_run(t0 + timedelta(hours=3)) is False
    # New day: available again.
    assert routine.should_run(datetime(2026, 7, 25, 7, 30)) is True


@pytest.mark.asyncio
async def test_mid_conversation_defers_without_running():
    planner = _FakePlanner()
    routine = _routine(_full_registry(), planner, calendar=_FakeCalendar([]))
    t0 = datetime(2026, 7, 24, 7, 30)

    result = await routine.run(now=t0, in_conversation=True)
    assert result is None
    assert planner.prompts == []                                  # never briefed
    assert routine.should_run(t0) is False                        # short retry deferral set
    assert routine.should_run(t0 + timedelta(minutes=6)) is True  # conversation_retry_minutes=5


# ===================== 5. single-confirmation setup offer ==================
@pytest.mark.asyncio
async def test_setup_is_offered_as_one_confirmation_then_runs_on_approval(monkeypatch):
    opened: List[str] = []

    async def fake_open(app):
        opened.append(app)
        return True

    monkeypatch.setattr(mr, "_open_app", fake_open)

    bus = EventBus()
    guardian = Guardian(event_bus=bus)
    requests: List[ConfirmationRequestedEvent] = []

    async def on_req(e: ConfirmationRequestedEvent):
        requests.append(e)

    bus.subscribe(ConfirmationRequestedEvent, on_req)

    planner = _FakePlanner()
    routine = _routine(
        _full_registry(), planner, guardian=guardian,
        config=_config(playlist="Deep Focus", work_apps=["Visual Studio Code", "Terminal"]),
        calendar=_FakeCalendar([]), bus=bus,
    )

    turn = asyncio.create_task(routine.run(now=datetime(2026, 7, 24, 7, 30)))
    await asyncio.sleep(0.05)

    # Exactly ONE confirmation, covering the whole set (playlist + all apps).
    assert len(requests) == 1
    summary = requests[0].summary
    assert "Deep Focus" in summary
    assert "Visual Studio Code" in summary and "Terminal" in summary
    assert not opened, "nothing should run before approval"

    await bus.emit(ConfirmationResponseEvent(request_id=requests[0].request_id, approved=True, source="test"))
    await asyncio.wait_for(turn, timeout=2)
    assert opened == ["Visual Studio Code", "Terminal"]           # whole set ran, once approved


@pytest.mark.asyncio
async def test_pre_approved_setup_runs_without_confirmation(monkeypatch):
    opened: List[str] = []

    async def fake_open(app):
        opened.append(app)
        return True

    monkeypatch.setattr(mr, "_open_app", fake_open)

    bus = EventBus()
    guardian = Guardian(event_bus=bus)
    requests: List[ConfirmationRequestedEvent] = []

    async def on_req(e):
        requests.append(e)

    bus.subscribe(ConfirmationRequestedEvent, on_req)

    routine = _routine(
        _full_registry(), _FakePlanner(), guardian=guardian,
        config=_config(pre_approved=True, work_apps=["Terminal"]),
        calendar=_FakeCalendar([]), bus=bus,
    )
    result = await routine.run(now=datetime(2026, 7, 24, 7, 30))
    assert result is not None
    assert opened == ["Terminal"]           # ran directly
    assert requests == []                    # no confirmation asked


# ============================== weather tool ===============================
@pytest.mark.asyncio
async def test_weather_tool_summarizes_open_meteo(monkeypatch):
    import tools.weather as weather

    def fake_get(url):
        if "geocoding" in url:
            return {"results": [{"latitude": 51.5, "longitude": -0.12, "name": "London"}]}
        return {
            "current": {"temperature_2m": 14.3, "weather_code": 3, "wind_speed_10m": 10},
            "daily": {"temperature_2m_max": [17.1], "temperature_2m_min": [9.8]},
        }

    monkeypatch.setattr(weather, "_http_get_json", fake_get)
    out = await weather.current_weather({"location": "London"}, {})
    assert "London" in out and "overcast" in out and "14°C" in out

    # An empty location with nothing configured is handled gracefully.
    assert "location" in (await weather.current_weather({"location": ""}, {})).lower()
