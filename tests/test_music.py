"""Music (PC1) — Spotify MCP registration (all safe), music-preference retrieval
influencing selection (real semantic store), and the focus-block offer + cooldown."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agents.memory_agent import MemoryAgent
from bus.event_bus import EventBus
from orchestrator.brain import Brain
from proactive.engine import ProactiveEngine
from schemas.events import ObservationEvent, UpcomingMeetingEvent
from tools.mcp_bridge import MCPBridge
from tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _fresh_bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


class _FakeConn:
    def __init__(self, names):
        self.tools = [{"name": n, "description": n, "inputSchema": {"type": "object", "properties": {}}} for n in names]


# --------------------------- Spotify MCP registration ----------------------
def test_spotify_tools_registered_all_safe_and_admin_not_exposed():
    reg = ToolRegistry()
    bridge = MCPBridge(config={}, registry=reg)
    exposed = ["current_track", "search", "list_playlists", "play", "pause", "next", "set_volume", "queue"]
    cfg = {"expose": exposed, "tiers": {n: "safe" for n in exposed}}
    bridge._register_tools("spotify", _FakeConn(exposed + ["transfer_account", "delete_playlist"]), cfg)

    for name in exposed:
        spec = reg.get(name)
        assert spec is not None, f"{name} not registered"
        assert spec.tier == "safe", f"{name} must be safe — music is never confirmation-gated"
        assert spec.category == "mcp:spotify"
    # playback controls specifically must not be confirm/dangerous
    assert all(reg.get(n).tier == "safe" for n in ("play", "pause", "next", "set_volume", "queue"))
    # non-exposed admin tools are never registered
    assert reg.get("transfer_account") is None
    assert reg.get("delete_playlist") is None


# ---------------- Preference retrieval influencing selection ---------------
@pytest.mark.asyncio
async def test_music_preference_is_retrieved_for_a_music_request(tmp_path: Path):
    brain = Brain(config={}, enable_voice_agent=False)
    memory_agent = MemoryAgent(
        event_bus=brain._event_bus,
        config={"memory": {
            "sqlite": {"database_path": str(tmp_path / "memory.db")},
            "vector_store": {"enabled": True, "persist_directory": str(tmp_path / "chroma")},
        }},
    )
    await memory_agent.start()
    brain.register_agent(memory_agent)
    try:
        await memory_agent.index_memory_text(
            text="The user's coding playlist is 'Deep Focus'; he prefers lo-fi and dislikes vocals while working.",
            intent="reflection_preference",
            metadata={"kind": "preference"},
        )
        # A music request retrieves the stored preference, so it reaches the
        # planner's context and influences what Vesper chooses.
        memories = await brain._retrieve_relevant_memories("put on my coding playlist")
        joined = " ".join(memories).lower()
        assert "deep focus" in joined or "lo-fi" in joined
    finally:
        await memory_agent.stop()


# ------------------------------ focus_block rule ---------------------------
class FakeClock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw: float) -> None:
        self.now += timedelta(**kw)


def _config(focus_block: Dict[str, Any]) -> Dict[str, Any]:
    return {"proactive": {
        "rules": {
            "context_switch": {"enabled": False},
            "meeting_reminder": {"enabled": False},  # isolate focus_block observations
            "focus_block": focus_block,
        },
        "schedule": {"morning_briefing": {"enabled": False}},
    }}


def _focus_observations(bus: EventBus) -> List[ObservationEvent]:
    out: List[ObservationEvent] = []

    async def on_obs(event: ObservationEvent) -> None:
        if event.kind == "focus_block":
            out.append(event)

    bus.subscribe(ObservationEvent, on_obs)
    return out


@pytest.mark.asyncio
async def test_focus_block_offers_then_cooldown_suppresses():
    bus = EventBus()
    clock = FakeClock(datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc))
    obs = _focus_observations(bus)
    engine = ProactiveEngine(event_bus=bus, config=_config(
        {"enabled": True, "cooldown_min": 120, "start_within_minutes": 1, "playlist": "Deep Focus"}), clock=clock)
    await engine.start()

    await bus.emit(UpcomingMeetingEvent(title="Deep Work: draft the report", minutes_until=0))
    assert len(obs) == 1
    assert "Deep Focus" in obs[0].detail
    assert "shall I" in obs[0].detail.lower() or "?" in obs[0].detail  # it OFFERS

    # immediate repeat -> suppressed by cooldown
    await bus.emit(UpcomingMeetingEvent(title="Deep Work: draft the report", minutes_until=0))
    assert len(obs) == 1

    # after cooldown -> may offer again
    clock.advance(minutes=121)
    await bus.emit(UpcomingMeetingEvent(title="Focus block", minutes_until=0))
    assert len(obs) == 2
    await engine.stop()


@pytest.mark.asyncio
async def test_focus_block_is_opt_in_and_ignores_non_focus_titles():
    bus = EventBus()
    clock = FakeClock(datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc))
    obs = _focus_observations(bus)

    # disabled by default -> no offer even for a focus-titled block
    engine = ProactiveEngine(event_bus=bus, config=_config({"enabled": False}), clock=clock)
    await engine.start()
    await bus.emit(UpcomingMeetingEvent(title="Deep Work", minutes_until=0))
    assert obs == []
    await engine.stop()

    # enabled, but a non-focus title -> no offer; and not-yet-starting -> no offer
    engine2 = ProactiveEngine(event_bus=bus, config=_config(
        {"enabled": True, "start_within_minutes": 1}), clock=clock)
    await engine2.start()
    await bus.emit(UpcomingMeetingEvent(title="Team standup", minutes_until=0))
    await bus.emit(UpcomingMeetingEvent(title="Deep Work", minutes_until=5))  # not starting yet
    assert obs == []
    await engine2.stop()
