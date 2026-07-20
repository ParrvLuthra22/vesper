"""
Tests for orchestrator/brain.py: the enable_voice_agent flag and
get_agents_status() (P08).

Not a full Brain.start() integration test -- that would require mocking
every agent plus the LLM router; those are exercised by each
component's own test file (test_planner.py, test_mcp_bridge.py,
test_inbox_sensor.py, ...). This just targets what P08 actually added.
"""

from __future__ import annotations

import pytest

from bus.event_bus import EventBus
from orchestrator.brain import Brain


@pytest.fixture(autouse=True)
def _fresh_bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


class _FakeAgent:
    """Minimal BaseAgent stand-in -- avoids Mock's reserved `name` kwarg."""

    def __init__(self, name: str, healthy: bool = True):
        self._name = name
        self._healthy = healthy

    @property
    def name(self) -> str:
        return self._name

    def is_healthy(self) -> bool:
        return self._healthy


@pytest.mark.asyncio
async def test_enable_voice_agent_false_skips_voice_agent() -> None:
    brain = Brain(config={}, enable_voice_agent=False)
    await brain._register_default_agents()

    assert brain.get_agent("VoiceAgent") is None
    assert brain.get_agent("MemoryAgent") is not None


@pytest.mark.asyncio
async def test_enable_voice_agent_true_registers_voice_agent() -> None:
    brain = Brain(config={}, enable_voice_agent=True)
    await brain._register_default_agents()

    assert brain.get_agent("VoiceAgent") is not None


def test_get_agents_status_reports_health() -> None:
    brain = Brain(config={}, enable_voice_agent=False)
    brain.register_agent(_FakeAgent("Healthy", healthy=True))
    brain.register_agent(_FakeAgent("Unhealthy", healthy=False))

    statuses = brain.get_agents_status()

    by_name = {s["name"]: s for s in statuses}
    assert by_name["Healthy"]["healthy"] is True
    assert by_name["Unhealthy"]["healthy"] is False
    assert by_name["Healthy"]["error_count"] == 0


def test_get_router_returns_configured_router() -> None:
    brain = Brain(config={}, enable_voice_agent=False)
    assert brain.get_router() is not None
