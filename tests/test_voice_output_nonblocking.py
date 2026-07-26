"""
D3 regression: speaking must NOT block the event bus.

The audit caught a VoiceOutputEvent handler that ran macOS `say` synchronously
and stalled the bus for ~6 seconds. _handle_voice_output must now return
almost immediately and let audio play on a background task.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from agents.voice_agent import VoiceAgent
from bus.event_bus import EventBus, get_event_bus
from schemas.events import VoiceOutputEvent


@pytest.fixture(autouse=True)
def _fresh_bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


class _SlowTTS:
    """Stand-in TTS whose playback takes a long time (simulates real audio)."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finished = False

    async def speak(self, text: str, voice: str = "", rate: float = 1.0) -> None:
        self.started.set()
        await asyncio.sleep(1.0)  # "audio" plays for a full second
        self.finished = True

    async def stop(self) -> None:
        pass


def _make_agent() -> VoiceAgent:
    agent = VoiceAgent(event_bus=get_event_bus(), config={"voice": {}})
    agent._tts = _SlowTTS()
    agent._wake_word_detector = None  # no self-trigger suppression needed here
    return agent


@pytest.mark.asyncio
async def test_handler_returns_immediately_while_audio_plays():
    agent = _make_agent()

    start = time.perf_counter()
    await agent._handle_voice_output(VoiceOutputEvent(text="Good evening, Sir.", source="test"))
    elapsed_ms = (time.perf_counter() - start) * 1000

    # The handler must return in well under 50ms even though playback takes 1s.
    assert elapsed_ms < 50, f"handler blocked for {elapsed_ms:.1f}ms"

    # And audio is genuinely playing on a background task that has NOT finished.
    assert agent._speak_task is not None
    await asyncio.wait_for(agent._tts.started.wait(), timeout=1.0)
    assert not agent._speak_task.done()
    assert agent._tts.finished is False

    # Let it complete cleanly.
    await asyncio.wait_for(agent._speak_task, timeout=2.0)
    assert agent._tts.finished is True


@pytest.mark.asyncio
async def test_new_utterance_cancels_the_previous_one():
    agent = _make_agent()

    await agent._handle_voice_output(VoiceOutputEvent(text="first", source="test"))
    first_task = agent._speak_task
    await asyncio.wait_for(agent._tts.started.wait(), timeout=1.0)

    # Second utterance should cancel the first (barge-in) — still non-blocking.
    start = time.perf_counter()
    await agent._handle_voice_output(VoiceOutputEvent(text="second", source="test"))
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 50, f"barge-in handler blocked for {elapsed_ms:.1f}ms"

    assert agent._speak_task is not first_task
    await asyncio.sleep(0)  # let the cancellation propagate
    assert first_task.cancelled() or first_task.done()

    agent._speak_task.cancel()
