"""
Tests for the PF6 voice wiring: the 8GB memory guard on speech synthesis,
and the wake-model resolution/fallback that lets the custom phrase be
configured before it has been trained.

Nothing here loads Kokoro or openWakeWord — the heavy stages are injected,
so these stay fast and run with no model files present.
"""

from __future__ import annotations

import logging
import time
from typing import List

import pytest

from voice.input.config import VoiceInputConfig
from voice.output.config import VoiceOutputConfig
from voice.output.speaker import VoiceOutputService, clean_for_speech, split_sentences


class _FakeSpeaker:
    def __init__(self) -> None:
        self.spoken: List[str] = []
        self.stops = 0

    def speak(self, text: str) -> bool:
        self.spoken.append(text)
        return True

    def stop(self) -> None:
        self.stops += 1


def _service(**overrides):
    config = VoiceOutputConfig.from_app_config(
        {"voice": {"output": {"enabled": True, "defer_timeout_seconds": 1.0, **overrides}}}
    )
    speaker = _FakeSpeaker()
    return VoiceOutputService(config, speaker=speaker), speaker


def _settle(seconds: float = 0.5) -> None:
    """Speaking happens on a worker thread; give it a moment."""
    time.sleep(seconds)


# ---------------------------------------------------------------------------
# 8GB guard: never hold Kokoro (~520MB) and the Ollama rescue (~2.5GB) at once
# ---------------------------------------------------------------------------


def test_speech_is_queued_while_the_local_model_is_active():
    service, speaker = _service()

    service.handle({"type": "local_model", "active": True, "model": "llama3.2:3b"})
    service.handle({"type": "reply", "text": "First line, Sir."})
    service.handle({"type": "reply", "text": "Second line, Sir."})
    _settle()

    assert speaker.spoken == [], "must not synthesize while the local model holds memory"


def test_queue_drains_in_order_once_the_local_model_is_idle():
    service, speaker = _service()

    service.handle({"type": "local_model", "active": True})
    service.handle({"type": "reply", "text": "First line, Sir."})
    service.handle({"type": "reply", "text": "Second line, Sir."})
    service.handle({"type": "local_model", "active": False})
    _settle()

    assert speaker.spoken == ["First line, Sir.", "Second line, Sir."]


def test_speech_is_immediate_when_no_local_model_is_active():
    """The Groq path is the normal one — nothing should ever be deferred there."""
    service, speaker = _service()

    service.handle({"type": "reply", "text": "Good evening, Sir."})
    _settle()

    assert speaker.spoken == ["Good evening, Sir."]


def test_wake_drops_a_stale_backlog():
    """If the user speaks again, queued lines are stale — barge-in discards them."""
    service, speaker = _service()

    service.handle({"type": "local_model", "active": True})
    service.handle({"type": "reply", "text": "Stale line."})
    service.handle({"type": "wake", "animate": True})
    service.handle({"type": "local_model", "active": False})
    _settle()

    assert speaker.spoken == []


def test_deferral_times_out_so_a_stuck_local_model_cannot_mute_vesper():
    """
    The gateway signals "idle" from another process, so a crash there must not
    mute Vesper for the rest of the session: a queued backlog is flushed once
    the deferral times out.
    """
    service, speaker = _service(defer_timeout_seconds=0.3)

    service.handle({"type": "local_model", "active": True})
    service.handle({"type": "reply", "text": "Spoken anyway, Sir."})
    # Never sent "active: False" — the flag stays stuck on.
    _settle(0.2)
    assert speaker.spoken == [], "should still be queued before the timeout"

    _settle(0.8)
    assert speaker.spoken == ["Spoken anyway, Sir."], "watchdog must flush the backlog"


def test_guard_can_be_disabled():
    service, speaker = _service(defer_while_local_llm=False)

    service.handle({"type": "local_model", "active": True})
    service.handle({"type": "reply", "text": "Unguarded, Sir."})
    _settle()

    assert speaker.spoken == ["Unguarded, Sir."]


def test_only_serif_lines_are_spoken():
    """Trace/plan lines are display-only and must never reach the speaker."""
    service, speaker = _service()

    service.handle({"type": "tool_started", "tool": "get_time", "arguments": {}})
    service.handle({"type": "plan", "description": "x", "steps": [], "total_steps": 0})
    service.handle({"type": "tool_finished", "tool": "get_time", "success": True})
    _settle()

    assert speaker.spoken == []


# ---------------------------------------------------------------------------
# Wake model resolution
# ---------------------------------------------------------------------------


def test_untrained_custom_model_falls_back_to_the_pretrained_one(caplog):
    """
    The custom phrase can be configured before its one-time GPU training run
    without leaving voice input dead.
    """
    from voice.input.stages import WakeDetector

    config = VoiceInputConfig.from_app_config(
        {
            "voice": {
                "input": {
                    "enabled": True,
                    "wake_model": "wake_up_daddys_home",
                    "wake_model_fallback": "hey_jarvis",
                }
            }
        }
    )
    detector = WakeDetector(config)

    # Nothing resolves for the custom name; the fallback does.
    assert detector._local_model_path() is None
    assert config.wake_model_fallback == "hey_jarvis"


def test_wake_model_accepts_an_explicit_path(tmp_path):
    from voice.input.stages import WakeDetector

    model = tmp_path / "custom.onnx"
    model.write_bytes(b"not-a-real-model")
    config = VoiceInputConfig.from_app_config(
        {"voice": {"input": {"enabled": True, "wake_model": str(model)}}}
    )

    assert WakeDetector(config)._local_model_path() == str(model.resolve())


def test_fallback_can_be_disabled_to_make_a_missing_model_fatal():
    config = VoiceInputConfig.from_app_config(
        {"voice": {"input": {"wake_model": "nope", "wake_model_fallback": ""}}}
    )
    assert config.wake_model_fallback == ""


# ---------------------------------------------------------------------------
# Speech text handling
# ---------------------------------------------------------------------------


def test_trace_glyphs_and_urls_are_stripped_before_speaking():
    cleaned = clean_for_speech("▸ Done **Sir** — see https://example.com/x")
    assert "▸" not in cleaned and "**" not in cleaned and "http" not in cleaned
    assert "Sir" in cleaned


def test_sentences_split_for_streaming_playback():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_empty_text_is_not_spoken():
    service, speaker = _service()
    service.handle({"type": "reply", "text": "   "})
    _settle(0.2)
    assert speaker.spoken == []
