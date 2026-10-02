"""Voice-input state machine tests — mocked audio/stages, no ML deps needed.

Covers: the full idle→wake→listen→transcribe→inject→idle path, no-speech
timeout, STT-failure recovery, and that the transcript reaches the (mocked)
planner via the sink.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pytest

from voice.input.config import VoiceInputConfig, VoiceState, VoiceTransition
from voice.input.pipeline import VoiceInputPipeline
from voice.input.sink import CallableSink

WAKE, SPEECH, QUIET = 1, 2, 0


def cfg() -> VoiceInputConfig:
    # Small timings so a handful of frames drives the machine.
    return VoiceInputConfig(
        frame_ms=30, sample_rate=16000,
        vad_silence_ms=60,       # 2 trailing-silence frames end an utterance
        listen_timeout_ms=150,   # 5 frames of no speech after wake -> idle
        max_utterance_ms=600,
    )


def frame(marker: int, c: VoiceInputConfig) -> np.ndarray:
    return np.full(c.frame_samples, marker, dtype=np.int16)


class FakeWake:
    def process(self, f: np.ndarray) -> bool:
        return int(f[0]) == WAKE


class FakeVad:
    def is_speech(self, f: np.ndarray) -> bool:
        return int(f[0]) == SPEECH


class FakeTranscriber:
    def __init__(self, text="what is my battery", boom=False):
        self.text, self.boom, self.calls = text, boom, 0

    def transcribe(self, audio: np.ndarray) -> str:
        self.calls += 1
        if self.boom:
            raise RuntimeError("whisper exploded")
        return self.text


def build(c, transcriber, sink, emitter):
    return VoiceInputPipeline(
        config=c, wake=FakeWake(), transcriber=transcriber, sink=sink,
        emitter=emitter, vad_factory=lambda: FakeVad(),
    )


def run(c, frames_markers, transcriber):
    transitions: List[VoiceTransition] = []
    injected: List[str] = []
    sink = CallableSink(lambda t: injected.append(t))  # the mocked planner boundary
    pipe = build(c, transcriber, sink, transitions.append)
    pipe.run_stream([frame(m, c) for m in frames_markers])
    return transitions, injected, pipe


def test_full_pipeline_wake_listen_transcribe_inject():
    c = cfg()
    markers = [QUIET, QUIET, WAKE, SPEECH, SPEECH, SPEECH, QUIET, QUIET]
    tx = FakeTranscriber(text="what is my battery")
    transitions, injected, pipe = run(c, markers, tx)

    assert [t.state for t in transitions] == [
        VoiceState.WAKE_DETECTED,
        VoiceState.LISTENING,
        VoiceState.TRANSCRIBING,
        VoiceState.INJECTED,
        VoiceState.IDLE,
    ]
    assert injected == ["what is my battery"]       # reached the (mocked) planner
    assert tx.calls == 1
    assert pipe.state == VoiceState.IDLE


def test_no_speech_after_wake_times_out():
    c = cfg()
    # wake, then only quiet -> timeout (>=5 quiet frames), never transcribes.
    markers = [WAKE, QUIET, QUIET, QUIET, QUIET, QUIET]
    tx = FakeTranscriber()
    transitions, injected, pipe = run(c, markers, tx)

    states = [t.state for t in transitions]
    assert VoiceState.LISTENING in states
    assert VoiceState.TRANSCRIBING not in states
    assert states[-1] == VoiceState.IDLE
    assert any(t.detail == "no speech" for t in transitions)
    assert injected == []
    assert tx.calls == 0
    assert pipe.state == VoiceState.IDLE


def test_stt_failure_returns_to_idle_without_crashing():
    c = cfg()
    markers = [WAKE, SPEECH, SPEECH, QUIET, QUIET]
    tx = FakeTranscriber(boom=True)
    transitions, injected, pipe = run(c, markers, tx)

    states = [t.state for t in transitions]
    assert VoiceState.TRANSCRIBING in states   # it tried
    assert tx.calls == 1                        # STT was invoked and raised
    assert injected == []                       # nothing injected
    assert states[-1] == VoiceState.IDLE        # recovered to idle
    assert pipe.state == VoiceState.IDLE


def test_empty_transcript_is_not_injected():
    c = cfg()
    markers = [WAKE, SPEECH, SPEECH, QUIET, QUIET]
    tx = FakeTranscriber(text="   ")  # whitespace -> empty
    transitions, injected, pipe = run(c, markers, tx)
    assert injected == []
    assert transitions[-1].state == VoiceState.IDLE


def test_two_wakes_in_one_stream():
    c = cfg()
    markers = [WAKE, SPEECH, SPEECH, QUIET, QUIET, QUIET, WAKE, SPEECH, SPEECH, QUIET, QUIET]
    tx = FakeTranscriber(text="hello")
    transitions, injected, pipe = run(c, markers, tx)
    assert injected == ["hello", "hello"]
    assert transitions.count(VoiceTransition(VoiceState.WAKE_DETECTED, VoiceState.IDLE)) == 2


def test_config_from_app_config_reads_voice_input_block():
    app = {
        "voice": {"input": {"enabled": True, "wake_model": "hey_jarvis", "whisper_model": "base.en"}},
        "gateway": {"host": "127.0.0.1", "port": 8760, "token": "tok"},
    }
    c = VoiceInputConfig.from_app_config(app)
    assert c.enabled is True
    assert c.wake_model == "hey_jarvis"
    assert c.whisper_model == "base.en"
    assert c.gateway_port == 8760
    assert c.frame_samples == 480  # 30 ms @ 16 kHz


# --------------------------------------------------------------------------- first-frame watchdog
# NOTE: patch MicSource.frames on the CLASS — _AnnouncingMicSource's base is bound at import,
# so replacing the module attribute would leave it reading the real microphone.

def test_silent_microphone_trips_the_first_frame_watchdog(monkeypatch):
    """A mic that never delivers (blocked permission prompt / held by another app) makes the
    blocking read hang forever; the watchdog must fire and exit EXIT_UNAVAILABLE."""
    import threading

    from voice.input import service as svc

    release = threading.Event()
    seen = []

    def silent_frames(self):
        release.wait(5)            # blocks like sounddevice's read() on a dead device
        return iter(())

    monkeypatch.setattr(svc.MicSource, "frames", silent_frames)
    src = svc._AnnouncingMicSource(16000, 480, None, first_frame_timeout=0.1,
                                   on_timeout=lambda t: (seen.append(t), release.set()))
    assert list(src.frames()) == []
    assert seen == [0.1]


def test_watchdog_is_cancelled_once_audio_flows(monkeypatch):
    import time

    import numpy as np

    from voice.input import service as svc

    def good_frames(self):
        yield np.zeros(480, dtype=np.int16)
        time.sleep(0.3)            # longer than the watchdog: it must already be cancelled
        yield np.zeros(480, dtype=np.int16)

    fired = []
    monkeypatch.setattr(svc.MicSource, "frames", good_frames)
    src = svc._AnnouncingMicSource(16000, 480, None, first_frame_timeout=0.1, on_timeout=lambda t: fired.append(t))
    assert len(list(src.frames())) == 2 and fired == []


def test_watchdog_disabled_with_zero_timeout(monkeypatch):
    import numpy as np

    from voice.input import service as svc

    monkeypatch.setattr(svc.MicSource, "frames", lambda self: iter([np.zeros(480, dtype=np.int16)]))
    src = svc._AnnouncingMicSource(16000, 480, None, first_frame_timeout=0.0)
    assert len(list(src.frames())) == 1


def test_watchdog_exit_uses_the_unavailable_status_and_a_clear_message(monkeypatch):
    from voice.input import service as svc

    exits, logged = [], []
    monkeypatch.setattr(svc.os, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(svc.logger, "error", lambda msg, *a: logged.append(msg % a))
    monkeypatch.setattr(svc.logging, "shutdown", lambda: None)
    svc._exit_no_audio(15.0)
    assert exits == [69] and svc.EXIT_UNAVAILABLE == 69
    assert "delivered no audio within 15s" in logged[0] and "Microphone" in logged[0]


def test_first_frame_timeout_is_configurable_and_survives_the_config_loader():
    from config.settings import load_config_dict
    from voice.input.config import VoiceInputConfig

    assert VoiceInputConfig().first_frame_timeout_seconds == 15.0
    assert VoiceInputConfig.from_app_config(
        {"voice": {"input": {"first_frame_timeout_seconds": 3}}}).first_frame_timeout_seconds == 3.0
    assert load_config_dict()["voice"]["input"]["first_frame_timeout_seconds"] == 15.0
