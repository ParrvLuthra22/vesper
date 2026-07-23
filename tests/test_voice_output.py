"""Voice-output tests — mocked audio. Covers: reply is spoken, trace lines are
NOT spoken, observations are spoken, barge-in stops playback, and speech
cleaning/streaming."""
from __future__ import annotations

import numpy as np

from voice.output.config import VoiceOutputConfig
from voice.output.speaker import Speaker, VoiceOutputService, clean_for_speech, split_sentences


def cfg(**kw) -> VoiceOutputConfig:
    return VoiceOutputConfig(streaming=True, **kw)


class FakeSpeaker:
    def __init__(self):
        self.spoken, self.stops = [], 0

    def speak(self, text):
        self.spoken.append(text)
        return True

    def stop(self):
        self.stops += 1


class FakeTTS:
    def __init__(self):
        self.texts = []

    def synthesize(self, text):
        self.texts.append(text)
        return np.zeros(16, dtype=np.float32), 24000


class FakePlayer:
    def __init__(self, interrupt_at=None):
        self.plays, self.stops = 0, 0
        self._interrupt_at = interrupt_at

    def play(self, samples, sr):
        self.plays += 1
        return not (self._interrupt_at is not None and self.plays >= self._interrupt_at)

    def stop(self):
        self.stops += 1


def _join(svc):
    if svc._worker is not None:
        svc._worker.join(2)


# ------------------------------- service routing --------------------------
def test_reply_is_spoken():
    sp = FakeSpeaker()
    svc = VoiceOutputService(cfg(), sp)
    svc.handle({"type": "reply", "text": "Good evening, Sir."})
    _join(svc)
    assert sp.spoken == ["Good evening, Sir."]


def test_trace_lines_are_not_spoken():
    sp = FakeSpeaker()
    svc = VoiceOutputService(cfg(), sp)
    for ev in ({"type": "tool_started", "tool": "get_battery"},
               {"type": "tool_finished", "tool": "get_battery", "result": "76%"},
               {"type": "plan"}, {"type": "snapshot", "greeting": "hi"},
               {"type": "confirmation_requested", "summary": "Close?"}):
        svc.handle(ev)
    _join(svc)
    assert sp.spoken == []  # only serif voice lines are spoken


def test_observation_is_spoken():
    sp = FakeSpeaker()
    svc = VoiceOutputService(cfg(), sp)
    svc.handle({"type": "observation", "detail": "You've been in four different apps."})
    _join(svc)
    assert sp.spoken == ["You've been in four different apps."]


def test_wake_barges_in():
    sp = FakeSpeaker()
    svc = VoiceOutputService(cfg(), sp)
    svc.handle({"type": "wake"})
    assert sp.stops >= 1  # playback stopped immediately


# ------------------------------- speaker engine ---------------------------
def test_speaker_streams_sentence_by_sentence_and_cleans():
    tts, player = FakeTTS(), FakePlayer()
    sp = Speaker(cfg(), tts, player)
    ok = sp.speak("▸ Good evening, Sir. It is one o'clock.")
    assert ok is True
    # glyphs stripped; split into two sentences -> two synth/play calls
    assert tts.texts == ["Good evening, Sir.", "It is one o'clock."]
    assert player.plays == 2


def test_barge_in_aborts_remaining_sentences():
    tts, player = FakeTTS(), FakePlayer(interrupt_at=1)  # first playback interrupted
    sp = Speaker(cfg(), tts, player)
    ok = sp.speak("First. Second. Third.")
    assert ok is False
    assert len(tts.texts) == 1  # stopped after the interrupted first sentence


def test_stop_stops_the_player():
    tts, player = FakeTTS(), FakePlayer()
    sp = Speaker(cfg(), tts, player)
    sp.stop()
    assert player.stops == 1


def test_clean_and_split():
    assert clean_for_speech("▸ get_battery — 76%") == "get battery 76%"  # glyph + underscore stripped
    assert clean_for_speech("✓ done ✕") == "done"
    assert clean_for_speech("   ") == ""
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
