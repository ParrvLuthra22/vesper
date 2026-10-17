"""The wake-word patience guard (>= N of the last M windows at/above the threshold) and the false-accept checks that
justify lowering the threshold to 0.35.

Fast tests use a fake model. The false-accept tests use the REAL openWakeWord model with synthetic audio only (speech-like
shaped noise, and Kokoro TTS of OTHER phrases when the local Kokoro files exist) — no microphone, no network."""

from __future__ import annotations

import dataclasses
import os
from fractions import Fraction

import numpy as np
import pytest

from voice.input.config import VoiceInputConfig
from voice.input.stages import WakeDetector

WINDOW = 1280


class FakeModel:
    def __init__(self, scores):
        self.scores, self.calls = list(scores), 0

    def predict(self, window):
        s = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return {"hey_jarvis": s}

    def reset(self):
        pass


def det(scores, hits=2, window=3, threshold=0.35):
    d = WakeDetector(VoiceInputConfig(wake_threshold=threshold, wake_patience_hits=hits, wake_patience_window=window))
    d._model, d._model_key = FakeModel(scores), "hey_jarvis"
    return d


def run(d, n):
    return d.process(np.zeros(WINDOW * n, dtype=np.int16))


def test_one_stray_window_above_the_threshold_does_not_wake():
    assert run(det([0.05, 0.9, 0.05, 0.05, 0.05]), 5) is False


def test_two_of_the_last_three_windows_wake_even_if_not_adjacent():
    assert run(det([0.05, 0.4, 0.1, 0.4]), 4) is True
    assert run(det([0.4, 0.1, 0.4]), 3) is True


def test_two_adjacent_windows_wake():
    assert run(det([0.36, 0.36]), 2) is True


def test_two_hits_spread_over_more_than_the_window_do_not_wake():
    assert run(det([0.4, 0.1, 0.1, 0.4, 0.1, 0.1, 0.4]), 7) is False


def test_just_below_the_threshold_never_wakes():
    assert run(det([0.349] * 10), 10) is False


def test_patience_state_survives_chunk_boundaries():
    d = det([0.4, 0.4])
    assert d.process(np.zeros(WINDOW, dtype=np.int16)) is False
    assert d.process(np.zeros(WINDOW, dtype=np.int16)) is True


def test_a_detection_clears_the_history_so_it_cannot_double_fire():
    d = det([0.9, 0.9, 0.9, 0.9])
    assert run(d, 2) is True
    assert run(d, 1) is False                                   # history cleared: needs 2 fresh hits again


def test_one_of_one_is_the_old_single_window_behaviour():
    assert run(det([0.9], hits=1, window=1), 1) is True


def test_patience_hits_larger_than_the_window_is_clamped_not_a_dead_detector():
    assert run(det([0.9, 0.9, 0.9], hits=5, window=3), 3) is True


# --------------------------------------------------------------------------- false accepts with the REAL model

def _shaped_noise(seconds: float, seed: int, syllable_hz: float) -> np.ndarray:
    """Speech-like noise: band-limited noise (300-3400 Hz) with a syllable-rate amplitude envelope."""
    from scipy import signal

    rng = np.random.default_rng(seed)
    n = int(16000 * seconds)
    x = rng.standard_normal(n)
    b, a = signal.butter(4, [300 / 8000, 3400 / 8000], btype="band")
    x = signal.lfilter(b, a, x)
    t = np.arange(n) / 16000
    env = 0.5 * (1 + np.sin(2 * np.pi * syllable_hz * t)) ** 2
    env *= (rng.random(int(seconds) + 2).repeat(16000)[:n] > 0.3)                 # words separated by gaps
    x = x * env
    x = x / (np.max(np.abs(x)) or 1.0) * 0.3
    return np.clip(np.rint(x * 32767), -32768, 32767).astype(np.int16)


def _real_detector(threshold=0.35, hits=2, window=3) -> WakeDetector:
    pytest.importorskip("openwakeword")
    d = WakeDetector(VoiceInputConfig(wake_threshold=threshold, wake_patience_hits=hits, wake_patience_window=window))
    try:
        d.load()
    except Exception as exc:                                    # model files not on this machine
        pytest.skip(f"wake model unavailable: {exc}")
    return d


def _fires(d: WakeDetector, pcm: np.ndarray) -> int:
    fires = 0
    for i in range(0, len(pcm) - 479, 480):
        if d.process(pcm[i:i + 480]):
            fires += 1
    return fires


def test_speech_like_noise_never_wakes_the_real_model_at_035_with_patience():
    d = _real_detector()
    total = sum(_fires(d, _shaped_noise(20.0, seed, hz)) for seed, hz in [(1, 3.0), (2, 4.5), (3, 6.0)])
    assert total == 0


def test_the_patience_guard_is_what_keeps_035_quiet_on_loud_noise():
    """Raw windows above 0.35 may occur on noise; the guard (not the threshold alone) is the false-accept defence."""
    d = _real_detector(hits=1, window=1)
    pcm = _shaped_noise(20.0, 11, 3.0)
    guarded = _real_detector(hits=2, window=3)
    assert _fires(guarded, pcm) <= _fires(d, pcm)                # never worse than a single window


KOKORO = "voice/models/kokoro-v1.0.onnx"
OTHER_PHRASES = ["What time is it today?", "Hey Siri, play some music.", "Hey Travis, are you coming to dinner?",
                 "Hey service, I would like a table.", "Jarvis is a name from a film", "Thanks, that is all."]


@pytest.mark.skipif(not os.path.exists(KOKORO), reason="local Kokoro model not present")
def test_tts_of_other_phrases_does_not_wake_the_real_model_at_035():
    from config.settings import load_config_dict
    from scipy import signal
    from voice.output.config import VoiceOutputConfig
    from voice.output.tts import KokoroTTS

    out = VoiceOutputConfig.from_app_config(load_config_dict())
    d = _real_detector()
    fires = []
    for voice in (out.tts_voice,):
        tts = KokoroTTS(dataclasses.replace(out, tts_voice=voice, speed=1.0))
        for phrase in OTHER_PHRASES:
            samples, rate = tts.synthesize(phrase)
            f = Fraction(16000, rate).limit_denominator(1000)
            x = signal.resample_poly(np.asarray(samples, dtype=np.float64), f.numerator, f.denominator)
            x = x / (np.max(np.abs(x)) or 1.0) * 0.5
            pcm = np.concatenate([np.zeros(8000), x, np.zeros(16000)])
            pcm = np.clip(np.rint(pcm * 32767), -32768, 32767).astype(np.int16)
            d.reset()
            fires.append((phrase, _fires(d, pcm)))
    assert [p for p, n in fires if n] == [], "these phrases falsely woke the detector"
