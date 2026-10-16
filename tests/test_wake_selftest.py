"""`vesper mic-test --selftest`: a synthesized phrase through the live conversion/resample/chunk/predict path, at three levels,
with a direct-16-kHz control and negative controls. Fake detectors and fake TTS: no Kokoro, no wake model, no audio stream."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice.input import mictest, selftest
from voice.input.selftest import (
    LEVELS, PEAK_FULL, VERDICT_OK, VERDICT_SUSPECT, control_signals, frames_via_live_path, prepare_signal, run_selftest)

RATE_TTS = 24000


def voice_tone(freq=450.0, seconds=0.8):
    t = np.arange(int(RATE_TTS * seconds)) / RATE_TTS
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32), RATE_TTS


def dominant_hz(frame_stack, rate=16000):
    y = np.concatenate(frame_stack).astype(np.float64)
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return float(np.argmax(spec)) * rate / len(y)


class ToneDetector:
    """Stands in for the wake model: 'hears the phrase' when a frame is mostly energy near 450 Hz."""

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def feed(self, frame):
        x = frame.astype(np.float64)
        if np.max(np.abs(x)) < 200:
            return [0.0]
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        hz = float(np.argmax(spec)) * 16000 / len(x)
        tonal = spec[max(0, np.argmax(spec) - 3): np.argmax(spec) + 4].sum() / max(spec.sum(), 1e-9) > 0.6     # a pure tone, not noise
        return [0.99 if (400 <= hz <= 500 and tonal) else 0.05]


class DeafDetector(ToneDetector):
    def feed(self, frame):
        return [0.01]


# ------------------------------------------------------------------ the signal path

def test_prepare_signal_normalises_levels_pads_silence_and_converts_the_rate():
    samples, rate = voice_tone()
    for label, level in LEVELS:
        x = prepare_signal(samples, rate, 48000, level)
        assert x.dtype == np.int16
        peak = np.max(np.abs(x)) / 32768.0
        assert peak == pytest.approx(PEAK_FULL * level, rel=0.03)
        assert len(x) == pytest.approx(48000 * (0.8 + 2.0), abs=200)              # phrase + 1 s lead + 1 s tail
        assert np.all(x[:47000] == 0)                                              # the lead is digital silence
    full, third, tenth = (20 * np.log10(np.max(np.abs(prepare_signal(samples, rate, 48000, l))) / 32768.0) for _, l in LEVELS)
    assert full == pytest.approx(-1.0, abs=0.3) and third == pytest.approx(-11.5, abs=0.4) and tenth == pytest.approx(-21.0, abs=0.5)


def test_frames_via_live_path_are_480_sample_int16_frames_that_keep_the_pitch():
    samples, rate = voice_tone(450.0)
    frames = list(frames_via_live_path(prepare_signal(samples, rate, 48000, 1.0), 48000))
    assert all(f.dtype == np.int16 and len(f) == 480 for f in frames)
    loud = [f for f in frames if np.max(np.abs(f)) > 3000]
    assert abs(dominant_hz(loud) - 450.0) < 8.0                                    # not 150 Hz (the mislabelled-rate failure)
    direct = list(frames_via_live_path(prepare_signal(samples, rate, 16000, 1.0), 16000))
    assert abs(dominant_hz([f for f in direct if np.max(np.abs(f)) > 3000]) - 450.0) < 8.0


def test_the_live_path_does_not_drop_the_phrase_at_the_end_of_the_clip():
    """A phrase that ends right at the end of the audio is still fully delivered: the resampler's look-ahead is flushed."""
    n = 1440                                                                    # one 30 ms block at 48 kHz
    t = np.arange(n * 6) / 48000.0
    x = np.concatenate([np.zeros(n, dtype=np.int16), (0.5 * 32767 * np.sin(2 * np.pi * 450 * t)).astype(np.int16)])
    frames = list(frames_via_live_path(x, 48000))
    out_energy = sum(float(np.sum(f.astype(np.float64) ** 2)) for f in frames)
    in_energy = float(np.sum(x.astype(np.float64) ** 2)) / 3.0                  # 3x fewer samples at 16 kHz
    assert out_energy == pytest.approx(in_energy, rel=0.12)                    # nothing lost off the end


def test_a_device_only_detection_is_reported_with_a_note(monkeypatch):
    """The synthetic phrase is detected through the device-rate path but the direct 16 kHz control misses it."""
    real = selftest.frames_via_live_path

    def only_device_path(audio, rate, frame_samples=480, target_rate=16000):
        if rate == 16000:
            return iter([np.zeros(480, dtype=np.int16)] * 50)                  # the direct control sees silence
        return real(audio, rate, frame_samples, target_rate)

    monkeypatch.setattr(selftest, "frames_via_live_path", only_device_path)
    res, out = run(ToneDetector())
    assert res.verdict == VERDICT_OK
    assert "only the device-rate path detected it" in res.diagnosis and any("only the device-rate path" in l for l in out)


# ------------------------------------------------------------------ the verdicts

def run(detector, voices=None, **kw):
    out = []
    res = run_selftest(voices or {"v1": voice_tone}, detector, out=out.append, **kw)
    return res, out


def test_pipeline_ok_when_the_phrase_is_detected_at_every_level_and_both_paths():
    res, out = run(ToneDetector(), other_phrase=lambda: voice_tone(150.0))
    assert res.verdict == VERDICT_OK == "pipeline OK (synthetic phrase detected)"
    assert len(res.rows) == 3 * 2 and {r.level for r in res.rows} == {"full", "0.3x", "0.1x"}
    assert {r.path for r in res.rows} == {"device-rate path", "direct 16 kHz"}
    assert all(r.max_score >= 0.5 for r in res.rows)
    assert any("control silence" in l for l in out) and any("control other phrase" in l for l in out)
    assert res.diagnosis == ""


def test_pipeline_suspect_when_nothing_is_detected_and_it_says_the_fault_is_not_the_mic_path():
    res, out = run(DeafDetector())
    assert res.verdict == VERDICT_SUSPECT == "pipeline suspect (synthetic phrase not detected)"
    assert "not even the direct 16 kHz audio is detected" in res.diagnosis


def test_a_broken_resampler_is_told_apart_from_a_broken_model(monkeypatch):
    class BrokenResampler:
        latency_frames = 1

        def __init__(self, src, dst, frame):
            self.block_in = frame * (src // dst)

        def push(self, block):
            return np.asarray(block, dtype=np.int16)[:480]            # keeps the first 480 raw samples: pitch / 3

    monkeypatch.setattr("voice.input.resample.StreamResampler", BrokenResampler)
    res, _ = run(ToneDetector())
    assert res.verdict == VERDICT_SUSPECT
    assert "direct 16 kHz audio IS detected but the device-rate path loses it" in res.diagnosis
    assert res.best_direct_full >= 0.5 > res.best_full


def test_controls_that_fire_make_the_result_untrustworthy():
    class FiresOnEverything(ToneDetector):
        def feed(self, frame):
            return [0.9]

    res, out = run(FiresOnEverything())
    assert res.verdict == VERDICT_SUSPECT and "controls silence" in res.diagnosis and "proves nothing" in res.diagnosis


def test_each_run_starts_from_a_reset_detector():
    det = ToneDetector()
    run(det, voices={"a": voice_tone, "b": voice_tone})
    assert det.resets == 2 * 3 * 2 + 2                                   # voices x levels x paths, plus the 2 controls (silence, noise)


def test_the_simulated_device_rate_is_honoured():
    for rate in (44100, 48000):
        res, _ = run(ToneDetector(), device_rate=rate)
        assert res.verdict == VERDICT_OK, rate


def test_control_signals():
    c = control_signals(48000)
    assert set(c) == {"silence", "noise -30 dBFS"} and np.all(c["silence"] == 0)
    assert 20 * np.log10(np.sqrt(np.mean((c["noise -30 dBFS"] / 32767.0) ** 2))) == pytest.approx(-30, abs=1.5)


# ------------------------------------------------------------------ the promises + the command

def test_the_selftest_cannot_touch_audio_hardware_or_the_network():
    src = Path(selftest.__file__).read_text()
    for forbidden in ("sounddevice", "InputStream", "OutputStream", "sd.play", "httpx", "requests", "socket", "wave.open",
                      "np.save", "write_bytes", "open("):
        assert forbidden not in src, forbidden


def test_selftest_command_with_fakes(monkeypatch, capsys):
    from voice.input import service, stages
    from voice.input.config import VoiceInputConfig
    from voice.output import tts

    monkeypatch.setattr(service, "load_config", lambda: VoiceInputConfig(enabled=True))
    monkeypatch.setattr(service, "resolve_device", lambda c, **k: (_ for _ in ()).throw(AssertionError("no device calls with --rate")))

    class FakeTTS:
        def __init__(self, cfg):
            self.voice = cfg.tts_voice

        def synthesize(self, text):
            return voice_tone(150.0 if "time" in text else 450.0)

    class FakeWake(ToneDetector):
        def __init__(self, config):
            super().__init__()

        def load(self):
            pass

    monkeypatch.setattr(tts, "KokoroTTS", FakeTTS)
    monkeypatch.setattr(stages, "WakeDetector", FakeWake)
    assert mictest.main(["--selftest", "--rate", "48000"]) == 0
    out = capsys.readouterr().out
    assert "no microphone, nothing played/recorded/sent" in out and "48000 Hz" in out
    assert "verdict: pipeline OK (synthetic phrase detected)" in out
    assert out.count("max wake score") >= 18 + 3

    monkeypatch.setattr(FakeWake, "feed", lambda self, f: [0.0])
    assert mictest.main(["--selftest", "--rate", "48000"]) == 4
    assert "pipeline suspect (synthetic phrase not detected)" in capsys.readouterr().out
