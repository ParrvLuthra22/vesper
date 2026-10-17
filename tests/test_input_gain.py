"""Optional input gain (voice.input.gain_db / auto_gain): OFF by default, then clearly bounded. Synthetic frames only."""

from __future__ import annotations

import numpy as np
import pytest

from voice.input.audio import AudioSource
from voice.input.config import VoiceInputConfig
from voice.input.gain import GainSource, GainStage, wrap_with_gain


def tone(peak, n=480, i=0):
    t = (np.arange(n) + i * n) / 16000.0
    return (peak * 32767 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)


def peak_of(x):
    return float(np.max(np.abs(x))) / 32767.0


class Src(AudioSource):
    open_rate, resampled = 48000, True

    def __init__(self, frames):
        self._frames, self.closed = frames, False

    def frames(self):
        yield from self._frames

    def close(self):
        self.closed = True


# ------------------------------------------------------------------ off by default

def test_the_defaults_are_off_and_the_path_is_untouched():
    cfg = VoiceInputConfig.from_app_config({})
    assert cfg.gain_db == 0.0 and cfg.auto_gain is False
    src = Src([tone(0.1)])
    assert wrap_with_gain(src, cfg.gain_db, cfg.auto_gain) is src               # the very same object: nothing in the way
    f = tone(0.1)
    assert GainStage()(f) is f                                                   # and frames pass through without a copy


def test_shipped_settings_ship_gain_off():
    from config.settings import load_config_dict
    raw = load_config_dict()
    cfg = VoiceInputConfig.from_app_config(raw)
    assert cfg.gain_db == 0.0 and cfg.auto_gain is False
    assert raw["voice"]["input"]["gain_db"] == 0 and raw["voice"]["input"]["auto_gain"] is False


def test_config_values_are_read():
    cfg = VoiceInputConfig.from_app_config({"voice": {"input": {"gain_db": 6, "auto_gain": True}}})
    assert cfg.gain_db == 6.0 and cfg.auto_gain is True


# ------------------------------------------------------------------ fixed gain

def test_fixed_gain_scales_and_clips():
    boosted = GainStage(gain_db=6.0)(tone(0.1))
    assert peak_of(boosted) == pytest.approx(0.1995, rel=0.03)
    assert peak_of(GainStage(gain_db=-6.0)(tone(0.4))) == pytest.approx(0.2005, rel=0.03)
    clipped = GainStage(gain_db=30.0)(tone(0.5))
    assert clipped.dtype == np.int16 and peak_of(clipped) == pytest.approx(1.0, abs=0.001) and clipped.min() >= -32768


# ------------------------------------------------------------------ auto gain

def run_auto(frames, **kw):
    stage = GainStage(auto_gain=True, **kw)
    return stage, [stage(f) for f in frames]


def test_auto_gain_brings_a_quiet_voice_up_toward_minus_12_dbfs_but_never_past_the_cap():
    stage, out = run_auto([tone(0.05, i=i) for i in range(120)])                  # -26 dBFS speech
    final = peak_of(out[-1])
    assert final == pytest.approx(10 ** (-12 / 20), rel=0.2)                      # ~0.25
    assert stage.last_gain_db <= 30.0 + 1e-6
    _, quiet = run_auto([tone(0.002, i=i) for i in range(400)])                   # -54 dBFS: above the floor but needs > +30 dB
    assert peak_of(quiet[-1]) <= 0.002 * 10 ** (30 / 20) * 1.02                   # capped at +30 dB


def test_auto_gain_does_not_amplify_silence_or_the_noise_floor():
    rng = np.random.default_rng(3)
    hiss = [(rng.normal(0, 0.0004, 480) * 32767).astype(np.int16) for _ in range(200)]      # about -68 dBFS rms
    stage, out = run_auto(hiss)
    assert all(np.array_equal(a, b) for a, b in zip(hiss[-20:], out[-20:]))       # passed through unchanged
    assert stage._auto == 1.0
    _, silence = run_auto([np.zeros(480, dtype=np.int16)] * 50)
    assert all(np.all(f == 0) for f in silence)


def test_a_sudden_loud_sound_backs_the_gain_off_at_once_so_it_does_not_clip():
    stage = GainStage(auto_gain=True)
    for i in range(100):
        stage(tone(0.01, i=i))
    assert stage._auto > 5                                                          # it had been boosting the quiet input
    burst = stage(tone(0.8, i=100))
    assert peak_of(burst) <= 0.82 and not np.any(np.abs(burst.astype(np.int32)) >= 32767)   # the loud frame is NOT clipped
    assert stage._auto == pytest.approx(1.0, abs=0.05)                              # and the gain went back to ~1 immediately


def test_the_gain_rises_gradually_not_in_one_jump():
    stage = GainStage(auto_gain=True)
    peaks = [peak_of(stage(tone(0.03, i=i))) for i in range(30)]
    assert peaks[0] < peaks[10] < peaks[29] and peaks[1] - peaks[0] < 0.1          # smooth, no sudden 10x step


def test_fixed_and_auto_gain_combine():
    stage = GainStage(gain_db=6.0, auto_gain=True)
    out = [stage(tone(0.02, i=i)) for i in range(150)]
    assert peak_of(out[-1]) == pytest.approx(10 ** (-12 / 20), rel=0.25)
    assert stage.last_gain_db > 6.0


# ------------------------------------------------------------------ the wrapper

def test_gain_source_applies_the_stage_passes_attributes_through_and_closes_the_inner_source():
    inner = Src([tone(0.1, i=i) for i in range(3)])
    wrapped = wrap_with_gain(inner, 6.0, False)
    assert isinstance(wrapped, GainSource)
    got = list(wrapped.frames())
    assert [peak_of(f) for f in got] == pytest.approx([0.1995] * 3, rel=0.05)
    assert wrapped.open_rate == 48000 and wrapped.resampled is True
    wrapped.close()
    assert inner.closed


def test_voice_input_wraps_its_source_only_when_a_gain_is_configured(monkeypatch):
    from voice.input import service
    from voice.input.devices import InputDevice
    calls = []

    class FakePipeline:
        disabled_reason = None

        def warm_up(self): pass

        def run(self, source): calls.append(source)

        def stop(self): pass

    class FakeMic:
        resampled, open_rate = False, 16000

        def __init__(self, *a, **k): pass

        def close(self): pass

    dev = InputDevice(2, "MacBook Pro Microphone", 1, 48000.0)
    for gain, auto, wrapped in ((0.0, False, False), (6.0, False, True), (0.0, True, True)):
        cfg = VoiceInputConfig(enabled=True, gain_db=gain, auto_gain=auto)
        monkeypatch.setattr(service, "load_config", lambda c=cfg: c)
        monkeypatch.setattr(service, "resolve_device", lambda c, **k: dev)
        monkeypatch.setattr(service, "build_pipeline", lambda c, **k: FakePipeline())
        monkeypatch.setattr(service, "_AnnouncingMicSource", FakeMic)
        calls.clear()
        assert service.main() == 0
        assert isinstance(calls[0]._inner, GainSource) is wrapped, (gain, auto)       # service adds a BufferedSource around it
