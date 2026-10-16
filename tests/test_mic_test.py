"""`vesper mic-test`: the verdicts, the 0.5 s report cadence, and the promises (records nothing, no STT, sends nothing).
Synthetic silence / noise / loud tone through a fake source; no microphone, no wake model, no network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice.input import mictest
from voice.input.devices import DeviceResolutionError, InputDevice
from voice.input.mictest import (
    NO_SIGNAL_PEAK, VERDICT_FIRED, VERDICT_NO_SIGNAL, VERDICT_QUIET, meter_bar, run_mic_test, to_dbfs, verdict_for)

FRAME = 480
N_10S = 334            # 334 x 30 ms frames = 10.02 s


def frames_of(kind, n=N_10S, seed=1):
    rng = np.random.default_rng(seed)
    for i in range(n):
        if kind == "silence":
            yield np.zeros(FRAME, dtype=np.int16)
        elif kind == "noise":            # a quiet room: ~-40 dBFS rms
            yield (rng.normal(0, 0.01, FRAME) * 32767).astype(np.int16)
        elif kind == "tone":             # loud: 0.8 full scale 440 Hz
            t = (np.arange(FRAME) + i * FRAME) / 16000.0
            yield (0.8 * 32767 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)


def const_score(v):
    return lambda frame: [v, v]


def run(kind, score=0.0, threshold=0.5, **kw):
    out = []
    res = run_mic_test(frames_of(kind), const_score(score), threshold=threshold, device="MacBook Pro Microphone",
                       out=out.append, **kw)
    return res, out


# ------------------------------------------------------------------ verdicts

def test_digital_silence_is_no_signal():
    res, _ = run("silence")
    assert res.verdict == VERDICT_NO_SIGNAL == "no signal" and res.peak == 0.0


def test_quiet_room_noise_is_signal_but_the_wake_word_never_scored():
    res, _ = run("noise", score=0.04)
    assert res.verdict == VERDICT_QUIET == "signal but wake word never above threshold"
    assert res.peak > NO_SIGNAL_PEAK and res.max_score == pytest.approx(0.04)


def test_a_loud_tone_with_a_low_wake_score_is_still_signal_but_not_fired():
    res, _ = run("tone", score=0.2)
    assert res.verdict == VERDICT_QUIET and res.peak == pytest.approx(0.8, abs=0.01)


def test_a_wake_score_at_the_threshold_is_fired_even_on_a_quiet_input():
    res, _ = run("noise", score=0.5)
    assert res.verdict == VERDICT_FIRED == "wake word fired"
    res2, _ = run("tone", score=0.93)
    assert res2.verdict == VERDICT_FIRED and res2.max_score == pytest.approx(0.93)


def test_threshold_is_honoured_not_hard_coded():
    assert run("tone", score=0.4, threshold=0.35)[0].verdict == VERDICT_FIRED
    assert run("tone", score=0.4, threshold=0.45)[0].verdict == VERDICT_QUIET


def test_verdict_function_boundaries():
    assert verdict_for(0.0, 0.0, 0.5) == VERDICT_NO_SIGNAL
    assert verdict_for(NO_SIGNAL_PEAK * 0.99, 0.0, 0.5) == VERDICT_NO_SIGNAL
    assert verdict_for(NO_SIGNAL_PEAK * 1.01, 0.0, 0.5) == VERDICT_QUIET
    assert verdict_for(0.0, 0.5, 0.5) == VERDICT_FIRED


# ------------------------------------------------------------------ the report

def test_one_line_every_half_second_plus_a_verdict_and_the_running_max():
    res, out = run("tone", score=0.3)
    reports = [l for l in out if "rms" in l and "peak" in l and l.lstrip()[0].isdigit()]
    assert len(reports) == 20                                          # 10 s / 0.5 s
    assert out[-1].startswith("verdict: signal but wake word never above threshold")
    assert all("wake" in l and "max" in l and "dBFS" in l for l in reports)
    assert "0.5s" in reports[0].replace(" ", "") or " 0.5s" in reports[0]
    assert res.seconds == pytest.approx(10.02, abs=0.05)


def test_the_max_score_is_a_running_maximum():
    scores = iter([0.1] * 40 + [0.7] + [0.2] * 400)
    res = run_mic_test(frames_of("noise"), lambda f: [next(scores)], threshold=0.5, out=lambda s: None)
    assert res.max_score == pytest.approx(0.7)


def test_the_meter_bar_follows_the_level():
    assert meter_bar(0.0) == "[" + "-" * 20 + "]"
    assert meter_bar(1.0) == "[" + "#" * 20 + "]"
    quiet, loud = meter_bar(0.003), meter_bar(0.3)
    assert quiet.count("#") < loud.count("#")
    assert to_dbfs(1.0) == pytest.approx(0.0) and to_dbfs(0.1) == pytest.approx(-20.0)


def test_it_stops_after_the_requested_seconds_even_if_the_source_goes_on():
    res = run_mic_test(frames_of("silence", n=10_000), const_score(0.0), threshold=0.5, seconds=3.0, out=lambda s: None)
    assert res.seconds == pytest.approx(3.0, abs=0.05)


# ------------------------------------------------------------------ the promises

def test_the_module_cannot_record_transcribe_or_send_anything():
    src = Path(mictest.__file__).read_text()
    for forbidden in ("faster_whisper", "Transcriber", "httpx", "requests", "urllib", "socket", "import wave", "soundfile",
                      "np.save", "tofile", "GatewayRestSink", ".inject(", "open(", "write_bytes"):
        assert forbidden not in src, forbidden
    assert "does NOT record or save audio" in src and "does NOT run speech-to-text" in src


def test_running_it_creates_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run("tone", score=0.9)
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------ the command, end to end with fakes

class FakeMic:
    resampled, open_rate = True, 48000

    def __init__(self, rate, frame_samples, device=None, **kw):
        self.device = device

    def frames(self):
        yield from frames_of("tone")

    def close(self):
        pass


class FakeWake:
    def __init__(self, config):
        pass

    def load(self):
        pass

    def feed(self, frame):
        return [0.9]


@pytest.fixture
def cmd(monkeypatch):
    from voice.input import audio, service, stages
    from voice.input.config import VoiceInputConfig
    monkeypatch.setattr(service, "load_config", lambda: VoiceInputConfig(enabled=True))
    builtin = InputDevice(2, "MacBook Pro Microphone", 1, 48000.0)
    monkeypatch.setattr(service, "resolve_device", lambda config, **k: builtin)
    monkeypatch.setattr(audio, "MicSource", FakeMic)
    monkeypatch.setattr(stages, "WakeDetector", FakeWake)
    return builtin


def test_the_command_prints_the_device_and_fires(cmd, capsys):
    assert mictest.main([]) == 0
    out = capsys.readouterr().out
    assert "MacBook Pro Microphone" in out and "index 2" in out and "native 48000 Hz" in out
    assert "nothing is recorded or saved" in out and "verdict: wake word fired" in out
    assert "converted to 16000 Hz" in out


def test_the_command_exits_3_on_no_signal(cmd, monkeypatch, capsys):
    monkeypatch.setattr(FakeMic, "frames", lambda self: frames_of("silence"))
    monkeypatch.setattr(FakeWake, "feed", lambda self, f: [0.0])
    assert mictest.main(["--seconds", "2"]) == 3
    assert "verdict: no signal" in capsys.readouterr().out


def test_the_command_exits_69_and_lists_the_devices_when_the_builtin_mic_is_missing(monkeypatch, capsys):
    from voice.input import service
    from voice.input.config import VoiceInputConfig
    monkeypatch.setattr(service, "load_config", lambda: VoiceInputConfig(enabled=True))

    def missing(config, **k):
        raise DeviceResolutionError("The built-in microphone was not found.", ["Parrv's AirPods Pro", "BlackHole 2ch"])
    monkeypatch.setattr(service, "resolve_device", missing)
    assert mictest.main([]) == 69
    err = capsys.readouterr().err
    assert "AirPods" in err and "BlackHole 2ch" in err


def test_the_command_is_reachable_as_vesper_mic_test(monkeypatch):
    from cli import app
    called = {}

    def fake_main(argv):
        called["argv"] = argv
        return 0
    monkeypatch.setattr("voice.input.mictest.main", fake_main)
    monkeypatch.setattr(app.sys, "argv", ["vesper", "mic-test", "--seconds", "3"])
    with pytest.raises(SystemExit) as exc:
        app.run()
    assert exc.value.code == 0 and called["argv"] == ["--seconds", "3"]
