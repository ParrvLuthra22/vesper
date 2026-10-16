"""The sample-rate path: a mic that only offers 44.1 / 48 kHz must still feed 16 kHz mono frames to the wake word, converted
with an anti-alias filter — never raw 48 kHz audio labelled as 16 kHz. Pure numpy; a fake sounddevice stands in for the
hardware, so no stream is ever opened."""

from __future__ import annotations

import numpy as np
import pytest

from voice.input.audio import MicSource
from voice.input.resample import StreamResampler, UnsupportedRate

FRAME = 480            # 30 ms at 16 kHz


def tone(rate, freq, seconds, amp=0.5, phase_samples=0):
    t = (np.arange(int(rate * seconds)) + phase_samples) / rate
    return (amp * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def push_all(rs: StreamResampler, x: np.ndarray):
    n = rs.block_in
    outs = [rs.push(x[i * n:(i + 1) * n]) for i in range(len(x) // n)]
    return [o for o in outs if o is not None]


def dominant_hz(y, rate=16000):
    spec = np.abs(np.fft.rfft(y.astype(np.float64) * np.hanning(len(y))))
    return float(np.argmax(spec)) * rate / len(y)


def rms(y):
    return float(np.sqrt(np.mean((y.astype(np.float64) / 32767.0) ** 2)))


# ------------------------------------------------------------------ the resampler

@pytest.mark.parametrize("src", [48000, 44100, 96000, 32000, 24000, 8000, 88200])
def test_a_tone_keeps_its_frequency_and_level_and_comes_out_in_480_sample_frames(src):
    rs = StreamResampler(src, 16000, FRAME)
    frames = push_all(rs, tone(src, 440.0, 2.0))
    assert all(f.dtype == np.int16 and len(f) == FRAME for f in frames)
    y = np.concatenate(frames[8:])                                    # skip the start-up
    assert abs(dominant_hz(y) - 440.0) < 3.0
    assert rms(y) == pytest.approx(0.5 / np.sqrt(2), rel=0.03)         # same loudness, not 3x louder or quieter


@pytest.mark.parametrize("src", [48000, 44100])
def test_a_tone_above_the_new_nyquist_is_filtered_not_aliased(src):
    """20 kHz does not exist at 16 kHz. Without the anti-alias filter it would fold into the audible band."""
    rs = StreamResampler(src, 16000, FRAME)
    y = np.concatenate(push_all(rs, tone(src, 20000.0, 2.0))[8:])
    assert rms(y) < 0.5 / np.sqrt(2) * 10 ** (-60 / 20)                # at least 60 dB down


def test_frame_boundaries_are_seamless():
    """No clicks at the joins between frames: the output is never steeper than the tone itself, and matches a one-shot
    resample of the whole signal."""
    from scipy import signal

    src = 48000
    x = tone(src, 1000.0, 1.0)
    rs = StreamResampler(src, 16000, FRAME)
    got = np.concatenate(push_all(rs, x)).astype(np.float64)
    seg = slice(3 * FRAME, len(got))
    assert np.max(np.abs(got[seg])) / 32767.0 == pytest.approx(0.5, rel=0.02)
    jumps = np.abs(np.diff(got[seg])) / 32767.0
    assert jumps.max() < 2 * np.pi * 1000.0 / 16000.0 * 0.5 * 1.05      # a click would be a step far larger than this
    # against the one-shot result, allowing for the streaming path's fixed delay (a whole number of samples)
    whole = signal.resample_poly(x.astype(np.float64), 1, 3)
    errs = [np.max(np.abs(got[seg] - whole[seg.start + k: seg.start + k + len(got[seg])])) / 32767.0
            for k in range(-FRAME, FRAME + 1) if seg.start + k >= 0 and seg.start + k + len(got[seg]) <= len(whole)]
    assert min(errs) < 0.01


def test_unsupported_and_misused_inputs():
    with pytest.raises(UnsupportedRate):
        StreamResampler(11025, 16000, FRAME)
    rs = StreamResampler(48000, 16000, FRAME)
    with pytest.raises(ValueError, match="expected 1440"):
        rs.push(np.zeros(100, dtype=np.int16))


def test_start_up_latency_is_one_to_two_frames():
    assert StreamResampler(48000, 16000, FRAME).latency_frames == 1
    assert StreamResampler(44100, 16000, FRAME).latency_frames <= 2
    rs = StreamResampler(48000, 16000, FRAME)
    assert rs.push(np.zeros(rs.block_in, dtype=np.int16)) is None       # nothing until the look-ahead has arrived


# ------------------------------------------------------------------ MicSource over a fake sounddevice

class FakeStream:
    def __init__(self, parent, kwargs):
        self.kwargs = kwargs
        self.parent = parent
        self.samplerate = parent.report_rate if parent.report_rate is not None else kwargs["samplerate"]
        self.started = self.closed = False
        self._n = 0

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True

    def read(self, n):
        x = tone(int(self.samplerate), 440.0, n / self.samplerate, phase_samples=self._n)[:n]
        self._n += n
        return x.reshape(-1, 1), False


class FakeSD:
    def __init__(self, accepts, native=48000.0, report_rate=None, devices=None):
        self.accepts, self.native, self.report_rate = set(accepts), native, report_rate
        self.streams = []
        self.checked = []

    def check_input_settings(self, device=None, channels=None, dtype=None, samplerate=None):
        self.checked.append((device, samplerate))
        if samplerate not in self.accepts:
            raise ValueError(f"Invalid sample rate {samplerate}")

    def query_devices(self, device=None, kind=None):
        return {"name": "MacBook Pro Microphone", "default_samplerate": self.native, "max_input_channels": 1}

    def InputStream(self, **kwargs):
        s = FakeStream(self, kwargs)
        self.streams.append(s)
        return s


def take(src: MicSource, n: int):
    it = src.frames()
    out = [next(it) for _ in range(n)]
    it.close()
    return out


def test_a_device_that_accepts_16k_is_opened_at_16k_with_no_resampling_and_the_pinned_index():
    sd = FakeSD(accepts={16000, 48000})
    src = MicSource(16000, FRAME, device=2, sd_module=sd)
    frames = take(src, 5)
    s = sd.streams[0]
    assert s.kwargs["samplerate"] == 16000 and s.kwargs["device"] == 2 and s.kwargs["blocksize"] == FRAME
    assert s.kwargs["channels"] == 1 and s.kwargs["dtype"] == "int16"
    assert src.open_rate == 16000 and src.resampled is False and s.closed
    assert all(len(f) == FRAME for f in frames)


def test_a_device_that_only_offers_48k_is_opened_at_48k_and_converted_to_16k_frames():
    sd = FakeSD(accepts={48000}, native=48000.0)
    src = MicSource(16000, FRAME, device=2, sd_module=sd)
    frames = take(src, 12)
    s = sd.streams[0]
    assert s.kwargs["samplerate"] == 48000 and s.kwargs["blocksize"] == 1440 and s.kwargs["device"] == 2
    assert src.open_rate == 48000 and src.resampled is True
    assert all(len(f) == FRAME and f.dtype == np.int16 for f in frames)
    y = np.concatenate(frames[4:])
    assert abs(dominant_hz(y) - 440.0) < 5.0                           # a 440 Hz tone at the mic is 440 Hz in the frames
    assert rms(y) == pytest.approx(0.5 / np.sqrt(2), rel=0.05)


def test_44k_is_converted_too():
    sd = FakeSD(accepts={44100}, native=44100.0)
    src = MicSource(16000, FRAME, device=1, sd_module=sd)
    frames = take(src, 14)
    assert sd.streams[0].kwargs["blocksize"] == 1323 and src.resampled
    assert abs(dominant_hz(np.concatenate(frames[5:])) - 440.0) < 5.0


def test_wrong_rate_would_otherwise_be_silent_this_is_the_failure_it_prevents():
    """Raw 48 kHz audio read as 16 kHz is the same speech slowed 3x and dropped to a third of its pitch — no error, just a
    deaf wake word."""
    raw = tone(48000, 440.0, 0.5)[:FRAME * 10]
    assert abs(dominant_hz(raw[: FRAME * 10]) - 440.0 / 3) < 5.0        # what the wake model would have been fed
    sd = FakeSD(accepts={48000})
    y = np.concatenate(take(MicSource(16000, FRAME, device=0, sd_module=sd), 12)[4:])
    assert abs(dominant_hz(y) - 440.0) < 5.0                            # what it is fed now


def test_a_device_that_accepts_nothing_we_can_use_fails_loudly():
    sd = FakeSD(accepts=set(), native=48000.0)
    with pytest.raises(RuntimeError, match="accepts neither 16000 Hz nor its native 48000 Hz"):
        next(MicSource(16000, FRAME, device=2, sd_module=sd).frames())


def test_portaudio_opening_at_a_different_rate_than_asked_is_corrected():
    sd = FakeSD(accepts={16000}, report_rate=44100)                     # asked for 16k, the stream says 44.1k
    src = MicSource(16000, FRAME, device=2, sd_module=sd)
    frames = take(src, 14)
    assert [s.kwargs["samplerate"] for s in sd.streams] == [16000, 44100]
    assert sd.streams[0].closed and src.open_rate == 44100 and src.resampled
    assert all(len(f) == FRAME for f in frames)
