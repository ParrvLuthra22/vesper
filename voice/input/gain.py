"""Optional input gain for a quiet microphone. OFF by default (gain_db = 0, auto_gain = false): the default audio path is
byte-for-byte what it was.

    voice.input.gain_db     a fixed boost (or cut) in dB applied to every frame before the wake word / VAD / Whisper
    voice.input.auto_gain   keep the recent PEAK near -12 dBFS (never more than +30 dB, never on near-silence)

Auto-gain is a slow envelope follower, not a compressor: the gain drops INSTANTLY when a loud sound arrives (so speech never
clips) and rises slowly. Frames whose recent peak is below the noise floor (-55 dBFS) are passed through untouched, so a
silent room is not amplified into hiss that the wake word then scores. The result is clipped to int16.

Raising the macOS Input volume (System Settings > Sound > Input) is better than software gain — it improves the signal before
the converter — but this exists for when that is already at maximum."""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from voice.input.audio import AudioSource


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


class GainStage:
    def __init__(self, gain_db: float = 0.0, auto_gain: bool = False, target_dbfs: float = -12.0, max_auto_db: float = 30.0,
                 floor_dbfs: float = -55.0, release: float = 0.995, rise: float = 0.15):
        self.fixed = db_to_lin(gain_db)
        self.auto = bool(auto_gain)
        self._target = db_to_lin(target_dbfs)
        self._max_auto = db_to_lin(max_auto_db)
        self._floor = db_to_lin(floor_dbfs)
        self._release, self._rise = release, rise
        self._env = 0.0               # recent peak, 0..1, after the fixed gain
        self._auto = 1.0              # current automatic gain (linear)
        self.last_gain_db = gain_db

    @property
    def active(self) -> bool:
        return self.auto or self.fixed != 1.0

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if not self.active:
            return frame
        x = frame.astype(np.float32) * self.fixed
        if self.auto:
            peak = float(np.max(np.abs(x))) / 32768.0 if len(x) else 0.0
            self._env = max(peak, self._env * self._release)             # instant attack, slow release
            if self._env < self._floor:
                want = 1.0                                               # near-silence: do not amplify the noise floor
            else:
                want = min(self._max_auto, max(1.0, self._target / self._env))
            if want < self._auto:
                self._auto = want                                        # a louder sound: back off at once
            else:
                self._auto += (want - self._auto) * self._rise           # a quieter one: come up gradually
            x = x * self._auto
            self.last_gain_db = 20.0 * np.log10(max(self.fixed * self._auto, 1e-9))
        return np.clip(np.rint(x), -32768, 32767).astype(np.int16)


class GainSource(AudioSource):
    """Wraps another source and applies a GainStage to every frame it yields."""

    def __init__(self, inner: AudioSource, stage: GainStage):
        self._inner, self._stage = inner, stage

    def frames(self) -> Iterator[np.ndarray]:
        for frame in self._inner.frames():
            yield self._stage(frame)

    def close(self) -> None:
        self._inner.close()

    def __getattr__(self, name):              # open_rate / resampled etc. of the wrapped MicSource
        return getattr(self._inner, name)


def wrap_with_gain(source: AudioSource, gain_db: float, auto_gain: bool) -> AudioSource:
    """`source` unchanged unless a gain is configured (so the default path is untouched)."""
    stage = GainStage(gain_db, auto_gain)
    return GainSource(source, stage) if stage.active else source
