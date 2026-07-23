"""Audio sources — a live microphone and a WAV file (for tests/verification).

Both yield fixed-size int16 mono frames at the configured sample rate. Heavy
imports (sounddevice) are lazy so the rest of the module imports without them.
"""
from __future__ import annotations

import wave
from typing import Iterator, Optional

import numpy as np


class AudioSource:
    """Yields fixed-size int16 mono frames of `frame_samples` at `sample_rate`."""

    def frames(self) -> Iterator[np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        pass


class MicSource(AudioSource):
    """Live microphone via sounddevice (imported lazily)."""

    def __init__(self, sample_rate: int, frame_samples: int, device: Optional[object] = None):
        self._sample_rate = sample_rate
        self._frame_samples = frame_samples
        self._device = device
        self._stream = None

    def frames(self) -> Iterator[np.ndarray]:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self._frame_samples,
            device=self._device,
        )
        self._stream.start()
        try:
            while True:
                data, _overflowed = self._stream.read(self._frame_samples)
                yield np.frombuffer(bytes(data), dtype=np.int16)
        finally:
            self.close()

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None


class FileSource(AudioSource):
    """Reads a 16 kHz mono 16-bit PCM WAV and yields frames. `loop_silence`
    appends trailing silence so a wake/utterance near the file's end still gets
    its end-of-speech detected."""

    def __init__(self, path: str, frame_samples: int, expected_rate: int = 16000, trailing_silence_ms: int = 1200):
        self._path = path
        self._frame_samples = frame_samples
        self._expected_rate = expected_rate
        self._trailing_silence_ms = trailing_silence_ms

    def frames(self) -> Iterator[np.ndarray]:
        with wave.open(self._path, "rb") as wf:
            if wf.getframerate() != self._expected_rate or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                raise ValueError(
                    f"{self._path}: expected {self._expected_rate}Hz mono 16-bit, got "
                    f"{wf.getframerate()}Hz {wf.getnchannels()}ch {wf.getsampwidth() * 8}-bit"
                )
            raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16)
        pad = int(self._expected_rate * self._trailing_silence_ms / 1000)
        samples = np.concatenate([samples, np.zeros(pad, dtype=np.int16)])
        for i in range(0, len(samples) - self._frame_samples + 1, self._frame_samples):
            yield samples[i : i + self._frame_samples]
