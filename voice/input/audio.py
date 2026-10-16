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
    """Live microphone via sounddevice (imported lazily, or injected for tests).

    Always yields `frame_samples` int16 mono samples at `sample_rate` (16 kHz for the wake word and Whisper):
      * if the device can be opened at that rate, it is opened at that rate;
      * if not (a mic that only offers 44.1 / 48 kHz), it is opened at its native rate and converted with an anti-alias
        filter (voice/input/resample.py). It never feeds 48 kHz audio labelled as 16 kHz (speech slowed 3x, a third of its pitch), which
        would make the wake word deaf without any error.
    `device` is a sounddevice index or name; voice input always passes an explicitly resolved one (voice/input/devices.py).
    After the stream is open, `open_rate` and `resampled` say what actually happened (for the startup log line)."""

    def __init__(self, sample_rate: int, frame_samples: int, device: Optional[object] = None, sd_module: object = None):
        self._sample_rate = sample_rate
        self._frame_samples = frame_samples
        self._device = device
        self._sd = sd_module
        self._stream = None
        self.open_rate: Optional[int] = None
        self.resampled = False

    def _negotiate(self, sd) -> int:
        """The rate to open the device at: the wanted one if it accepts it, else its native rate."""
        def accepts(rate: int) -> bool:
            try:
                sd.check_input_settings(device=self._device, channels=1, dtype="int16", samplerate=rate)
                return True
            except Exception:  # noqa: BLE001 — PortAudio raises PortAudioError / ValueError for an unsupported rate
                return False

        if accepts(self._sample_rate):
            return self._sample_rate
        info = sd.query_devices(self._device, "input") if self._device is None else sd.query_devices(self._device)
        native = int(round(float(info["default_samplerate"])))
        if not accepts(native):
            raise RuntimeError(f"the input device accepts neither {self._sample_rate} Hz nor its native {native} Hz")
        return native

    def frames(self) -> Iterator[np.ndarray]:
        if self._sd is None:
            import sounddevice as sd_real

            sd = sd_real
        else:
            sd = self._sd
        rate = self._negotiate(sd)
        resampler = None
        for _attempt in range(2):
            resampler = None
            if rate != self._sample_rate:
                from voice.input.resample import StreamResampler

                resampler = StreamResampler(rate, self._sample_rate, self._frame_samples)
            blocksize = resampler.block_in if resampler is not None else self._frame_samples
            self._stream = sd.InputStream(samplerate=rate, channels=1, dtype="int16", blocksize=blocksize, device=self._device)
            actual = int(round(float(getattr(self._stream, "samplerate", rate))))
            if actual == rate:
                break
            # PortAudio gave a different rate than asked for: trust the stream, not the request.
            self.close()
            rate = actual
        self.open_rate, self.resampled = rate, resampler is not None
        self._stream.start()
        try:
            while True:
                data, _overflowed = self._stream.read(blocksize)
                block = np.frombuffer(bytes(data), dtype=np.int16)
                if resampler is None:
                    yield block
                else:
                    out = resampler.push(block)
                    if out is not None:
                        yield out
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
