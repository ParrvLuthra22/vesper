"""Audio sources — a live microphone and a WAV file (for tests/verification).

Both yield fixed-size int16 mono frames at the configured sample rate. Heavy
imports (sounddevice) are lazy so the rest of the module imports without them.
"""
from __future__ import annotations

import queue
import threading
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


class BufferedSource(AudioSource):
    """Reads the wrapped source on its own thread, so the microphone keeps being drained while the pipeline is busy
    (speech-to-text, the HTTP calls to the gateway, waiting for a reply to be spoken).

    Without this the pipeline thread was also the audio reader: during every transcription (0.3-5 s) and every POST the stream
    was not read, PortAudio overflowed and dropped audio, and a wake word said right after a reply was simply lost.
    The backlog is BOUNDED (`backlog_s`, default 2 s): when the consumer falls further behind the oldest frames are dropped, so
    it never replays minutes-old audio. `flush()` discards the backlog (used after a reply, so Vesper's own voice is not
    heard as the user)."""

    _END = object()

    def __init__(self, inner: AudioSource, frame_ms: int = 30, backlog_s: float = 2.0):
        self._inner = inner
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=max(2, int(backlog_s * 1000 / frame_ms)))
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._error: Optional[BaseException] = None
        self.dropped = 0

    def __getattr__(self, name):                      # open_rate / resampled / ... of the wrapped source
        return getattr(self._inner, name)

    def _put(self, item: object) -> None:
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()          # drop the OLDEST frame
                    self.dropped += 1
                except queue.Empty:
                    pass

    def _reader(self) -> None:
        try:
            for frame in self._inner.frames():
                if self._stop.is_set():
                    break
                self._put(frame)
        except BaseException as exc:                  # noqa: BLE001 — handed to the consumer thread
            self._error = exc
        finally:
            self._put(self._END)

    def flush(self) -> int:
        """Drop everything queued so far; returns how many frames were discarded."""
        n = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return n
            if item is self._END:                     # keep the end marker: the source really is finished
                self._queue.put_nowait(item)
                return n
            n += 1

    def frames(self) -> Iterator[np.ndarray]:
        self._thread = threading.Thread(target=self._reader, daemon=True, name="mic-reader")
        self._thread.start()
        while True:
            item = self._queue.get()
            if item is self._END:
                if self._error is not None:
                    raise self._error
                return
            yield item                                # type: ignore[misc]

    def close(self) -> None:
        self._stop.set()
        self._inner.close()


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
