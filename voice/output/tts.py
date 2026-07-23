"""Kokoro-82M synthesis and interruptible local playback. Heavy imports
(kokoro_onnx, sounddevice) are lazy so the module imports without them."""
from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np

from voice.output.config import VoiceOutputConfig


class KokoroTTS:
    def __init__(self, config: VoiceOutputConfig):
        self._config = config
        self._kokoro = None

    @staticmethod
    def _ensure_espeak_data() -> None:
        """Kokoro's phonemizer needs espeak-ng data. On macOS with
        `brew install espeak-ng`, point it there if not already set."""
        import os

        if os.environ.get("ESPEAK_DATA_PATH"):
            return
        for path in ("/opt/homebrew/share/espeak-ng-data", "/usr/local/share/espeak-ng-data"):
            if os.path.isdir(path):
                os.environ["ESPEAK_DATA_PATH"] = path
                os.environ.setdefault("ESPEAKNG_DATA_PATH", path)
                break

    def load(self) -> None:
        if self._kokoro is not None:
            return
        self._ensure_espeak_data()
        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro(self._config.model_path, self._config.voices_path)

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """text -> (float32 mono samples in [-1, 1], sample_rate)."""
        self.load()
        samples, sample_rate = self._kokoro.create(
            text, voice=self._config.tts_voice, speed=self._config.speed, lang=self._config.lang
        )
        return np.asarray(samples, dtype=np.float32), int(sample_rate)


class AudioPlayer:
    """Plays float32 audio and can be stopped mid-playback (barge-in)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._active = False

    def play(self, samples: np.ndarray, sample_rate: int) -> bool:
        """Play blocking. Returns False if interrupted by stop()."""
        import sounddevice as sd

        with self._lock:
            self._stop.clear()
            self._active = True
        try:
            sd.play(samples, sample_rate)
            # Interruptible wait: stop() sets the event and calls sd.stop().
            while sd.get_stream().active:
                if self._stop.wait(0.02):
                    return False
            return True
        finally:
            with self._lock:
                self._active = False

    def stop(self) -> None:
        import sounddevice as sd

        self._stop.set()
        try:
            sd.stop()
        except Exception:
            pass

    @property
    def active(self) -> bool:
        return self._active
