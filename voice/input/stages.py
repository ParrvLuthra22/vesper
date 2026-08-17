"""The three ears: wake detection (openWakeWord), speech segmentation (VAD),
and transcription (faster-whisper).

Every heavy dependency is imported lazily inside the stage that needs it, so
the pipeline and its tests import cleanly with none of them installed.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from voice.input.config import VoiceInputConfig

logger = logging.getLogger("vesper.voice")

#: openWakeWord processes 80 ms (1280-sample) windows at 16 kHz.
_WAKE_WINDOW = 1280


class VoiceInputUnavailable(RuntimeError):
    """
    A stage cannot work at all and never will this session — a missing
    dependency, or a model that will not load.

    Distinct from a transient per-frame error on purpose. Stage `load()`
    is called on every frame, so without this distinction a missing
    `openwakeword` raised ModuleNotFoundError 16,000 times a second into
    the pipeline's catch-all, which logged a full traceback for each. The
    pipeline treats this exception as terminal: one line, then stop.
    """


def _permanent(stage: str, exc: BaseException, fix: str) -> "VoiceInputUnavailable":
    return VoiceInputUnavailable(f"{stage} unavailable: {exc}. {fix}")


class WakeDetector:
    """Wraps openWakeWord. `wake_model` is either a bundled pretrained name
    (e.g. "hey_jarvis") or a path to a custom .onnx (e.g. a trained
    "wake up daddy's home"). Buffers arbitrary frames into 80 ms windows."""

    def __init__(self, config: VoiceInputConfig):
        self._config = config
        self._model = None
        self._model_key: Optional[str] = None
        self._buf = np.zeros(0, dtype=np.int16)
        #: Set once a load has permanently failed, so the next frame re-raises
        #: from memory instead of retrying the same import.
        self._load_error: Optional[VoiceInputUnavailable] = None

    def _local_model_path(self) -> Optional[str]:
        """A custom model on disk (explicit path or under models_dir)?"""
        wm = self._config.wake_model
        candidates = [Path(wm), Path(self._config.models_dir) / wm]
        if not wm.endswith(".onnx"):
            candidates.append(Path(self._config.models_dir) / f"{wm}.onnx")
        for c in candidates:
            if c.exists():
                return str(c.resolve())
        return None

    @staticmethod
    def _resolve_bundled(name: str) -> Optional[str]:
        """Map a bundled name ("hey_jarvis") to its shipped .onnx
        (hey_jarvis_v0.1.onnx), skipping the shared feature models."""
        import glob

        import openwakeword

        base = os.path.dirname(openwakeword.__file__)
        stem = name[:-5] if name.endswith(".onnx") else name
        skip = ("melspectrogram", "embedding", "silero")
        matches = [
            p
            for p in glob.glob(os.path.join(base, "resources", "models", f"*{stem}*.onnx"))
            if not any(s in os.path.basename(p).lower() for s in skip)
        ]
        return matches[0] if matches else None

    def load(self) -> None:
        if self._model is not None:
            return
        # Called per frame: a previous permanent failure must re-raise
        # immediately rather than re-attempting the import.
        if self._load_error is not None:
            raise self._load_error

        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError as exc:
            self._load_error = _permanent(
                "wake word (openwakeword)", exc, "Install it: pip install openwakeword"
            )
            raise self._load_error from exc

        try:
            model_ref = self._local_model_path()
            if model_ref is None:
                # Bundled pretrained name. Resolve from disk FIRST: the model
                # is usually already there, and download_models() is an
                # unbounded network call that otherwise ran on every single
                # start — it hung startup for >10 minutes when the release
                # host was slow. Only reach for the network when the file is
                # genuinely absent.
                model_ref = self._resolve_bundled(self._config.wake_model)

                # A custom name (e.g. "wake_up_daddys_home") that is neither a
                # file on disk nor a bundled model has simply not been trained
                # yet. Fall back to the configured pretrained model rather than
                # leaving voice input dead — one line, then carry on.
                if model_ref is None and self._config.wake_model_fallback:
                    fallback = self._config.wake_model_fallback
                    if fallback != self._config.wake_model:
                        resolved = self._resolve_bundled(fallback)
                        if resolved is not None:
                            logger.warning(
                                "wake model %r not found (not trained yet?); falling back to %r. "
                                "See voice/TRAINING.md.",
                                self._config.wake_model, fallback,
                            )
                            model_ref = resolved

                if model_ref is None:
                    logger.info(
                        "wake model %r not found locally; downloading openWakeWord "
                        "pretrained models (one time)", self._config.wake_model,
                    )
                    try:
                        openwakeword.utils.download_models()
                    except Exception as exc:
                        logger.warning("openWakeWord model download failed: %s", exc)
                    model_ref = self._resolve_bundled(self._config.wake_model) or self._config.wake_model

            self._model = Model(wakeword_models=[model_ref], inference_framework="onnx")
            # The dict key openWakeWord reports scores under (basename without ext).
            self._model_key = Path(model_ref).stem if model_ref.endswith(".onnx") else model_ref
        except Exception as exc:
            self._load_error = _permanent(
                f"wake model '{self._config.wake_model}'",
                exc,
                "Check voice.input.wake_model in settings.yaml.",
            )
            raise self._load_error from exc

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.int16)
        if self._model is not None and hasattr(self._model, "reset"):
            try:
                self._model.reset()
            except Exception:
                pass

    def score(self, predictions: dict) -> float:
        if self._model_key and self._model_key in predictions:
            return float(predictions[self._model_key])
        return float(max(predictions.values())) if predictions else 0.0

    def process(self, frame: np.ndarray) -> bool:
        """Feed one frame. Returns True when the wake word is detected."""
        self.load()
        self._buf = np.concatenate([self._buf, frame.astype(np.int16)])
        detected = False
        while len(self._buf) >= _WAKE_WINDOW:
            window = self._buf[:_WAKE_WINDOW]
            self._buf = self._buf[_WAKE_WINDOW:]
            predictions = self._model.predict(window)
            if self.score(predictions) >= self._config.wake_threshold:
                detected = True
                self.reset()
                break
        return detected


class _WebrtcVad:
    def __init__(self, config: VoiceInputConfig):
        import webrtcvad

        self._vad = webrtcvad.Vad(config.vad_aggressiveness)
        self._rate = config.sample_rate

    def is_speech(self, frame: np.ndarray) -> bool:
        return self._vad.is_speech(frame.astype(np.int16).tobytes(), self._rate)


class _SileroVad:
    def __init__(self, config: VoiceInputConfig):
        from silero_vad import load_silero_vad

        self._model = load_silero_vad(onnx=True)
        self._rate = config.sample_rate
        self._threshold = config.vad_speech_prob

    def is_speech(self, frame: np.ndarray) -> bool:
        import torch  # silero's callable expects a torch tensor

        audio = torch.from_numpy((frame.astype(np.float32) / 32768.0))
        prob = float(self._model(audio, self._rate).item())
        return prob >= self._threshold


def make_vad(config: VoiceInputConfig):
    return _SileroVad(config) if config.vad_backend == "silero" else _WebrtcVad(config)


class SpeechSegmenter:
    """After wake, captures audio until end-of-speech. Frame-driven so it is
    trivially testable: push frames, get one of CONTINUE / DONE / TIMEOUT."""

    CONTINUE = "continue"
    DONE = "done"
    TIMEOUT = "timeout"

    def __init__(self, config: VoiceInputConfig, vad=None):
        self._config = config
        self._vad = vad if vad is not None else make_vad(config)
        self._frame_ms = config.frame_ms
        self._silence_frames_needed = max(1, config.vad_silence_ms // config.frame_ms)
        self._timeout_frames = max(1, config.listen_timeout_ms // config.frame_ms)
        self._max_frames = max(1, config.max_utterance_ms // config.frame_ms)
        self._preroll_frames = max(1, 300 // config.frame_ms)

        self._preroll: List[np.ndarray] = []
        self._captured: List[np.ndarray] = []
        self._started = False
        self._elapsed = 0
        self._trailing_silence = 0

    @property
    def audio(self) -> np.ndarray:
        if not self._captured:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self._captured)

    def push(self, frame: np.ndarray) -> str:
        self._elapsed += 1
        speech = self._vad.is_speech(frame)

        if not self._started:
            # Keep a short pre-roll so we don't clip the utterance's start.
            self._preroll.append(frame)
            if len(self._preroll) > self._preroll_frames:
                self._preroll.pop(0)
            if speech:
                self._started = True
                self._captured.extend(self._preroll)
                self._captured.append(frame)
                self._trailing_silence = 0
                return self.CONTINUE
            if self._elapsed >= self._timeout_frames:
                return self.TIMEOUT
            return self.CONTINUE

        # Speech has started: accumulate until trailing silence ends it.
        self._captured.append(frame)
        if speech:
            self._trailing_silence = 0
        else:
            self._trailing_silence += 1
            if self._trailing_silence >= self._silence_frames_needed:
                return self.DONE
        if len(self._captured) >= self._max_frames:
            return self.DONE
        return self.CONTINUE


class Transcriber:
    """Wraps faster-whisper. int16 audio -> text."""

    def __init__(self, config: VoiceInputConfig):
        self._config = config
        self._model = None
        self._load_error: Optional[VoiceInputUnavailable] = None

    def load(self) -> None:
        if self._model is not None:
            return
        if self._load_error is not None:
            raise self._load_error

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            self._load_error = _permanent(
                "transcription (faster-whisper)", exc, "Install it: pip install faster-whisper"
            )
            raise self._load_error from exc

        try:
            self._model = WhisperModel(
                self._config.whisper_model,
                device=self._config.whisper_device,
                compute_type=self._config.whisper_compute_type,
            )
        except Exception as exc:
            self._load_error = _permanent(
                f"whisper model '{self._config.whisper_model}'",
                exc,
                "Check voice.input.whisper_model in settings.yaml.",
            )
            raise self._load_error from exc

    def transcribe(self, audio_int16: np.ndarray) -> str:
        self.load()
        audio = audio_int16.astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(
            audio,
            language=self._config.whisper_language,
            beam_size=self._config.whisper_beam_size,
            vad_filter=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
