"""Configuration and shared types for the voice-input module.

This module is deliberately independent of the old VoiceAgent. Voice input is
JUST ANOTHER CLIENT: it wakes, listens, transcribes, and injects the transcript
into the Brain through the same path the CLI/gateway use — no special planner
path.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class VoiceState(str, enum.Enum):
    """The voice-input state machine."""

    IDLE = "idle"
    WAKE_DETECTED = "wake_detected"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    INJECTED = "injected"


@dataclass(frozen=True)
class VoiceTransition:
    """A single state-machine transition, emitted so a UI (the HUD, PV4) can
    reflect what the ears are doing."""

    state: VoiceState
    previous: VoiceState
    detail: str = ""


@dataclass
class VoiceInputConfig:
    """All voice-input settings. Everything is off by default — the user opts
    in via settings.yaml (voice.input.*)."""

    enabled: bool = False

    # Wake stage (openWakeWord). `wake_model` is either a bundled pretrained
    # name ("hey_jarvis", "alexa", "hey_mycroft") or a path to a custom .onnx.
    wake_model: str = "hey_jarvis"
    #: Used when `wake_model` names a custom .onnx that is not on disk yet —
    #: e.g. "wake_up_daddys_home" before its one-time training run has been
    #: done (voice/TRAINING.md). Lets the custom phrase be configured ahead of
    #: time without leaving voice input dead in the meantime. Set to "" to
    #: make a missing custom model a hard failure instead.
    wake_model_fallback: str = "hey_jarvis"
    wake_threshold: float = 0.5
    wake_vad_threshold: float = 0.0  # openWakeWord's optional built-in VAD gate

    # Listen stage (VAD): capture until end-of-speech.
    vad_backend: str = "webrtcvad"  # "webrtcvad" | "silero"
    vad_aggressiveness: int = 2  # webrtcvad 0..3
    vad_silence_ms: int = 800  # trailing silence that ends an utterance
    vad_speech_prob: float = 0.5  # silero speech-probability threshold
    listen_timeout_ms: int = 6000  # no speech after wake -> back to idle
    max_utterance_ms: int = 15000  # hard cap on a single utterance

    # STT stage (faster-whisper).
    whisper_model: str = "base.en"
    whisper_compute_type: str = "int8"
    whisper_device: str = "cpu"
    whisper_beam_size: int = 1
    whisper_language: str = "en"

    # Audio.
    sample_rate: int = 16000
    frame_ms: int = 30  # 30 ms frames (480 samples @ 16 kHz) — webrtcvad-legal
    mic_device: Optional[Any] = None  # sounddevice device index/name; None=default

    # Where to inject the transcript (the gateway — "just another client").
    gateway_host: str = "127.0.0.1"
    gateway_port: int = 8760
    gateway_token: str = ""

    #: Directory holding custom wake models (relative to repo root).
    models_dir: str = "voice/models"

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @classmethod
    def from_app_config(cls, config: Dict[str, Any]) -> "VoiceInputConfig":
        """Build from the app config dict (settings.yaml). Reads voice.input.*
        and the top-level gateway.* for the injection target; env overrides the
        gateway token."""
        import os

        vi = ((config.get("voice", {}) or {}).get("input", {}) or {})
        gw = config.get("gateway", {}) or {}

        def g(key: str, default: Any) -> Any:
            val = vi.get(key)
            return default if val is None else val

        token = os.getenv("VESPER_GATEWAY_TOKEN") or gw.get("token") or ""
        return cls(
            enabled=bool(g("enabled", cls.enabled)),
            wake_model=str(g("wake_model", cls.wake_model)),
            wake_model_fallback=str(g("wake_model_fallback", cls.wake_model_fallback)),
            wake_threshold=float(g("wake_threshold", cls.wake_threshold)),
            wake_vad_threshold=float(g("wake_vad_threshold", cls.wake_vad_threshold)),
            vad_backend=str(g("vad_backend", cls.vad_backend)),
            vad_aggressiveness=int(g("vad_aggressiveness", cls.vad_aggressiveness)),
            vad_silence_ms=int(g("vad_silence_ms", cls.vad_silence_ms)),
            vad_speech_prob=float(g("vad_speech_prob", cls.vad_speech_prob)),
            listen_timeout_ms=int(g("listen_timeout_ms", cls.listen_timeout_ms)),
            max_utterance_ms=int(g("max_utterance_ms", cls.max_utterance_ms)),
            whisper_model=str(g("whisper_model", cls.whisper_model)),
            whisper_compute_type=str(g("whisper_compute_type", cls.whisper_compute_type)),
            whisper_device=str(g("whisper_device", cls.whisper_device)),
            whisper_beam_size=int(g("whisper_beam_size", cls.whisper_beam_size)),
            whisper_language=str(g("whisper_language", cls.whisper_language)),
            sample_rate=int(g("sample_rate", cls.sample_rate)),
            frame_ms=int(g("frame_ms", cls.frame_ms)),
            mic_device=g("mic_device", cls.mic_device),
            gateway_host=str(gw.get("host", cls.gateway_host)),
            gateway_port=int(gw.get("port", cls.gateway_port)),
            gateway_token=str(token),
        )
