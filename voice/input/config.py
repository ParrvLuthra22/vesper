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
    #: Between an injected command and the next utterance of the same conversation: waiting for the reply to be spoken.
    FOLLOWUP = "followup"


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
    #: 0.35 with the patience guard below (>= `wake_patience_hits` of the last `wake_patience_window` 80 ms windows at or
    #: above it). openWakeWord's own default of 0.5 needed the phrase said slowly and clearly in the live test.
    wake_threshold: float = 0.35
    wake_patience_hits: int = 2
    wake_patience_window: int = 3
    wake_vad_threshold: float = 0.0  # openWakeWord's optional built-in VAD gate

    # Listen stage (VAD): capture until end-of-speech.
    vad_backend: str = "webrtcvad"  # "webrtcvad" | "silero"
    vad_aggressiveness: int = 2  # webrtcvad 0..3
    vad_silence_ms: int = 1200  # trailing silence that ends an utterance (voice.vad.silence_ms)
    vad_speech_prob: float = 0.5  # silero speech-probability threshold
    listen_timeout_ms: int = 6000  # no speech after wake -> back to idle
    #: The mic must deliver its first frame within this long or voice input exits as
    #: "unavailable" (blocked permission prompt, another app holding the device). 0 = off.
    first_frame_timeout_seconds: float = 15.0
    max_utterance_ms: int = 30000  # hard cap on a single utterance (voice.vad.max_utterance_s)
    vad_preroll_ms: int = 300  # audio kept from before speech is detected, so the first word is not clipped
    vad_min_speech_ms: int = 250  # a voiced burst shorter than this (wake-word tail, chime, click) is not an utterance
    vad_onset_ms: int = 90  # speech must persist this long before an utterance counts as started

    # Conversation session (voice.session.*): see voice/input/session.py.
    followup_s: float = 8.0
    greet_after_idle_min: float = 60.0
    chime: bool = True
    reply_wait_s: float = 25.0
    end_phrases: tuple = ("thanks", "thank you", "thats all", "that is all", "stop", "nevermind", "never mind", "cancel",
                          "goodbye")

    # STT stage (faster-whisper).
    whisper_model: str = "base.en"
    whisper_compute_type: str = "int8"
    whisper_device: str = "cpu"
    whisper_beam_size: int = 1
    whisper_language: str = "en"

    # Audio.
    sample_rate: int = 16000
    frame_ms: int = 30  # 30 ms frames (480 samples @ 16 kHz) — webrtcvad-legal
    #: Which microphone. "builtin" (default) = the MacBook's own mic, resolved explicitly at startup — never whatever macOS
    #: has as default. Or an exact device name, a sounddevice index, or "default" (the system default, only by asking).
    device: Any = "builtin"
    mic_device: Optional[Any] = None  # LEGACY alias of `device` (used only when `device` is not set)
    #: Optional input gain, applied before the wake word / VAD / Whisper. OFF by default (see voice/input/gain.py).
    gain_db: float = 0.0
    auto_gain: bool = False

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
        voice = (config.get("voice", {}) or {})
        wake = (voice.get("wake", {}) or {})
        vad = (voice.get("vad", {}) or {})
        session = (voice.get("session", {}) or {})

        def pick(section: Dict[str, Any], key: str, legacy: Any) -> Any:
            val = section.get(key)
            return legacy if val is None else val

        # voice.wake.threshold is the documented knob; voice.input.wake_threshold is honoured for old config files.
        threshold = wake.get("threshold")
        threshold = g("wake_threshold", cls.wake_threshold) if threshold is None else threshold
        legacy_mic = g("mic_device", cls.mic_device)
        device = g("device", legacy_mic if legacy_mic is not None else cls.device)
        return cls(
            enabled=bool(g("enabled", cls.enabled)),
            wake_model=str(wake.get("model_path") or g("wake_model", cls.wake_model)),
            wake_model_fallback=str(g("wake_model_fallback", cls.wake_model_fallback)),
            wake_threshold=float(threshold),
            wake_patience_hits=int(pick(wake, "patience_hits", cls.wake_patience_hits)),
            wake_patience_window=int(pick(wake, "patience_window", cls.wake_patience_window)),
            wake_vad_threshold=float(g("wake_vad_threshold", cls.wake_vad_threshold)),
            vad_backend=str(g("vad_backend", cls.vad_backend)),
            vad_aggressiveness=int(g("vad_aggressiveness", cls.vad_aggressiveness)),
            vad_silence_ms=int(pick(vad, "silence_ms", g("vad_silence_ms", cls.vad_silence_ms))),
            vad_preroll_ms=int(pick(vad, "preroll_ms", cls.vad_preroll_ms)),
            vad_min_speech_ms=int(pick(vad, "min_speech_ms", cls.vad_min_speech_ms)),
            vad_onset_ms=int(pick(vad, "onset_ms", cls.vad_onset_ms)),
            followup_s=float(pick(session, "followup_s", cls.followup_s)),
            greet_after_idle_min=float(pick(session, "greet_after_idle_min", cls.greet_after_idle_min)),
            chime=bool(pick(session, "chime", cls.chime)),
            reply_wait_s=float(pick(session, "reply_wait_s", cls.reply_wait_s)),
            end_phrases=tuple(str(p) for p in (session.get("end_phrases") or cls.end_phrases)),
            vad_speech_prob=float(g("vad_speech_prob", cls.vad_speech_prob)),
            listen_timeout_ms=int(g("listen_timeout_ms", cls.listen_timeout_ms)),
            first_frame_timeout_seconds=float(g("first_frame_timeout_seconds", cls.first_frame_timeout_seconds)),
            max_utterance_ms=int(float(vad["max_utterance_s"]) * 1000) if vad.get("max_utterance_s") is not None
            else int(g("max_utterance_ms", cls.max_utterance_ms)),
            whisper_model=str(g("whisper_model", cls.whisper_model)),
            whisper_compute_type=str(g("whisper_compute_type", cls.whisper_compute_type)),
            whisper_device=str(g("whisper_device", cls.whisper_device)),
            whisper_beam_size=int(g("whisper_beam_size", cls.whisper_beam_size)),
            whisper_language=str(g("whisper_language", cls.whisper_language)),
            sample_rate=int(g("sample_rate", cls.sample_rate)),
            frame_ms=int(g("frame_ms", cls.frame_ms)),
            device=device,
            gain_db=float(g("gain_db", cls.gain_db)),
            auto_gain=bool(g("auto_gain", cls.auto_gain)),
            mic_device=legacy_mic,
            gateway_host=str(gw.get("host", cls.gateway_host)),
            gateway_port=int(gw.get("port", cls.gateway_port)),
            gateway_token=str(token),
        )
