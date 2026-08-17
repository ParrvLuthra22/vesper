"""Configuration for voice output (Kokoro TTS)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class VoiceOutputConfig:
    enabled: bool = False

    # Kokoro-82M. A measured, lower-register voice fits the butler restraint.
    # bm_lewis measured lowest among the British male voices (median F0 ~100Hz
    # vs bm_george's ~142Hz) — see voice/output/README.md for the full table.
    tts_voice: str = "bm_lewis"
    speed: float = 0.92  # a touch slower — composed, unhurried
    lang: str = "en-gb"

    streaming: bool = True  # speak sentence-by-sentence as the reply streams

    # Also speak proactive call-outs (observations are serif voice, not traces).
    speak_observations: bool = True

    # Kokoro model files (downloaded once — see voice/output/README).
    model_path: str = "voice/models/kokoro-v1.0.onnx"
    voices_path: str = "voice/models/voices-v1.0.bin"
    sample_rate: int = 24000  # Kokoro output rate

    # 8GB guard (PF6): queue synthesis while the local LLM rescue holds ~2.5GB,
    # rather than loading Kokoro's ~520MB alongside it and swapping the machine.
    defer_while_local_llm: bool = True
    defer_timeout_seconds: float = 20.0

    # Gateway to subscribe to (VoiceOutputEvent -> wire "reply").
    gateway_host: str = "127.0.0.1"
    gateway_port: int = 8760
    gateway_token: str = ""

    @classmethod
    def from_app_config(cls, config: Dict[str, Any]) -> "VoiceOutputConfig":
        vo = ((config.get("voice", {}) or {}).get("output", {}) or {})
        gw = config.get("gateway", {}) or {}

        def g(key: str, default: Any) -> Any:
            val = vo.get(key)
            return default if val is None else val

        token = os.getenv("VESPER_GATEWAY_TOKEN") or gw.get("token") or ""
        return cls(
            enabled=bool(g("enabled", cls.enabled)),
            tts_voice=str(g("tts_voice", cls.tts_voice)),
            speed=float(g("speed", cls.speed)),
            lang=str(g("lang", cls.lang)),
            streaming=bool(g("streaming", cls.streaming)),
            speak_observations=bool(g("speak_observations", cls.speak_observations)),
            model_path=str(g("model_path", cls.model_path)),
            voices_path=str(g("voices_path", cls.voices_path)),
            sample_rate=int(g("sample_rate", cls.sample_rate)),
            defer_while_local_llm=bool(g("defer_while_local_llm", cls.defer_while_local_llm)),
            defer_timeout_seconds=float(g("defer_timeout_seconds", cls.defer_timeout_seconds)),
            gateway_host=str(gw.get("host", cls.gateway_host)),
            gateway_port=int(gw.get("port", cls.gateway_port)),
            gateway_token=str(token),
        )
