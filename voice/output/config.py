"""Configuration for voice output (Kokoro TTS)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class VoiceOutputConfig:
    enabled: bool = False

    # Kokoro-82M. A measured, lower-register voice fits the butler restraint;
    # bm_* are British male (see voice/output/README notes / voices test).
    tts_voice: str = "bm_george"
    speed: float = 0.92  # a touch slower — composed, unhurried
    lang: str = "en-gb"

    streaming: bool = True  # speak sentence-by-sentence as the reply streams

    # Also speak proactive call-outs (observations are serif voice, not traces).
    speak_observations: bool = True

    # Kokoro model files (downloaded once — see voice/output/README).
    model_path: str = "voice/models/kokoro-v1.0.onnx"
    voices_path: str = "voice/models/voices-v1.0.bin"
    sample_rate: int = 24000  # Kokoro output rate

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
            gateway_host=str(gw.get("host", cls.gateway_host)),
            gateway_port=int(gw.get("port", cls.gateway_port)),
            gateway_token=str(token),
        )
