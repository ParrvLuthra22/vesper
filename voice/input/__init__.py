"""Vesper's ears — a clean voice-input module, independent of the old
VoiceAgent. Wake (openWakeWord) → listen (VAD) → transcribe (faster-whisper) →
inject the transcript into the Brain as just another client.
"""
from voice.input.config import VoiceInputConfig, VoiceState, VoiceTransition
from voice.input.pipeline import VoiceInputPipeline
from voice.input.sink import CallableSink, GatewayRestSink, TranscriptSink

__all__ = [
    "VoiceInputConfig",
    "VoiceState",
    "VoiceTransition",
    "VoiceInputPipeline",
    "TranscriptSink",
    "CallableSink",
    "GatewayRestSink",
]
