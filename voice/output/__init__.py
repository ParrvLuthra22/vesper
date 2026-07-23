"""Vesper's voice — Kokoro-82M TTS. Speaks the serif voice lines the HUD shows;
mono trace lines are never spoken. Barge-in on wake.
"""
from voice.output.config import VoiceOutputConfig
from voice.output.speaker import Speaker, VoiceOutputService, clean_for_speech, split_sentences

__all__ = ["VoiceOutputConfig", "Speaker", "VoiceOutputService", "clean_for_speech", "split_sentences"]
