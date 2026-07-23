"""The speaking engine and the gateway-client service that drives it.

Voice output is JUST ANOTHER CLIENT: it subscribes to the gateway and speaks
Vesper's serif voice lines (wire "reply", and proactive "observation"). Mono
trace lines (tool_started/tool_finished/plan) are NEVER spoken — the service
simply doesn't route them. Barge-in: a "wake" event stops playback at once.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Iterable, List, Optional

from voice.output.config import VoiceOutputConfig
from voice.output.tts import AudioPlayer, KokoroTTS

logger = logging.getLogger("vesper.voice.out")

#: Wire event types that are Vesper *speaking* (serif). Everything else — traces,
#: plans, confirmation cards, snapshots — is display-only and never spoken.
SPOKEN_TYPES = {"reply", "observation"}

_STRIP_CHARS = "▸✓✕○◦»«•·—▾►"
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def clean_for_speech(text: str) -> str:
    """Strip anything unspeakable (trace glyphs, markdown, stray symbols)."""
    t = text or ""
    for ch in _STRIP_CHARS:
        t = t.replace(ch, " ")
    t = re.sub(r"[*_`#>]+", " ", t)          # markdown
    t = re.sub(r"https?://\S+", "", t)        # URLs
    t = re.sub(r"\s+", " ", t).strip()
    return t


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


class Speaker:
    """Synthesizes + plays speech, sentence-by-sentence (so the first words
    come fast), and can be stopped instantly for barge-in."""

    def __init__(self, config: VoiceOutputConfig, tts: Optional[KokoroTTS] = None, player: Optional[AudioPlayer] = None):
        self._config = config
        self._tts = tts if tts is not None else KokoroTTS(config)
        self._player = player if player is not None else AudioPlayer()
        self._barge = threading.Event()

    def stop(self) -> None:
        """Barge-in: abort the current utterance immediately."""
        self._barge.set()
        self._player.stop()

    def speak(self, text: str) -> bool:
        """Speak one line. Returns False if cleaned to nothing or barged-in."""
        cleaned = clean_for_speech(text)
        if not cleaned:
            return False
        self._barge.clear()
        parts: Iterable[str] = split_sentences(cleaned) if self._config.streaming else [cleaned]
        for sentence in parts:
            if self._barge.is_set():
                return False
            samples, rate = self._tts.synthesize(sentence)
            if not self._player.play(samples, rate):  # interrupted
                return False
        return True


class VoiceOutputService:
    """Gateway WS client: routes spoken events to the Speaker; barges in on
    wake. Speaking runs on a worker thread so the receive loop never blocks."""

    def __init__(self, config: VoiceOutputConfig, speaker: Optional[Speaker] = None):
        self._config = config
        self._speaker = speaker if speaker is not None else Speaker(config)
        self._worker: Optional[threading.Thread] = None

    def handle(self, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == "wake":
            # Barge-in: whatever Vesper was saying, stop and listen.
            self._speaker.stop()
            return
        if mtype in SPOKEN_TYPES:
            text = str(msg.get("text") or msg.get("detail") or "").strip()
            if text:
                self._speak_async(text)

    def _speak_async(self, text: str) -> None:
        # A new line supersedes the last: barge the current, then speak.
        self._speaker.stop()
        self._worker = threading.Thread(target=self._speaker.speak, args=(text,), daemon=True)
        self._worker.start()
