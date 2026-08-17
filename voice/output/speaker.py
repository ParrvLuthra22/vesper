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
import time
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
    wake. Speaking runs on a worker thread so the receive loop never blocks.

    Also honours the 8GB guard: while the gateway reports the local LLM as
    active (wire type "local_model"), synthesis is queued instead of run, so
    Kokoro's ~520MB never loads alongside the ~2.5GB Ollama rescue model. The
    queue drains as soon as the local model goes idle.
    """

    def __init__(self, config: VoiceOutputConfig, speaker: Optional[Speaker] = None):
        self._config = config
        self._speaker = speaker if speaker is not None else Speaker(config)
        self._worker: Optional[threading.Thread] = None
        #: Set while a local LLM call is in flight (memory pressure).
        self._local_model_active = threading.Event()
        #: Lines received while deferred, spoken in order once it clears.
        self._deferred: List[str] = []
        self._deferred_lock = threading.Lock()
        self._defer_watchdog: Optional[threading.Thread] = None

    def handle(self, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == "wake":
            # Barge-in: whatever Vesper was saying, stop and listen. A queued
            # backlog is stale the moment the user speaks again — drop it.
            self._speaker.stop()
            with self._deferred_lock:
                self._deferred.clear()
            return
        if mtype == "local_model":
            self._set_local_model_active(bool(msg.get("active")))
            return
        if mtype in SPOKEN_TYPES:
            text = str(msg.get("text") or msg.get("detail") or "").strip()
            if text:
                self._speak_async(text)

    def _arm_defer_watchdog(self) -> None:
        """
        Guarantee a queued backlog is eventually spoken.

        The gateway signals "local model idle" from a different process, so a
        crash or a dropped websocket there would otherwise leave lines queued
        forever — muting Vesper permanently on a failure that has nothing to
        do with speech. One watchdog at a time; it flushes on timeout.
        """
        with self._deferred_lock:
            if self._defer_watchdog is not None and self._defer_watchdog.is_alive():
                return
            self._defer_watchdog = threading.Thread(
                target=self._defer_watchdog_run, daemon=True, name="tts-defer-watchdog"
            )
            self._defer_watchdog.start()

    def _defer_watchdog_run(self) -> None:
        deadline = time.monotonic() + self._config.defer_timeout_seconds
        while time.monotonic() < deadline:
            if not self._local_model_active.is_set():
                return  # cleared normally; _set_local_model_active drains it
            time.sleep(0.05)

        with self._deferred_lock:
            backlog, self._deferred = self._deferred, []
        if backlog:
            logger.warning(
                "local model still active after %.0fs — speaking %d queued line(s) anyway",
                self._config.defer_timeout_seconds, len(backlog),
            )
            for text in backlog:
                self._speaker.stop()
                threading.Thread(
                    target=self._speaker.speak, args=(text,), daemon=True
                ).start()

    def _set_local_model_active(self, active: bool) -> None:
        if not self._config.defer_while_local_llm:
            return
        if active:
            self._local_model_active.set()
            logger.debug("local model active — deferring speech synthesis")
            return

        self._local_model_active.clear()
        with self._deferred_lock:
            backlog, self._deferred = self._deferred, []
        if backlog:
            logger.info("local model idle — speaking %d deferred line(s)", len(backlog))
            for text in backlog:
                self._speak_async(text)

    def _speak_async(self, text: str) -> None:
        if self._config.defer_while_local_llm and self._local_model_active.is_set():
            with self._deferred_lock:
                self._deferred.append(text)
            logger.debug("queued while local model is active: %r", text[:60])
            self._arm_defer_watchdog()
            return

        # A new line supersedes the last: barge the current, then speak.
        self._speaker.stop()
        self._worker = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
        self._worker.start()

    def _speak_worker(self, text: str) -> None:
        """
        Speak on the worker thread.

        A local-model call can begin *while* we are already synthesizing. The
        deferral above only catches lines that arrive during one, so wait here
        too — bounded, so a local model that never reports idle (a crash, a
        dropped link) can't mute Vesper permanently.
        """
        if self._config.defer_while_local_llm:
            self._wait_for_local_model_idle()
        self._speaker.speak(text)

    def _wait_for_local_model_idle(self) -> None:
        """
        Block until the local model reports idle, or the deferral times out.

        Polled rather than Event-waited on purpose: threading.Event can only
        wait for a flag to be *set*, and the condition here is the opposite —
        waiting for `_local_model_active` to be CLEARED.
        """
        if not self._local_model_active.is_set():
            return

        deadline = time.monotonic() + self._config.defer_timeout_seconds
        while self._local_model_active.is_set():
            if time.monotonic() >= deadline:
                logger.warning(
                    "local model still active after %.0fs — speaking anyway",
                    self._config.defer_timeout_seconds,
                )
                return
            time.sleep(0.05)
