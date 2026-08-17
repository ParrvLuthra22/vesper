"""The voice-input state machine.

    idle → wake_detected → listening → transcribing → injected → idle

Each transition is emitted so a UI (the HUD, PV4) can reflect what the ears are
doing.

Failures never crash the loop, and are split by whether they can recover:
a transient one (no speech after wake, a bad frame, an STT hiccup) returns to
idle and keeps listening; a permanent one (VoiceInputUnavailable — a missing
package or an unloadable model) disables voice input for the session after a
single log line. Stage `load()` runs per frame, so without that split a
missing dependency raised — and logged a full traceback — for every frame the
mic produced.


The pipeline is synchronous and dependency-injected: wake detector, a VAD
factory, transcriber, sink, and emitter are all provided, so tests drive it with
fakes and no ML deps.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional

import numpy as np

from voice.input.audio import AudioSource
from voice.input.config import VoiceInputConfig, VoiceState, VoiceTransition
from voice.input.sink import TranscriptSink
from voice.input.stages import (
    SpeechSegmenter,
    Transcriber,
    VoiceInputUnavailable,
    WakeDetector,
    make_vad,
)

logger = logging.getLogger("vesper.voice")

Emitter = Callable[[VoiceTransition], None]

#: How many recoverable stage errors get a full traceback before the rest are
#: merely counted. A fault repeating on every frame is a defect, not news.
_TRANSIENT_LOG_LIMIT = 3


class VoiceInputPipeline:
    def __init__(
        self,
        config: VoiceInputConfig,
        wake: WakeDetector,
        transcriber: Transcriber,
        sink: TranscriptSink,
        emitter: Optional[Emitter] = None,
        vad_factory: Optional[Callable[[], object]] = None,
        on_wake: Optional[Callable[[], None]] = None,
    ):
        self._config = config
        self._wake = wake
        self._transcriber = transcriber
        self._sink = sink
        self._emitter = emitter
        self._on_wake = on_wake
        self._vad_factory = vad_factory or (lambda: make_vad(config))
        self._state = VoiceState.IDLE
        self._stop = False
        #: Set when a permanent failure shut the pipeline down; None while healthy.
        self._disabled_reason: Optional[str] = None
        self._transient_errors = 0

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def disabled_reason(self) -> Optional[str]:
        """Why voice input switched itself off, or None if it did not."""
        return self._disabled_reason

    def stop(self) -> None:
        self._stop = True

    def _transition(self, new_state: VoiceState, detail: str = "") -> None:
        previous, self._state = self._state, new_state
        logger.info("voice %s -> %s%s", previous.value, new_state.value, f" ({detail})" if detail else "")
        if self._emitter is not None:
            try:
                self._emitter(VoiceTransition(state=new_state, previous=previous, detail=detail))
            except Exception:
                logger.exception("voice emitter failed")

    # ------------------------------------------------------------------ #
    def run(self, source: AudioSource) -> None:
        """Drive the machine over a live source until stopped/exhausted."""
        try:
            self.run_stream(source.frames())
        finally:
            source.close()

    def run_stream(self, frames: Iterable[np.ndarray]) -> None:
        """Drive the machine over any frame iterator (a mic, or a test list)."""
        it = iter(frames)
        while not self._stop:
            if not self._await_wake(it):
                break  # source exhausted
            self._transition(VoiceState.WAKE_DETECTED)
            self._flare_hint()

            self._transition(VoiceState.LISTENING)
            audio = self._listen(it)
            if audio is None:
                self._transition(VoiceState.IDLE, "no speech")
                continue

            self._transition(VoiceState.TRANSCRIBING)
            text = self._transcribe(audio).strip()
            if not text:
                self._transition(VoiceState.IDLE, "empty transcript")
                continue

            self._sink.inject(text)
            self._transition(VoiceState.INJECTED, text)
            self._transition(VoiceState.IDLE)

    # ------------------------------------------------------------------ #
    def _await_wake(self, it) -> bool:
        for frame in it:
            if self._stop:
                return False
            try:
                if self._wake.process(frame):
                    return True
            except VoiceInputUnavailable as exc:
                # Permanent: a missing package or an unloadable model will not
                # fix itself between frames. Log ONE line and stop, rather than
                # re-raising the same import failure for every frame the mic
                # produces (which is what made this a hot loop).
                self._disable(str(exc))
                return False
            except Exception:
                # Transient (a malformed frame): stay idle and keep listening,
                # but never let a repeating fault flood the log.
                self._log_transient("wake stage error; staying idle")
        return False

    def _disable(self, reason: str) -> None:
        """Shut the pipeline down for the rest of the session, with one line."""
        self._stop = True
        self._disabled_reason = reason
        logger.warning("Voice input disabled for this session — %s", reason)

    def _log_transient(self, message: str) -> None:
        """
        Log a recoverable stage error at most `_TRANSIENT_LOG_LIMIT` times.

        A fault that recurs on every frame is a defect, not news; past the
        limit it is counted and reported once at shutdown instead.
        """
        self._transient_errors += 1
        if self._transient_errors <= _TRANSIENT_LOG_LIMIT:
            logger.exception(message)
            if self._transient_errors == _TRANSIENT_LOG_LIMIT:
                logger.warning(
                    "further voice stage errors will be counted, not logged "
                    "(suppressing repeat tracebacks)"
                )

    def _listen(self, it) -> Optional[np.ndarray]:
        segmenter = SpeechSegmenter(self._config, self._vad_factory())
        for frame in it:
            if self._stop:
                return None
            result = segmenter.push(frame)
            if result == SpeechSegmenter.DONE:
                return segmenter.audio
            if result == SpeechSegmenter.TIMEOUT:
                return None
        # Source ended mid-utterance: use whatever we captured, if any.
        audio = segmenter.audio
        return audio if len(audio) else None

    def _transcribe(self, audio: np.ndarray) -> str:
        try:
            return self._transcriber.transcribe(audio)
        except VoiceInputUnavailable as exc:
            # No transcriber means wake detection is pointless — the pipeline
            # can hear but never understand. Stop rather than looping.
            self._disable(str(exc))
            return ""
        except Exception as exc:
            logger.error("STT failed, returning to idle: %s", exc)
            return ""

    def _flare_hint(self) -> None:
        """Wake → fire the cinematic reveal: signal the gateway (HUD star flare,
        spoken greeting, voice-output barge-in). PV4."""
        if self._on_wake is not None:
            try:
                self._on_wake()
            except Exception:
                logger.exception("on_wake hook failed")
