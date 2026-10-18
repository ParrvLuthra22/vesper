"""The voice-input state machine.

    idle → wake_detected → listening → transcribing → injected → followup → listening → … → idle

One wake word opens a CONVERSATION (voice/input/session.py): a chime (no spoken greeting), you speak, the reply is spoken,
then a follow-up window (voice.session.followup_s, default 8 s) in which the next utterance needs no wake word. The window
ends on silence, or when you say "thanks" / "that's all" / "stop". With followup_s = 0 it is the old single-shot machine:
… injected → idle.

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
import time
from typing import Callable, Iterable, Optional

import numpy as np

from voice.input.audio import AudioSource
from voice.input.config import VoiceInputConfig, VoiceState, VoiceTransition
from voice.input.echo import EchoGate
from voice.input.session import Chime, SessionClock, clean_transcript, is_end_phrase
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
        on_wake: Optional[Callable[..., object]] = None,
        chime: Optional[Chime] = None,
        session_clock: Optional[SessionClock] = None,
        reply_busy: Optional[Callable[[], Optional[bool]]] = None,
        drain: Optional[Callable[[], object]] = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        gate: Optional[EchoGate] = None,
    ):
        self._config = config
        self._wake = wake
        self._transcriber = transcriber
        self._sink = sink
        self._emitter = emitter
        self._on_wake = on_wake
        self._chime = chime if chime is not None else Chime(enabled=False)
        self._session_clock = session_clock if session_clock is not None else SessionClock(config.greet_after_idle_min)
        #: Asks the gateway whether a reply is still being worked out / spoken. None = never wait for replies.
        self._reply_busy = reply_busy if reply_busy is not None else getattr(sink, "reply_busy", None)
        #: Drops audio queued while we were busy (BufferedSource.flush), so Vesper's own voice is not heard as the user.
        self._drain = drain
        self._mono, self._sleep = monotonic, sleep
        #: Half-duplex gate (voice/input/echo.py): None = no gating.
        self._gate = gate
        self.gated_frames = 0
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
    def warm_up(self) -> None:
        """Load the wake model eagerly. Raises VoiceInputUnavailable if it can't load."""
        self._wake.load()

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
            self._flare_hint(greet=self._session_clock.should_greet())
            self._chime.play()
            self._conversation(it)
            self._reset_wake()

    def _conversation(self, it) -> None:
        """One wake = one conversation: listen, answer, then keep listening for `followup_s` without the wake word."""
        cfg = self._config
        first = True
        while not self._stop:
            timeout_ms = None if first else int(cfg.followup_s * 1000)
            self._transition(VoiceState.LISTENING, "" if first else "follow-up")
            audio = self._listen(it, timeout_ms)
            if audio is None:
                self._transition(VoiceState.IDLE, "no speech" if first else "follow-up window closed")
                return

            self._transition(VoiceState.TRANSCRIBING)
            text = clean_transcript(self._transcribe(audio))
            if not text:
                self._transition(VoiceState.IDLE, "empty transcript")
                return

            if is_end_phrase(text, cfg.end_phrases):
                if "stop" in text.lower().split():
                    self._flare_hint(greet=False, animate=False)          # barge-in: Vesper stops talking
                self._transition(VoiceState.IDLE, "conversation ended")
                return

            accepted = self._sink.inject(text)
            if accepted is False:                                 # the gateway refused it (echo of our own reply / voice paused)
                self._transition(VoiceState.IDLE, "transcript not accepted")
                return
            self._session_clock.touch()
            self._transition(VoiceState.INJECTED, text)
            first = False
            if cfg.followup_s <= 0:
                self._transition(VoiceState.IDLE)
                return
            self._transition(VoiceState.FOLLOWUP, "waiting for the reply")
            self._wait_for_reply()

    def _wait_for_reply(self) -> None:
        """Hold the follow-up window until the reply has been spoken (or `reply_wait_s` passes), then discard the audio that
        piled up meanwhile — it is mostly Vesper's own voice."""
        if self._reply_busy is not None:
            deadline = self._mono() + self._config.reply_wait_s
            self._sleep(0.4)                                   # let the gateway start the turn
            quiet = 0
            while self._mono() < deadline and not self._stop:
                try:
                    busy = self._reply_busy()
                except Exception:
                    busy = None
                if busy is None:                               # gateway not answering: do not wait for it
                    break
                quiet = 0 if busy else quiet + 1
                if quiet >= 2:
                    break
                self._sleep(0.2)
            self._sleep(0.3)                                   # room tail of the last syllable
        if self._drain is not None:
            try:
                self._drain()
            except Exception:
                logger.debug("drain failed", exc_info=True)

    def _reset_wake(self) -> None:
        reset = getattr(self._wake, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    def _await_wake(self, it) -> bool:
        for frame in it:
            if self._stop:
                return False
            if self._gate is not None:
                if self._gate.paused_for() > 0:                  # loop breaker: deaf for a while, wake word included
                    continue
                self._wake.speaking = self._gate.speaking()      # raise the wake bar while it talks (barge-in still possible)
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

    def _listen(self, it, timeout_ms: Optional[int] = None) -> Optional[np.ndarray]:
        segmenter = SpeechSegmenter(self._config, self._vad_factory(), timeout_ms=timeout_ms)
        for frame in it:
            if self._stop:
                return None
            if self._gate is not None and (self._gate.active() or self._gate.paused_for() > 0):
                segmenter.hold()                                  # never START capturing while it speaks (or just spoke)
                self.gated_frames += 1
                continue
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

    def _flare_hint(self, greet: bool = False, animate: bool = True) -> None:
        """Wake → signal the gateway: HUD star flare and voice-output barge-in always; the spoken persona greeting only when
        `greet` (after a long idle period, never on every wake)."""
        if self._on_wake is not None:
            try:
                try:
                    self._on_wake(greet=greet, animate=animate)
                except TypeError:                              # a plain zero-argument hook
                    self._on_wake()
            except Exception:
                logger.exception("on_wake hook failed")
