"""The voice-input state machine.

    idle → wake_detected → listening → transcribing → injected → idle

Each transition is emitted so a UI (the HUD, PV4) can reflect what the ears are
doing. Failures never crash the loop: no speech after wake times out back to
idle; an STT error logs and returns to idle.

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
from voice.input.stages import SpeechSegmenter, Transcriber, WakeDetector, make_vad

logger = logging.getLogger("vesper.voice")

Emitter = Callable[[VoiceTransition], None]


class VoiceInputPipeline:
    def __init__(
        self,
        config: VoiceInputConfig,
        wake: WakeDetector,
        transcriber: Transcriber,
        sink: TranscriptSink,
        emitter: Optional[Emitter] = None,
        vad_factory: Optional[Callable[[], object]] = None,
    ):
        self._config = config
        self._wake = wake
        self._transcriber = transcriber
        self._sink = sink
        self._emitter = emitter
        self._vad_factory = vad_factory or (lambda: make_vad(config))
        self._state = VoiceState.IDLE
        self._stop = False

    @property
    def state(self) -> VoiceState:
        return self._state

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
            except Exception:
                logger.exception("wake stage error; staying idle")
        return False

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
        except Exception as exc:
            logger.error("STT failed, returning to idle: %s", exc)
            return ""

    def _flare_hint(self) -> None:
        """Wake intent → the HUD should flare the star. PV4 wires the flare;
        for now the WAKE_DETECTED transition (emitted above) carries the intent
        and we simply log it."""
        logger.info("wake detected — (HUD star flare intent; wired in PV4)")
