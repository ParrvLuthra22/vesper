"""A conversation after ONE wake word (voice.session.*).

    wake -> chime (no greeting) -> you speak -> STT -> reply is spoken -> follow-up window (no wake word needed)
         -> you speak again ... -> silence for followup_s, or "thanks" / "that's all" / "stop" -> back to idle

This module holds the small, pure pieces so they are testable without audio: the greeting clock, the end-phrase test, the
cleaning of a transcript (wake-word residue, Whisper's hallucinated one-word clips) and the chime.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import time
import wave
from typing import Callable, Iterable, Optional

import numpy as np

logger = logging.getLogger("vesper.voice")

_NON_WORD = re.compile(r"[^a-z0-9 ]+")


def _norm(text: str) -> str:
    return " ".join(_NON_WORD.sub(" ", (text or "").lower().replace("'", "").replace("’", "")).split())


# --------------------------------------------------------------------------- transcript hygiene

#: "Hey Jarvis, what time is it" -> "what time is it". Only a LEADING wake phrase is removed.
_WAKE_RESIDUE = re.compile(r"^(?:(?:hey|hi|ok|okay)\s+)?(?:jarvis|vesper|travis|service)\b[\s,.!?:;-]*", re.IGNORECASE)

#: What Whisper emits for a breath, a click or the tail of the wake word. Never a command.
_JUNK = {"", "you", "uh", "um", "hmm", "mm", "mhm", "ah", "oh", "the", "so", "and", "i", "a", "huh", "bye", "okay", "ok"}


def strip_wake_residue(text: str) -> str:
    t = (text or "").strip()
    stripped = _WAKE_RESIDUE.sub("", t, count=1).strip()
    return stripped if stripped != t else t


def is_junk(text: str) -> bool:
    """True for transcripts that are not a request: empty, one filler word, or only punctuation."""
    return _norm(text) in _JUNK


def clean_transcript(text: str) -> str:
    """The text to inject: wake-word residue removed; "" when what is left is junk."""
    t = strip_wake_residue(text)
    return "" if is_junk(t) else t


_POLITE = {"jarvis", "vesper", "sir", "ok", "okay", "please", "then", "alright"}


def is_end_phrase(text: str, phrases: Iterable[str]) -> bool:
    """True when the WHOLE utterance is made of end-of-conversation phrases ("thanks", "that's all", "ok thanks Jarvis",
    "thank you, that's all"). A phrase inside a longer request ("stop the music") is NOT an end phrase — that is a request
    for the planner."""
    words = [w for w in _norm(text).split() if w not in _POLITE]
    if not words:
        return False
    ends = [tuple(_norm(p).split()) for p in phrases if _norm(p)]
    ok = [False] * (len(words) + 1)
    ok[0] = True
    for i in range(1, len(words) + 1):                       # can words[:i] be segmented into end phrases?
        ok[i] = any(ok[i - len(e)] and tuple(words[i - len(e):i]) == e for e in ends if len(e) <= i)
    return ok[len(words)]


# --------------------------------------------------------------------------- greeting clock

class SessionClock:
    """Decides when the persona greeting may play: at most once per `greet_after_idle_min` of idleness, never per wake.

    `last_activity` starts at construction time: the gateway already speaks a greeting when it boots, so the first wake of
    a fresh run does not repeat it. Activity (a wake, an utterance, the end of a session) pushes the idle timer forward."""

    def __init__(self, greet_after_idle_min: float, clock: Callable[[], float] = time.monotonic):
        self._idle_s = max(0.0, float(greet_after_idle_min)) * 60.0
        self._clock = clock
        self._last = clock()

    def touch(self) -> None:
        self._last = self._clock()

    def should_greet(self) -> bool:
        """True once, when the user comes back after the idle period. Calling it counts as activity."""
        now = self._clock()
        due = self._idle_s > 0 and now - self._last >= self._idle_s
        self._last = now
        return due


# --------------------------------------------------------------------------- chime

def _chime_samples(rate: int = 24000) -> np.ndarray:
    """A short rising two-note ping (~0.22 s), soft, with a fast fade so it does not click. Pure numpy."""
    def note(freq: float, seconds: float) -> np.ndarray:
        t = np.arange(int(rate * seconds)) / rate
        env = np.minimum(1.0, t / 0.01) * np.exp(-t * 14.0)
        return np.sin(2 * np.pi * freq * t) * env

    x = np.concatenate([note(880.0, 0.11), note(1320.0, 0.14)])
    x = x / (np.max(np.abs(x)) or 1.0) * 0.35
    return np.clip(np.rint(x * 32767), -32768, 32767).astype(np.int16)


CHIME_SECONDS = 0.25


class Chime:
    """Plays the wake chime without blocking. `player(path)` is injectable (tests); the default is macOS `afplay`.
    Any failure is swallowed: no chime must never mean no listening."""

    def __init__(self, enabled: bool = True, player: Optional[Callable[[str], object]] = None, directory: Optional[str] = None):
        self.enabled = enabled
        self._player = player or self._afplay
        self._dir = directory or tempfile.gettempdir()
        self._path: Optional[str] = None

    @staticmethod
    def _afplay(path: str) -> object:
        return subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _ensure_file(self) -> str:
        if self._path and os.path.exists(self._path):
            return self._path
        path = os.path.join(self._dir, "vesper-wake-chime.wav")
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(_chime_samples(24000).tobytes())
        self._path = path
        return path

    def play(self) -> bool:
        if not self.enabled:
            return False
        try:
            self._player(self._ensure_file())
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("wake chime not played: %s", type(exc).__name__)
            return False
