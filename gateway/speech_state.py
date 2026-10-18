"""Is Vesper still working out or saying a reply? Voice input asks (GET /speech) before it opens the follow-up window, so the
window starts AFTER the reply is spoken, never in the middle of it.

Sources of truth:
  * turns   — the gateway itself: a user turn is in flight from /message until the Brain finished it;
  * speech  — voice output reports (POST /speech {"speaking": true|false}) when it starts and finishes a line;
  * queued  — a reply was forwarded to voice output but synthesis has not started yet (a grace period covers Kokoro's
              first-sentence latency, so a quick gap between "turn done" and "speaking" is not read as "all quiet").
Everything is time-bounded, so a crashed voice output cannot hold the window shut: `speaking` expires after `max_speaking_s`.
"""
from __future__ import annotations

import time
from typing import Callable, Dict


class SpeechState:
    def __init__(self, clock: Callable[[], float] = time.monotonic, synth_grace_s: float = 4.0, tail_s: float = 0.5,
                 max_speaking_s: float = 120.0):
        self._clock = clock
        self._grace, self._tail, self._max = synth_grace_s, tail_s, max_speaking_s
        self._turns = 0
        self._speaking_since = None
        self._hold_until = 0.0
        self.ended_at = None                      # when the assistant last STOPPED speaking (our clock)

    def turn_started(self) -> None:
        self._turns += 1

    def turn_finished(self) -> None:
        self._turns = max(0, self._turns - 1)

    def reply_queued(self) -> None:
        """A VoiceOutputEvent was forwarded: speech is expected within `synth_grace_s`."""
        self._hold_until = max(self._hold_until, self._clock() + self._grace)

    def set_speaking(self, speaking: bool) -> None:
        now = self._clock()
        if speaking:
            self._speaking_since = now
            self._hold_until = 0.0
        else:
            self._speaking_since = None
            self.ended_at = now
            self._hold_until = now + self._tail                    # the next line of the same reply may follow at once

    @property
    def speaking(self) -> bool:
        return self._speaking_since is not None and self._clock() - self._speaking_since < self._max

    def audible(self) -> bool:
        """Is Vesper speaking, or about to (a reply was queued for voice output)? A turn that is merely thinking is not audible."""
        return self.speaking or self._clock() < self._hold_until

    def busy(self) -> bool:
        return self._turns > 0 or self.speaking or self._clock() < self._hold_until

    def since_speech_end(self) -> "float | None":
        """0 while it speaks; seconds since it stopped; None if it has never spoken."""
        if self.audible():
            return 0.0
        return None if self.ended_at is None else self._clock() - self.ended_at

    def snapshot(self) -> Dict[str, object]:
        return {"busy": self.busy(), "turns_active": self._turns, "speaking": self.speaking, "audible": self.audible()}
