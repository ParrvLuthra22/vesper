"""Half-duplex gating: Vesper does not listen for a command while it is talking (nor for `tail_ms` after).

The built-in mic hears the speaker. Without this, the end of Vesper's own reply — or its failure message — was captured as the "user's"
next utterance, transcribed, and sent to the planner: a self-triggered turn, which also spent Groq's tokens-per-minute budget.

What the gate does NOT do: it never stops the wake-word detector (you must still be able to interrupt), it just raises the bar while
speaking (voice.echo.wake_threshold_while_speaking). Whether the assistant is speaking comes from the gateway (GET /speech `audible`:
voice output reports start/finish of each line, plus a short grace while a reply is queued), polled in the background so per-frame checks
cost nothing. If the gateway cannot be asked the gate stays open (fail open): an unreachable gateway has nothing to say anyway.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional


class EchoGate:
    def __init__(self, tail_ms: int = 700, clock: Callable[[], float] = time.monotonic,
                 source: Optional[Callable[[], Optional[Dict[str, object]]]] = None):
        self._tail = max(0, int(tail_ms)) / 1000.0
        self._clock, self._source = clock, source
        self._audible = False
        self._last_audible = float("-inf")
        self._pause_until = 0.0

    def update(self) -> None:
        """Refresh from the gateway snapshot (called by the poller thread; tests call it directly)."""
        snap = self._source() if self._source is not None else None
        if snap is None:
            return                                           # fail open: keep the last known state, never invent speech
        now = self._clock()
        self._audible = bool(snap.get("audible"))
        if self._audible:
            self._last_audible = now
        pause = float(snap.get("pause_s") or 0.0)
        self._pause_until = now + pause if pause > 0 else 0.0

    def active(self) -> bool:
        """True while it speaks and for tail_ms after: no utterance may START."""
        return self._audible or (self._clock() - self._last_audible) < self._tail

    def speaking(self) -> bool:
        """For the wake threshold: true during speech AND its tail (the room still rings)."""
        return self.active()

    def paused_for(self) -> float:
        """Seconds of loop-breaker pause left (voice input ignores everything, wake word included)."""
        return max(0.0, self._pause_until - self._clock())


class SpeechPoller:
    """Background thread: gate.update() every `interval` seconds."""

    def __init__(self, gate: EchoGate, interval: float = 0.15):
        self._gate, self._interval = gate, interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "SpeechPoller":
        self._thread = threading.Thread(target=self._run, daemon=True, name="echo-gate-poller")
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._gate.update()
            except Exception:  # noqa: BLE001 — never let the poller die
                pass
            self._stop.wait(self._interval)

    def stop(self) -> None:
        self._stop.set()
