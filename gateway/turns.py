"""One active turn at a time per session — latest wins.

Before this, every /message started its own planner loop: two utterances meant two turns in flight, each emitting its own tool
events and reply, rendered by the HUD and the voice in whatever order they finished (3 turns in flight at once in the 2026-10-17
sessions). Now:

  * a new utterance while a turn is THINKING (no reply yet)  -> the in-flight turn is cancelled and replaced;
  * a new utterance while Vesper is SPEAKING                 -> barge-in: speech stops (a WakeEvent, which voice output obeys),
                                                                the in-flight turn is cancelled, the new one starts;
  * a cancelled turn emits nothing more: its task is cancelled (asyncio cancels the Brain handler awaiting inside bus.emit) AND
    the gateway drops any reply whose correlation id belongs to a cancelled turn, so nothing reaches the HUD or voice output.

End phrases ("thanks", "that's all", "stop") never get here: voice input ends the session itself (voice/input/session.py).
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any, Awaitable, Callable, Deque, Optional, Set
from uuid import UUID

logger = logging.getLogger(__name__)


class TurnManager:
    def __init__(self, emit: Callable[[Any], Awaitable[None]], make_event: Callable[[str], Any],
                 barge_in: Callable[[], Awaitable[None]], is_speaking: Callable[[], bool],
                 on_start: Callable[[], None] = lambda: None, on_end: Callable[[], None] = lambda: None):
        self._emit, self._make_event, self._barge_in, self._is_speaking = emit, make_event, barge_in, is_speaking
        self._on_start, self._on_end = on_start, on_end
        self._task: Optional["asyncio.Task[None]"] = None
        self._current: Optional[UUID] = None
        self._cancelled: Set[UUID] = set()
        self._order: Deque[UUID] = deque(maxlen=256)
        #: What the last submit() did: "started" | "replaced" (was thinking) | "barge-in" (was speaking). For logs and tests.
        self.last_action = ""
        self.cancelled_count = 0

    def is_cancelled(self, correlation_id: Optional[UUID]) -> bool:
        return correlation_id is not None and correlation_id in self._cancelled

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    def submit(self, text: str) -> UUID:
        """Start a turn for `text`; any turn still in flight is cancelled first. Returns the new turn's id."""
        thinking = self.active
        speaking = self._is_speaking()
        if thinking:
            self._cancel_current()
        event = self._make_event(text)
        turn_id = event.event_id
        self._current = turn_id
        self.last_action = "barge-in" if speaking else "replaced" if thinking else "started"
        self._task = asyncio.create_task(self._run(event, barge=speaking))
        return turn_id

    def _cancel_current(self) -> None:
        if self._current is not None:
            self._cancelled.add(self._current)
            self._order.append(self._current)
            while len(self._cancelled) > self._order.maxlen:        # bounded memory
                self._cancelled.discard(self._order.popleft())
            self.cancelled_count += 1
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def _run(self, event: Any, barge: bool) -> None:
        self._on_start()
        try:
            if barge:
                await self._barge_in()                              # stop speech BEFORE the new turn can produce any
            await self._emit(event)
        except asyncio.CancelledError:
            logger.info("turn cancelled (a newer utterance replaced it)")
            raise
        finally:
            self._on_end()
