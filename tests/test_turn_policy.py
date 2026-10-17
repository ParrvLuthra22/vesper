"""One active turn at a time, latest wins; barge-in while speaking; a cancelled turn emits nothing. Fake clock, fake Brain
(no LLM), the real gateway/bus on an ephemeral port, a real websocket client reading what the HUD / voice output would get."""
from __future__ import annotations

import asyncio
import json
from typing import Dict, List

import httpx
import pytest
import websockets

from gateway.speech_state import SpeechState
from schemas.events import ToolCallStartedEvent, VoiceInputEvent, VoiceOutputEvent
from tests.test_gateway import _auth, _running_gateway


@pytest.fixture(autouse=True)
def _fresh_bus():
    from bus.event_bus import EventBus
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeBrain:
    """Handles VoiceInputEvent like the Brain: waits (per-text gate), then replies with the event id as correlation."""

    def __init__(self, bus):
        self.bus, self.gates, self.started, self.finished = bus, {}, [], []
        bus.subscribe(VoiceInputEvent, self.handle)

    def gate(self, text):
        self.gates[text] = asyncio.Event()
        return self.gates[text]

    async def handle(self, event):
        self.started.append(event.text)
        if event.text in self.gates:
            await self.gates[event.text].wait()
        await self.bus.emit(ToolCallStartedEvent(tool_name="x", arguments={}, source="Planner"))
        await self.bus.emit(VoiceOutputEvent(text=f"reply:{event.text}", source="Brain", correlation_id=event.event_id))
        self.finished.append(event.text)


async def settle(n=40):
    for _ in range(n):
        await asyncio.sleep(0)
    await asyncio.sleep(0.05)


async def drain(ws) -> List[Dict]:
    out = []
    while True:
        try:
            out.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=0.15)))
        except asyncio.TimeoutError:
            return out


def replies(msgs):
    return [m["text"] for m in msgs if m["type"] == "reply"]


@pytest.mark.asyncio
async def test_overlapping_utterances_produce_exactly_one_reply_stream():
    async with _running_gateway() as (gw, port, bus, stub):
        brain = FakeBrain(bus)
        gw._speech = SpeechState(clock=FakeClock())
        ga, gb = brain.gate("first"), brain.gate("second")
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws", additional_headers=_auth()) as ws:
            await ws.recv()
            await drain(ws)
            async with httpx.AsyncClient() as c:
                url = f"http://127.0.0.1:{port}"
                await c.post(f"{url}/message", json={"text": "first"}, headers=_auth())
                await settle()
                await c.post(f"{url}/message", json={"text": "second"}, headers=_auth())     # while "first" is still thinking
                await settle()
                ga.set(); gb.set()                                                              # releasing the OLD turn changes nothing
                await settle()
            msgs = await drain(ws)
        assert replies(msgs) == ["reply:second"]
        assert brain.finished == ["second"]                                                   # the cancelled turn never finished
        assert gw._turns.cancelled_count == 1 and gw._turns.last_action == "replaced"
        assert not [m for m in msgs if m["type"] == "wake"]                                    # thinking != speaking: no barge-in


@pytest.mark.asyncio
async def test_a_cancelled_turn_emits_nothing_after_cancel_even_if_it_replies_late():
    async with _running_gateway() as (gw, port, bus, stub):
        brain = FakeBrain(bus)
        ga = brain.gate("old")
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws", additional_headers=_auth()) as ws:
            await ws.recv(); await drain(ws)
            gw.inject_message("old"); await settle()
            gw.inject_message("new"); await settle()
            first = await drain(ws)
            # Even a reply that does get emitted for the cancelled turn is dropped at the gateway (correlation id filter).
            old_id = next(iter(gw._turns._cancelled))
            await bus.emit(VoiceOutputEvent(text="reply:old-late", source="Brain", correlation_id=old_id))
            await settle()
            late = await drain(ws)
        assert replies(first) == ["reply:new"] and replies(late) == []


@pytest.mark.asyncio
async def test_speech_while_speaking_is_a_barge_in_and_the_order_is_deterministic():
    clock = FakeClock()
    async with _running_gateway() as (gw, port, bus, stub):
        brain = FakeBrain(bus)
        gw._speech = SpeechState(clock=clock)
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws", additional_headers=_auth()) as ws:
            await ws.recv(); await drain(ws)
            gw.inject_message("tell me a story"); await settle()
            gw._speech.set_speaking(True)                                   # voice output reports it is talking
            gb = brain.gate("stop talking and check mail")
            gw.inject_message("stop talking and check mail"); await settle()
            gb.set(); await settle()
            msgs = await drain(ws)
        kinds = [m["type"] for m in msgs]
        assert kinds.count("wake") == 1 and msgs[kinds.index("wake")].get("animate") is False
        last_reply = len(kinds) - 1 - kinds[::-1].index("reply")
        assert last_reply > kinds.index("wake")                              # speech is stopped BEFORE the new reply is sent
        assert replies(msgs) == ["reply:tell me a story", "reply:stop talking and check mail"]   # the story had already finished thinking
        assert gw._turns.last_action == "barge-in" and gw._speech.speaking is False


@pytest.mark.asyncio
async def test_a_turn_that_already_finished_is_not_cancelled_but_speech_is_still_interrupted():
    clock = FakeClock()
    async with _running_gateway() as (gw, port, bus, stub):
        brain = FakeBrain(bus)
        gw._speech = SpeechState(clock=clock)
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws", additional_headers=_auth()) as ws:
            await ws.recv(); await drain(ws)
            gw.inject_message("one"); await settle()
            assert brain.finished == ["one"] and not gw._turns.active
            gw._speech.set_speaking(True)
            gw.inject_message("two"); await settle()
            msgs = await drain(ws)
        assert gw._turns.cancelled_count == 0 and gw._turns.last_action == "barge-in"
        assert replies(msgs) == ["reply:one", "reply:two"] and "wake" in [m["type"] for m in msgs]


@pytest.mark.asyncio
async def test_three_rapid_utterances_leave_exactly_one_reply_and_a_balanced_busy_counter():
    async with _running_gateway() as (gw, port, bus, stub):
        brain = FakeBrain(bus)
        g = [brain.gate(t) for t in ("a", "b", "c")]
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws", additional_headers=_auth()) as ws:
            await ws.recv(); await drain(ws)
            for t in ("a", "b", "c"):
                gw.inject_message(t); await settle(5)
            for ev in g:
                ev.set()
            await settle()
            msgs = await drain(ws)
        assert replies(msgs) == ["reply:c"] and brain.finished == ["c"] and gw._turns.cancelled_count == 2
        assert gw._speech.snapshot()["turns_active"] == 0


def test_turn_manager_memory_is_bounded():
    from gateway.turns import TurnManager
    import uuid

    tm = TurnManager(emit=None, make_event=None, barge_in=None, is_speaking=lambda: False)
    for _ in range(1000):
        tm._current = uuid.uuid4()
        tm._cancel_current()
    assert len(tm._cancelled) <= 256
