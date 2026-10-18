"""POST /wake only speaks the persona greeting when asked; GET/POST /speech tell voice input when a reply is over.
Real gateway on an ephemeral port, stub brain (no LLM) — the same harness as test_gateway.py."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from tests.test_gateway import TOKEN, _auth, _running_gateway


@pytest.mark.asyncio
async def test_wake_with_greet_false_does_not_call_the_brain_greeting():
    async with _running_gateway() as (gw, port, bus, brain):
        calls = []

        async def greet():
            calls.append(1)
            return "x"

        brain.greet = greet
        async with httpx.AsyncClient() as c:
            r = await c.post(f"http://127.0.0.1:{port}/wake", json={"greet": False}, headers=_auth())
            assert r.status_code == 200 and r.json() == {"status": "waking", "greeting": False}
            await asyncio.sleep(0.1)
            assert calls == []
            r = await c.post(f"http://127.0.0.1:{port}/wake", json={"greet": True}, headers=_auth())
            assert r.json()["greeting"] is True
            await asyncio.sleep(0.1)
            assert calls == [1]
            await c.post(f"http://127.0.0.1:{port}/wake", headers=_auth())          # old clients: no body = greet, as before
            await asyncio.sleep(0.1)
            assert calls == [1, 1]


@pytest.mark.asyncio
async def test_speech_endpoint_reports_a_turn_in_flight_and_voice_outputs_speaking_state():
    async with _running_gateway() as (gw, port, bus, brain):
        async with httpx.AsyncClient() as c:
            url = f"http://127.0.0.1:{port}"
            base = (await c.get(f"{url}/speech", headers=_auth())).json()
            assert set(base) == {"busy", "turns_active", "speaking", "audible", "pause_s"}
            await c.post(f"{url}/speech", json={"speaking": True}, headers=_auth())
            assert (await c.get(f"{url}/speech", headers=_auth())).json()["speaking"] is True
            await c.post(f"{url}/speech", json={"speaking": False}, headers=_auth())


@pytest.mark.asyncio
async def test_speech_endpoints_require_the_token():
    async with _running_gateway() as (gw, port, bus, brain):
        async with httpx.AsyncClient() as c:
            url = f"http://127.0.0.1:{port}"
            assert (await c.get(f"{url}/speech")).status_code == 401
            assert (await c.post(f"{url}/speech", json={"speaking": True})).status_code == 401
            assert (await c.post(f"{url}/wake", json={"greet": False})).status_code == 401


@pytest.mark.asyncio
async def test_a_message_marks_the_turn_busy_until_the_brain_has_finished_it():
    from schemas.events import VoiceInputEvent

    async with _running_gateway() as (gw, port, bus, brain):
        gate = asyncio.Event()

        async def slow(event):
            await gate.wait()

        bus.subscribe(VoiceInputEvent, slow)
        async with httpx.AsyncClient() as c:
            url = f"http://127.0.0.1:{port}"
            await c.post(f"{url}/message", json={"text": "hello there"}, headers=_auth())
            await asyncio.sleep(0.2)
            state = (await c.get(f"{url}/speech", headers=_auth())).json()
            assert state["busy"] is True and state["turns_active"] == 1
            gate.set()
            await asyncio.sleep(0.3)
            assert (await c.get(f"{url}/speech", headers=_auth())).json()["turns_active"] == 0
