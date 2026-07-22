"""Gateway (PV0) tests — websocket auth, inbound injection, outbound forward,
and the reconnect snapshot. A real uvicorn server runs on an ephemeral port
against an injected STUB brain (no LLM), driven by the websockets client."""
from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from typing import Any, Dict, List

import httpx
import pytest
import uvicorn
import websockets

from bus.event_bus import EventBus
from gateway.server import Gateway
from schemas.events import ObservationEvent, VoiceInputEvent, VoiceOutputEvent

TOKEN = "test-token-abc"
GREETING = "Good evening, Sir."


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _StubRouter:
    async def provider_status(self) -> List[Dict[str, Any]]:
        return [{"tier": "primary", "provider": "groq", "model": "stub", "available": True}]


class _StubBrain:
    """Stands in for the real Brain: mirrors its VoiceInputEvent subscription so
    an injected message provably 'reaches the planner', and answers status."""

    def __init__(self, bus: EventBus) -> None:
        self.received: List[str] = []
        self.message_arrived = asyncio.Event()
        bus.subscribe(VoiceInputEvent, self._on_voice_input)

    async def _on_voice_input(self, event: VoiceInputEvent) -> None:
        self.received.append(event.text)
        self.message_arrived.set()

    def get_agents_status(self) -> List[Dict[str, Any]]:
        return [{"name": "MemoryAgent", "healthy": True, "error_count": 0}]

    def get_router(self) -> _StubRouter:
        return _StubRouter()


@contextlib.asynccontextmanager
async def _running_gateway():
    bus = EventBus()
    await bus.start()
    brain = _StubBrain(bus)
    config = {"gateway": {"host": "127.0.0.1", "port": 0, "token": TOKEN}}
    gw = Gateway(config=config, brain=brain, bus=bus, manage_brain=False)
    await gw.startup()  # injected brain/bus => only attaches subscriptions
    # Simulate the boot greeting the real Brain emits during start().
    await bus.emit(VoiceOutputEvent(text=GREETING, source="Brain"))

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(gw.app, host="127.0.0.1", port=port, log_level="error", lifespan="off")
    )
    serve_task = asyncio.create_task(server.serve())
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.02)
        assert server.started, "uvicorn did not start"
        yield gw, port, bus, brain
    finally:
        server.should_exit = True
        await serve_task
        await gw.shutdown()
        await bus.stop()


def _auth(token: str = TOKEN) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_ws_auth_reject_then_accept():
    async with _running_gateway() as (gw, port, bus, brain):
        uri = f"ws://127.0.0.1:{port}/ws"

        # No token -> handshake rejected.
        with pytest.raises(Exception):
            await websockets.connect(uri)

        # Wrong token -> handshake rejected.
        with pytest.raises(Exception):
            await websockets.connect(uri, additional_headers=_auth("wrong"))

        # Correct token -> accepted, first frame is a snapshot.
        async with websockets.connect(uri, additional_headers=_auth()) as ws:
            snap = json.loads(await ws.recv())
            assert snap["type"] == "snapshot"


@pytest.mark.asyncio
async def test_ws_message_reaches_planner():
    async with _running_gateway() as (gw, port, bus, brain):
        uri = f"ws://127.0.0.1:{port}/ws"
        async with websockets.connect(uri, additional_headers=_auth()) as ws:
            await ws.recv()  # snapshot
            await ws.send(json.dumps({"type": "message", "text": "what time is it"}))
            await asyncio.wait_for(brain.message_arrived.wait(), timeout=3)
            assert "what time is it" in brain.received


@pytest.mark.asyncio
async def test_outbound_event_reaches_connected_client():
    async with _running_gateway() as (gw, port, bus, brain):
        uri = f"ws://127.0.0.1:{port}/ws"
        async with websockets.connect(uri, additional_headers=_auth()) as ws:
            await ws.recv()  # snapshot
            await bus.emit(ObservationEvent(kind="context_switch", detail="hello there", source="test"))
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
            assert msg["type"] == "observation"
            assert msg["detail"] == "hello there"
            assert msg["kind"] == "context_switch"


@pytest.mark.asyncio
async def test_reconnect_delivers_fresh_snapshot():
    async with _running_gateway() as (gw, port, bus, brain):
        uri = f"ws://127.0.0.1:{port}/ws"

        async with websockets.connect(uri, additional_headers=_auth()) as ws1:
            snap1 = json.loads(await ws1.recv())
            assert snap1["type"] == "snapshot"

        # Disconnected. Reconnect -> a NEW snapshot (Brain is source of truth;
        # the client lost nothing).
        async with websockets.connect(uri, additional_headers=_auth()) as ws2:
            snap2 = json.loads(await ws2.recv())
            assert snap2["type"] == "snapshot"
            assert snap2["greeting"] == GREETING
            assert "agents" in snap2["status"]
            assert "providers" in snap2["status"]


@pytest.mark.asyncio
async def test_rest_status_auth():
    async with _running_gateway() as (gw, port, bus, brain):
        base = f"http://127.0.0.1:{port}"
        async with httpx.AsyncClient(base_url=base) as client:
            # Missing token -> 401.
            r = await client.get("/status")
            assert r.status_code == 401
            # With token -> agents + providers.
            r = await client.get("/status", headers=_auth())
            assert r.status_code == 200
            body = r.json()
            assert body["agents"][0]["name"] == "MemoryAgent"
            assert body["providers"][0]["provider"] == "groq"
