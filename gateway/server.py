"""FastAPI gateway over the Vesper event bus — WebSocket + REST.

The gateway runs in the SAME process as the Brain and reimplements none of
its logic. It:

  * subscribes to a fixed set of outbound events (see gateway/wire.py) and
    pushes each, serialized to compact JSON, to every connected WebSocket
    client;
  * injects inbound client messages onto the bus exactly as any other input
    surface would — a typed message becomes a ``VoiceInputEvent`` (the same
    event the Brain already routes to the Planner), a confirmation becomes a
    ``ConfirmationResponseEvent`` (exactly as the CLI emits it).

The Brain is the single source of truth; WebSocket clients are stateless
views. A client may disconnect and reconnect freely: on every connect it
receives a ``snapshot`` (current status + last greeting) so it can render
immediately without the Brain having held any per-client state.

SECURITY: binds 127.0.0.1 by default and MUST NOT be exposed beyond
localhost in v2. The bearer token gates local access only; exposing this
surface to a network is a v3 decision that requires real auth (per-user
credentials, TLS, rate limiting) — none of which exists here.
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from gateway.wire import FORWARDED_EVENT_TYPES, to_wire
from schemas.events import (
    ConfirmationResponseEvent,
    VoiceInputEvent,
    VoiceOutputEvent,
    WakeEvent,
)
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8760
#: WebSocket close code for a failed/absent bearer token (RFC 6455 "policy
#: violation"). Sent before accept so the handshake is rejected outright.
WS_POLICY_VIOLATION = 1008


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MessageIn(BaseModel):
    text: str


class ConfirmIn(BaseModel):
    request_id: str
    approved: bool


class ConnectionManager:
    """Tracks connected WebSocket clients and broadcasts wire messages to all
    of them. Clients are interchangeable stateless views onto one Brain."""

    def __init__(self) -> None:
        self._active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        payload = json.dumps(message, default=str)
        for ws in list(self._active):
            try:
                await ws.send_text(payload)
            except Exception:
                # A send failure means the socket is gone; drop it and keep
                # broadcasting to the rest.
                self._active.discard(ws)

    @property
    def count(self) -> int:
        return len(self._active)


class Gateway:
    """Owns the FastAPI app, the bus subscriptions, and the (optionally
    self-managed) Brain lifecycle.

    When ``brain``/``bus`` are injected (tests), the caller owns their
    lifecycle and the gateway only bridges. Otherwise the gateway creates and
    manages both.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        brain: Optional[Any] = None,
        bus: Optional[Any] = None,
        manage_brain: Optional[bool] = None,
    ) -> None:
        self._config = config or {}
        gw = self._config.get("gateway", {}) or {}
        self._host = str(gw.get("host", DEFAULT_HOST))
        self._port = int(gw.get("port", DEFAULT_PORT))
        # The exact env var name the deploy contract promises; falls back to
        # the YAML value. Empty => auth fails closed (all connections rejected).
        self._token = os.getenv("VESPER_GATEWAY_TOKEN") or gw.get("token") or ""

        self._bus = bus
        self._brain = brain
        self._manage_brain = manage_brain if manage_brain is not None else (brain is None)

        self._manager = ConnectionManager()
        self._greeting: Optional[str] = None
        self._subscriptions: List[Any] = []
        self._attached = False

        self.app = self._build_app()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def startup(self) -> None:
        """Create/attach whatever isn't injected, then start the Brain if we
        own it. Safe to run under uvicorn's lifespan or called directly."""
        if self._bus is None:
            from bus.event_bus import EventBus

            self._bus = EventBus()
            await self._bus.start()
        if self._brain is None:
            from orchestrator.brain import Brain

            self._brain = Brain(config=self._config, event_bus=self._bus, enable_voice_agent=False)

        self._attach()

        if self._manage_brain:
            logger.info("Gateway starting Brain (in-process)...")
            await self._brain.start()

        if not self._token:
            logger.warning(
                "gateway.token is empty (VESPER_GATEWAY_TOKEN unset) — all "
                "gateway connections will be REJECTED. Set a token to use it."
            )
        logger.info(f"Gateway ready on http://{self._host}:{self._port} (localhost only)")

    async def shutdown(self) -> None:
        for token in self._subscriptions:
            token.unsubscribe()
        self._subscriptions.clear()
        self._attached = False
        if self._manage_brain and self._brain is not None:
            await self._brain.stop("gateway shutdown")

    def _attach(self) -> None:
        """Subscribe the single forwarding handler to every exposed event
        type. Idempotent."""
        if self._attached:
            return
        for event_type in FORWARDED_EVENT_TYPES:
            self._subscriptions.append(self._bus.subscribe(event_type, self._forward))
        self._attached = True

    async def _forward(self, event: Any) -> None:
        """One handler for every forwarded type — remembers the boot greeting,
        serializes via the shared mapping, and broadcasts."""
        if isinstance(event, VoiceOutputEvent) and self._greeting is None:
            self._greeting = event.text
        wire = to_wire(event)
        if wire is not None:
            await self._manager.broadcast(wire)

    # ------------------------------------------------------------------ #
    # Bus injection — mirrors how the CLI feeds the same bus
    # ------------------------------------------------------------------ #
    def inject_message(self, text: str) -> None:
        """Enqueue a user turn. Emitting VoiceInputEvent awaits the whole turn
        (LLM + tools), so it MUST be fire-and-forget — never block the WS loop
        or the HTTP response on it."""
        asyncio.create_task(self._bus.emit(VoiceInputEvent(text=text, source="gateway")))

    def inject_confirm(self, request_id: str, approved: bool) -> None:
        asyncio.create_task(
            self._bus.emit(
                ConfirmationResponseEvent(
                    request_id=request_id, approved=approved, source="gateway"
                )
            )
        )

    # ------------------------------------------------------------------ #
    # Snapshot / status
    # ------------------------------------------------------------------ #
    async def _status_payload(self) -> Dict[str, Any]:
        agents: List[Dict[str, Any]] = []
        providers: List[Dict[str, Any]] = []
        if self._brain is not None:
            agents = self._brain.get_agents_status()
            providers = await self._brain.get_router().provider_status()
        return {"agents": agents, "providers": providers}

    async def _snapshot(self) -> Dict[str, Any]:
        return {
            "type": "snapshot",
            "greeting": self._greeting,
            "status": await self._status_payload(),
            "ts": _now_iso(),
        }

    # ------------------------------------------------------------------ #
    # Auth
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bearer(header: str) -> Optional[str]:
        if header and header.lower().startswith("bearer "):
            return header[7:].strip()
        return None

    def _authorized(self, provided: Optional[str]) -> bool:
        # Fail closed: an unset token authorizes nobody.
        return bool(self._token) and provided == self._token

    # ------------------------------------------------------------------ #
    # App wiring
    # ------------------------------------------------------------------ #
    def _build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self.startup()
            try:
                yield
            finally:
                await self.shutdown()

        app = FastAPI(title="Vesper Gateway", version="0.1.0", lifespan=lifespan)

        async def require_token(authorization: str = Header(default="")) -> None:
            if not self._authorized(self._bearer(authorization)):
                raise HTTPException(status_code=401, detail="invalid or missing bearer token")

        @app.get("/status")
        async def get_status(_: None = Depends(require_token)) -> Dict[str, Any]:
            return await self._status_payload()

        @app.post("/message")
        async def post_message(body: MessageIn, _: None = Depends(require_token)) -> Dict[str, Any]:
            text = (body.text or "").strip()
            if text:
                self.inject_message(text)
            return {"status": "queued", "text": text}

        @app.post("/confirm")
        async def post_confirm(body: ConfirmIn, _: None = Depends(require_token)) -> Dict[str, Any]:
            self.inject_confirm(body.request_id, body.approved)
            return {"status": "ok"}

        @app.post("/wake")
        async def post_wake(_: None = Depends(require_token)) -> Dict[str, Any]:
            # The wake word fired (from the voice-input client). Broadcast the
            # WakeEvent (HUD flare + voice-output barge-in) and speak the
            # time-appropriate greeting — the cinematic reveal.
            animate = bool((self._config.get("wake_flow", {}) or {}).get("animation", True))
            await self._bus.emit(WakeEvent(animate=animate, source="voice"))
            if self._brain is not None:
                asyncio.create_task(self._brain.greet())
            return {"status": "waking"}

        @app.websocket("/ws")
        async def ws_endpoint(ws: WebSocket) -> None:
            provided = self._bearer(ws.headers.get("authorization", "")) or ws.query_params.get("token")
            if not self._authorized(provided):
                # Reject before accept -> handshake fails; client sees refusal.
                await ws.close(code=WS_POLICY_VIOLATION)
                return

            await self._manager.connect(ws)
            try:
                # Reconnection contract: every fresh connection renders from a
                # snapshot; the Brain never held per-client state to lose.
                await ws.send_text(json.dumps(await self._snapshot(), default=str))
                while True:
                    raw = await ws.receive_text()
                    await self._handle_inbound(raw)
            except WebSocketDisconnect:
                pass
            except Exception:
                logger.exception("WebSocket handler error")
            finally:
                self._manager.disconnect(ws)

        return app

    async def _handle_inbound(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Ignoring non-JSON inbound WS frame")
            return
        if not isinstance(msg, dict):
            return
        mtype = msg.get("type")
        if mtype == "message":
            text = (msg.get("text") or "").strip()
            if text:
                self.inject_message(text)
        elif mtype == "confirm":
            request_id = msg.get("request_id")
            if request_id:
                self.inject_confirm(str(request_id), bool(msg.get("approved")))
        else:
            logger.warning(f"Ignoring unknown inbound WS message type: {mtype!r}")

    # ------------------------------------------------------------------ #
    # Server entry
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        import uvicorn

        if self._host not in ("127.0.0.1", "localhost", "::1"):
            # Hard stop: v2 is localhost-only by contract. Exposing beyond
            # localhost needs real auth (v3), which does not exist yet.
            raise RuntimeError(
                f"Refusing to bind gateway to {self._host!r}: v2 is localhost-only. "
                "Exposing the gateway to a network is a v3 decision requiring real auth."
            )
        uvicorn.run(self.app, host=self._host, port=self._port, log_level="info")


def create_app(
    config: Optional[Dict[str, Any]] = None,
    brain: Optional[Any] = None,
    bus: Optional[Any] = None,
    manage_brain: Optional[bool] = None,
) -> FastAPI:
    """Build (but do not start) the gateway's FastAPI app."""
    return Gateway(config=config, brain=brain, bus=bus, manage_brain=manage_brain).app


def _load_config() -> Dict[str, Any]:
    from pathlib import Path

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass

    from config.settings import load_config_dict

    config = load_config_dict()
    # Headless server: never spawn the native macOS HUD overlay window (the
    # CLI disables it for the same reason). Clients render their own UI.
    config.setdefault("ui", {}).setdefault("hud", {})["enabled"] = False
    return config


def main() -> None:
    """`python -m gateway.server` — start the gateway and its Brain."""
    from utils.logger import init_from_config

    config = _load_config()
    init_from_config(config.get("general", {}))
    Gateway(config=config).run()


if __name__ == "__main__":
    main()
