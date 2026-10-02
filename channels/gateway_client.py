"""The channel process's client for the gateway's restricted /channel/* endpoints."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from channels.message import InboundMessage


class GatewayUnavailable(Exception):
    """The gateway cannot be reached (not running, restarting, or the stream broke)."""


class GatewayRefused(Exception):
    """The gateway answered 401/403/422: a configuration problem, not a transient one."""

    def __init__(self, status: int):
        super().__init__(f"gateway refused the request (HTTP {status})")
        self.status = status


class GatewayClient:
    def __init__(self, base_url: str, token: str, transport: Optional[httpx.AsyncBaseTransport] = None,
                 read_timeout: float = 60.0):
        self._http = httpx.AsyncClient(
            base_url=base_url, headers={"Authorization": f"Bearer {token}"}, transport=transport,
            timeout=httpx.Timeout(connect=5.0, read=read_timeout, write=10.0, pool=5.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def stream_turn(self, msg: InboundMessage) -> AsyncIterator[Dict[str, Any]]:
        """Yield confirm/reply/error/busy events; keep-alive pings are swallowed."""
        try:
            async with self._http.stream("POST", "/channel/turn", json=msg.to_dict()) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise GatewayRefused(resp.status_code)
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict) and event.get("type") != "ping":
                        yield event
        except httpx.HTTPError as exc:
            raise GatewayUnavailable(type(exc).__name__) from None

    async def confirm(self, request_id: str, approved: bool, channel: str, user_id: str) -> str:
        try:
            resp = await self._http.post("/channel/confirm", json={
                "request_id": request_id, "approved": approved, "channel": channel, "user_id": user_id})
        except httpx.HTTPError as exc:
            raise GatewayUnavailable(type(exc).__name__) from None
        if resp.status_code == 403:
            return "forbidden"
        if resp.status_code != 200:
            raise GatewayRefused(resp.status_code)
        return str(resp.json().get("status", "unknown"))

    async def audit(self, event: str, detail: Dict[str, Any], channel: str, user_id: str) -> None:
        try:
            await self._http.post("/channel/audit", json={
                "event": event, "detail": detail, "channel": channel, "user_id": user_id})
        except httpx.HTTPError:
            pass        # best effort: a failed audit post must never break the poll loop
