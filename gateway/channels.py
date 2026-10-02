"""The gateway's restricted door for chat channels (Telegram today).

A channel adapter runs in its OWN process (supervised by the launcher) and talks to the gateway
over three bearer-token endpoints that live here:

    POST /channel/turn      one normalized message -> NDJSON stream: confirm cards, then the reply
    POST /channel/confirm   answer a confirmation card the channel is showing
    POST /channel/audit     record a rejected callback (stale / reused / foreign user)

Everything that decides what a channel message may DO is enforced HERE, not in the adapter, so a
second adapter (iMessage, Slack) inherits it and a buggy adapter cannot loosen it:

  * the sender must be in `channels.<name>.allowed_user_ids` (config), else 403 + an audit entry;
  * the session is ALWAYS restricted — `allow_dangerous=False`, regardless of what the request or the
    config says (the Guardian refuses dangerous-tier tools outright for the turn);
  * third-party content (forward / quote / caption / file) seeds the turn's taint, and only the typed
    part counts as "said by the user" for the tainted-input rule;
  * the channel name and user id travel in a contextvar, so every Guardian audit entry — and every
    confirmation event — records where the turn came from;
  * a confirmation may only be answered by the channel and user it was raised for;
  * the reply is NOT spoken at the desk (`speak=False`).
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, Optional, Tuple

from channels.limits import LogThrottle
from channels.message import InboundMessage, TRUST_THIRD_PARTY
from guardian.gate import SessionPolicy, channel_context, session_policy
from schemas.events import ConfirmationRequestedEvent, ConfirmationResponseEvent
from utils.logger import get_logger

logger = get_logger(__name__)

#: How long the stream may be silent before a keep-alive line is sent (so the adapter's HTTP read does
#: not time out while the planner works or a confirmation card waits for a tap).
PING_SECONDS = 10.0
#: Audit events a channel may write through /channel/audit (anything else is refused).
AUDIT_EVENTS = frozenset({"callback_rejected"})
_DETAIL_KEYS = frozenset({"reason", "request_id", "outcome"})


class ChannelForbidden(Exception):
    """The sender is not allowed on this channel."""


def taint_label(channel: str) -> str:
    return f"third-party content received via {channel}"


class ChannelTurnService:
    def __init__(self, brain: Any, bus: Any, config: Optional[Dict[str, Any]] = None,
                 guardian: Any = None, ping_seconds: float = PING_SECONDS):
        self._brain = brain
        self._bus = bus
        self._config = config or {}
        self._guardian = guardian if guardian is not None else getattr(brain, "guardian", None)
        self._ping = ping_seconds
        self._locks: Dict[Tuple[str, str], asyncio.Lock] = {}
        self._running: "set[asyncio.Task]" = set()
        self._audit_throttle = LogThrottle(interval=60.0)
        self._audit_throttle_cb = LogThrottle(interval=3.0)

    # ------------------------------------------------------------- authorization
    def _channel_cfg(self, channel: str) -> Optional[Dict[str, Any]]:
        cfg = (self._config.get("channels", {}) or {}).get(channel)
        return cfg if isinstance(cfg, dict) else None

    def authorize(self, channel: str, user_id: str) -> None:
        cfg = self._channel_cfg(channel)
        allowed = {str(u) for u in ((cfg or {}).get("allowed_user_ids") or [])}
        if cfg is None or not cfg.get("enabled") or str(user_id) not in allowed:
            go, suppressed = self._audit_throttle.ready(f"{channel}:{user_id}")
            if go:
                self._audit("rejected_user", {"reason": "not on the channel allowlist",
                                              "suppressed_since_last": suppressed}, channel, user_id)
            raise ChannelForbidden(channel)

    # ------------------------------------------------------------- audit
    def _audit(self, event: str, detail: Dict[str, Any], channel: str, user_id: str = "") -> None:
        if self._guardian is None:
            return
        try:
            self._guardian.record_channel_event(event, detail, channel=channel, user_id=user_id)
        except Exception:
            logger.exception("[channels] could not write an audit entry")

    def channel_enabled(self, channel: str) -> bool:
        cfg = self._channel_cfg(channel)
        return bool(cfg and cfg.get("enabled"))

    def audit_from_adapter(self, event: str, detail: Dict[str, Any], channel: str, user_id: str) -> bool:
        """An adapter reports something it rejected — typically a button press by a user who is NOT the
        owner, so `user_id` is the offender and is deliberately not required to be allowlisted. Only known
        events, only ids/reasons (never text), and at most a few per minute per user."""
        if event not in AUDIT_EVENTS or not self.channel_enabled(channel):
            return False
        go, _ = self._audit_throttle_cb.ready(f"{channel}:{user_id}")
        if not go:
            return True     # accepted, deliberately not written: a flood must not flood the audit log
        clean = {k: str(v)[:80] for k, v in (detail or {}).items() if k in _DETAIL_KEYS}
        self._audit(event, clean, channel, user_id)
        return True

    # ------------------------------------------------------------- the turn
    async def stream_turn(self, msg: InboundMessage) -> AsyncIterator[Dict[str, Any]]:
        self.authorize(msg.channel, msg.user_id)        # raises ChannelForbidden
        lock = self._locks.setdefault((msg.channel, msg.user_id), asyncio.Lock())
        if lock.locked():
            yield {"type": "busy"}
            return

        third_party = msg.effective_trust == TRUST_THIRD_PARTY
        input_taint = taint_label(msg.channel) if third_party else None
        trusted_text = msg.supplied_text if third_party else None

        queue: "asyncio.Queue[Optional[Dict[str, Any]]]" = asyncio.Queue()

        async def on_confirmation(event: ConfirmationRequestedEvent) -> None:
            # Only this turn's own confirmations: same channel AND same user.
            if event.channel == msg.channel and event.channel_user == msg.user_id:
                queue.put_nowait({"type": "confirm", "request_id": event.request_id,
                                  "summary": event.summary, "tool_name": event.tool_name})

        async def run() -> None:
            try:
                # allow_dangerous is hard-wired False: nothing in the request or config can raise it.
                with channel_context(msg.channel, msg.user_id), \
                        session_policy(SessionPolicy(name=msg.channel, allow_dangerous=False)):
                    result = await self._brain.handle_user_text(
                        msg.text, input_taint=input_taint, trusted_text=trusted_text, speak=False)
                queue.put_nowait({"type": "reply", "text": getattr(result, "text", "") or "",
                                  "tainted": bool(getattr(result, "tainted", False))})
            except Exception:
                logger.exception("[channels] turn failed")
                queue.put_nowait({"type": "error", "message": "Something went wrong handling that."})
            finally:
                queue.put_nowait(None)

        async with lock:
            token = self._bus.subscribe(ConfirmationRequestedEvent, on_confirmation)
            task = asyncio.create_task(run())
            self._running.add(task)                      # keep a reference until it finishes
            task.add_done_callback(self._running.discard)
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=self._ping)
                    except asyncio.TimeoutError:
                        yield {"type": "ping"}
                        continue
                    if item is None:
                        break
                    yield item
            finally:
                # If the adapter disconnected mid-turn the turn still finishes; an unanswered
                # confirmation is denied by the Guardian when it expires.
                token.unsubscribe()

    # ------------------------------------------------------------- confirmations
    async def confirm(self, request_id: str, approved: bool, channel: str, user_id: str) -> str:
        """"ok" | "unknown" | "forbidden". A confirmation may only be answered by the channel and user it
        was raised for."""
        self.authorize(channel, user_id)
        owner = self._guardian.pending_channel(request_id) if self._guardian is not None else None
        if owner is None:
            return "unknown"
        if owner != (channel, str(user_id)):
            self._audit("callback_rejected", {"reason": "confirmation belongs to another channel or user",
                                              "request_id": request_id}, channel, user_id)
            return "forbidden"
        await self._bus.emit(ConfirmationResponseEvent(
            request_id=request_id, approved=bool(approved), source=f"{channel}:{user_id}"))
        return "ok"
