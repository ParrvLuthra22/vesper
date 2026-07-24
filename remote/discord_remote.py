"""
remote/discord_remote.py — the Discord DM control channel (PC3).

Mobile access to Vesper without a mobile app: messages the owner sends in a
designated private Discord channel are injected as ordinary user turns
(brain.handle_user_text) and Vesper's replies are posted back. It is just
another input surface, like the CLI — but a RESTRICTED one, with rails that are
enforced, not advisory:

  1. Owner-only. Every message whose author isn't the configured owner_user_id
     is ignored outright — no turn, no reply.
  2. Dangerous tools are DISABLED for remote sessions entirely. Each remote turn
     runs under a Guardian SessionPolicy(allow_dangerous=remote.allow_dangerous),
     and that flag DEFAULTS FALSE — so dangerous-tier tools are denied at the
     gate, never even offered for confirmation (see guardian/gate.py).
  3. Confirm-tier tools still work remotely, but approval must be EXPLICIT: the
     request id or a ✅ reaction on the specific prompt. A bare "yes" is refused.

The core (DiscordRemote) is transport-agnostic and fully unit-testable; the real
discord.py client (DiscordTransport / run_remote) is a thin adapter wired up in
main.py only when remote.enabled.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:  # Protocol lives in typing on 3.8+, but keep the import defensive.
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore

from bus.event_bus import EventBus, get_event_bus
from guardian.gate import SessionPolicy, session_policy
from schemas.events import ConfirmationRequestedEvent, ConfirmationResponseEvent
from utils.logger import get_logger

logger = get_logger(__name__)

REMOTE_SESSION_NAME = "remote"
APPROVE_REACTIONS = {"✅", "👍", "☑️", "✔️"}
DENY_REACTIONS = {"❌", "👎", "🚫"}
DENY_WORDS = {"no", "cancel", "deny", "stop", "abort", "nope", "nevermind", "never mind"}
SHORT_ID_LEN = 8


@dataclass
class RemoteConfig:
    enabled: bool = False
    owner_user_id: str = ""
    channel_id: str = ""
    allow_dangerous: bool = False
    token_env: str = "VESPER_DISCORD_TOKEN"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RemoteConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            owner_user_id=str(data.get("owner_user_id", "") or ""),
            channel_id=str(data.get("channel_id", "") or ""),
            allow_dangerous=bool(data.get("allow_dangerous", False)),
            token_env=str(data.get("token_env", "VESPER_DISCORD_TOKEN") or "VESPER_DISCORD_TOKEN"),
        )


@dataclass
class IncomingMessage:
    """One event from the Discord side, normalized. A normal message has
    `content`; a reaction sets is_reaction with the target message + emoji."""

    author_id: str
    channel_id: str = ""
    content: str = ""
    message_id: str = ""
    is_reaction: bool = False
    reaction_emoji: str = ""
    reaction_target_id: str = ""


class Transport(Protocol):
    async def send(self, channel_id: str, text: str) -> str:
        """Post `text` to `channel_id`; return the sent message's id."""
        ...


class DiscordRemote:
    """Transport-agnostic core of the remote interface (all the safety logic)."""

    def __init__(
        self,
        brain: Any,
        config: RemoteConfig,
        transport: Transport,
        event_bus: Optional[EventBus] = None,
    ):
        self._brain = brain
        self._config = config
        self._transport = transport
        self._event_bus = event_bus or get_event_bus()
        #: request_id -> {"summary", "prompt_msg_id"} for confirmations raised
        #: during a remote turn and awaiting the owner's explicit approval.
        self._pending: Dict[str, Dict[str, str]] = {}
        #: >0 while a remote turn is executing — so we only intercept
        #: confirmations that belong to THIS surface, never a local CLI turn's.
        self._active_turns = 0
        #: the channel the current/last remote turn is talking on.
        self._channel = config.channel_id or ""
        self._event_bus.subscribe(ConfirmationRequestedEvent, self._on_confirmation_requested)

    # ------------------------------- inbound -------------------------------
    async def on_incoming(self, msg: IncomingMessage) -> None:
        if str(msg.author_id) != str(self._config.owner_user_id):
            logger.warning(f"[remote] ignoring message from non-owner author_id={msg.author_id!r}")
            return
        if self._config.channel_id and str(msg.channel_id) != str(self._config.channel_id):
            logger.debug(f"[remote] ignoring message outside the designated channel ({msg.channel_id})")
            return

        # While a confirmation is pending, the owner's messages are ANSWERS to
        # it, never new turns — so a plain "yes" can't accidentally start work.
        if self._pending:
            await self._handle_approval_reply(msg)
            return
        if msg.is_reaction:
            return  # a reaction with nothing pending — nothing to answer

        text = (msg.content or "").strip()
        if not text:
            return
        await self._run_turn(text, reply_channel=str(msg.channel_id))

    async def _run_turn(self, text: str, reply_channel: str) -> None:
        self._channel = self._config.channel_id or reply_channel
        policy = SessionPolicy(name=REMOTE_SESSION_NAME, allow_dangerous=bool(self._config.allow_dangerous))
        self._active_turns += 1
        try:
            with session_policy(policy):
                result = await self._brain.handle_user_text(text)
        except Exception as exc:  # never let a bad turn kill the listener loop
            logger.error(f"[remote] turn failed: {exc}", exc_info=True)
            await self._transport.send(self._channel, f"Something went wrong handling that, Sir: {exc}")
            return
        finally:
            self._active_turns -= 1
        reply = (getattr(result, "text", "") or "").strip() or "(no reply)"
        await self._transport.send(self._channel, reply)

    # ---------------------------- confirmations ----------------------------
    async def _on_confirmation_requested(self, event: ConfirmationRequestedEvent) -> None:
        # Only intercept confirmations raised during one of OUR turns; a local
        # surface (CLI/voice) owns its own confirmations.
        if self._active_turns <= 0:
            return
        short = event.request_id[:SHORT_ID_LEN]
        prompt = (
            f"Confirmation required (id: {short})\n{event.summary}\n\n"
            f"Reply `{short}` or react ✅ to approve; reply `no` to cancel. "
            "A plain 'yes' will not approve it."
        )
        prompt_msg_id = await self._transport.send(self._channel, prompt)
        self._pending[event.request_id] = {"summary": event.summary, "prompt_msg_id": str(prompt_msg_id or "")}

    async def _handle_approval_reply(self, msg: IncomingMessage) -> None:
        if msg.is_reaction:
            request_id = self._request_for_prompt(msg.reaction_target_id)
            if request_id is None:
                return  # reaction on some unrelated message
            if msg.reaction_emoji in APPROVE_REACTIONS:
                await self._resolve(request_id, approved=True)
            elif msg.reaction_emoji in DENY_REACTIONS:
                await self._resolve(request_id, approved=False)
            return

        content = (msg.content or "").strip()
        lowered = content.lower()
        request_id = self._request_for_text(content)
        if request_id is not None:
            approved = not any(w in lowered for w in DENY_WORDS)
            await self._resolve(request_id, approved=approved)
            return
        if any(w in lowered for w in DENY_WORDS):
            for rid in list(self._pending):  # a bare "no"/"cancel" cancels everything pending
                await self._resolve(rid, approved=False)
            return

        # No id, no reaction, not a cancel — i.e. a bare "yes" or unrelated text.
        # This is the rule: approval must be explicit. Refuse and re-prompt.
        handles = ", ".join(f"`{rid[:SHORT_ID_LEN]}`" for rid in self._pending)
        await self._transport.send(
            self._channel,
            f"That won't approve it, Sir. Reply with the request id ({handles}) or react ✅ to "
            "the specific message. A plain 'yes' isn't enough.",
        )

    def _request_for_prompt(self, prompt_msg_id: str) -> Optional[str]:
        for rid, info in self._pending.items():
            if info.get("prompt_msg_id") and str(info["prompt_msg_id"]) == str(prompt_msg_id):
                return rid
        return None

    def _request_for_text(self, content: str) -> Optional[str]:
        if not content:
            return None
        for rid in self._pending:
            if rid in content or rid[:SHORT_ID_LEN] in content:
                return rid
        return None

    async def _resolve(self, request_id: str, approved: bool) -> None:
        self._pending.pop(request_id, None)
        await self._event_bus.emit(
            ConfirmationResponseEvent(request_id=request_id, approved=approved, source="discord_remote")
        )
        await self._transport.send(self._channel, "Approved." if approved else "Cancelled.")


# =========================================================================
# Real discord.py adapter (untested infra — wired up only when remote.enabled).
# =========================================================================
class DiscordTransport:
    """Transport backed by a live discord.py client."""

    def __init__(self, client: Any):
        self._client = client

    async def send(self, channel_id: str, text: str) -> str:
        channel = self._client.get_channel(int(channel_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(channel_id))
        message = await channel.send(text[:1900])  # Discord's 2000-char ceiling
        return str(message.id)


async def run_remote(brain: Any, config: Dict[str, Any]) -> None:
    """Start the live Discord listener. No-op (with a clear log) if the interface
    is disabled, discord.py isn't installed, or no token is configured."""
    remote_cfg = RemoteConfig.from_dict((config or {}).get("remote", {}))
    if not remote_cfg.enabled:
        return
    if not remote_cfg.owner_user_id:
        logger.error("[remote] remote.enabled but no owner_user_id configured; not starting")
        return
    try:
        import discord  # type: ignore
    except ImportError:
        logger.error("[remote] discord.py is not installed; remote interface disabled (pip install discord.py)")
        return
    token = os.getenv(remote_cfg.token_env)
    if not token:
        logger.error(f"[remote] no bot token in ${remote_cfg.token_env}; remote interface disabled")
        return

    intents = discord.Intents.default()
    intents.message_content = True
    intents.reactions = True
    client = discord.Client(intents=intents)
    remote = DiscordRemote(brain=brain, config=remote_cfg, transport=DiscordTransport(client))

    @client.event
    async def on_message(message: Any) -> None:  # pragma: no cover - live adapter
        if message.author.id == client.user.id:
            return
        await remote.on_incoming(IncomingMessage(
            author_id=str(message.author.id),
            channel_id=str(message.channel.id),
            content=message.content or "",
            message_id=str(message.id),
        ))

    @client.event
    async def on_raw_reaction_add(payload: Any) -> None:  # pragma: no cover - live adapter
        if client.user is not None and str(payload.user_id) == str(client.user.id):
            return
        await remote.on_incoming(IncomingMessage(
            author_id=str(payload.user_id),
            channel_id=str(payload.channel_id),
            is_reaction=True,
            reaction_emoji=str(payload.emoji),
            reaction_target_id=str(payload.message_id),
        ))

    logger.info("[remote] Discord remote interface connecting…")
    await client.start(token)


def create_discord_remote(brain: Any, config: Dict[str, Any], transport: Transport) -> DiscordRemote:
    """Construct the core with an explicit transport (used by tests and by any
    non-discord.py transport)."""
    return DiscordRemote(brain=brain, config=RemoteConfig.from_dict((config or {}).get("remote", {})), transport=transport)
