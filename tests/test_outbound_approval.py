"""Outbound-action rule: content that would reach another person (or land in a mail account) and was requested over a
remote channel can only be approved on the Mac. Real Guardian + real bus; the Brain is a stub. No network, no mail."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Dict, List

import httpx
import pytest
import pytest_asyncio

from bus.event_bus import EventBus
from channels.message import InboundMessage
from gateway.channels import ChannelForbidden, ChannelTurnService
from gateway.server import Gateway
from guardian.gate import (
    LOCAL_APPROVER_SOURCES, Guardian, SessionPolicy, VerdictType, bump_tier, channel_context, session_policy)
from schemas.events import ConfirmationRequestedEvent, ConfirmationResponseEvent
from tools.registry import ToolRegistry, ToolSpec

OWNER, OTHER = "111", "222"
CONFIG = {"channels": {"telegram": {"enabled": True, "allowed_user_ids": [int(OWNER)]}}, "gateway": {"token": "tok"}}


async def _noop(arguments, context):
    return "ok"


def spec(tier="confirm", name="post_it", **kw):
    return ToolSpec(name=name, description="d", tier=tier, handler=_noop, **kw)


@pytest_asyncio.fixture
async def env(tmp_path):
    EventBus.reset_instance()
    bus = EventBus()
    await bus.start()
    audit = tmp_path / "audit.jsonl"
    guardian = Guardian(event_bus=bus, audit_log_path=audit, confirmation_timeout_seconds=30)
    requested: List[ConfirmationRequestedEvent] = []

    async def on_req(e):
        requested.append(e)

    bus.subscribe(ConfirmationRequestedEvent, on_req)

    def entries():
        return [json.loads(l) for l in audit.read_text().splitlines()] if audit.exists() else []

    async def raise_from(channel, tool, args=None, user=OWNER, **ctx):
        """Raise a confirmation the way a turn on `channel` would (channel "local" = no channel context)."""
        if channel == "local":
            verdict = await guardian.check(tool, args or {}, context=ctx)
        else:
            with channel_context(channel, user), session_policy(SessionPolicy(name=channel, allow_dangerous=False)):
                verdict = await guardian.check(tool, args or {}, context=ctx)
        assert verdict.outcome == VerdictType.NEEDS_CONFIRMATION
        await asyncio.sleep(0)
        return verdict

    async def respond(request_id, approved, source):
        await bus.emit(ConfirmationResponseEvent(request_id=request_id, approved=approved, source=source))
        await asyncio.sleep(0.01)

    yield SimpleNamespace(bus=bus, guardian=guardian, requested=requested, entries=entries, raise_from=raise_from,
                          respond=respond, audit=audit)
    await bus.stop()
    EventBus.reset_instance()


# ------------------------------------------------------------------ registry

def test_an_outbound_tool_can_never_be_safe_tier():
    with pytest.raises(ValueError, match="cannot be 'safe'"):
        spec(tier="safe", outbound=True)
    assert spec(tier="confirm", outbound=True).outbound is True
    assert spec(tier="draft").tier == "draft"


def test_draft_tier_bumps_to_confirm_not_dangerous():
    assert bump_tier("draft") == "confirm" and bump_tier("safe") == "confirm"
    assert bump_tier("confirm") == "dangerous" and bump_tier("dangerous") == "dangerous"


# ------------------------------------------------------------------ the rule

@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["telegram", "remote", "slack-adapter"])
async def test_a_remote_channel_cannot_approve_an_outbound_action(env, channel):
    v = await env.raise_from(channel, spec(outbound=True))
    ev = env.requested[-1]
    assert ev.mac_only is True and "only be approved on your Mac" in ev.summary and ev.channel == channel
    await env.respond(v.request_id, True, f"{channel}:{OWNER}")
    assert env.guardian.pending_mac_only(v.request_id) is True               # still pending: not approved
    refusal = [e for e in env.entries() if e["tool"] == "channel:approval_needs_mac"]
    assert len(refusal) == 1 and refusal[0]["channel"] == channel and refusal[0]["verdict"] == "rejected"
    assert refusal[0]["args"]["approver"] == f"{channel}:{OWNER}" and refusal[0]["args"]["tool"] == "post_it"


@pytest.mark.asyncio
async def test_the_mac_can_still_approve_after_a_refused_remote_attempt(env):
    v = await env.raise_from("telegram", spec(outbound=True), {"text": "hi"})
    await env.respond(v.request_id, True, f"telegram:{OWNER}")
    await env.respond(v.request_id, True, "gateway")                          # the HUD, via the authenticated gateway
    verdict = await env.guardian.await_resolution(v.request_id)
    assert verdict.outcome == VerdictType.ALLOW
    last = [e for e in env.entries() if e["tool"] == "post_it"][-1]
    assert last["who_approved"] == "gateway" and last["channel"] == "telegram" and last["mac_only"] is True
    assert last["verdict"] == "allow"


@pytest.mark.asyncio
@pytest.mark.parametrize("source", sorted(LOCAL_APPROVER_SOURCES))
async def test_every_local_surface_can_approve(env, source):
    v = await env.raise_from("telegram", spec(outbound=True))
    await env.respond(v.request_id, True, source)
    assert (await env.guardian.await_resolution(v.request_id)).outcome == VerdictType.ALLOW


@pytest.mark.asyncio
async def test_a_remote_deny_is_always_honoured(env):
    v = await env.raise_from("telegram", spec(outbound=True))
    await env.respond(v.request_id, False, f"telegram:{OWNER}")
    verdict = await env.guardian.await_resolution(v.request_id)
    assert verdict.outcome == VerdictType.DENY
    assert [e for e in env.entries() if e["tool"] == "post_it"][-1]["verdict"] == "deny"


@pytest.mark.asyncio
async def test_a_local_turn_is_unchanged_and_needs_no_mac_only_flag(env):
    v = await env.raise_from("local", spec(outbound=True))
    assert env.requested[-1].mac_only is False
    await env.respond(v.request_id, True, "cli")
    assert (await env.guardian.await_resolution(v.request_id)).outcome == VerdictType.ALLOW


@pytest.mark.asyncio
async def test_remote_approval_is_still_fine_for_local_only_actions(env):
    v = await env.raise_from("telegram", spec(outbound=False))                # e.g. a reminder, a git commit
    assert env.requested[-1].mac_only is False
    await env.respond(v.request_id, True, f"telegram:{OWNER}")
    assert (await env.guardian.await_resolution(v.request_id)).outcome == VerdictType.ALLOW


@pytest.mark.asyncio
async def test_replayed_approval_cannot_flip_a_resolved_request(env):
    v = await env.raise_from("telegram", spec(outbound=True))
    await env.respond(v.request_id, False, "gateway")                          # the Mac denied
    await env.respond(v.request_id, True, "gateway")                           # a replayed / late approval
    assert (await env.guardian.await_resolution(v.request_id)).outcome == VerdictType.DENY


@pytest.mark.asyncio
async def test_an_approval_for_an_unknown_request_does_nothing(env):
    await env.respond("no-such-request", True, "gateway")
    assert env.entries() == []


@pytest.mark.asyncio
async def test_a_request_nobody_approves_expires_as_a_denial(tmp_path):
    EventBus.reset_instance()
    bus = EventBus()
    await bus.start()
    g = Guardian(event_bus=bus, audit_log_path=tmp_path / "a.jsonl", confirmation_timeout_seconds=0.05)
    with channel_context("telegram", OWNER), session_policy(SessionPolicy(name="telegram", allow_dangerous=False)):
        v = await g.check(spec(outbound=True), {}, context={})
    await bus.emit(ConfirmationResponseEvent(request_id=v.request_id, approved=True, source=f"telegram:{OWNER}"))
    verdict = await g.await_resolution(v.request_id)
    assert verdict.outcome == VerdictType.DENY and "expired" in verdict.reason
    await bus.stop()
    EventBus.reset_instance()


@pytest.mark.asyncio
async def test_a_tainted_outbound_call_from_a_remote_channel_is_denied_outright(env):
    """confirm -> dangerous by the taint rule, and a restricted channel session refuses dangerous tools."""
    with channel_context("telegram", OWNER), session_policy(SessionPolicy(name="telegram", allow_dangerous=False)):
        v = await env.guardian.check(spec(outbound=True), {}, context={"tainted_input": True, "taint_reason": "email"})
    assert v.outcome == VerdictType.DENY


# ------------------------------------------------------------------ the channel door (gateway/channels.py)

@pytest_asyncio.fixture
async def door(env):
    class Brain:
        guardian = env.guardian

        async def handle_user_text(self, *a, **k):
            return SimpleNamespace(text="r", tainted=False)

    return ChannelTurnService(Brain(), env.bus, CONFIG, guardian=env.guardian, ping_seconds=0.05)


@pytest.mark.asyncio
async def test_the_door_refuses_a_remote_approval_and_tells_the_user_to_use_the_mac(env, door):
    v = await env.raise_from("telegram", spec(outbound=True))
    assert await door.confirm(v.request_id, True, "telegram", OWNER) == "needs_mac"
    assert env.guardian.pending_mac_only(v.request_id) is True
    audit = [e for e in env.entries() if e["tool"] == "channel:approval_needs_mac"]
    assert len(audit) == 1 and audit[0]["channel"] == "telegram" and audit[0]["channel_user"] == OWNER
    assert "text" not in json.dumps(audit[0]).lower().replace("context", "")


@pytest.mark.asyncio
async def test_the_door_still_accepts_a_remote_deny_and_a_remote_approve_of_a_local_action(env, door):
    d = await env.raise_from("telegram", spec(outbound=True), user=OWNER)
    assert await door.confirm(d.request_id, False, "telegram", OWNER) == "ok"
    assert (await env.guardian.await_resolution(d.request_id)).outcome == VerdictType.DENY
    a = await env.raise_from("telegram", spec(outbound=False, name="remind"), user=OWNER)
    assert await door.confirm(a.request_id, True, "telegram", OWNER) == "ok"
    assert (await env.guardian.await_resolution(a.request_id)).outcome == VerdictType.ALLOW


@pytest.mark.asyncio
async def test_a_foreign_user_cannot_answer_someone_elses_outbound_card(env, door):
    v = await env.raise_from("telegram", spec(outbound=True), user=OWNER)
    with pytest.raises(ChannelForbidden):
        await door.confirm(v.request_id, True, "telegram", OTHER)              # not on the allowlist
    assert env.guardian.pending_mac_only(v.request_id) is True


@pytest.mark.asyncio
async def test_the_confirm_card_streamed_to_the_channel_carries_mac_only(env, door):
    async def raise_in_turn():
        with channel_context("telegram", OWNER), session_policy(SessionPolicy(name="telegram", allow_dangerous=False)):
            await env.guardian.check(spec(outbound=True), {}, context={})

    class Brain:
        guardian = env.guardian

        async def handle_user_text(self, *a, **k):
            await raise_in_turn()
            return SimpleNamespace(text="r", tainted=False)

    service = ChannelTurnService(Brain(), env.bus, CONFIG, guardian=env.guardian, ping_seconds=0.05)
    events = [e async for e in service.stream_turn(InboundMessage(text="x", channel="telegram", user_id=OWNER))
              if e["type"] != "ping"]
    card = next(e for e in events if e["type"] == "confirm")
    assert card["mac_only"] is True


@pytest.mark.asyncio
async def test_over_http_a_channel_cannot_forge_a_local_approver(env, tmp_path, monkeypatch):
    """/channel/confirm stamps source="<channel>:<user>" itself; nothing in the body can make it "gateway"."""
    class Brain:
        guardian = env.guardian

        async def handle_user_text(self, *a, **k):
            return SimpleNamespace(text="r", tainted=False)

    monkeypatch.setenv("VESPER_GATEWAY_TOKEN", "tok")
    gw = Gateway(config=CONFIG, brain=Brain(), bus=env.bus, manage_brain=False)
    await gw.startup()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw",
                               headers={"Authorization": "Bearer tok"})
    v = await env.raise_from("telegram", spec(outbound=True))
    r = await client.post("/channel/confirm", json={"request_id": v.request_id, "approved": True, "channel": "telegram",
                                                    "user_id": OWNER, "source": "gateway"})
    assert r.status_code == 200 and r.json() == {"status": "needs_mac"}
    assert env.guardian.pending_mac_only(v.request_id) is True
    await client.aclose()


# ------------------------------------------------------------------ the telegram card + discord prompt

@pytest.mark.asyncio
async def test_telegram_card_for_a_mac_only_action_has_no_approve_button():
    from channels.telegram import CALLBACK_PREFIX, TelegramChannel
    sent = {}

    class Fake:
        broker = __import__("channels.confirm", fromlist=["x"]).ConfirmationBroker(120)
        cfg = SimpleNamespace(summary_chars=500, confirm_ttl=120.0)
        _cards = {}

        async def _send(self, uid, text, markup=None):
            sent.update(text=text, markup=markup)
            return 7

    await TelegramChannel._send_card(Fake(), 111, {"request_id": "r1", "summary": "Save a draft", "mac_only": True})
    labels = [b["text"] for row in sent["markup"]["inline_keyboard"] for b in row]
    assert labels == ["Deny"] and "approval on your Mac" in sent["text"]
    await TelegramChannel._send_card(Fake(), 111, {"request_id": "r2", "summary": "Do a thing"})
    assert [b["text"] for row in sent["markup"]["inline_keyboard"] for b in row] == ["Approve", "Deny"]


@pytest.mark.asyncio
async def test_discord_remote_tells_the_owner_to_use_the_mac_and_does_not_resolve(env):
    from remote.discord_remote import DiscordRemote, IncomingMessage, RemoteConfig

    sent: List[str] = []

    class Transport:
        async def send(self, channel_id, text):
            sent.append(text)
            return "m1"

    class Brain:
        async def handle_user_text(self, text):
            v = await env.guardian.check(spec(outbound=True), {}, context={})
            return SimpleNamespace(text="done")

    remote = DiscordRemote(Brain(), RemoteConfig(enabled=True, owner_user_id="9", channel_id="c"), Transport(), env.bus)
    task = asyncio.create_task(remote.on_incoming(IncomingMessage(author_id="9", channel_id="c", content="post it")))
    for _ in range(200):
        if remote._pending:
            break
        await asyncio.sleep(0.005)
    rid = next(iter(remote._pending))
    assert "only be approved on your Mac" in sent[0]
    await remote.on_incoming(IncomingMessage(author_id="9", channel_id="c", content=rid[:8]))
    assert any("only be approved on your Mac" in t for t in sent[1:]) and rid in remote._pending
    assert env.guardian.pending_mac_only(rid) is True
    await remote.on_incoming(IncomingMessage(author_id="9", channel_id="c", content="no"))
    await task
