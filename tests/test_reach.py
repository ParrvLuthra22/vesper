"""Reach (PC3) — Slack + Discord MCP registration, and the Discord remote
interface with its restricted permissions.

Covers the promised contracts:
  1. Slack and Discord register through the bridge with reads=safe, sends=confirm,
     admin tools never exposed; two servers with the same tool names don't collide.
  2. A confirm-tier send shows the exact channel + full text in the Guardian summary.
  3. Remote sessions disable dangerous-tier tools ENTIRELY (denied at the gate,
     no confirmation offered).
  4. A non-owner message is rejected (no turn, no reply).
  5. A confirm-tier tool over the remote channel requires EXPLICIT approval — a
     bare "yes" is refused; the request id (or a ✅ reaction) approves it.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from bus.event_bus import EventBus
from guardian.gate import Guardian, SessionPolicy, VerdictType, session_policy
from remote.discord_remote import DiscordRemote, IncomingMessage, RemoteConfig
from schemas.events import ConfirmationRequestedEvent
from tools.mcp_bridge import MCPBridge
from tools.registry import ToolRegistry, ToolSpec


@pytest.fixture(autouse=True)
def _fresh_bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


class _FakeConn:
    def __init__(self, names):
        self.tools = [
            {"name": n, "description": n, "inputSchema": {"type": "object", "properties": {}}} for n in names
        ]


READS = ["list_unreads", "get_channel_messages", "search", "get_mentions"]
SENDS = ["post_message", "reply_thread"]
ADMIN = ["delete_channel", "kick_user"]


def _server_cfg():
    exposed = READS + SENDS
    return {
        "expose": exposed,
        "tiers": {**{n: "safe" for n in READS}, **{n: "confirm" for n in SENDS}},
    }


# ============================ 1. bridge registration =======================
@pytest.mark.parametrize("server", ["slack", "discord"])
def test_reads_safe_sends_confirm_admin_not_exposed(server):
    reg = ToolRegistry()
    bridge = MCPBridge(config={}, registry=reg)
    bridge._register_tools(server, _FakeConn(READS + SENDS + ADMIN), _server_cfg())

    for name in READS:
        spec = reg.get(name)
        assert spec is not None and spec.tier == "safe", f"{name} must be a safe read"
        assert spec.category == f"mcp:{server}"
    for name in SENDS:
        spec = reg.get(name)
        assert spec is not None and spec.tier == "confirm", f"{name} must be confirm-gated"
    for name in ADMIN:
        assert reg.get(name) is None, f"{name} must never be exposed"


def test_slack_and_discord_coexist_without_name_collision():
    # Both servers expose identically named tools; the first keeps the bare
    # names, the second is disambiguated with a server prefix (no crash).
    reg = ToolRegistry()
    bridge = MCPBridge(config={}, registry=reg)
    bridge._register_tools("slack", _FakeConn(READS + SENDS), _server_cfg())
    bridge._register_tools("discord", _FakeConn(READS + SENDS), _server_cfg())

    assert reg.get("get_mentions").category == "mcp:slack"          # first wins the bare name
    assert reg.get("discord_get_mentions") is not None              # second is prefixed
    assert reg.get("discord_get_mentions").category == "mcp:discord"
    assert reg.get("discord_post_message").tier == "confirm"        # tier preserved on the prefixed tool


@pytest.mark.asyncio
async def test_send_confirmation_shows_exact_channel_and_full_text():
    reg = ToolRegistry()
    bridge = MCPBridge(config={}, registry=reg)
    bridge._register_tools("slack", _FakeConn(READS + SENDS), _server_cfg())
    post = reg.get("post_message")

    guardian = Guardian(event_bus=EventBus())
    text = "Ship it at 5pm, and tell the team the incident is resolved."
    verdict = await guardian.check(post, {"channel": "#eng-general", "text": text})
    assert verdict.outcome == VerdictType.NEEDS_CONFIRMATION
    assert "#eng-general" in verdict.reason      # exact channel
    assert text in verdict.reason                # full text, verbatim (not paraphrased)
    for p in list(guardian._pending.values()):
        p.expiry_task.cancel()


# --------------------------------- doubles ---------------------------------
class _FakeTransport:
    def __init__(self):
        self.sent: List[Dict[str, str]] = []
        self._n = 0

    async def send(self, channel_id: str, text: str) -> str:
        self._n += 1
        self.sent.append({"channel": channel_id, "text": text})
        return f"msg-{self._n}"


async def _noop_handler(arguments, context):
    return ""


def _dangerous_tool():
    return ToolSpec(name="run_shell", description="d", tier="dangerous", handler=_noop_handler, category="creator")


def _confirm_tool():
    return ToolSpec(name="post_message", description="c", tier="confirm", handler=_noop_handler, category="mcp:slack")


class _GuardianCallingBrain:
    """Stands in for the real Brain+Planner: a turn attempts ONE tool through
    the Guardian (as the planner would), honoring the active session policy."""

    def __init__(self, guardian: Guardian, tool_spec: ToolSpec, args: Dict[str, Any]):
        self._g = guardian
        self._spec = tool_spec
        self._args = args
        self.calls: List[str] = []
        self.last_outcome: str = ""

    async def handle_user_text(self, text: str, **_: Any):
        self.calls.append(text)
        verdict = await self._g.check(self._spec, self._args)
        if verdict.outcome == VerdictType.NEEDS_CONFIRMATION:
            verdict = await self._g.await_resolution(verdict.request_id)
        self.last_outcome = verdict.outcome.value
        if verdict.outcome == VerdictType.ALLOW:
            return SimpleNamespace(text="Done, Sir.")
        return SimpleNamespace(text=f"Not done ({verdict.outcome.value}).")


class _EchoBrain:
    def __init__(self):
        self.calls: List[str] = []

    async def handle_user_text(self, text: str, **_: Any):
        self.calls.append(text)
        return SimpleNamespace(text=f"Ack: {text}")


def _remote(brain, transport, bus, *, allow_dangerous=False, owner="111", channel="222"):
    cfg = RemoteConfig(enabled=True, owner_user_id=owner, channel_id=channel, allow_dangerous=allow_dangerous)
    return DiscordRemote(brain=brain, config=cfg, transport=transport, event_bus=bus)


# =========================== 3. remote tier ceiling ========================
@pytest.mark.asyncio
async def test_dangerous_tool_denied_under_remote_policy_no_confirmation():
    guardian = Guardian(event_bus=EventBus())
    dangerous = _dangerous_tool()

    requested: List[ConfirmationRequestedEvent] = []

    async def on_req(e: ConfirmationRequestedEvent):
        requested.append(e)

    EventBus().subscribe(ConfirmationRequestedEvent, on_req)

    # Local: dangerous is confirmable (a prompt is emitted).
    v_local = await guardian.check(dangerous, {"command": "echo hi"})
    assert v_local.outcome == VerdictType.NEEDS_CONFIRMATION
    assert len(requested) == 1

    # Remote (allow_dangerous=False): denied at the gate, NO confirmation emitted.
    with session_policy(SessionPolicy(name="remote", allow_dangerous=False)):
        v_remote = await guardian.check(dangerous, {"command": "echo hi"})
    assert v_remote.outcome == VerdictType.DENY
    assert "disabled for remote" in v_remote.reason
    assert len(requested) == 1  # unchanged — nothing new was offered for confirmation

    for p in list(guardian._pending.values()):
        p.expiry_task.cancel()


@pytest.mark.asyncio
async def test_remote_turn_refuses_dangerous_tool_end_to_end():
    bus = EventBus()
    guardian = Guardian(event_bus=bus)
    transport = _FakeTransport()
    brain = _GuardianCallingBrain(guardian, _dangerous_tool(), {"command": "rm -rf ~/x"})
    remote = _remote(brain, transport, bus, allow_dangerous=False)

    await remote.on_incoming(IncomingMessage(author_id="111", channel_id="222", content="wipe my home dir"))

    assert brain.last_outcome == "deny"                      # gate refused it
    assert transport.sent and transport.sent[-1]["text"].startswith("Not done")
    # No confirmation prompt was ever posted for a dangerous tool over remote.
    assert not any("Confirmation required" in m["text"] for m in transport.sent)


# ============================ 4. owner-only access =========================
@pytest.mark.asyncio
async def test_non_owner_message_is_rejected():
    bus = EventBus()
    transport = _FakeTransport()
    brain = _EchoBrain()
    remote = _remote(brain, transport, bus, owner="111", channel="222")

    await remote.on_incoming(IncomingMessage(author_id="999", channel_id="222", content="what are my emails"))

    assert brain.calls == []          # no turn ran
    assert transport.sent == []       # and nothing was posted back


@pytest.mark.asyncio
async def test_owner_message_runs_a_turn_and_replies():
    bus = EventBus()
    transport = _FakeTransport()
    brain = _EchoBrain()
    remote = _remote(brain, transport, bus, owner="111", channel="222")

    await remote.on_incoming(IncomingMessage(author_id="111", channel_id="222", content="any mentions I've missed?"))

    assert brain.calls == ["any mentions I've missed?"]
    assert transport.sent[-1]["text"] == "Ack: any mentions I've missed?"


# ===================== 5. explicit remote confirmation =====================
@pytest.mark.asyncio
async def test_confirm_over_remote_requires_explicit_approval_bare_yes_refused():
    bus = EventBus()
    guardian = Guardian(event_bus=bus)
    transport = _FakeTransport()
    brain = _GuardianCallingBrain(guardian, _confirm_tool(), {"channel": "#general", "text": "hello team"})
    remote = _remote(brain, transport, bus, allow_dangerous=False)

    # The turn blocks awaiting the owner's approval — run it in the background.
    turn = asyncio.create_task(
        remote.on_incoming(IncomingMessage(author_id="111", channel_id="222", content="post hello to #general"))
    )
    await asyncio.sleep(0.05)

    # A confirmation prompt with a request-id handle was posted.
    prompts = [m for m in transport.sent if "Confirmation required" in m["text"]]
    assert len(prompts) == 1
    prompt_text = prompts[0]["text"]
    short_id = prompt_text.split("id: ")[1].split(")")[0].strip()

    # A bare "yes" must NOT approve it — the turn keeps waiting.
    await remote.on_incoming(IncomingMessage(author_id="111", channel_id="222", content="yes"))
    await asyncio.sleep(0.02)
    assert not turn.done(), "a bare 'yes' must not approve a remote confirmation"
    assert any("won't approve it" in m["text"] for m in transport.sent)

    # The explicit request id approves it — the turn now completes as allowed.
    await remote.on_incoming(IncomingMessage(author_id="111", channel_id="222", content=f"approve {short_id}"))
    await asyncio.wait_for(turn, timeout=2)
    assert brain.last_outcome == "allow"
    assert any(m["text"] == "Approved." for m in transport.sent)


@pytest.mark.asyncio
async def test_confirm_over_remote_approved_by_reaction():
    bus = EventBus()
    guardian = Guardian(event_bus=bus)
    transport = _FakeTransport()
    brain = _GuardianCallingBrain(guardian, _confirm_tool(), {"channel": "#general", "text": "hi"})
    remote = _remote(brain, transport, bus, allow_dangerous=False)

    turn = asyncio.create_task(
        remote.on_incoming(IncomingMessage(author_id="111", channel_id="222", content="post hi to #general"))
    )
    await asyncio.sleep(0.05)
    prompt = [m for m in transport.sent if "Confirmation required" in m["text"]]
    assert len(prompt) == 1
    prompt_msg_id = "msg-1"  # the fake transport's id for the first sent message (the prompt)

    # A ✅ reaction ON THE PROMPT approves it.
    await remote.on_incoming(IncomingMessage(
        author_id="111", channel_id="222", is_reaction=True, reaction_emoji="✅", reaction_target_id=prompt_msg_id,
    ))
    await asyncio.wait_for(turn, timeout=2)
    assert brain.last_outcome == "allow"


# ============================ briefing integration =========================
@pytest.mark.asyncio
async def test_briefing_includes_slack_section_at_most_two_lines():
    from proactive.briefing import gather_slack

    reg = ToolRegistry()

    async def _mentions(arguments, context):
        return '[{"from": "ceo"}, {"from": "pm"}, {"from": "lead"}]'

    async def _unreads(arguments, context):
        return '[{"channel": "dm-1"}, {"channel": "dm-2"}]'

    reg.register(ToolSpec(name="get_mentions", description="m", handler=_mentions, category="mcp:slack"))
    reg.register(ToolSpec(name="list_unreads", description="u", handler=_unreads, category="mcp:slack"))

    section = await gather_slack(reg)
    assert section is not None
    assert "Slack" in section
    assert "3" in section and "2" in section          # mention count + dm count surfaced
    body_lines = [ln for ln in section.splitlines() if ln.strip().startswith("-")]
    assert len(body_lines) <= 2                        # 2 lines max


@pytest.mark.asyncio
async def test_briefing_slack_absent_when_not_wired():
    from proactive.briefing import gather_slack

    reg = ToolRegistry()
    assert await gather_slack(reg) is None
    # A same-named tool from a DIFFERENT server must not be mistaken for Slack.
    reg.register(ToolSpec(name="get_mentions", description="m", handler=_noop_handler, category="mcp:discord"))
    assert await gather_slack(reg) is None


@pytest.mark.asyncio
async def test_confirm_over_remote_cancelled_by_no():
    bus = EventBus()
    guardian = Guardian(event_bus=bus)
    transport = _FakeTransport()
    brain = _GuardianCallingBrain(guardian, _confirm_tool(), {"channel": "#general", "text": "hi"})
    remote = _remote(brain, transport, bus, allow_dangerous=False)

    turn = asyncio.create_task(
        remote.on_incoming(IncomingMessage(author_id="111", channel_id="222", content="post hi to #general"))
    )
    await asyncio.sleep(0.05)
    await remote.on_incoming(IncomingMessage(author_id="111", channel_id="222", content="no"))
    await asyncio.wait_for(turn, timeout=2)
    assert brain.last_outcome == "deny"
    assert any(m["text"] == "Cancelled." for m in transport.sent)
