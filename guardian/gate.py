"""
Guardian — the safety gate between the planner and tool execution.

The Guardian never executes tools itself; it only decides whether a given
tool call may proceed:
    - "safe" tier tools are always allowed immediately.
    - "confirm" and "dangerous" tier tools require explicit approval: the
      Guardian emits a ConfirmationRequestedEvent (for a voice/HUD/UI
      surface to present to the user) and waits for a matching
      ConfirmationResponseEvent. If none arrives within the configured
      timeout, the request is denied.

Every non-safe execution's final outcome (approved, denied, or expired) is
appended to a JSONL audit log.

Tainted-input rule
------------------
Text written by third parties (email, web pages, calendar entries, chat
messages) can contain instructions aimed at the model. When the Planner has read
such content earlier in the turn AND a later tool call carries an argument the
user did not say, the Planner marks the call tainted
(`context={"tainted_input": True, ...}`) and the Guardian raises its tier by one:

    safe -> confirm        confirm -> dangerous        dangerous -> dangerous

so a `safe` tool steered by an email still needs an explicit approval, with a
summary that says why.

Outbound-action rule (and the "draft" tier)
-------------------------------------------
A tool marked `outbound=True` would deliver content to another person (an email, a message, a post). When
such a call is raised by a turn that arrived over a NON-LOCAL channel (Telegram, Discord, any adapter, or any
restricted session policy), it is "Mac-only": the confirmation is shown on the remote channel as information,
but an APPROVAL is accepted only from a local surface (`LOCAL_APPROVER_SOURCES`: the HUD via the gateway, the
CLI, local voice). An approval from the remote channel itself is refused, audited
(`channel:approval_needs_mac`) and the request stays pending for the real approver until it expires; a remote
DENY is always honoured. Remote approval remains fine for local-only actions (reminders, git, files...). The
`outbound` flag also forbids the "safe" tier, so a tool that delivers content can never skip confirmation.
The "draft" tier (a tool that only PREPARES something, e.g. an email draft in Gmail) follows the same Mac-only
rule for remote channels, always needs an explicit approval, and is raised to "confirm" — not "dangerous" — by the
tainted-input rule (including the planner's recipient check: `ToolSpec.recipient_args`).

Trust boundary: `ConfirmationResponseEvent.source` is set by the in-process component that emits it (the
gateway stamps "gateway" only on its authenticated local /confirm endpoint and `"<channel>:<user>"` on
/channel/confirm), so a channel adapter cannot claim to be local over HTTP.

A turn starts tainted if it was handed untrusted text up front: a calendar-title
observation (`ObservationEvent.untrusted`) or a retrieved memory flagged
`tainted` (written by reflection from a tainted turn; a missing flag means
untainted).

Known limitation — cross-turn history: the rule works at turn granularity. The
Planner replays the last few turns of the conversation, and an assistant reply
that summarized an email or web page in an EARLIER turn is trusted text in a
later turn; taint does not follow it across turns. (Memories derived from such
turns are flagged, but the replayed history itself is not.) This is a one-rung
bump, deliberately not a data-flow tracking system.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from uuid import uuid4

from bus.event_bus import EventBus, get_event_bus
from schemas.events import ConfirmationRequestedEvent, ConfirmationResponseEvent
from tools.registry import ToolSpec
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_AUDIT_LOG_PATH = Path("data/audit.jsonl")
#: Overrides the default audit path (an explicit `audit_log_path=` still wins).
#: The test suite points this at a temp file so tests can never write to the
#: real audit trail.
AUDIT_LOG_ENV_VAR = "VESPER_AUDIT_LOG"
DEFAULT_CONFIRMATION_TIMEOUT_SECONDS = 120.0
#: Sources that count as "approved on the Mac": the HUD / local clients through the authenticated gateway
#: endpoint, the CLI, and local voice. Anything else (a channel adapter, a remote interface) is remote.
LOCAL_APPROVER_SOURCES = frozenset({"gateway", "cli", "voice_agent", "voice", "hud"})


@dataclass(frozen=True)
class SessionPolicy:
    """Extra restrictions the Guardian applies for the current session.

    Local surfaces (CLI, voice, HUD) run with no policy — full capability, with
    dangerous-tier calls still individually confirmed. A restricted surface —
    the Discord remote interface (remote/discord_remote.py) — activates a policy
    around each turn so the Guardian can enforce that surface's ceiling
    regardless of what the planner decides to call."""

    name: str = "local"
    #: When False, every dangerous-tier tool is DENIED outright (never even
    #: offered for confirmation) while this policy is active. The remote
    #: interface sets this from `remote.allow_dangerous`, which defaults False.
    allow_dangerous: bool = True


#: Task-scoped so a restricted turn's policy cannot leak into a concurrent
#: local turn — contextvars are per-asyncio-task and propagate across `await`
#: within the same task (the whole handle_user_text → planner → check chain).
_active_policy: ContextVar[Optional[SessionPolicy]] = ContextVar("guardian_session_policy", default=None)


def current_session_policy() -> Optional[SessionPolicy]:
    """The SessionPolicy in force for the current task, or None (local/unrestricted)."""
    return _active_policy.get()


@contextmanager
def session_policy(policy: SessionPolicy) -> Iterator[None]:
    """Activate `policy` for the duration of the `with` block (and everything it
    awaits within this task). Restored on exit, even on exception."""
    token = _active_policy.set(policy)
    try:
        yield
    finally:
        _active_policy.reset(token)


@dataclass(frozen=True)
class ChannelInfo:
    """Which inbound channel the current turn came from (docs/CHANNELS.md)."""

    name: str
    user_id: str = ""


_active_channel: ContextVar[Optional[ChannelInfo]] = ContextVar("guardian_channel", default=None)

#: What an audit entry records when the turn has neither a channel nor a session policy: CLI,
#: voice, the HUD and the local gateway /message path.
LOCAL_CHANNEL = "local"


def current_channel() -> Optional[ChannelInfo]:
    return _active_channel.get()


@contextmanager
def channel_context(channel: str, user_id: str = "") -> Iterator[None]:
    """Mark everything awaited inside the block as arriving over `channel`, so every Guardian audit
    entry (and the confirmation it raises) records it."""
    token = _active_channel.set(ChannelInfo(name=str(channel), user_id=str(user_id)))
    try:
        yield
    finally:
        _active_channel.reset(token)


def audit_channel_name() -> str:
    """The channel to stamp on an audit entry: the explicit channel, else the session policy's
    name (e.g. the Discord remote's "remote"), else "local"."""
    info = _active_channel.get()
    if info is not None:
        return info.name
    policy = _active_policy.get()
    return policy.name if policy is not None else LOCAL_CHANNEL


_TIER_ORDER = ("safe", "confirm", "dangerous")


def bump_tier(tier: str) -> str:
    """One rung up the tier ladder (dangerous stays dangerous). "draft" sits beside the ladder: one rung up from it is
    "confirm" (a draft whose recipient came from untrusted content is a full confirmation, not a dangerous call)."""
    if tier == "draft":
        return "confirm"
    idx = _TIER_ORDER.index(tier)
    return _TIER_ORDER[min(idx + 1, len(_TIER_ORDER) - 1)]


class VerdictType(str, Enum):
    ALLOW = "allow"
    NEEDS_CONFIRMATION = "needs_confirmation"
    DENY = "deny"


@dataclass
class Verdict:
    """The Guardian's decision for one tool call."""

    outcome: VerdictType
    reason: str = ""
    request_id: Optional[str] = None

    @property
    def allowed(self) -> bool:
        return self.outcome == VerdictType.ALLOW


@dataclass
class _PendingConfirmation:
    request_id: str
    tool_name: str
    arguments: Dict[str, Any]
    summary: str
    future: "asyncio.Future[Verdict]"
    expiry_task: asyncio.Task
    tainted: bool = False
    #: Captured when the confirmation is raised, inside the turn's context — the response
    #: arrives later on another task, where the contextvars are no longer set.
    channel: str = LOCAL_CHANNEL
    channel_user: str = ""
    #: outbound tool raised from a non-local channel: approvable only from LOCAL_APPROVER_SOURCES
    mac_only: bool = False


class Guardian:
    """Safety gate: decides allow / needs_confirmation / deny for a tool call."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        audit_log_path: Optional[Path] = None,
        confirmation_timeout_seconds: float = DEFAULT_CONFIRMATION_TIMEOUT_SECONDS,
    ):
        self._event_bus = event_bus or get_event_bus()
        if audit_log_path is None:
            audit_log_path = os.environ.get(AUDIT_LOG_ENV_VAR) or DEFAULT_AUDIT_LOG_PATH
        self._audit_log_path = Path(audit_log_path)
        self._confirmation_timeout_seconds = confirmation_timeout_seconds
        self._pending: Dict[str, _PendingConfirmation] = {}
        self._event_bus.subscribe(ConfirmationResponseEvent, self._handle_confirmation_response)

    async def check(
        self,
        tool_spec: ToolSpec,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Verdict:
        """Decide whether `tool_spec(arguments)` may run.

        Returns immediately for "safe" tier tools. For "confirm"/"dangerous"
        tools, emits a ConfirmationRequestedEvent and returns a
        NEEDS_CONFIRMATION verdict carrying a `request_id`; call
        `await_resolution(request_id)` to wait for the final allow/deny.
        """
        tainted = bool((context or {}).get("tainted_input"))
        tier = bump_tier(tool_spec.tier) if tainted else tool_spec.tier

        if tier == "safe":
            return Verdict(VerdictType.ALLOW, reason="safe tier - no confirmation required")

        # Session ceiling: a restricted surface (the remote interface) disables
        # dangerous-tier tools ENTIRELY — refused outright, never even offered
        # for confirmation. Confirm-tier still goes through the normal flow.
        policy = current_session_policy()
        if tier == "dangerous" and policy is not None and not policy.allow_dangerous:
            verdict = Verdict(
                VerdictType.DENY,
                reason=f"dangerous-tier tools are disabled for {policy.name} sessions",
            )
            self._audit(tool_spec.name, arguments, verdict, who_approved=None, tainted=tainted)
            logger.warning(
                f"[Guardian] denied dangerous tool '{tool_spec.name}' — "
                f"disabled for {policy.name} session"
            )
            return verdict

        summary = await self._render_summary(tool_spec, arguments)
        mac_only = (bool(tool_spec.outbound) or tool_spec.tier == "draft") and audit_channel_name() != LOCAL_CHANNEL
        if mac_only:
            what = "delivers content to another person" if tool_spec.outbound else "writes into your mail account"
            summary = (f"Requested from {audit_channel_name()} — this {what}, so it can only be "
                       f"approved on your Mac.\n{summary}")
        if tainted:
            reason = (context or {}).get("taint_reason") or "external content"
            summary = (
                f"⚠ Raised from {tool_spec.tier} to {tier}: an argument did not come from "
                f"you and this turn read {reason}.\n{summary}"
            )
        request_id = str(uuid4())

        loop = asyncio.get_running_loop()
        future: "asyncio.Future[Verdict]" = loop.create_future()
        expiry_task = asyncio.create_task(self._expire_after_timeout(request_id))

        self._pending[request_id] = _PendingConfirmation(
            request_id=request_id,
            tool_name=tool_spec.name,
            arguments=arguments,
            summary=summary,
            future=future,
            expiry_task=expiry_task,
            tainted=tainted,
            channel=audit_channel_name(),
            channel_user=(current_channel().user_id if current_channel() is not None else ""),
            mac_only=mac_only,
        )

        await self._event_bus.emit(
            ConfirmationRequestedEvent(
                summary=summary,
                tool_name=tool_spec.name,
                arguments=arguments,
                request_id=request_id,
                source="Guardian",
                channel=audit_channel_name(),
                channel_user=(current_channel().user_id if current_channel() is not None else ""),
                mac_only=mac_only,
            )
        )

        return Verdict(VerdictType.NEEDS_CONFIRMATION, reason=summary, request_id=request_id)

    def pending_channel(self, request_id: str) -> Optional[tuple]:
        """(channel, channel_user) the pending confirmation was raised for, or None if there is no such
        unresolved request. Lets a channel endpoint refuse to answer a confirmation that is not its own."""
        pending = self._pending.get(request_id)
        if pending is None or pending.future.done():
            return None
        return pending.channel, pending.channel_user

    def pending_mac_only(self, request_id: str) -> bool:
        """True if the unresolved request can only be approved on the Mac (outbound + remote origin)."""
        pending = self._pending.get(request_id)
        return bool(pending is not None and not pending.future.done() and pending.mac_only)

    async def await_resolution(self, request_id: str) -> Verdict:
        """Wait for a pending confirmation to resolve (approved, denied, or expired).

        Safe to call before or after resolution, and safe to call multiple
        times — resolved entries are kept (not popped) so every caller sees
        the same outcome.
        """
        pending = self._pending.get(request_id)
        if pending is None:
            return Verdict(VerdictType.DENY, reason="unknown or already-resolved request_id")
        return await pending.future

    @staticmethod
    async def _render_summary(tool_spec: ToolSpec, arguments: Dict[str, Any]) -> str:
        # A tool may supply its own summary (e.g. git_commit shows the exact
        # repo + branch + the message it generated from the diff).
        if tool_spec.confirm_summary is not None:
            return await tool_spec.confirm_summary(arguments)
        rendered_args = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        return f"{tool_spec.name}({rendered_args})"

    async def _expire_after_timeout(self, request_id: str) -> None:
        try:
            await asyncio.sleep(self._confirmation_timeout_seconds)
        except asyncio.CancelledError:
            return

        # Deliberately `.get()`, not `.pop()` — the entry stays around so
        # `await_resolution` can find it (and see the already-done future)
        # no matter when it's called relative to this expiry firing.
        pending = self._pending.get(request_id)
        if pending is None or pending.future.done():
            return

        verdict = Verdict(
            VerdictType.DENY,
            reason=f"confirmation expired after {self._confirmation_timeout_seconds:.0f}s",
            request_id=request_id,
        )
        self._write_audit(pending, verdict, who_approved=None)
        pending.future.set_result(verdict)

    async def _handle_confirmation_response(self, event: ConfirmationResponseEvent) -> None:
        # `.get()`, not `.pop()` — see _expire_after_timeout.
        pending = self._pending.get(event.request_id)
        if pending is None:
            return

        if pending.future.done():
            return
        if event.approved and pending.mac_only and str(event.source) not in LOCAL_APPROVER_SOURCES:
            # A remote channel tried to approve something that delivers content to another person. Refuse, keep the
            # request pending for the real approver (the expiry timer is untouched), and leave a trail.
            self.record_channel_event(
                "approval_needs_mac",
                {"request_id": event.request_id, "tool": pending.tool_name, "approver": str(event.source)[:80]},
                channel=pending.channel, user_id=pending.channel_user)
            logger.warning(f"[Guardian] refused remote approval of outbound tool '{pending.tool_name}' "
                           f"from '{event.source}': approve on the Mac")
            return
        pending.expiry_task.cancel()

        outcome = VerdictType.ALLOW if event.approved else VerdictType.DENY
        reason = "approved by user" if event.approved else "denied by user"
        verdict = Verdict(outcome, reason=reason, request_id=event.request_id)
        self._write_audit(pending, verdict, who_approved=event.source if event.approved else None)
        pending.future.set_result(verdict)

    def _write_audit(
        self,
        pending: _PendingConfirmation,
        verdict: Verdict,
        who_approved: Optional[str],
    ) -> None:
        self._audit(pending.tool_name, pending.arguments, verdict, who_approved, tainted=pending.tainted,
                    channel=pending.channel, mac_only=pending.mac_only)

    def record_channel_event(self, event: str, detail: Dict[str, Any], channel: str, user_id: str = "") -> None:
        """Audit something a channel refused or noticed that never reached a tool call — a stale, reused or
        foreign-user confirmation callback, a message from a non-allowlisted user. `detail` must not
        contain message content (callers pass ids, counts and reasons only)."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": f"channel:{event}",
            "args": dict(detail),
            "verdict": "rejected",
            "who_approved": None,
            "channel": str(channel),
        }
        if user_id:
            entry["channel_user"] = str(user_id)
        self._write_entry(entry)

    def _audit(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        verdict: Verdict,
        who_approved: Optional[str],
        tainted: bool = False,
        channel: Optional[str] = None,
        mac_only: bool = False,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "args": arguments,
            "verdict": verdict.outcome.value,
            "who_approved": who_approved,
            "channel": channel if channel is not None else audit_channel_name(),
        }
        if tainted:
            entry["tainted_input"] = True
        if mac_only:
            entry["mac_only"] = True          # outbound tool raised from a non-local channel
        self._write_entry(entry)

    def _write_entry(self, entry: Dict[str, Any]) -> None:
        try:
            self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            logger.error(f"[Guardian] failed to write audit log entry: {exc}")
