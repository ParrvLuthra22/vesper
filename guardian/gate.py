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


_TIER_ORDER = ("safe", "confirm", "dangerous")


def bump_tier(tier: str) -> str:
    """One rung up the tier ladder (dangerous stays dangerous)."""
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
        )

        await self._event_bus.emit(
            ConfirmationRequestedEvent(
                summary=summary,
                tool_name=tool_spec.name,
                arguments=arguments,
                request_id=request_id,
                source="Guardian",
            )
        )

        return Verdict(VerdictType.NEEDS_CONFIRMATION, reason=summary, request_id=request_id)

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

        pending.expiry_task.cancel()
        if pending.future.done():
            return

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
        self._audit(pending.tool_name, pending.arguments, verdict, who_approved, tainted=pending.tainted)

    def _audit(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        verdict: Verdict,
        who_approved: Optional[str],
        tainted: bool = False,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "args": arguments,
            "verdict": verdict.outcome.value,
            "who_approved": who_approved,
        }
        if tainted:
            entry["tainted_input"] = True
        try:
            self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            logger.error(f"[Guardian] failed to write audit log entry: {exc}")
