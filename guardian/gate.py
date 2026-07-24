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
"""

from __future__ import annotations

import asyncio
import json
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


class Guardian:
    """Safety gate: decides allow / needs_confirmation / deny for a tool call."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        audit_log_path: Path = DEFAULT_AUDIT_LOG_PATH,
        confirmation_timeout_seconds: float = DEFAULT_CONFIRMATION_TIMEOUT_SECONDS,
    ):
        self._event_bus = event_bus or get_event_bus()
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
        if tool_spec.tier == "safe":
            return Verdict(VerdictType.ALLOW, reason="safe tier - no confirmation required")

        # Session ceiling: a restricted surface (the remote interface) disables
        # dangerous-tier tools ENTIRELY — refused outright, never even offered
        # for confirmation. Confirm-tier still goes through the normal flow.
        policy = current_session_policy()
        if tool_spec.tier == "dangerous" and policy is not None and not policy.allow_dangerous:
            verdict = Verdict(
                VerdictType.DENY,
                reason=f"dangerous-tier tools are disabled for {policy.name} sessions",
            )
            self._audit(tool_spec.name, arguments, verdict, who_approved=None)
            logger.warning(
                f"[Guardian] denied dangerous tool '{tool_spec.name}' — "
                f"disabled for {policy.name} session"
            )
            return verdict

        summary = await self._render_summary(tool_spec, arguments)
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
        self._audit(pending.tool_name, pending.arguments, verdict, who_approved)

    def _audit(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        verdict: Verdict,
        who_approved: Optional[str],
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "args": arguments,
            "verdict": verdict.outcome.value,
            "who_approved": who_approved,
        }
        try:
            self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            logger.error(f"[Guardian] failed to write audit log entry: {exc}")
