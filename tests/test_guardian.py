"""
Tests for guardian/gate.py (the Guardian safety gate).

Each test builds its own EventBus (after resetting the process-wide
singleton) so Guardian instances across tests never share subscriptions.
"""

from __future__ import annotations

import json

import pytest

from bus.event_bus import EventBus
from guardian.gate import Guardian, VerdictType
from schemas.events import ConfirmationRequestedEvent, ConfirmationResponseEvent
from tools.registry import ToolSpec


@pytest.fixture(autouse=True)
def _fresh_bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


def _make_tool(tier: str, name: str = "some_tool") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="test tool",
        target_agent="SystemAgent",
        action=name,
        tier=tier,
    )


# =============================================================================
# safe tier
# =============================================================================

@pytest.mark.asyncio
async def test_safe_tier_allows_immediately(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    guardian = Guardian(event_bus=EventBus(), audit_log_path=audit_path)

    verdict = await guardian.check(_make_tool("safe"), {}, context={})

    assert verdict.outcome == VerdictType.ALLOW
    assert verdict.allowed
    assert not audit_path.exists()  # safe tier is never audited


# =============================================================================
# confirm / dangerous tiers: request-confirmation path
# =============================================================================

@pytest.mark.asyncio
async def test_confirm_tier_requests_confirmation_and_emits_event(tmp_path):
    bus = EventBus()
    received = []

    async def on_confirmation_requested(event):
        received.append(event)

    bus.subscribe(ConfirmationRequestedEvent, on_confirmation_requested)

    guardian = Guardian(event_bus=bus, audit_log_path=tmp_path / "audit.jsonl")
    verdict = await guardian.check(
        _make_tool("confirm", name="close_app"), {"app_name": "Safari"}, context={}
    )

    assert verdict.outcome == VerdictType.NEEDS_CONFIRMATION
    assert verdict.request_id is not None
    assert len(received) == 1
    assert received[0].tool_name == "close_app"
    assert received[0].request_id == verdict.request_id
    assert "close_app" in verdict.reason
    assert "Safari" in verdict.reason


@pytest.mark.asyncio
async def test_dangerous_tier_also_needs_confirmation(tmp_path):
    guardian = Guardian(event_bus=EventBus(), audit_log_path=tmp_path / "audit.jsonl")

    verdict = await guardian.check(
        _make_tool("dangerous", name="run_shell"), {"cmd": "ls"}, context={}
    )

    assert verdict.outcome == VerdictType.NEEDS_CONFIRMATION
    assert verdict.request_id is not None


# =============================================================================
# confirm tier: resolution via ConfirmationResponseEvent
# =============================================================================

@pytest.mark.asyncio
async def test_confirm_tier_approved_resolves_to_allow_and_audits(tmp_path):
    bus = EventBus()
    audit_path = tmp_path / "audit.jsonl"
    guardian = Guardian(event_bus=bus, audit_log_path=audit_path)

    verdict = await guardian.check(
        _make_tool("confirm", name="close_app"), {"app_name": "Safari"}, context={}
    )
    await bus.emit(
        ConfirmationResponseEvent(request_id=verdict.request_id, approved=True, source="voice_agent")
    )

    final = await guardian.await_resolution(verdict.request_id)

    assert final.outcome == VerdictType.ALLOW
    entry = json.loads(audit_path.read_text().strip().splitlines()[0])
    assert entry["tool"] == "close_app"
    assert entry["args"] == {"app_name": "Safari"}
    assert entry["verdict"] == "allow"
    assert entry["who_approved"] == "voice_agent"


@pytest.mark.asyncio
async def test_confirm_tier_denied_resolves_to_deny_and_audits(tmp_path):
    bus = EventBus()
    audit_path = tmp_path / "audit.jsonl"
    guardian = Guardian(event_bus=bus, audit_log_path=audit_path)

    verdict = await guardian.check(
        _make_tool("confirm", name="close_app"), {"app_name": "Safari"}, context={}
    )
    await bus.emit(
        ConfirmationResponseEvent(request_id=verdict.request_id, approved=False, source="voice_agent")
    )

    final = await guardian.await_resolution(verdict.request_id)

    assert final.outcome == VerdictType.DENY
    entry = json.loads(audit_path.read_text().strip().splitlines()[0])
    assert entry["verdict"] == "deny"
    assert entry["who_approved"] is None


@pytest.mark.asyncio
async def test_await_resolution_on_unknown_request_id_denies(tmp_path):
    guardian = Guardian(event_bus=EventBus(), audit_log_path=tmp_path / "audit.jsonl")

    verdict = await guardian.await_resolution("does-not-exist")

    assert verdict.outcome == VerdictType.DENY


# =============================================================================
# confirmation expiry
# =============================================================================

@pytest.mark.asyncio
async def test_confirmation_expires_after_timeout_and_denies(tmp_path):
    bus = EventBus()
    audit_path = tmp_path / "audit.jsonl"
    guardian = Guardian(event_bus=bus, audit_log_path=audit_path, confirmation_timeout_seconds=0.05)

    verdict = await guardian.check(
        _make_tool("confirm", name="close_app"), {"app_name": "Safari"}, context={}
    )
    assert verdict.outcome == VerdictType.NEEDS_CONFIRMATION

    final = await guardian.await_resolution(verdict.request_id)

    assert final.outcome == VerdictType.DENY
    assert "expired" in final.reason
    entry = json.loads(audit_path.read_text().strip().splitlines()[0])
    assert entry["verdict"] == "deny"
    assert entry["who_approved"] is None


@pytest.mark.asyncio
async def test_late_response_after_expiry_is_ignored(tmp_path):
    bus = EventBus()
    audit_path = tmp_path / "audit.jsonl"
    guardian = Guardian(event_bus=bus, audit_log_path=audit_path, confirmation_timeout_seconds=0.05)

    verdict = await guardian.check(
        _make_tool("confirm", name="close_app"), {"app_name": "Safari"}, context={}
    )
    final = await guardian.await_resolution(verdict.request_id)
    assert final.outcome == VerdictType.DENY  # expired

    # A response that arrives after expiry must not resurrect the request
    # or append a second audit entry.
    await bus.emit(
        ConfirmationResponseEvent(request_id=verdict.request_id, approved=True, source="voice_agent")
    )

    lines = audit_path.read_text().strip().splitlines()
    assert len(lines) == 1
