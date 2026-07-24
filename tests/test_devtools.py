"""Developer tools (PC0) — registration, MCP expose/tier mapping, run_tests as a
slow task, and git_commit message generation + guardian summary."""
from __future__ import annotations

from pathlib import Path

import pytest

import tools  # noqa: F401 — registers builtin + dev tools into the shared registry
import tools.devtools as dt
from bus.event_bus import EventBus
from guardian.gate import Guardian, VerdictType
from tools.mcp_bridge import MCPBridge
from tools.registry import ToolRegistry, ToolSpec, get_registry


class _FakeConn:
    def __init__(self, names):
        self.tools = [{"name": n, "description": f"{n} desc", "inputSchema": {"type": "object", "properties": {}}} for n in names]


# ------------------------- GitHub MCP: expose + tiers ----------------------
def test_bridge_expose_allowlist_and_tier_mapping():
    reg = ToolRegistry()
    bridge = MCPBridge(config={}, registry=reg)
    cfg = {
        "expose": ["list_pull_requests", "add_issue_comment"],
        "tiers": {"list_pull_requests": "safe", "add_issue_comment": "confirm"},
    }
    bridge._register_tools("github", _FakeConn(["list_pull_requests", "merge_pull_request", "add_issue_comment"]), cfg)

    assert reg.get("list_pull_requests").tier == "safe"
    assert reg.get("add_issue_comment").tier == "confirm"
    assert reg.get("merge_pull_request") is None  # not in expose -> never registered
    assert reg.get("list_pull_requests").category == "mcp:github"


def test_bridge_without_expose_registers_all_with_default_tier():
    reg = ToolRegistry()
    bridge = MCPBridge(config={}, registry=reg)
    bridge._register_tools("s", _FakeConn(["a", "b"]), {"tiers": {"a": "safe"}})
    assert reg.get("a").tier == "safe"
    assert reg.get("b").tier == "confirm"  # DEFAULT_UNMAPPED_TIER


# ------------------------------ run_tests (slow) ---------------------------
def test_run_tests_registered_as_slow_safe():
    spec = get_registry().get("run_tests")
    assert spec is not None
    assert spec.slow is True   # -> planner routes it through the TaskQueue
    assert spec.tier == "safe"


@pytest.mark.asyncio
async def test_run_tests_reports_pass(monkeypatch):
    async def fake_repo_root():
        return "/repo"
    async def fake_run(cmd, cwd=None, timeout=30.0):
        return (0, "collected 5 items\n\n5 passed in 1.20s\n", "")
    monkeypatch.setattr(dt, "_repo_root", fake_repo_root)
    monkeypatch.setattr(dt, "_run", fake_run)
    out = await dt.run_tests({}, {})
    assert out.startswith("Tests passed") and "5 passed" in out


@pytest.mark.asyncio
async def test_run_tests_reports_first_failure(monkeypatch):
    async def fake_repo_root():
        return "/repo"
    async def fake_run(cmd, cwd=None, timeout=30.0):
        return (1, "FAILED tests/test_x.py::test_y - assert 1 == 2\n1 failed, 4 passed in 1s\n", "")
    monkeypatch.setattr(dt, "_repo_root", fake_repo_root)
    monkeypatch.setattr(dt, "_run", fake_run)
    out = await dt.run_tests({}, {})
    assert "Tests FAILED" in out and "first failure" in out and "test_y" in out


# ----------------- git_commit: message generation + summary ----------------
@pytest.mark.asyncio
async def test_commit_message_generation_mocked_llm(monkeypatch):
    class FakeResp:
        text = "feat: add developer tools\n(second line ignored)"

    class FakeRouter:
        async def complete(self, messages, purpose):
            assert purpose == "planning"
            assert "staged changes" in messages[0]["content"]
            return FakeResp()

    monkeypatch.setattr(dt, "get_router", lambda: FakeRouter())
    msg = await dt._generate_commit_message("diff --git a/x b/x")
    assert msg == "feat: add developer tools"  # single line, cleaned


@pytest.mark.asyncio
async def test_commit_summary_shows_repo_branch_message_and_injects(monkeypatch):
    async def fake_repo_root():
        return "/repo/vesper"
    async def fake_branch():
        return "v3-devtools"
    async def fake_run(cmd, cwd=None, timeout=30.0):
        return (0, "some staged diff", "")
    async def fake_gen(diff):
        return "chore: tidy the tools"

    monkeypatch.setattr(dt, "_repo_root", fake_repo_root)
    monkeypatch.setattr(dt, "_current_branch", fake_branch)
    monkeypatch.setattr(dt, "_run", fake_run)
    monkeypatch.setattr(dt, "_generate_commit_message", fake_gen)

    args = {}  # no message given
    summary = await dt._git_commit_summary(args)
    assert "/repo/vesper" in summary            # exact repo
    assert "v3-devtools" in summary             # exact branch
    assert "chore: tidy the tools" in summary   # generated message shown
    assert args["message"] == "chore: tidy the tools"  # injected so handler commits it


# --------------------- Guardian uses the custom summary --------------------
@pytest.mark.asyncio
async def test_guardian_uses_custom_confirm_summary(tmp_path: Path):
    bus = EventBus()
    await bus.start()
    guardian = Guardian(event_bus=bus, audit_log_path=tmp_path / "audit.jsonl")

    async def summary(arguments):
        return f"CUSTOM SUMMARY for x={arguments.get('x')}"

    async def _noop(a, c):
        return ""

    spec = ToolSpec(name="t", description="", tier="confirm", handler=_noop, confirm_summary=summary)
    verdict = await guardian.check(spec, {"x": 42}, context={})
    assert verdict.outcome == VerdictType.NEEDS_CONFIRMATION
    assert verdict.reason == "CUSTOM SUMMARY for x=42"


@pytest.mark.asyncio
async def test_git_status_and_diff_handle_non_repo(monkeypatch):
    async def not_a_repo():
        return None
    monkeypatch.setattr(dt, "_repo_root", not_a_repo)
    assert "Not a git repository" in await dt.git_status({}, {})
    assert "Not a git repository" in await dt.git_diff({"staged": True}, {})
