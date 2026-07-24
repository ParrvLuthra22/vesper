"""Vesper's developer tools (PC0) — local git / test / editor tools.

Registered into the shared tool registry on import (see tools/__init__.py).
Native async handlers; no planner/gateway/HUD/voice changes. Git tools operate
ONLY on the current working directory's repository — never a repo the user
didn't name. `git_commit` is confirm-tier and shows the exact repo + branch +
message (generating one from the staged diff via the LLM if none was given).

The GitHub half of the dev tools is the official GitHub MCP server, connected +
tiered + exposed through tools/mcp_bridge.py and config (mcp.servers.github).
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List, Optional, Tuple

from tools.registry import ToolSpec, get_registry
from utils.logger import get_logger

logger = get_logger(__name__)
registry = get_registry()

GIT = "git"
_COMMIT_MSG_MAX = 100


async def _run(cmd: List[str], cwd: Optional[str] = None, timeout: float = 30.0) -> Tuple[int, str, str]:
    """Run a command as an arg list (no shell) → (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, "", f"timed out after {timeout:.0f}s"
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


async def _repo_root() -> Optional[str]:
    rc, out, _ = await _run([GIT, "rev-parse", "--show-toplevel"])
    return out.strip() if rc == 0 and out.strip() else None


async def _current_branch() -> str:
    rc, out, _ = await _run([GIT, "rev-parse", "--abbrev-ref", "HEAD"])
    return out.strip() if rc == 0 else "(unknown)"


# --- LLM commit-message generation (get_router is split out so tests mock it) ---
def get_router():
    from config.settings import load_config_dict
    from llm.router import ModelRouter

    return ModelRouter(config=load_config_dict())


async def _generate_commit_message(diff: str) -> str:
    if not diff.strip():
        return "chore: update"
    prompt = (
        "Write ONE Conventional Commits message: a single line, imperative mood, "
        "under 72 characters, no body. Summarize these staged changes. Reply with "
        f"only the message.\n\n{diff[:6000]}"
    )
    try:
        resp = await get_router().complete(
            messages=[{"role": "user", "content": prompt}], purpose="planning"
        )
        first = (getattr(resp, "text", "") or "").strip().splitlines()
        msg = (first[0].strip().strip('"`').strip() if first else "")
        return msg[:_COMMIT_MSG_MAX] if msg else "chore: update"
    except Exception as exc:
        logger.warning(f"[devtools] commit-message generation failed: {exc}")
        return "chore: update"


# ================================ run_tests =================================
async def run_tests(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Run the project's test suite (pytest); return a pass/fail summary +
    the first failure. slow=True → runs on the task queue."""
    path = str(arguments.get("path") or "").strip()
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", "--tb=line"]
    if path:
        cmd.append(path)
    rc, out, err = await _run(cmd, cwd=await _repo_root(), timeout=600.0)
    lines = [ln for ln in (out + "\n" + err).splitlines() if ln.strip()]
    summary = next(
        (ln for ln in reversed(lines) if any(w in ln for w in ("passed", "failed", "error"))),
        lines[-1] if lines else "no output",
    ).strip()
    if rc == 0:
        return f"Tests passed — {summary}"
    first_fail = next((ln for ln in lines if ln.startswith("FAILED") or "Error" in ln or "assert" in ln), "")
    return f"Tests FAILED — {summary}" + (f"\nfirst failure: {first_fail.strip()}" if first_fail else "")


# ================================ git tools ================================
async def git_status(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    root = await _repo_root()
    if not root:
        return "Not a git repository (in the current working directory)."
    _, out, _ = await _run([GIT, "status", "--short", "--branch"], cwd=root)
    return out.strip() or "Working tree clean."


async def git_diff(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    root = await _repo_root()
    if not root:
        return "Not a git repository."
    staged = ["--staged"] if arguments.get("staged") else []
    _, stat, _ = await _run([GIT, "diff", "--stat", *staged], cwd=root)
    _, patch, _ = await _run([GIT, "diff", *staged], cwd=root)
    if not stat.strip():
        return "No staged changes." if staged else "No changes in the working tree."
    body = patch.strip()
    if len(body) > 4000:
        body = body[:4000] + "\n… (truncated)"
    return f"{stat.strip()}\n\n{body}"


async def _git_commit_summary(arguments: Dict[str, Any]) -> str:
    """Guardian confirmation summary: exact repo + branch + the message (which we
    generate from the staged diff if none was given, and inject so the handler
    commits with the very message the user approved)."""
    root = await _repo_root() or "(current directory — not a git repo)"
    branch = await _current_branch()
    msg = str(arguments.get("message") or "").strip()
    if not msg:
        _, diff, _ = await _run([GIT, "diff", "--staged"], cwd=await _repo_root())
        msg = await _generate_commit_message(diff)
        arguments["message"] = msg
    return f'Commit to {root} on branch "{branch}" with message:\n  "{msg}"'


async def git_commit(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    root = await _repo_root()
    if not root:
        return "Not a git repository."
    _, staged, _ = await _run([GIT, "diff", "--staged", "--name-only"], cwd=root)
    if not staged.strip():
        return "Nothing staged to commit — stage changes first (git add)."
    msg = str(arguments.get("message") or "").strip()
    if not msg:
        _, diff, _ = await _run([GIT, "diff", "--staged"], cwd=root)
        msg = await _generate_commit_message(diff)
    rc, out, err = await _run([GIT, "commit", "-m", msg], cwd=root)
    if rc != 0:
        return f"Commit failed: {(err or out).strip()[:200]}"
    _, short, _ = await _run([GIT, "rev-parse", "--short", "HEAD"], cwd=root)
    return f"Committed {short.strip()} on {await _current_branch()}: {msg}"


# ============================== open_in_editor =============================
async def open_in_editor(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    path = str(arguments.get("path") or "").strip()
    if not path:
        return "No path given."
    rc, out, err = await _run(["code", path])
    if rc != 0:
        return f"Could not open in VS Code (is the `code` command installed?): {(err or out).strip()[:120]}"
    return f"Opened {path} in VS Code."


# ============================== registration ==============================
def _register() -> None:
    registry.register(ToolSpec(
        name="run_tests",
        description="Run the project's test suite (pytest) and report a pass/fail summary plus the first failure. Optionally scope to a path/file.",
        parameters={"type": "object", "properties": {
            "path": {"type": "string", "description": "Optional test path or file; omit to run the whole suite."}}},
        tier="safe", handler=run_tests, category="dev", slow=True,
    ))
    registry.register(ToolSpec(
        name="git_status",
        description="Show `git status` (short) for the current working directory's repository.",
        tier="safe", handler=git_status, category="dev",
    ))
    registry.register(ToolSpec(
        name="git_diff",
        description="Show the current repo's diff (diffstat + patch). Set staged=true for the staged diff.",
        parameters={"type": "object", "properties": {
            "staged": {"type": "boolean", "description": "Show staged changes instead of the working tree."}}},
        tier="safe", handler=git_diff, category="dev",
    ))
    registry.register(ToolSpec(
        name="git_commit",
        description=(
            "Commit the currently STAGED changes in the current working directory's git repo. "
            "If the user did not dictate a message, omit `message` — one will be generated from "
            "the staged diff and shown for confirmation. Operates only on the current repo."
        ),
        parameters={"type": "object", "properties": {
            "message": {"type": "string", "description": "Commit message; omit to auto-generate from the staged diff."}}},
        tier="confirm", handler=git_commit, confirm_summary=_git_commit_summary, category="dev",
    ))
    registry.register(ToolSpec(
        name="open_in_editor",
        description="Open a file or folder in VS Code (via the `code` command).",
        parameters={"type": "object", "properties": {
            "path": {"type": "string", "description": "File or directory to open."}}, "required": ["path"]},
        tier="safe", handler=open_in_editor, category="dev",
    ))


_register()
