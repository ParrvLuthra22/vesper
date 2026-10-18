"""The ignore rules that keep personal data and secrets out of git (added across the earlier branches)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MUST_BE_IGNORED = [
    "data/briefing.db", "data/briefing.db-wal", "data/briefing.db-shm", "data/briefing_rules.json",
    "config/briefing.local.yaml", ".env", ".env.local", ".env.production", ".env.test",
    ".claude/settings.local.json", "data/audit.jsonl", "data/run/token", "data/notifications.json",
    # the repo is public: token files at ANY depth, and local working notes
    "google_token.json", "data/google_token.json", "mcp_servers/gmail/google_token.json", "a/b/c/google_token.json",
    "slack.token", "mcp_servers/slack/bot.token", "gmail.token",
    "DIAGNOSIS.md", "DIAGNOSIS2.md", "DIAGNOSIS3.md", "docs/DIAGNOSIS-notes.md",
    "PROGRESS.md", "BLOCKED.md", "VESPER_STATUS.md",
]

#: look-alikes that MUST stay trackable (the new rules are not allowed to swallow real code or docs)
MUST_NOT_BE_IGNORED = ["llm/token_meter.py", "tests/test_token_budget.py", "scripts/smoke_token_budget.py", "channels/credentials.py",
                       "docs/README.md", "docs/ROUTER.md", "DIAGNOSIS_TOOL.py", "docs/progress_report.md", "tools/google_tokens.py"]

pytestmark = pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")


@pytest.mark.parametrize("path", MUST_BE_IGNORED)
def test_path_is_ignored(path):
    r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    assert r.returncode == 0, f"{path} is not git-ignored"


def test_none_of_them_is_tracked():
    r = subprocess.run(["git", "ls-files", "--", *MUST_BE_IGNORED], cwd=ROOT, capture_output=True, text=True)
    assert r.stdout.strip() == ""


@pytest.mark.parametrize("path", MUST_NOT_BE_IGNORED)
def test_lookalikes_are_not_swallowed(path):
    r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    assert r.returncode == 1, f"{path} is wrongly ignored"


def test_the_rules_live_in_the_repo_gitignore_not_only_in_a_local_exclude():
    """DIAGNOSIS*.md used to be hidden only through .git/info/exclude, which a fresh clone does not have."""
    text = (ROOT / ".gitignore").read_text()
    for rule in ("google_token.json", "*.token", "DIAGNOSIS*.md", "PROGRESS.md", "BLOCKED.md", "VESPER_STATUS.md"):
        assert any(line.strip() == rule for line in text.splitlines()), rule


def test_no_tracked_file_matches_the_new_rules():
    """(Only the NEW rules: voice/models/.gitkeep is a long-standing tracked file under an ignored directory.)"""
    import fnmatch

    names = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
    patterns = ["google_token.json", "*.token", "DIAGNOSIS*.md", "PROGRESS.md", "BLOCKED.md", "VESPER_STATUS.md"]
    hits = [n for n in names if n and any(fnmatch.fnmatch(Path(n).name, pat) for pat in patterns)]
    assert hits == [], f"{len(hits)} tracked file(s) match the new ignore rules"
