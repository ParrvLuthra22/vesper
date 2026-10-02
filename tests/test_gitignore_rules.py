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
    ".claude/settings.local.json", "data/audit.jsonl", "data/run/token",
]

pytestmark = pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")


@pytest.mark.parametrize("path", MUST_BE_IGNORED)
def test_path_is_ignored(path):
    r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    assert r.returncode == 0, f"{path} is not git-ignored"


def test_none_of_them_is_tracked():
    r = subprocess.run(["git", "ls-files", "--", *MUST_BE_IGNORED], cwd=ROOT, capture_output=True, text=True)
    assert r.stdout.strip() == ""
