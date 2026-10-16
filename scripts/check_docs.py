#!/usr/bin/env python3
"""Check that every relative link in the Markdown docs points at a file that exists (and, for #anchors, at a heading that
exists), and that every backticked repo path mentioned in the README exists. Offline and read-only.

    python scripts/check_docs.py          # exit 1 and list the problems if any link is broken
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|!\[[^\]]*\]\(([^)\s]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$", re.M)
FENCE = re.compile(r"^```.*?^```", re.M | re.S)


def slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    h = re.sub(r"[`*_]", "", heading.strip().lower())
    h = re.sub(r"[^\w\- ]", "", h, flags=re.UNICODE)
    return h.replace(" ", "-")


def anchors(path: Path) -> set:
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    seen, out = {}, set()
    for _, title in HEADING.findall(text):
        s = slug(title)
        n = seen.get(s, 0)
        out.add(s if n == 0 else f"{s}-{n}")
        seen[s] = n + 1
    return out


def check(files: List[Path]) -> List[str]:
    problems = []
    for md in files:
        text = FENCE.sub("", md.read_text(encoding="utf-8"))
        for m in LINK.finditer(text):
            target = m.group(1) or m.group(2)
            if re.match(r"^(https?:|mailto:|tel:)", target):
                continue
            path_part, _, frag = target.partition("#")
            dest = md if not path_part else (md.parent / path_part).resolve()
            if not dest.exists():
                problems.append(f"{md.relative_to(ROOT)}: missing file  {target}")
            elif frag and dest.suffix == ".md" and frag not in anchors(dest):
                problems.append(f"{md.relative_to(ROOT)}: missing anchor  {target}")
    return problems


CODE_PATH = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|md|yaml|yml|ts|json|sh))(?:::[A-Za-z0-9_]+)?`")
#: Files the run writes (git-ignored) or that appear in examples; not expected to exist in a fresh checkout.
RUNTIME_PREFIXES = ("data/", "logs/", "current.json", "config/briefing.local.yaml", "briefing.yaml", "rules.json",
                    "mcp_servers/.venv", "voice/models/")


def check_paths(files: List[Path]) -> List[str]:
    """Backticked repo paths (tests/test_router.py, briefing/redact.py ...) in the docs must exist."""
    problems = []
    for md in files:
        for m in CODE_PATH.finditer(FENCE.sub("", md.read_text(encoding="utf-8"))):
            rel = m.group(1)
            if "*" in rel or rel.startswith(RUNTIME_PREFIXES) or "/" not in rel:
                continue
            if not (ROOT / rel).exists() and not (md.parent / rel).exists():
                problems.append(f"{md.relative_to(ROOT)}: backticked path does not exist  {rel}")
    return problems


def main() -> int:
    files = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md")) + sorted(ROOT.glob("voice/**/*.md"))
    files = [f for f in files if f.exists()]
    fresh = [ROOT / n for n in ("README.md", "docs/DESIGN.md", "docs/SECURITY.md", "docs/RESULTS.md", "docs/DEMO.md",
                                "docs/ROUTER.md", "docs/PROACTIVE.md", "docs/MESSAGES.md")]
    problems = check(files) + check_paths(fresh)
    for p in problems:
        print(p)
    print(f"{len(files)} files checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
