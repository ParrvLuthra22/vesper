"""The docs must not rot: every relative link and #anchor in the README and docs/, and every backticked repo path in the
new docs, has to exist. (scripts/check_docs.py, offline.)"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("check_docs", ROOT / "scripts" / "check_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_repository_docs_have_no_broken_links_or_paths():
    mod = _load()
    files = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md")) + sorted(ROOT.glob("voice/**/*.md"))
    fresh = [ROOT / n for n in ("README.md", "docs/DESIGN.md", "docs/SECURITY.md", "docs/RESULTS.md", "docs/DEMO.md",
                                "docs/ROUTER.md", "docs/PROACTIVE.md", "docs/MESSAGES.md")]
    assert mod.check(files) + mod.check_paths(fresh) == []


def test_the_checker_actually_catches_broken_links_anchors_and_paths(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "B.md").write_text("# Real heading\n\ntext\n")
    a = tmp_path / "docs" / "A.md"
    a.write_text("[ok](B.md) [ok anchor](B.md#real-heading) [bad file](Nope.md) [bad anchor](B.md#not-there) "
                 "[web](https://example.com/x) `src/missing.py` `src/real.py`\n\n```\n[inside fence](Ignored.md)\n```\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "real.py").write_text("")
    problems = mod.check([a]) + mod.check_paths([a])
    assert sorted(p.split("  ")[-1] for p in problems) == sorted(["Nope.md", "B.md#not-there", "src/missing.py"])


def test_github_style_anchors():
    mod = _load()
    assert mod.slug("What it does today") == "what-it-does-today"
    assert mod.slug("Hardware `constraints`") == "hardware-constraints"
    assert mod.slug("8. Observability: how a bad plan gets debugged") == "8-observability-how-a-bad-plan-gets-debugged"
