"""Runs the HUD's own vitest/jsdom tests (hud/tests) as part of the suite when the frontend deps are installed."""
import shutil
import subprocess
from pathlib import Path

import pytest

HUD = Path(__file__).resolve().parents[1] / "hud"


@pytest.mark.skipif(not (HUD / "node_modules" / ".bin" / "vitest").exists() or shutil.which("node") is None,
                    reason="hud/node_modules (npm install in hud/) or node not available")
def test_hud_dom_tests_pass():
    r = subprocess.run([str(HUD / "node_modules" / ".bin" / "vitest"), "run"], cwd=HUD, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
