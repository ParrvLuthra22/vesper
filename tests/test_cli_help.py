"""`vesper --help`, `-h` and `help` print usage and exit 0 without booting the Brain or touching any data file."""
from __future__ import annotations

import builtins
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def snapshot(root: Path):
    out = {}
    for p in sorted((root / "data").rglob("*")) if (root / "data").exists() else []:
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


@pytest.mark.parametrize("flag", ["--help", "-h", "help"])
def test_help_prints_usage_and_exits_zero_in_process(flag, monkeypatch, capsys):
    import cli.app as app

    opened = []
    real_open, real_connect = builtins.open, sqlite3.connect

    def spy_open(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    def spy_connect(database, *a, **k):
        opened.append(str(database))
        return real_connect(database, *a, **k)

    def no_boot(*a, **k):
        raise AssertionError("the Brain / event loop must not start for --help")

    monkeypatch.setattr(sys, "argv", ["vesper", flag])
    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(sqlite3, "connect", spy_connect)
    monkeypatch.setattr(app.asyncio, "run", no_boot)
    monkeypatch.setattr(app, "Brain", no_boot)
    with pytest.raises(SystemExit) as exc:
        app.run()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage: vesper" in out and "doctor" in out and "up | down | status" in out
    assert [f for f in opened if "data" in Path(f).parts or f.endswith((".db", ".jsonl"))] == []


def test_help_in_a_fresh_process_leaves_the_data_directory_untouched():
    before = snapshot(ROOT)
    for flag in ("--help", "-h", "help"):
        r = subprocess.run([sys.executable, "-c", f"import sys; sys.argv=['vesper','{flag}']; from cli.app import run; run()"],
                           cwd=ROOT, capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONWARNINGS": "ignore"})
        assert r.returncode == 0 and "usage: vesper" in r.stdout, (flag, r.stderr[-300:])
        assert "Good evening" not in r.stdout and "You:" not in r.stdout           # the REPL never started
    assert snapshot(ROOT) == before
