"""Run-state files shared by `vesper up` (writer) and `vesper status|down` (readers).

Everything lives under `data/run/` (git-ignored):
    launcher.json   supervisor pid + per-component state, rewritten on every change
    token           the session's gateway bearer token, mode 0600
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_RUN_DIR = Path("data/run")
STATE_FILE = "launcher.json"
TOKEN_FILE = "token"


def pid_alive(pid: Optional[int]) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours
    return True


class RunState:
    def __init__(self, run_dir: Path = DEFAULT_RUN_DIR):
        self.run_dir = Path(run_dir)

    @property
    def state_path(self) -> Path:
        return self.run_dir / STATE_FILE

    @property
    def token_path(self) -> Path:
        return self.run_dir / TOKEN_FILE

    def write(self, payload: Dict[str, Any]) -> None:
        """Atomic rewrite, so a concurrent `status` never reads a torn file."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.run_dir, prefix=".launcher-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            os.replace(tmp, self.state_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def read(self) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def write_token(self, token: str) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(token)

    def read_token(self) -> Optional[str]:
        try:
            return self.token_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def running_supervisor_pid(self) -> Optional[int]:
        """pid of a live supervisor, or None (a stale file from a crash is ignored)."""
        state = self.read()
        pid = (state or {}).get("supervisor_pid")
        return pid if pid_alive(pid) else None

    def clear(self) -> None:
        for path in (self.state_path, self.token_path):
            try:
                path.unlink()
            except OSError:
                pass
