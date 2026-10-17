"""Is the HUD the launcher starts actually built from the sources in the repo?

`vesper up` starts the Tauri release binary `hud/src-tauri/target/release/hud`. That binary has the frontend (hud/dist) BAKED IN at
compile time, so editing hud/src, or even running `npm run build`, changes nothing the user sees until the binary itself is
rebuilt (`npm run tauri build`). On 2026-10-17 the binary was from 16 Aug while hud/src had been fixed on 17 Oct: the screenshot
showed the old layout. This module compares the newest source file with the binary, rebuilds it (or refuses to start the HUD, loudly)
and notices other HUD copies the launcher must never use.

Everything here is a pure function over paths / mtimes, with the build runner injectable, so it is tested with fake mtimes.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

#: Build + embed the frontend + compile the binary, without making a .app bundle. Slow the first time (cargo).
BUILD_COMMAND: Tuple[str, ...] = ("npm", "run", "tauri", "build", "--", "--no-bundle")
BUILD_COMMAND_TEXT = "cd hud && npm install && npm run tauri build -- --no-bundle"
DEFAULT_BUILD_TIMEOUT = 900.0

#: Anything under these changes what the user sees (frontend sources, the shell that embeds them, build config).
SOURCE_GLOBS: Tuple[str, ...] = ("src/**/*", "index.html", "vite.config.ts", "tsconfig.json", "src-tauri/src/**/*",
                                 "src-tauri/tauri.conf.json", "src-tauri/Cargo.toml", "src-tauri/capabilities/**/*")
FRESH, STALE, MISSING = "fresh", "stale", "missing"

Runner = Callable[[Sequence[str], Path, float, Dict[str, str]], Tuple[int, str, bool]]   # (returncode, output tail, timed_out)


@dataclass
class HudState:
    state: str                       # fresh | stale | missing
    binary: Path
    binary_mtime: Optional[float] = None
    newest_source: Optional[str] = None          # file NAME only
    newest_source_mtime: Optional[float] = None
    build_info: Dict[str, str] = field(default_factory=dict)

    @property
    def behind_by(self) -> float:
        if self.binary_mtime is None or self.newest_source_mtime is None:
            return 0.0
        return max(0.0, self.newest_source_mtime - self.binary_mtime)


def newest_source(hud_dir: Path) -> Tuple[Optional[str], Optional[float]]:
    best: Tuple[Optional[str], Optional[float]] = (None, None)
    for pattern in SOURCE_GLOBS:
        for p in hud_dir.glob(pattern):
            if not p.is_file():
                continue
            m = p.stat().st_mtime
            if best[1] is None or m > best[1]:
                best = (p.name, m)
    return best


def sidecar(binary: Path) -> Path:
    return binary.with_name(binary.name + ".build.json")


def read_build_info(binary: Path) -> Dict[str, str]:
    try:
        data = json.loads(sidecar(binary).read_text())
        return {k: str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def check(binary: Path, hud_dir: Path) -> HudState:
    name, src_m = newest_source(hud_dir)
    if not binary.exists():
        return HudState(MISSING, binary, None, name, src_m)
    bin_m = binary.stat().st_mtime
    info = read_build_info(binary)
    if src_m is not None and src_m > bin_m + 1.0:          # a second of slack for coarse filesystems
        return HudState(STALE, binary, bin_m, name, src_m, info)
    return HudState(FRESH, binary, bin_m, name, src_m, info)


def git_short(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def default_runner(cmd: Sequence[str], cwd: Path, timeout: float, env: Dict[str, str]) -> Tuple[int, str, bool]:
    if os.environ.get("PYTEST_CURRENT_TEST"):          # a test must never start a real cargo build: inject a runner instead
        return 125, "refused: running under pytest", False
    try:
        r = subprocess.run(list(cmd), cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr)[-600:], False
    except subprocess.TimeoutExpired:
        return 124, "", True
    except OSError as exc:
        return 127, f"{type(exc).__name__}: {exc}", False


def build_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ if base is None else base)
    cargo = str(Path.home() / ".cargo" / "bin")
    if cargo not in env.get("PATH", ""):
        env["PATH"] = cargo + os.pathsep + env.get("PATH", "")
    return env


@dataclass
class Outcome:
    ok: bool
    message: str                 # one clear line for the log
    state: HudState


def ensure_fresh(binary: Path, hud_dir: Path, root: Path, *, autobuild: bool = True, timeout: float = DEFAULT_BUILD_TIMEOUT,
                 runner: Runner = default_runner, log: Callable[[str], None] = print, now: Callable[[], float] = time.time,
                 env: Optional[Dict[str, str]] = None) -> Outcome:
    """Fresh -> ok. Stale/missing -> rebuild (autobuild) with a timeout; if that is off or fails, ok=False and the message names
    the exact command, so the caller refuses to start a HUD that is not the code in the repo."""
    state = check(binary, hud_dir)
    if state.state == FRESH:
        return Outcome(True, "HUD build is current", state)
    why = ("is not built" if state.state == MISSING else
           f"is older than hud/{state.newest_source} by {int(state.behind_by // 60)} min — it would show an OLD version")
    if not autobuild:
        return Outcome(False, f"WARN: the HUD {why}; not starting it. Build it: {BUILD_COMMAND_TEXT}", state)
    log(f"[launcher] the HUD {why}; rebuilding it (up to {int(timeout)} s): {' '.join(BUILD_COMMAND)}")
    rc, tail, timed_out = runner(BUILD_COMMAND, hud_dir, timeout, env if env is not None else build_env())
    if timed_out:
        return Outcome(False, f"WARN: the HUD rebuild did not finish within {int(timeout)} s; not starting the HUD. "
                              f"Run it yourself: {BUILD_COMMAND_TEXT}", state)
    after = check(binary, hud_dir)
    if rc != 0 or after.state != FRESH:
        reason = f"exit {rc}" if rc != 0 else "the binary is still older than the sources"
        return Outcome(False, f"WARN: the HUD rebuild failed ({reason}); not starting the HUD. Run: {BUILD_COMMAND_TEXT}", after)
    info = {"commit": git_short(root), "built_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now()))}
    try:
        sidecar(binary).write_text(json.dumps(info))
    except OSError:
        pass
    after.build_info = info
    return Outcome(True, f"HUD rebuilt (commit {info['commit']})", after)


# ------------------------------------------------------------------ other HUD copies

def other_copies(binary: Path, hud_dir: Path, home: Optional[Path] = None,
                 extra_dirs: Optional[Sequence[Path]] = None) -> List[Path]:
    """HUD executables / apps that are NOT the repo release binary and that something could launch instead (the debug build,
    a bundled .app, an installed copy)."""
    home = home or Path.home()
    found: List[Path] = []
    cands = [hud_dir / "src-tauri" / "target" / "debug" / "hud"]
    bundle = hud_dir / "src-tauri" / "target" / "release" / "bundle"
    if bundle.exists():
        cands += list(bundle.glob("**/*.app"))
    for d in [Path("/Applications"), home / "Applications", *(extra_dirs or [])]:
        if d.exists():
            cands += [p for p in d.glob("*.app") if "vesper" in p.name.lower() or "hud" in p.name.lower()]
    for p in cands:
        if p.exists() and p.resolve() != binary.resolve() and p not in found:
            found.append(p)
    return found


def stale_copies(copies: Sequence[Path], hud_dir: Path) -> List[Tuple[Path, float]]:
    _, src_m = newest_source(hud_dir)
    if src_m is None:
        return []
    return [(p, p.stat().st_mtime) for p in copies if p.stat().st_mtime + 1.0 < src_m]
