"""The HUD binary embeds the frontend at compile time: stale/missing builds are rebuilt by `vesper up` (or the HUD is refused,
loudly), `vesper doctor` reports the same, other HUD copies are never used. Fake mtimes and fake build runners only — no npm, no
cargo, no real HUD."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pytest

from launcher import doctor, hudbuild, stack

T0 = 1_800_000_000.0


def touch(p: Path, t: float, text: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    os.utime(p, (t, t))
    return p


def repo(tmp_path: Path, src_t: float, bin_t=None) -> Path:
    touch(tmp_path / "hud" / "src" / "render.ts", src_t - 100)
    touch(tmp_path / "hud" / "src" / "styles.css", src_t)
    touch(tmp_path / "hud" / "index.html", src_t - 500)
    if bin_t is not None:
        touch(tmp_path / "hud" / "src-tauri" / "target" / "release" / "hud", bin_t)
    return tmp_path


def binary(root: Path) -> Path:
    return root / "hud" / "src-tauri" / "target" / "release" / "hud"


class Runner:
    """Fake build: records calls; optionally 'builds' by touching the binary newer than every source."""

    def __init__(self, result="ok", rc=0):
        self.calls: List[tuple] = []
        self.result, self.rc = result, rc

    def __call__(self, cmd, cwd, timeout, env):
        self.calls.append((tuple(cmd), Path(cwd), timeout))
        if self.result == "timeout":
            return 124, "", True
        if self.result == "ok":
            touch(binary(cwd.parent), T0 + 10_000)
        return self.rc, "tail", False


def ensure(root, runner, **kw):
    logs: List[str] = []
    out = hudbuild.ensure_fresh(binary(root), root / "hud", root, runner=runner, log=logs.append, now=lambda: T0 + 20_000, **kw)
    return out, logs


# ---------------------------------------------------------------- check()

def test_fresh_when_the_binary_is_newer_than_every_source(tmp_path):
    root = repo(tmp_path, T0, bin_t=T0 + 50)
    st = hudbuild.check(binary(root), root / "hud")
    assert st.state == hudbuild.FRESH and st.newest_source == "styles.css"


def test_stale_when_a_source_is_newer_and_it_names_the_file_and_the_gap(tmp_path):
    root = repo(tmp_path, T0 + 3600, bin_t=T0)
    st = hudbuild.check(binary(root), root / "hud")
    assert st.state == hudbuild.STALE and st.newest_source == "styles.css" and st.behind_by == pytest.approx(3600)


def test_missing_build(tmp_path):
    root = repo(tmp_path, T0)
    assert hudbuild.check(binary(root), root / "hud").state == hudbuild.MISSING


def test_the_tauri_shell_and_config_count_as_sources(tmp_path):
    root = repo(tmp_path, T0, bin_t=T0 + 50)
    touch(root / "hud" / "src-tauri" / "src" / "main.rs", T0 + 500)
    assert hudbuild.check(binary(root), root / "hud").state == hudbuild.STALE


def test_dist_alone_does_not_make_the_binary_fresh(tmp_path):
    """The reported bug: `npm run build` refreshed hud/dist, but the binary embeds the OLD dist."""
    root = repo(tmp_path, T0 + 100, bin_t=T0)
    touch(root / "hud" / "dist" / "index.html", T0 + 5000)
    assert hudbuild.check(binary(root), root / "hud").state == hudbuild.STALE


def test_a_second_of_clock_slack(tmp_path):
    root = repo(tmp_path, T0 + 0.5, bin_t=T0)
    assert hudbuild.check(binary(root), root / "hud").state == hudbuild.FRESH


# ---------------------------------------------------------------- ensure_fresh()

def test_fresh_build_runs_nothing(tmp_path):
    root = repo(tmp_path, T0, bin_t=T0 + 50)
    r = Runner()
    out, _ = ensure(root, r)
    assert out.ok and r.calls == []


def test_stale_build_is_rebuilt_with_a_timeout_and_a_build_record(tmp_path):
    root = repo(tmp_path, T0 + 3600, bin_t=T0)
    r = Runner()
    out, logs = ensure(root, r, timeout=321.0)
    assert out.ok and len(r.calls) == 1
    cmd, cwd, timeout = r.calls[0]
    assert cmd == hudbuild.BUILD_COMMAND and cwd == root / "hud" and timeout == 321.0
    assert any("rebuilding it (up to 321 s)" in l and "OLD version" in l for l in logs)
    info = json.loads(hudbuild.sidecar(binary(root)).read_text())
    assert set(info) == {"commit", "built_at"} and info["built_at"].startswith("2027-")
    assert hudbuild.check(binary(root), root / "hud").build_info == info


def test_missing_build_is_built(tmp_path):
    root = repo(tmp_path, T0)
    r = Runner()
    out, logs = ensure(root, r)
    assert out.ok and len(r.calls) == 1 and "is not built" in logs[0]


def test_build_command_failing_refuses_to_start_and_names_the_command(tmp_path):
    root = repo(tmp_path, T0 + 3600, bin_t=T0)
    out, _ = ensure(root, Runner(result="fail", rc=1))
    assert not out.ok and "rebuild failed (exit 1)" in out.message
    assert out.message.startswith("WARN") and hudbuild.BUILD_COMMAND_TEXT in out.message and "not starting the HUD" in out.message


def test_build_that_exits_zero_but_leaves_the_binary_stale_is_a_failure(tmp_path):
    root = repo(tmp_path, T0 + 3600, bin_t=T0)
    out, _ = ensure(root, Runner(result="noop", rc=0))
    assert not out.ok and "still older than the sources" in out.message


def test_build_timeout_refuses_to_start(tmp_path):
    root = repo(tmp_path, T0 + 3600, bin_t=T0)
    out, _ = ensure(root, Runner(result="timeout"), timeout=60.0)
    assert not out.ok and "did not finish within 60 s" in out.message and hudbuild.BUILD_COMMAND_TEXT in out.message


def test_autobuild_off_refuses_without_running_anything(tmp_path):
    root = repo(tmp_path, T0 + 3600, bin_t=T0)
    r = Runner()
    out, _ = ensure(root, r, autobuild=False)
    assert not out.ok and r.calls == [] and out.message.startswith("WARN") and hudbuild.BUILD_COMMAND_TEXT in out.message


def test_the_real_runner_refuses_to_build_under_pytest():
    rc, tail, timed_out = hudbuild.default_runner(["npm", "run", "x"], Path("."), 1.0, {})
    assert rc == 125 and "pytest" in tail and not timed_out


# ---------------------------------------------------------------- launcher integration

def cfg(**launcher):
    return {"launcher": {"components": {"hud": True}, **launcher}}


def test_prepare_hud_returns_a_skip_reason_when_the_rebuild_fails(tmp_path):
    root = repo(tmp_path, T0 + 3600, bin_t=T0)
    logs: List[str] = []
    reason = stack.prepare_hud(cfg(), root, logs.append, Runner(result="fail", rc=2))
    assert reason and "rebuild failed" in reason and any("WARN" in l for l in logs)
    specs = {s.name: s for s in stack.build_specs(cfg(), "t", python="/py", root=root, hud_skip=reason)}
    assert specs["hud"].skip_reason == reason


def test_prepare_hud_is_silent_and_ok_when_fresh(tmp_path):
    root = repo(tmp_path, T0, bin_t=T0 + 50)
    logs: List[str] = []
    assert stack.prepare_hud(cfg(), root, logs.append, Runner()) is None
    assert len(logs) == 1 and logs[0].startswith("[launcher] HUD build:")


def test_prepare_hud_does_nothing_when_the_hud_is_disabled(tmp_path):
    root = repo(tmp_path, T0 + 3600, bin_t=T0)
    r = Runner()
    assert stack.prepare_hud({"launcher": {"components": {"hud": False}}}, root, lambda m: None, r) is None and r.calls == []


def test_the_launcher_only_ever_runs_a_build_inside_this_repos_hud_dir(tmp_path):
    root = repo(tmp_path, T0, bin_t=T0 + 50)
    elsewhere = touch(tmp_path / "Applications" / "Vesper HUD.app" / "Contents" / "MacOS" / "hud", T0 - 99999)
    assert stack.resolve_hud_binary(cfg(hud_binary=str(elsewhere)), root) == binary(root)
    assert "ignored" in stack.hud_override_ignored(cfg(hud_binary=str(elsewhere)), root)
    inside = root / "hud" / "src-tauri" / "target" / "debug" / "hud"
    assert stack.resolve_hud_binary(cfg(hud_binary=str(inside)), root) == inside
    assert stack.hud_override_ignored(cfg(), root) is None
    logs: List[str] = []
    stack.prepare_hud(cfg(hud_binary=str(elsewhere)), root, logs.append, Runner())
    assert any("ignored" in l and "WARN" in l for l in logs)


def test_stale_second_copies_are_found_and_warned_about(tmp_path):
    root = repo(tmp_path, T0 + 3600, bin_t=T0 + 4000)
    debug = touch(root / "hud" / "src-tauri" / "target" / "debug" / "hud", T0 - 5000)
    app = touch(tmp_path / "home" / "Applications" / "Vesper HUD.app" / "x", T0 + 9000)
    (tmp_path / "home" / "Applications" / "Vesper HUD.app").touch()
    copies = hudbuild.other_copies(binary(root), root / "hud", home=tmp_path / "home")
    assert debug in copies and binary(root) not in copies
    stale = [p for p, _ in hudbuild.stale_copies(copies, root / "hud")]
    assert debug in stale
    logs: List[str] = []
    stack.prepare_hud(cfg(), root, logs.append, Runner())
    assert any("another HUD copy exists (hud" in l and "ignores it" in l for l in logs)


# ---------------------------------------------------------------- doctor

def labels(checks):
    return {c.label: c for c in checks}


def test_doctor_fresh_reports_the_commit_and_timestamp(tmp_path):
    root = repo(tmp_path, T0, bin_t=T0 + 50)
    hudbuild.sidecar(binary(root)).write_text(json.dumps({"commit": "abc1234", "built_at": "2027-01-15 08:00:00"}))
    [c] = doctor.check_hud_build(cfg(), root)
    assert c.status == doctor.PASS and "commit abc1234" in c.detail and "built 20" in c.detail


def test_doctor_stale_warns_with_the_command(tmp_path):
    root = repo(tmp_path, T0 + 3600, bin_t=T0)
    [c] = doctor.check_hud_build(cfg(), root)
    assert c.status == doctor.WARN and "OLD version" in c.detail and "styles.css" in c.detail and "tauri build" in c.fix


def test_doctor_missing_fails_with_the_command(tmp_path):
    root = repo(tmp_path, T0)
    [c] = doctor.check_hud_build(cfg(), root)
    assert c.status == doctor.FAIL and "npm run tauri build" in c.fix


def test_doctor_reports_stale_copies_and_an_ignored_override(tmp_path):
    root = repo(tmp_path, T0 + 3600, bin_t=T0 + 4000)
    touch(root / "hud" / "src-tauri" / "target" / "debug" / "hud", T0 - 5000)
    checks = doctor.check_hud_build(cfg(hud_binary="/somewhere/else/hud"), root, home=tmp_path / "nohome")
    assert [c.label for c in checks] == ["HUD build", "HUD build: config", "HUD build: other copy"]
    assert all(c.status in (doctor.PASS, doctor.WARN) for c in checks)
