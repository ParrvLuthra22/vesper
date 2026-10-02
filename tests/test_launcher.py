"""
Launcher tests (launcher/): startup ordering, readiness, restart-with-backoff,
crash-storm give-up, clean shutdown order, state files, CLI. All with fake
processes and a virtual clock — no real subprocess is started except one
throwaway `sleep` in the `vesper down` test.

The real voice round trip (mic -> wake word -> STT -> planner -> TTS -> speaker)
is manual; see the checklist in docs/RESOURCES.md / the PR description.
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from launcher import cli as launcher_cli
from launcher import stack
from launcher.state import RunState, pid_alive
from launcher.supervisor import (
    ComponentSpec,
    ProcessHandle,
    ProbeContext,
    RestartPolicy,
    State,
    Supervisor,
    line_probe,
)


# ----------------------------------------------------------------------------
# Fake world: virtual time + scripted processes
# ----------------------------------------------------------------------------

class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: List[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)  # let other tasks run

    async def advance(self, seconds: float) -> None:
        """Virtual time passing for a fake process's lifetime — not a supervisor sleep."""
        self.now += seconds
        await asyncio.sleep(0)


class FakeProc(ProcessHandle):
    """A process whose lifetime is measured in VIRTUAL time: it exits `lifetime`
    seconds after spawn, and `wait()` is what lets virtual time pass."""

    def __init__(self, pid: int, name: str, world: "World", script: Dict[str, Any]):
        self.pid = pid
        self.name = name
        self.recent_lines = collections.deque(maxlen=50)
        self._world = world
        self._rc: Optional[int] = None
        self.ignore_sigterm = bool(script.get("ignore_sigterm"))
        for line in script.get("lines", []):
            self.recent_lines.append(line)
        self._die_at: Optional[float] = None
        self._die_rc = script.get("exit_rc")
        if self._die_rc is not None:
            self._die_at = world.clock.now + script.get("lifetime", 0.0)

    @property
    def returncode(self) -> Optional[int]:  # type: ignore[override]
        if self._rc is None and self._die_at is not None and self._world.clock.now >= self._die_at:
            self.finish(self._die_rc)
        return self._rc

    def finish(self, rc: int) -> None:
        if self._rc is None:
            self._rc = rc
            self._world.events.append(f"exit {self.name} {rc}")

    async def wait(self) -> int:
        while self.returncode is None:
            await self._world.clock.advance(0.1)
        return self._rc  # type: ignore[return-value]

    def terminate(self) -> None:
        self._world.events.append(f"term {self.name}")
        if not self.ignore_sigterm:
            self.finish(-signal.SIGTERM)

    def kill(self) -> None:
        self._world.events.append(f"kill {self.name}")
        self.finish(-signal.SIGKILL)


class World:
    """Per-component scripts: a list of dicts consumed one per spawn (last repeats)."""

    def __init__(self) -> None:
        self.clock = Clock()
        self.events: List[str] = []
        self.scripts: Dict[str, List[Dict[str, Any]]] = {}
        self.spawns: Dict[str, int] = collections.defaultdict(int)
        self.procs: List[FakeProc] = []
        self._pid = 1000

    def script(self, name: str, *steps: Dict[str, Any]) -> None:
        self.scripts[name] = list(steps)

    async def spawn(self, spec: ComponentSpec) -> FakeProc:
        steps = self.scripts.get(spec.name) or [{}]
        script = steps[min(self.spawns[spec.name], len(steps) - 1)]
        self.spawns[spec.name] += 1
        self._pid += 1
        self.events.append(f"spawn {spec.name}")
        proc = FakeProc(self._pid, spec.name, self, {"lines": ["ready"], **script})
        self.procs.append(proc)
        return proc

    def supervisor(self, names: List[str], *, policy: Optional[RestartPolicy] = None,
                   specs: Optional[List[ComponentSpec]] = None, grace: float = 2.0, **kw: Any) -> Supervisor:
        if specs is None:
            specs = [ComponentSpec(name=n, argv=[n], probe=line_probe("ready"), ready_timeout=5.0) for n in names]
        self.logs: List[str] = []
        return Supervisor(
            specs, policy or RestartPolicy(), spawn=self.spawn, sleep=self.clock.sleep,
            monotonic=self.clock.monotonic, log=self.logs.append, shutdown_grace=grace, **kw,
        )


def test_describe_exit_names_signals():
    from launcher.supervisor import describe_exit

    assert describe_exit(-6) == "signal SIGABRT (-6)"
    assert describe_exit(-9) == "signal SIGKILL (-9)"
    assert describe_exit(1) == "exit 1"
    assert describe_exit(None) == "unknown"


def _backoffs(world: World) -> List[float]:
    """The restart delays (poll sleeps are 0.25)."""
    return [s for s in world.clock.sleeps if s != 0.25]


# ----------------------------------------------------------------------------
# Ordering + readiness
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_components_start_in_order_each_after_the_previous_is_ready():
    world = World()
    names = ["gateway", "voice_output", "hud", "voice_input"]
    # the gateway only becomes ready after a few polls: nothing may start before that
    ready_after = {"polls": 0}

    async def gateway_probe(ctx: ProbeContext) -> bool:
        ready_after["polls"] += 1
        if ready_after["polls"] >= 4:
            world.events.append("ready gateway")
            return True
        return False

    specs = [ComponentSpec(name="gateway", argv=["g"], probe=gateway_probe, ready_timeout=30.0, required=True)] + [
        ComponentSpec(name=n, argv=[n], probe=line_probe("ready"), ready_timeout=5.0) for n in names[1:]
    ]
    sup = world.supervisor(names, specs=specs)
    assert await sup.start() is True
    spawn_order = [e.split()[1] for e in world.events if e.startswith("spawn")]
    assert spawn_order == names
    assert world.events.index("ready gateway") < world.events.index("spawn voice_output")
    assert all(c.state == State.READY for c in sup.components)
    await sup.stop()


@pytest.mark.asyncio
async def test_not_ready_until_the_probe_passes():
    world = World()
    world.script("voice_input", {"lines": ["loading models..."]})  # never prints "ready"
    spec = ComponentSpec(name="voice_input", argv=["x"], probe=line_probe("ready"), ready_timeout=3.0)
    sup = world.supervisor([], specs=[spec], policy=RestartPolicy(max_startup_failures=1))
    assert await sup.start() is True            # optional component: stack continues
    comp = sup.component("voice_input")
    assert comp.state == State.FAILED
    assert any("not ready after" in line for line in world.logs)
    assert ("kill voice_input" in world.events) or ("term voice_input" in world.events)  # it was stopped
    await sup.stop()


@pytest.mark.asyncio
async def test_skipped_component_is_reported_not_spawned():
    world = World()
    specs = [
        ComponentSpec(name="gateway", argv=["g"], required=True, probe=line_probe("ready")),
        ComponentSpec(name="hud", argv=["h"], skip_reason="HUD is not built (x)"),
    ]
    sup = world.supervisor([], specs=specs)
    assert await sup.start() is True
    assert sup.component("hud").state == State.SKIPPED
    assert world.spawns["hud"] == 0
    assert any("hud: SKIPPED" in line and "not built" in line for line in world.logs)
    await sup.stop()


# ----------------------------------------------------------------------------
# Restart behaviour
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_crash_is_restarted_with_exponential_backoff():
    world = World()
    world.script("voice_output",
                 {"exit_rc": 137, "lifetime": 1.0}, {"exit_rc": 137, "lifetime": 1.0}, {})  # 2 crashes then stable
    sup = world.supervisor(["gateway", "voice_output"])
    assert await sup.start() is True
    for _ in range(200):
        if world.spawns["voice_output"] >= 3 and sup.component("voice_output").state == State.READY:
            break
        await asyncio.sleep(0)
    comp = sup.component("voice_output")
    assert world.spawns["voice_output"] == 3
    assert comp.restarts == 2 and comp.state == State.READY
    assert _backoffs(world)[:2] == [1.0, 2.0]
    assert world.spawns["gateway"] == 1            # the others were not touched
    await sup.stop()


@pytest.mark.asyncio
async def test_backoff_is_capped_and_crash_storm_gives_up():
    world = World()
    world.script("hud", {"exit_rc": 1, "lifetime": 0.5})        # crashes forever, after becoming ready
    policy = RestartPolicy(initial_backoff=1.0, max_backoff=4.0, max_crashes=5, window_seconds=1000.0)
    sup = world.supervisor(["hud"], policy=policy)
    await sup.start()
    comp = sup.component("hud")
    for _ in range(500):
        if comp.state == State.FAILED:
            break
        await asyncio.sleep(0)
    assert comp.state == State.FAILED
    assert "crashed 5×" in comp.detail and "exit 1" in comp.detail
    assert _backoffs(world) == [1.0, 2.0, 4.0, 4.0]             # doubled, then capped
    assert world.spawns["hud"] == 5                             # and stopped retrying
    await sup.stop()


@pytest.mark.asyncio
async def test_repeated_startup_failure_gives_up_without_a_restart_loop():
    world = World()
    world.script("voice_output", {"exit_rc": 2, "lines": ["Traceback...", "ImportError: kokoro_onnx"]})
    sup = world.supervisor(["gateway", "voice_output"], policy=RestartPolicy(max_startup_failures=3))
    assert await sup.start() is True                            # optional => continue, loudly
    comp = sup.component("voice_output")
    assert comp.state == State.FAILED and world.spawns["voice_output"] == 3
    assert "ImportError: kokoro_onnx" in comp.detail
    assert any("Continuing WITHOUT voice_output" in line for line in world.logs)
    await sup.stop()


@pytest.mark.asyncio
async def test_clean_exit_zero_is_not_restarted():
    world = World()
    world.script("voice_output", {"lines": ["ready"], "exit_rc": 0, "lifetime": 1.0})
    sup = world.supervisor(["voice_output"])
    await sup.start()
    comp = sup.component("voice_output")
    for _ in range(100):
        if comp.state == State.EXITED:
            break
        await asyncio.sleep(0)
    assert comp.state == State.EXITED and world.spawns["voice_output"] == 1
    await sup.stop()


@pytest.mark.asyncio
async def test_permanent_exit_code_is_failed_loudly_and_never_retried():
    world = World()
    world.script("voice_input", {"exit_rc": 69, "lines": ["Voice input disabled for this session — mic busy"]})
    spec = ComponentSpec(name="voice_input", argv=["v"], probe=line_probe("ready"),
                         permanent_exit_codes=frozenset({69}), hint="Check mic permission.")
    sup = world.supervisor([], specs=[spec])
    assert await sup.start() is True
    comp = sup.component("voice_input")
    assert comp.state == State.FAILED and world.spawns["voice_input"] == 1
    assert "mic busy" in comp.detail
    assert any("FAILED" in line and "Check mic permission" in line for line in world.logs)
    await sup.stop()


@pytest.mark.asyncio
async def test_healthy_stretch_resets_the_backoff():
    world = World()
    # runs 100s (> healthy_reset), crashes, runs 100s, crashes ... then stable
    world.script("hud", {"exit_rc": 1, "lifetime": 100.0}, {"exit_rc": 1, "lifetime": 100.0}, {})
    policy = RestartPolicy(healthy_reset_seconds=60.0, max_crashes=2, window_seconds=10_000.0)
    sup = world.supervisor(["hud"], policy=policy)
    await sup.start()
    comp = sup.component("hud")
    for _ in range(20000):                      # 100 virtual seconds at 0.1s per tick, twice
        if world.spawns["hud"] >= 3 and comp.state == State.READY:
            break
        await asyncio.sleep(0)
    # without the reset, max_crashes=2 would have declared it FAILED on the 2nd crash
    assert comp.state == State.READY and world.spawns["hud"] == 3
    assert _backoffs(world)[:2] == [1.0, 1.0]
    await sup.stop()


@pytest.mark.asyncio
async def test_required_component_failing_at_startup_aborts_the_stack():
    world = World()
    world.script("gateway", {"exit_rc": 1, "lines": ["Address already in use"]})
    specs = [
        ComponentSpec(name="gateway", argv=["g"], required=True, probe=line_probe("ready")),
        ComponentSpec(name="voice_output", argv=["v"], probe=line_probe("ready")),
    ]
    sup = world.supervisor([], specs=specs, policy=RestartPolicy(max_startup_failures=2))
    assert await sup.start() is False
    assert world.spawns["voice_output"] == 0            # never reached
    assert "Address already in use" in (sup.fatal_reason or "")
    assert any("FATAL" in line for line in world.logs)


@pytest.mark.asyncio
async def test_required_component_failing_later_wakes_wait_closed():
    world = World()
    world.script("gateway", {"exit_rc": 1, "lifetime": 5.0})     # becomes ready, then crashes forever
    specs = [ComponentSpec(name="gateway", argv=["g"], required=True, probe=line_probe("ready"))]
    sup = world.supervisor([], specs=specs, policy=RestartPolicy(max_crashes=2, window_seconds=1000.0))
    assert await sup.start() is True
    await asyncio.wait_for(sup.wait_closed(), timeout=5)
    assert sup.component("gateway").state == State.FAILED
    assert "gateway failed" in (sup.fatal_reason or "")
    await sup.stop()


# ----------------------------------------------------------------------------
# Shutdown
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_terminates_in_reverse_order_and_escalates_to_kill():
    world = World()
    world.script("hud", {"ignore_sigterm": True})                # a stuck component
    sup = world.supervisor(["gateway", "voice_output", "hud", "voice_input"], grace=1.0)
    assert await sup.start() is True
    world.events.clear()
    await sup.stop()
    stops = [e for e in world.events if e.startswith(("term", "kill"))]
    assert stops == ["term voice_input", "term hud", "kill hud", "term voice_output", "term gateway"]
    assert all(c.state == State.STOPPED for c in sup.components)
    assert any("ignored SIGTERM" in line and "hud" in line for line in world.logs)


@pytest.mark.asyncio
async def test_stop_during_startup_aborts_and_stops_what_started():
    world = World()
    specs = [
        ComponentSpec(name="gateway", argv=["g"], probe=line_probe("ready"), required=True),
        ComponentSpec(name="voice_output", argv=["v"], probe=line_probe("never"), ready_timeout=1000.0),
        ComponentSpec(name="hud", argv=["h"], probe=line_probe("ready")),
    ]
    sup = world.supervisor([], specs=specs)
    starter = asyncio.ensure_future(sup.start())
    for _ in range(100):
        if world.spawns["voice_output"]:
            break
        await asyncio.sleep(0)
    sup.request_stop()                                           # Ctrl+C mid-startup
    assert await asyncio.wait_for(starter, timeout=5) is False
    assert world.spawns["hud"] == 0
    assert all(p.returncode is not None for p in world.procs)


# ----------------------------------------------------------------------------
# Probes
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_line_probe_matches_output():
    proc = FakeProc(1, "x", World(), {"lines": ["booting", "voice output connected to gateway"]})
    probe = line_probe(r"connected to gateway")
    assert await probe(ProbeContext(handle=proc)) is True
    assert await line_probe("nope")(ProbeContext(handle=proc)) is False


class _StatusHandler(BaseHTTPRequestHandler):
    clients = 1

    def do_GET(self):  # noqa: N802
        if self.headers.get("Authorization") != "Bearer secret-token":
            self.send_response(401); self.end_headers(); return
        body = json.dumps({"agents": [], "providers": [], "clients": type(self).clients}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence
        pass


@pytest.fixture
def fake_gateway():
    server = HTTPServer(("127.0.0.1", 0), _StatusHandler)
    _StatusHandler.clients = 1
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield {"gateway": {"host": "127.0.0.1", "port": server.server_address[1]}}
    server.shutdown()


@pytest.mark.asyncio
async def test_gateway_probe_requires_the_right_token(fake_gateway):
    ctx = ProbeContext(handle=FakeProc(1, "g", World(), {}))
    assert await stack.gateway_probe(fake_gateway, "secret-token")(ctx) is True
    assert await stack.gateway_probe(fake_gateway, "wrong")(ctx) is False


@pytest.mark.asyncio
async def test_hud_probe_waits_for_the_client_count_to_rise(fake_gateway):
    probe = stack.hud_attached_probe(fake_gateway, "secret-token")
    ctx = ProbeContext(handle=FakeProc(1, "hud", World(), {}))
    assert await probe(ctx) is False            # baseline recorded (voice output is already attached)
    assert await probe(ctx) is False
    _StatusHandler.clients = 2                  # the HUD connects
    assert await probe(ctx) is True


# ----------------------------------------------------------------------------
# Component list from config
# ----------------------------------------------------------------------------

def _cfg(**over: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "gateway": {"host": "127.0.0.1", "port": 8760},
        "voice": {"input": {"enabled": True}, "output": {"enabled": True}},
        "launcher": {"components": {"voice_output": True, "voice_input": True, "hud": True}, "offline_models": True},
    }
    for k, v in over.items():
        cfg[k] = v
    return cfg


def test_build_specs_order_env_and_requirements(tmp_path):
    hud = tmp_path / "hud"
    hud.write_text("#!/bin/sh\n")
    cfg = _cfg(launcher={"hud_binary": str(hud), "components": {"voice_output": True, "voice_input": True, "hud": True}})
    specs = stack.build_specs(cfg, "tok123", python="/py", root=tmp_path)
    assert [s.name for s in specs] == ["gateway", "voice_output", "hud", "voice_input"]
    assert specs[0].required and not any(s.required for s in specs[1:])
    assert specs[0].argv == ["/py", "-m", "gateway.server"]
    assert specs[3].argv == ["/py", "-m", "voice.input"]
    assert all(s.env["VESPER_GATEWAY_TOKEN"] == "tok123" for s in specs)
    assert all(s.env["HF_HUB_OFFLINE"] == "1" for s in specs)
    assert specs[3].permanent_exit_codes == frozenset({69})
    assert not any(s.skip_reason for s in specs)


def test_build_specs_skips_are_explicit(tmp_path):
    cfg = _cfg(
        voice={"input": {"enabled": False}, "output": {"enabled": True}},
        launcher={"hud_binary": str(tmp_path / "missing"), "components": {"voice_output": False, "voice_input": True, "hud": True},
                  "offline_models": False},
    )
    specs = {s.name: s for s in stack.build_specs(cfg, "t", python="/py", root=tmp_path)}
    assert "launcher.components.voice_output" in specs["voice_output"].skip_reason
    assert "not built" in specs["hud"].skip_reason and "npm run tauri build" in specs["hud"].skip_reason
    assert "voice.input.enabled is false" in specs["voice_input"].skip_reason
    assert "HF_HUB_OFFLINE" not in specs["gateway"].env


def test_voice_input_unavailable_exit_code_is_shared():
    from voice.input.service import EXIT_UNAVAILABLE

    assert EXIT_UNAVAILABLE == stack.VOICE_INPUT_UNAVAILABLE == 69


def test_policy_from_config_reads_the_launcher_block():
    from config.settings import load_config_dict

    policy = stack.policy_from_config(load_config_dict())
    assert (policy.initial_backoff, policy.max_backoff, policy.max_crashes) == (1.0, 30.0, 5)


# ----------------------------------------------------------------------------
# State files
# ----------------------------------------------------------------------------

def test_run_state_roundtrip_and_token_permissions(tmp_path):
    state = RunState(tmp_path / "run")
    assert state.read() is None and state.read_token() is None
    state.write({"supervisor_pid": os.getpid(), "components": []})
    state.write_token("s3cret")
    assert state.read()["supervisor_pid"] == os.getpid()
    assert state.read_token() == "s3cret"
    assert stat.S_IMODE(state.token_path.stat().st_mode) == 0o600
    assert state.running_supervisor_pid() == os.getpid()
    state.clear()
    assert state.read() is None and state.read_token() is None


def test_stale_state_file_is_ignored(tmp_path):
    state = RunState(tmp_path)
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    state.write({"supervisor_pid": proc.pid})
    assert pid_alive(proc.pid) is False
    assert state.running_supervisor_pid() is None


# ----------------------------------------------------------------------------
# run_up (end to end with fake processes)
# ----------------------------------------------------------------------------

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.asyncio
async def test_run_up_refuses_when_the_gateway_port_is_taken(tmp_path):
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    cfg = _cfg(gateway={"host": "127.0.0.1", "port": blocker.getsockname()[1]})
    logs: List[str] = []
    spawned: List[str] = []

    async def spawn(spec):  # pragma: no cover - must not be reached
        spawned.append(spec.name)
        raise AssertionError("spawned despite the port being in use")

    try:
        rc = await stack.run_up(cfg, run_state=RunState(tmp_path), spawn=spawn, log=logs.append,
                                install_signal_handlers=False)
    finally:
        blocker.close()
    assert rc == 1 and spawned == []
    assert any("already in use" in line for line in logs)


@pytest.mark.asyncio
async def test_run_up_refuses_a_second_instance(tmp_path):
    state = RunState(tmp_path)
    state.write({"supervisor_pid": os.getpid()})                 # "running"
    logs: List[str] = []
    rc = await stack.run_up(_cfg(gateway={"host": "127.0.0.1", "port": _free_port()}),
                            run_state=state, log=logs.append, install_signal_handlers=False)
    assert rc == 1 and any("already running" in line for line in logs)


@pytest.mark.asyncio
async def test_run_up_runs_until_sigterm_then_stops_everything_and_cleans_state(tmp_path, monkeypatch):
    world = World()
    cfg = _cfg(gateway={"host": "127.0.0.1", "port": _free_port()},
               launcher={"hud_binary": str(tmp_path / "nohud"),
                         "components": {"voice_output": True, "voice_input": True, "hud": True}})
    # swap the real readiness probes for ones that read the fake processes' output
    real_build = stack.build_specs

    def fake_build(cfg_, token, *a, **kw):
        out = real_build(cfg_, token, *a, **kw)
        for s in out:
            s.probe = line_probe("ready")
        return out

    monkeypatch.setattr(stack, "build_specs", fake_build)
    state = RunState(tmp_path / "run")
    logs: List[str] = []
    asyncio.get_running_loop().call_later(0.3, os.kill, os.getpid(), signal.SIGTERM)

    async def spawn(spec):
        return await world.spawn(spec)

    rc = await asyncio.wait_for(
        stack.run_up(cfg, run_state=state, spawn=spawn, log=logs.append, install_signal_handlers=True), timeout=20)
    assert rc == 0
    started = [e.split()[1] for e in world.events if e.startswith("spawn")]
    assert started == ["gateway", "voice_output", "voice_input"]          # hud skipped: not built
    assert any("hud" in l and "SKIPPED" in l for l in logs)
    assert all(p.returncode is not None for p in world.procs)             # everything stopped
    assert state.read() is None and state.read_token() is None            # run files removed


@pytest.mark.asyncio
async def test_strict_mode_fails_when_an_optional_component_is_skipped(tmp_path, monkeypatch):
    world = World()
    cfg = _cfg(gateway={"host": "127.0.0.1", "port": _free_port()},
               launcher={"hud_binary": str(tmp_path / "nohud"), "components": {"hud": True, "voice_output": True, "voice_input": True}})
    real_build = stack.build_specs
    monkeypatch.setattr(stack, "build_specs", lambda c, t, *a, **k: [
        setattr(s, "probe", line_probe("ready")) or s for s in real_build(c, t, *a, **k)])
    logs: List[str] = []

    async def spawn(spec):
        return await world.spawn(spec)

    rc = await stack.run_up(cfg, run_state=RunState(tmp_path / "r"), spawn=spawn, strict=True,
                            log=logs.append, install_signal_handlers=False)
    assert rc == 1 and any("--strict" in l for l in logs)
    assert all(p.returncode is not None for p in world.procs)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def test_cli_parses_subcommands():
    parser = launcher_cli.build_parser()
    assert parser.parse_args(["up"]).command == "up"
    assert parser.parse_args(["up", "--strict"]).strict is True
    assert parser.parse_args(["down", "--timeout", "5"]).timeout == 5
    assert parser.parse_args(["status", "--json"]).json is True
    assert parser.parse_args(["logs", "hud", "-n", "5"]).component == "hud"
    with pytest.raises(SystemExit):
        parser.parse_args(["frobnicate"])


def test_status_and_down_when_nothing_is_running(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launcher_cli, "PROJECT_ROOT", tmp_path)
    assert launcher_cli.main(["status"]) == launcher_cli.EXIT_NOT_RUNNING
    assert "not running" in capsys.readouterr().out
    assert launcher_cli.main(["down"]) == 0


def test_down_signals_the_supervisor_and_waits(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launcher_cli, "PROJECT_ROOT", tmp_path)
    victim = subprocess.Popen(["sleep", "60"])
    try:
        RunState(tmp_path / "data" / "run").write({"supervisor_pid": victim.pid, "components": []})
        reaper = threading.Thread(target=victim.wait, daemon=True)   # reap so pid_alive turns False
        reaper.start()
        assert launcher_cli.main(["down", "--timeout", "10"]) == 0
        assert victim.poll() is not None
    finally:
        if victim.poll() is None:
            victim.kill()
    assert "Stopped." in capsys.readouterr().out


def test_status_prints_components(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launcher_cli, "PROJECT_ROOT", tmp_path)
    RunState(tmp_path / "data" / "run").write({
        "supervisor_pid": os.getpid(), "started_at": 0,
        "components": [
            {"name": "gateway", "state": "ready", "pid": 11, "restarts": 0,
             "ready_since": __import__("time").time() - 3700, "detail": ""},
            {"name": "voice_input", "state": "failed", "pid": None, "restarts": 0, "ready_since": None,
             "detail": "mic busy"},
        ],
    })
    assert launcher_cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "gateway" in out and "READY" in out and "FAILED" in out and "mic busy" in out
    assert "1h01m" in out                                   # uptime is computed at read time, not frozen
    assert "attention: voice_input (failed)" in out


def test_vesper_console_script_dispatches_launcher_commands(monkeypatch):
    import cli.app as app

    seen: Dict[str, Any] = {}
    monkeypatch.setattr("launcher.cli.main", lambda argv: seen.setdefault("argv", argv) and 0)
    monkeypatch.setattr(sys, "argv", ["vesper", "status", "--json"])
    with pytest.raises(SystemExit):
        app.run()
    assert seen["argv"] == ["status", "--json"]


@pytest.mark.asyncio
async def test_summary_does_not_promise_a_wake_word_when_voice_input_failed():
    world = World()
    world.script("voice_input", {"exit_rc": 69, "lines": ["Voice input unavailable: no audio"]})
    specs = [
        ComponentSpec(name="gateway", argv=["g"], required=True, probe=line_probe("ready")),
        ComponentSpec(name="voice_input", argv=["v"], probe=line_probe("ready"),
                      permanent_exit_codes=frozenset({69})),
    ]
    sup = world.supervisor([], specs=specs)
    assert await sup.start() is True
    logs: List[str] = []
    stack._print_summary(sup, logs.append)
    text = "\n".join(logs)
    assert "voice input is NOT running" in text and "say the wake word" not in text
    await sup.stop()

    world2 = World()
    sup2 = world2.supervisor(["gateway", "voice_input"])
    await sup2.start()
    logs2: List[str] = []
    stack._print_summary(sup2, logs2.append)
    assert "say the wake word" in "\n".join(logs2)
    await sup2.stop()
