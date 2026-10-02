"""
Process supervisor — starts Vesper's components in dependency order, waits for
each to be genuinely ready, restarts crashes with exponential backoff, and shuts
everything down in reverse order.

Process model
-------------
Each component is a separate OS process (the gateway/Brain, voice output, the
HUD, voice input). A component runs under its own `_supervise` task:

    spawn -> wait until READY (or it exits / times out) -> monitor -> on exit decide

Exit decisions:
    exit 0                          -> EXITED: the component turned itself off
                                       (e.g. voice.output.enabled=false). Not restarted.
    exit code in permanent_exit_codes
                                    -> FAILED: it said "unavailable for this session"
                                       (voice input with no usable mic/model). Not
                                       restarted — retrying a missing device is a loop.
    anything else (crash, signal)   -> BACKOFF then restart, 1s, 2s, 4s ... capped.
                                       Too many crashes in a window, or repeated
                                       failures before ever becoming ready -> FAILED.

A FAILED *required* component (the gateway) takes the whole stack down: without
the Brain there is nothing to supervise. A FAILED optional one is reported loudly
and the rest keep running — never a silent degradation.

Everything time- or process-related is injectable (spawn, sleep, monotonic) so the
ordering, backoff and shutdown logic is unit-tested with fake processes.
"""

from __future__ import annotations

import asyncio
import collections
import os
import re
import signal
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, FrozenSet, List, Optional, Sequence


def describe_exit(rc: Optional[int]) -> str:
    """`-6` -> "signal SIGABRT (-6)"; positive codes stay as "exit 1"."""
    if rc is None:
        return "unknown"
    if rc < 0:
        try:
            return f"signal {signal.Signals(-rc).name} ({rc})"
        except ValueError:
            return f"signal {-rc} ({rc})"
    return f"exit {rc}"


class State(str, Enum):
    PENDING = "pending"
    STARTING = "starting"
    READY = "ready"
    BACKOFF = "backoff"
    FAILED = "failed"
    EXITED = "exited"      # turned itself off with exit 0
    SKIPPED = "skipped"    # never started (disabled in config, binary missing, ...)
    STOPPED = "stopped"    # stopped by a shutdown


TERMINAL = frozenset({State.FAILED, State.EXITED, State.SKIPPED, State.STOPPED})


@dataclass
class RestartPolicy:
    initial_backoff: float = 1.0
    max_backoff: float = 30.0
    #: Crashes (after having been ready) within `window_seconds` before giving up.
    max_crashes: int = 5
    window_seconds: float = 120.0
    #: A component that stays ready this long gets a clean slate.
    healthy_reset_seconds: float = 60.0
    #: Consecutive failures BEFORE ever becoming ready before giving up.
    max_startup_failures: int = 3


@dataclass
class ProbeContext:
    """What a readiness probe may look at."""

    handle: "ProcessHandle"
    #: Free-form scratch space supplied by the spec's `probe_data`.
    data: Dict[str, Any] = field(default_factory=dict)


Probe = Callable[[ProbeContext], Awaitable[bool]]


@dataclass
class ComponentSpec:
    name: str
    argv: List[str]
    description: str = ""
    env: Dict[str, str] = field(default_factory=dict)
    #: A FAILED required component brings the whole stack down.
    required: bool = False
    probe: Optional[Probe] = None
    probe_data: Dict[str, Any] = field(default_factory=dict)
    ready_timeout: float = 60.0
    #: Exit codes that mean "unavailable this session" -> FAILED, never restarted.
    permanent_exit_codes: FrozenSet[int] = frozenset()
    #: Set to skip the component entirely; the reason is printed, never silent.
    skip_reason: Optional[str] = None
    #: Shown when the component fails, e.g. how to fix it.
    hint: str = ""


class ProcessHandle:
    """Minimal surface the supervisor needs from a running process."""

    pid: int
    returncode: Optional[int]
    recent_lines: Deque[str]

    async def wait(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def terminate(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def kill(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


Spawn = Callable[[ComponentSpec], Awaitable[ProcessHandle]]


# --------------------------------------------------------------------------- real processes

class AsyncProcess(ProcessHandle):
    """A real subprocess in its own session, output pumped to a log file."""

    def __init__(self, proc: "asyncio.subprocess.Process", log_path: Optional[Path]):
        self._proc = proc
        self.pid = proc.pid
        self.recent_lines = collections.deque(maxlen=200)
        self._log = open(log_path, "ab", buffering=0) if log_path else None
        self._pump = asyncio.ensure_future(self._pump_output())

    @property
    def returncode(self) -> Optional[int]:  # type: ignore[override]
        return self._proc.returncode

    async def _pump_output(self) -> None:
        assert self._proc.stdout is not None
        try:
            async for raw in self._proc.stdout:
                self.recent_lines.append(raw.decode("utf-8", errors="replace").rstrip("\n"))
                if self._log is not None:
                    self._log.write(raw)
        except Exception:
            pass
        finally:
            if self._log is not None:
                self._log.close()

    async def wait(self) -> int:
        rc = await self._proc.wait()
        try:  # let the pump drain the final lines (the failure reason lives there)
            await asyncio.wait_for(asyncio.shield(self._pump), timeout=1.0)
        except Exception:
            pass
        return rc

    def _signal(self, sig: int) -> None:
        try:
            os.killpg(self.pid, sig)  # the child leads its own session/group
        except (ProcessLookupError, PermissionError):
            pass

    def terminate(self) -> None:
        self._signal(signal.SIGTERM)

    def kill(self) -> None:
        self._signal(signal.SIGKILL)


def make_spawner(cwd: Path, log_dir: Optional[Path], base_env: Optional[Dict[str, str]] = None) -> Spawn:
    async def spawn(spec: ComponentSpec) -> ProcessHandle:
        env = dict(os.environ if base_env is None else base_env)
        env.update(spec.env)
        log_path = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{spec.name}.log"
            with open(log_path, "ab") as fh:
                fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start: {' '.join(spec.argv)} =====\n".encode())
        proc = await asyncio.create_subprocess_exec(
            *spec.argv,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,  # Ctrl+C reaches only the supervisor, which stops children in order
        )
        return AsyncProcess(proc, log_path)

    return spawn


# --------------------------------------------------------------------------- probes

def line_probe(pattern: str) -> Probe:
    """Ready once the process has printed a line matching `pattern`."""
    rx = re.compile(pattern)

    async def probe(ctx: ProbeContext) -> bool:
        return any(rx.search(line) for line in list(ctx.handle.recent_lines))

    return probe


def alive_probe(settle_seconds: float, monotonic: Callable[[], float] = time.monotonic) -> Probe:
    """Ready once the process has stayed alive for `settle_seconds` (no better signal)."""
    started: Dict[int, float] = {}

    async def probe(ctx: ProbeContext) -> bool:
        t0 = started.setdefault(ctx.handle.pid, monotonic())
        return ctx.handle.returncode is None and (monotonic() - t0) >= settle_seconds

    return probe


# --------------------------------------------------------------------------- supervisor

@dataclass
class Component:
    spec: ComponentSpec
    state: State = State.PENDING
    handle: Optional[ProcessHandle] = None
    restarts: int = 0
    crash_times: List[float] = field(default_factory=list)
    startup_failures: int = 0
    started_at: Optional[float] = None
    ready_at: Optional[float] = None
    ready_wall: Optional[float] = None   # epoch seconds, for `vesper status` (monotonic isn't portable across processes)
    last_exit: Optional[int] = None
    detail: str = ""
    task: Optional["asyncio.Task[None]"] = None
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)


class Supervisor:
    def __init__(
        self,
        specs: Sequence[ComponentSpec],
        policy: Optional[RestartPolicy] = None,
        *,
        spawn: Spawn,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        on_change: Optional[Callable[[Dict[str, Any]], None]] = None,
        log: Callable[[str], None] = print,
        shutdown_grace: float = 8.0,
        poll_interval: float = 0.25,
    ):
        self._order = [Component(spec=s) for s in specs]
        self._by_name = {c.spec.name: c for c in self._order}
        self._policy = policy or RestartPolicy()
        self._spawn = spawn
        self._sleep = sleep
        self._mono = monotonic
        self._on_change = on_change
        self._log = log
        self._grace = shutdown_grace
        self._poll = poll_interval
        self._stopping = False
        self._fatal: Optional[str] = None
        self._closed = asyncio.Event()
        self._started_wall = time.time()

    # ------------------------------------------------------------------ public

    @property
    def components(self) -> List[Component]:
        return list(self._order)

    def component(self, name: str) -> Component:
        return self._by_name[name]

    @property
    def fatal_reason(self) -> Optional[str]:
        return self._fatal

    async def start(self) -> bool:
        """Start every component in order, each only after the previous is ready.
        Returns False (after stopping everything) if a required one failed."""
        self._emit_state()
        for comp in self._order:
            if self._closed.is_set():  # Ctrl+C / `vesper down` during startup
                await self.stop()
                return False
            if comp.spec.skip_reason:
                self._set(comp, State.SKIPPED, comp.spec.skip_reason)
                self._log(f"[launcher] {comp.spec.name}: SKIPPED — {comp.spec.skip_reason}")
                comp.done_event.set()
                continue
            comp.task = asyncio.ensure_future(self._supervise(comp))
            await self._until_ready_or_terminal(comp)
            if comp.state == State.READY:
                continue
            if comp.state == State.EXITED:
                self._log(f"[launcher] {comp.spec.name}: exited cleanly during startup (it turned itself off)")
                continue
            if self._closed.is_set() and comp.state not in (State.FAILED, State.EXITED):
                await self.stop()
                return False
            reason = f"{comp.spec.name} failed to start: {comp.detail or comp.state.value}"
            if comp.spec.required:
                self._fatal = reason
                self._log(f"[launcher] FATAL — {reason}")
                await self.stop()
                return False
            self._log(f"[launcher] WARNING — {reason}. Continuing WITHOUT {comp.spec.name}.")
        return True

    async def wait_closed(self) -> None:
        """Block until a shutdown (stop()) or a fatal component failure."""
        await self._closed.wait()

    def request_stop(self) -> None:
        """Signal-handler friendly: ask the owner of wait_closed() to wind down."""
        self._closed.set()

    async def stop(self) -> None:
        """Stop everything, last-started first: SIGTERM, wait the grace period, SIGKILL."""
        self._stopping = True
        for comp in reversed(self._order):
            handle = comp.handle
            if handle is not None and handle.returncode is None:
                self._log(f"[launcher] stopping {comp.spec.name} (pid {handle.pid})")
                handle.terminate()
                if not await self._wait_exit(handle, self._grace):
                    self._log(f"[launcher] {comp.spec.name} ignored SIGTERM for {self._grace:g}s — killing")
                    handle.kill()
                    await self._wait_exit(handle, 2.0)
            if comp.state not in (State.FAILED, State.SKIPPED, State.EXITED):
                self._set(comp, State.STOPPED, comp.detail)
        for comp in self._order:
            if comp.task is not None and not comp.task.done():
                comp.task.cancel()
        for comp in self._order:
            if comp.task is not None:
                try:
                    await comp.task
                except (asyncio.CancelledError, Exception):
                    pass
            comp.done_event.set()
        self._closed.set()
        self._emit_state()

    def snapshot(self) -> Dict[str, Any]:
        now = self._mono()
        return {
            "supervisor_pid": os.getpid(),
            "started_at": self._started_wall,
            "stopping": self._stopping,
            "fatal": self._fatal,
            "components": [
                {
                    "name": c.spec.name,
                    "description": c.spec.description,
                    "state": c.state.value,
                    "pid": c.handle.pid if c.handle is not None and c.handle.returncode is None else None,
                    "restarts": c.restarts,
                    "ready_since": c.ready_wall if c.state == State.READY else None,
                    "last_exit": c.last_exit,
                    "detail": c.detail,
                    "required": c.spec.required,
                }
                for c in self._order
            ],
        }

    # ------------------------------------------------------------------ internals

    def _emit_state(self) -> None:
        if self._on_change is not None:
            try:
                self._on_change(self.snapshot())
            except Exception:  # a state-file problem must never break supervision
                pass

    def _set(self, comp: Component, state: State, detail: str = "") -> None:
        comp.state = state
        comp.detail = detail
        self._emit_state()

    async def _until_ready_or_terminal(self, comp: Component) -> None:
        waits = [
            asyncio.ensure_future(comp.ready_event.wait()),
            asyncio.ensure_future(comp.done_event.wait()),
            asyncio.ensure_future(self._closed.wait()),
        ]
        try:
            await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for w in waits:
                w.cancel()

    async def _wait_exit(self, handle: ProcessHandle, timeout: float) -> bool:
        deadline = self._mono() + timeout
        while handle.returncode is None:
            if self._mono() >= deadline:
                return False
            await self._sleep(self._poll)
        return True

    async def _await_ready(self, comp: Component, handle: ProcessHandle) -> bool:
        probe = comp.spec.probe
        if probe is None:
            return True
        deadline = self._mono() + comp.spec.ready_timeout
        ctx = ProbeContext(handle=handle, data=comp.spec.probe_data)
        while self._mono() < deadline:
            if handle.returncode is not None:
                return False
            try:
                if await probe(ctx):
                    return True
            except Exception:
                pass  # a probe error is "not ready yet"
            await self._sleep(self._poll)
        return False

    async def _supervise(self, comp: Component) -> None:
        spec, policy = comp.spec, self._policy
        try:
            while not self._stopping:
                self._set(comp, State.STARTING)
                try:
                    handle = await self._spawn(spec)
                except Exception as exc:
                    handle = None
                    rc = -1
                    comp.last_exit = rc
                    comp.detail = f"could not spawn: {exc}"
                    became_ready = False
                else:
                    comp.handle = handle
                    comp.started_at = self._mono()
                    wait_task = asyncio.ensure_future(handle.wait())
                    probe_task = asyncio.ensure_future(self._await_ready(comp, handle))
                    became_ready = False
                    try:
                        await asyncio.wait({wait_task, probe_task}, return_when=asyncio.FIRST_COMPLETED)
                        if probe_task.done() and not probe_task.cancelled() and probe_task.result() and not wait_task.done():
                            became_ready = True
                            comp.ready_at = self._mono()
                            comp.ready_wall = time.time()
                            comp.ready_event.set()
                            self._set(comp, State.READY)
                            self._log(f"[launcher] {spec.name}: ready (pid {handle.pid})")
                        elif not wait_task.done():
                            # probe gave up (timeout) while the process is still alive
                            self._log(f"[launcher] {spec.name}: not ready after {spec.ready_timeout:g}s — killing")
                            handle.terminate()
                            if not await self._wait_exit(handle, self._grace):
                                handle.kill()
                        rc = await wait_task
                    finally:
                        probe_task.cancel()
                    comp.last_exit = rc
                    if not became_ready:
                        comp.detail = self._tail(handle)

                if self._stopping:
                    return

                lived = (self._mono() - comp.ready_at) if (became_ready and comp.ready_at is not None) else 0.0
                comp.ready_event.clear()

                if handle is not None and rc == 0:
                    self._set(comp, State.EXITED, "exited with status 0")
                    self._log(f"[launcher] {spec.name}: exited cleanly (status 0) — not restarting")
                    return
                if rc in spec.permanent_exit_codes:
                    detail = self._tail(handle) or f"exit status {rc}"
                    self._fail(comp, f"unavailable this session (exit {rc}): {detail}")
                    return

                # a crash
                if became_ready:
                    comp.startup_failures = 0
                    if lived >= policy.healthy_reset_seconds:
                        comp.crash_times.clear()
                    now = self._mono()
                    comp.crash_times = [t for t in comp.crash_times if now - t <= policy.window_seconds]
                    comp.crash_times.append(now)
                    if len(comp.crash_times) >= policy.max_crashes:
                        self._fail(comp, f"crashed {len(comp.crash_times)}× within {policy.window_seconds:g}s "
                                         f"(last: {describe_exit(rc)}): {self._tail(handle)}")
                        return
                    attempt = len(comp.crash_times)
                else:
                    comp.startup_failures += 1
                    if comp.startup_failures >= policy.max_startup_failures:
                        self._fail(comp, f"failed to become ready {comp.startup_failures}× in a row "
                                         f"(last: {describe_exit(rc)}): {comp.detail or self._tail(handle)}")
                        return
                    attempt = comp.startup_failures

                backoff = min(policy.initial_backoff * (2 ** (attempt - 1)), policy.max_backoff)
                self._log(f"[launcher] {spec.name}: died ({describe_exit(rc)}); restarting in {backoff:g}s "
                          f"(restart #{comp.restarts + 1})")
                self._set(comp, State.BACKOFF, f"{describe_exit(rc)}; restart in {backoff:g}s")
                await self._sleep(backoff)
                comp.restarts += 1
        except asyncio.CancelledError:
            raise
        finally:
            if comp.state in TERMINAL or self._stopping:
                comp.done_event.set()

    def _fail(self, comp: Component, detail: str) -> None:
        self._set(comp, State.FAILED, detail)
        hint = f" {comp.spec.hint}" if comp.spec.hint else ""
        self._log(f"[launcher] {comp.spec.name}: FAILED — {detail}.{hint}")
        comp.done_event.set()
        if comp.spec.required and not self._stopping:
            self._fatal = f"{comp.spec.name} failed: {detail}"
            self._closed.set()  # wake wait_closed(); the owner calls stop()

    @staticmethod
    def _tail(handle: Optional[ProcessHandle], n: int = 3) -> str:
        if handle is None:
            return ""
        lines = [ln.strip() for ln in list(handle.recent_lines) if ln.strip()]
        return " | ".join(lines[-n:])[:400]
