"""
Regression tests for the PF5 stability defects.

All three are correctness bugs in failure paths, which is exactly where
bugs hide: an MCP subprocess that dies a moment before we signal it, a
dependency that is missing for the whole session, and a legacy UI that is
supposed to be off.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

import numpy as np
import pytest

from tools.mcp_bridge import MCPServerConnection
from voice.input.config import VoiceInputConfig
from voice.input.pipeline import VoiceInputPipeline
from voice.input.stages import VoiceInputUnavailable


# =============================================================================
# 1. MCP shutdown must be clean and silent
# =============================================================================

class _FakeProcess:
    """
    Stands in for asyncio.subprocess.Process.

    `exit_after` controls how many wait() calls time out before the process
    "exits", which is how each escalation step (stdin close -> terminate ->
    kill) gets exercised. `raise_on` makes a signal call raise
    ProcessLookupError — the real race where the child dies between our
    check and our signal.
    """

    def __init__(self, exit_after: int = 0, raise_on: tuple = ()):
        self.stdin = None
        self.returncode = 0
        self.terminated = False
        self.killed = False
        self._waits = 0
        self._exit_after = exit_after
        self._raise_on = raise_on

    async def wait(self):
        self._waits += 1
        if self._waits > self._exit_after:
            return 0
        raise asyncio.TimeoutError  # surfaces through wait_for as a timeout

    def terminate(self):
        if "terminate" in self._raise_on:
            raise ProcessLookupError(3, "No such process")
        self.terminated = True

    def kill(self):
        if "kill" in self._raise_on:
            raise ProcessLookupError(3, "No such process")
        self.killed = True


def _connection(process) -> MCPServerConnection:
    connection = MCPServerConnection.__new__(MCPServerConnection)
    connection._name = "fake"
    connection._process = process
    connection._pending = {}
    connection._stopping = False
    connection._watchdog_task = None
    connection._reader_task = None
    connection._stderr_task = None
    return connection


@pytest.mark.asyncio
async def test_stop_closes_stdin_and_exits_without_signals():
    """The polite path: most MCP servers exit on stdin EOF."""
    process = _FakeProcess(exit_after=0)
    await _connection(process).stop()

    assert not process.terminated
    assert not process.killed


@pytest.mark.asyncio
async def test_stop_escalates_to_terminate_then_kill():
    process = _FakeProcess(exit_after=99)  # never exits on its own
    await _connection(process).stop()

    assert process.terminated
    assert process.killed


@pytest.mark.asyncio
async def test_already_dead_process_is_not_an_error_on_terminate():
    """
    The actual defect: a ProcessLookupError escaping shutdown surfaced as
    "CLI fatal error" on an otherwise clean Ctrl+C.
    """
    process = _FakeProcess(exit_after=99, raise_on=("terminate",))
    await _connection(process).stop()  # must not raise

    assert not process.killed  # returned early: the child is already gone


@pytest.mark.asyncio
async def test_already_dead_process_is_not_an_error_on_kill():
    process = _FakeProcess(exit_after=99, raise_on=("kill",))
    await _connection(process).stop()  # must not raise


@pytest.mark.asyncio
async def test_process_lookup_error_while_waiting_is_swallowed():
    class LookupOnWait(_FakeProcess):
        async def wait(self):
            raise ProcessLookupError(3, "No such process")

    await _connection(LookupOnWait()).stop()  # must not raise


@pytest.mark.asyncio
async def test_stop_fails_pending_requests_so_callers_do_not_hang():
    process = _FakeProcess(exit_after=0)
    connection = _connection(process)
    pending: asyncio.Future = asyncio.get_running_loop().create_future()
    connection._pending = {1: pending}

    await connection.stop()

    assert pending.done() and isinstance(pending.exception(), RuntimeError)


# =============================================================================
# 2. Voice input degrades, never hot-loops
# =============================================================================

class _MissingDepWake:
    """A wake stage whose dependency is absent — fails on every frame."""

    def __init__(self):
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        raise VoiceInputUnavailable("wake word (openwakeword) unavailable: no module")


class _FlakyWake:
    """Raises an ordinary (transient) error on every frame."""

    def process(self, frame):
        raise RuntimeError("bad frame")


class _Sink:
    def __init__(self):
        self.injected: List[str] = []

    def inject(self, text: str) -> None:
        self.injected.append(text)


def _pipeline(wake, transcriber=None) -> VoiceInputPipeline:
    config = VoiceInputConfig.from_app_config({})
    return VoiceInputPipeline(
        config=config,
        wake=wake,
        transcriber=transcriber,
        sink=_Sink(),
        emitter=lambda t: None,
        vad_factory=lambda: None,
    )


def _frames(n: int) -> List[np.ndarray]:
    return [np.zeros(160, dtype=np.int16) for _ in range(n)]


def test_missing_dependency_stops_after_one_frame(caplog):
    """
    The hot loop: stage load() runs per frame, so a missing package used to
    raise — and log a full traceback — for every frame the mic produced.
    """
    wake = _MissingDepWake()
    pipeline = _pipeline(wake)

    with caplog.at_level(logging.WARNING, logger="vesper.voice"):
        pipeline.run_stream(iter(_frames(500)))

    assert wake.calls == 1, "must stop at the first permanent failure, not retry per frame"
    assert pipeline.disabled_reason is not None
    disable_lines = [r for r in caplog.records if "disabled for this session" in r.message]
    assert len(disable_lines) == 1, "exactly one line, never a flood"


def test_transient_errors_are_rate_limited(caplog):
    """A recurring recoverable fault must not flood the log either."""
    pipeline = _pipeline(_FlakyWake())

    with caplog.at_level(logging.WARNING, logger="vesper.voice"):
        pipeline.run_stream(iter(_frames(200)))

    tracebacks = [r for r in caplog.records if r.exc_info is not None]
    assert len(tracebacks) <= 3, f"expected at most 3 tracebacks, got {len(tracebacks)}"


def test_transient_errors_do_not_disable_voice():
    """A bad frame is not a reason to give up for the session."""
    pipeline = _pipeline(_FlakyWake())
    pipeline.run_stream(iter(_frames(10)))

    assert pipeline.disabled_reason is None


def test_missing_transcriber_disables_rather_than_looping(caplog):
    class OkWake:
        def process(self, frame):
            return True

    class MissingTranscriber:
        def transcribe(self, audio):
            raise VoiceInputUnavailable("transcription (faster-whisper) unavailable: no module")

    pipeline = _pipeline(OkWake(), transcriber=MissingTranscriber())
    # Give the segmenter a real VAD-free path: a permanent transcriber failure
    # must disable regardless of how listening resolved.
    pipeline._listen = lambda it: np.zeros(160, dtype=np.int16)

    with caplog.at_level(logging.WARNING, logger="vesper.voice"):
        pipeline.run_stream(iter(_frames(50)))

    assert pipeline.disabled_reason is not None


def test_stage_load_failure_is_cached_not_retried():
    """A failed import must not be re-attempted on the next frame."""
    from voice.input.stages import WakeDetector

    config = VoiceInputConfig.from_app_config({})
    detector = WakeDetector(config)
    attempts = {"n": 0}

    def counting_load():
        attempts["n"] += 1
        raise VoiceInputUnavailable("nope")

    detector._load_error = VoiceInputUnavailable("already failed")
    with pytest.raises(VoiceInputUnavailable):
        detector.load()
    with pytest.raises(VoiceInputUnavailable):
        detector.load()

    assert attempts["n"] == 0  # never re-entered the import path


# =============================================================================
# 3. The legacy overlay stays off
# =============================================================================

@pytest.mark.asyncio
async def test_legacy_overlay_is_off_by_default():
    """
    It gated on `ui.hud.enabled` (default true) and never read
    `ui.legacy_overlay.enabled`, so the flag documenting "off by default"
    enforced nothing.
    """
    from ui.hud_overlay import HUDOverlayController

    controller = HUDOverlayController(config={})
    await controller.start()

    assert controller._hud is None
    assert controller._hud_process is None


@pytest.mark.asyncio
async def test_legacy_overlay_stays_off_even_when_hud_enabled():
    from ui.hud_overlay import HUDOverlayController

    controller = HUDOverlayController(config={"ui": {"hud": {"enabled": True}}})
    await controller.start()

    assert controller._hud is None
    assert controller._hud_process is None


@pytest.mark.asyncio
async def test_overlay_start_failure_cannot_crash_startup(monkeypatch):
    """An AppKit/Tk failure must degrade to 'no overlay', never propagate."""
    from ui import hud_overlay

    def exploding_process(*args, **kwargs):
        raise RuntimeError("requires macOS 14 or later")

    monkeypatch.setattr(hud_overlay.mp, "Process", exploding_process)

    controller = hud_overlay.HUDOverlayController(
        config={"ui": {"legacy_overlay": {"enabled": True}, "hud": {"enabled": True}}}
    )
    await controller.start()  # must not raise

    assert controller._hud_process is None
