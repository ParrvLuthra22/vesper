"""`--refresh --redact`: nothing but the deliberate output may reach the terminal. Logging (MCP bridge, httpx, warnings,
child-process stderr) goes to the log FILE for the duration of the command, so the guard scans what is actually printed
and a stored word that also appears in a log line cannot trip it. No log-shape exemptions, guard untouched.
Fake MCP bridge and collectors only: nothing real is started."""

from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path

import pytest

from briefing import cli
from briefing.cache import BriefingCache
from briefing.items import Item
from briefing.redact import Redactor
from briefing.service import BriefingService, CollectorReport
from utils import logger_core

# Words that appear in the log lines the fake bridge emits AND are stored as a sender here.
SENDER = "Connected Registered <tools@words.test>"


class FakeBridge:
    def __init__(self, *a, **k):
        pass

    async def start(self):
        logging.getLogger("tools.mcp_bridge").info("[MCPBridge] 'gmail' connected: 12 tool(s) registered")
        logging.getLogger("httpx").info('HTTP Request: POST https://example.invalid/v1 "HTTP/1.1 200 OK"')
        warnings.warn("a dependency warning that mentions tool registered connected", UserWarning)
        os.write(2, b"child process stderr: tool registered connected\n")        # what a subprocess/native lib would do

    async def stop(self):
        pass


@pytest.fixture
def env(tmp_path, monkeypatch, capfd):
    logs = tmp_path / "logs"
    logger_core.configure_logging(level="INFO", log_dir=str(logs), console_level="INFO", file_level="DEBUG")
    monkeypatch.setenv("VESPER_BRIEFING_DB", str(tmp_path / "b.db"))
    (tmp_path / "briefing.yaml").write_text(f'collect:\n  rules_path: "{tmp_path / "rules.json"}"\n')
    monkeypatch.setenv("VESPER_BRIEFING_CONFIG", str(tmp_path / "briefing.yaml"))
    now = time.time()
    cache = BriefingCache(tmp_path / "b.db")
    cache.upsert_items([Item(id="gmail:fake00000a", source="gmail", sender=SENDER, title="Quarterly numbers",
                             snippet="see attached", timestamp=now - 600)], now=now)
    cache.record_success("gmail", "1", now=now)
    cache.record_success("calendar", "1", now=now)
    cache.close()

    monkeypatch.setattr("tools.mcp_bridge.MCPBridge", FakeBridge)
    monkeypatch.setattr("launcher.stack.load_app_config", lambda: {})

    def from_registry(registry=None, cfg=None, cache=None):
        svc = BriefingService([], cfg, BriefingCache(cfg.cache_path))

        async def refresh():
            return {"gmail": CollectorReport("gmail", ok=True, new_items=1)}
        svc.refresh = refresh
        return svc

    monkeypatch.setattr(BriefingService, "from_registry", staticmethod(from_registry))
    capfd.readouterr()
    yield SimpleLogs(logs / "vesper.log")
    logging.shutdown()


class SimpleLogs:
    def __init__(self, path: Path):
        self.path = path

    def text(self) -> str:
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.FileHandler):
                h.flush()
        return self.path.read_text(errors="replace") if self.path.exists() else ""


def run(capfd, *argv):
    code = cli.main(list(argv))
    cap = capfd.readouterr()
    return code, cap.out, cap.err


# (a) -------------------------------------------------------------------------------------------------------

def test_a_stored_word_that_is_also_in_the_log_lines_does_not_trip_refresh_redact(env, capfd):
    code, out, err = run(capfd, "--refresh", "--explain", "--redact")
    assert code == 0, err
    assert "GUARD TRIPPED" not in err and "<sender:1>" in out and "<subject:1>" in out
    for word in ("connected", "registered", "words.test"):
        assert word not in (out + err).lower()


# (b) -------------------------------------------------------------------------------------------------------

def test_a_real_leak_in_a_dynamic_field_still_trips_the_guard(env, capfd, monkeypatch):
    monkeypatch.setattr(Redactor, "sender", lambda self, raw: str(raw))                  # a broken redaction: raw sender
    code, out, err = run(capfd, "--refresh", "--explain", "--redact")
    assert code == 3 and out == "" and "GUARD TRIPPED" in err
    assert "words.test" not in err and "Connected" not in err


# (c) -------------------------------------------------------------------------------------------------------

def test_the_leaky_formatter_probe_still_trips_the_guard(env, capfd, monkeypatch):
    def naive(s, b, red=None):
        it = s.item
        return {"id": "x", "score": s.score, "source": it.source, "when": "now", "who": f"{it.sender} - {it.title}",
                "why": "x", "flag": ""}
    monkeypatch.setattr(cli, "row_fields", naive)
    code, out, err = run(capfd, "--refresh", "--explain", "--redact")
    assert code == 3 and out == "" and "GUARD TRIPPED" in err
    assert "words.test" not in err and "Quarterly" not in err


# (d) -------------------------------------------------------------------------------------------------------

def test_nothing_from_the_refresh_reaches_the_terminal_under_redact_and_all_of_it_is_in_the_log_file(env, capfd):
    code, out, err = run(capfd, "--refresh", "--explain", "--redact")
    assert code == 0
    assert err.strip() == ""                                  # NOTHING on stderr: no progress, no MCP, no httpx, no warnings
    log = env.text()
    for fragment in ("'gmail' connected: 12 tool(s) registered", "HTTP Request: POST", "dependency warning",
                     "child process stderr", "[briefing] connecting MCP servers", "[briefing] gmail: ok (1 item(s) fetched)"):
        assert fragment in log, fragment


def test_logging_and_stderr_are_restored_afterwards(env, capfd):
    root = logging.getLogger()
    before = [h for h in root.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
    assert before, "the console handler should exist for this test to mean anything"
    run(capfd, "--refresh", "--explain", "--redact")
    after = [h for h in root.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
    assert after == before
    logging.getLogger("tools.mcp_bridge").info("after the command: visible again")
    os.write(2, b"fd 2 works again\n")
    err = capfd.readouterr().err
    assert "fd 2 works again" in err and "after the command: visible again" in err


def test_without_redact_the_progress_notes_still_go_to_stderr(env, capfd):
    code, out, err = run(capfd, "--refresh", "--explain")
    assert code == 0
    assert "[briefing] connecting MCP servers" in err and "[briefing] gmail: ok (1 item(s) fetched)" in err
    assert "Connected Registered" in out or "Connected" in out                       # unredacted view shows the real sender


def test_an_error_inside_the_command_is_reported_by_type_only_and_logging_is_restored(env, capfd, monkeypatch):
    async def boom(self):
        raise RuntimeError("secret words.test Connected")
    monkeypatch.setattr(FakeBridge, "start", boom)
    code, out, err = run(capfd, "--refresh", "--explain", "--redact")
    assert code == 1 and "RuntimeError" in err and "words.test" not in err and "Connected" not in err
    os.write(2, b"still alive\n")
    assert "still alive" in capfd.readouterr().err


# ---- each protection on its own (they used to mask one another) --------------------------------------------------

def test_a_live_console_handler_bound_to_stderr_is_detached_during_the_command(env, capfd):
    import sys
    live = logging.StreamHandler(sys.stderr)                    # a working console handler, as in a real run
    live.setLevel(logging.INFO)
    logging.getLogger().addHandler(live)
    try:
        code, out, err = run(capfd, "--refresh", "--explain", "--redact")
        assert code == 0 and err.strip() == "" and "registered" not in err.lower()
        assert live in logging.getLogger().handlers              # and put back afterwards
    finally:
        logging.getLogger().removeHandler(live)


def test_a_failing_handler_cannot_print_the_record_it_failed_on(env, capfd):
    """Python prints "--- Logging error ---" WITH the message to stderr when a handler raises. Under --redact that would
    land in the scanned buffer (and could carry a stored word); it must stay silent."""
    class Broken(logging.Handler):
        def emit(self, record):
            try:
                raise ValueError("boom")
            except ValueError:
                self.handleError(record)

    broken = Broken()
    logging.getLogger().addHandler(broken)
    try:
        code, out, err = run(capfd, "--refresh", "--explain", "--redact")
        assert code == 0 and "Logging error" not in err and "registered" not in err.lower()
        assert logging.raiseExceptions is True                   # restored
    finally:
        logging.getLogger().removeHandler(broken)
