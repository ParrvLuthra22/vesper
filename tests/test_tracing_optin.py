"""
LangSmith tracing is opt-in: a LANGSMITH_API_KEY in .env alone ships nothing
(docs/PRIVACY.md). Local JSONL tracing is unaffected.
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock


def _tracer(monkeypatch, tmp_path, *, key: bool, flag: bool):
    import tracing.tracer as tr

    if key:
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    else:
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    created: List[Any] = []
    monkeypatch.setattr(tr, "Client", lambda **kw: created.append(kw) or MagicMock())
    cfg = {"tracing": {"enabled": True, "local_dir": str(tmp_path), "langsmith_enabled": flag}}
    # Assert on the tracer's own logger directly: this repo's loggers don't always
    # propagate to pytest's caplog handler.
    monkeypatch.setattr(tr, "logger", MagicMock())
    return tr.Tracer(config=cfg), created


def _logged(monkeypatch) -> str:
    import tracing.tracer as tr

    return " ".join(str(c.args[0]) for c in tr.logger.method_calls if c.args)


def test_langsmith_stays_off_when_only_a_key_is_present(monkeypatch, tmp_path):
    tracer, created = _tracer(monkeypatch, tmp_path, key=True, flag=False)
    assert created == [] and tracer.remote_enabled is False
    assert "traces stay on this machine" in _logged(monkeypatch)


def test_langsmith_turns_on_only_with_flag_and_key(monkeypatch, tmp_path):
    tracer, created = _tracer(monkeypatch, tmp_path, key=True, flag=True)
    assert len(created) == 1 and tracer.remote_enabled is True
    assert "LangSmith tracing is ON" in _logged(monkeypatch)


def test_langsmith_flag_without_key_stays_local(monkeypatch, tmp_path):
    tracer, created = _tracer(monkeypatch, tmp_path, key=False, flag=True)
    assert created == [] and tracer.remote_enabled is False


def test_langsmith_is_off_by_default_in_config_and_code():
    from config.settings import TracingSettings, load_config_dict

    assert TracingSettings().langsmith_enabled is False
    assert load_config_dict()["tracing"]["langsmith_enabled"] is False
    assert load_config_dict()["tracing"]["enabled"] is True      # local JSONL tracing is unchanged


def test_local_tracing_still_writes_when_langsmith_is_off(monkeypatch, tmp_path):
    tracer, _ = _tracer(monkeypatch, tmp_path, key=True, flag=False)
    turn = tracer.start_turn("hello")
    turn.end(final_reply="hi", aborted=False, total_latency_ms=1.0)
    assert (tmp_path / "traces.jsonl").exists()


