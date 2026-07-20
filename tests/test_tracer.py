"""
Tests for tracing/tracer.py and tracing/last_trace.py.

All tests point the local JSONL trace file at tmp_path, never the real
data/traces/ directory. No real LangSmith network calls are made here —
tracing.enabled without an API key naturally falls back to local-only,
which is exactly the path these tests exercise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from langsmith.run_trees import RunTree

from llm.types import LLMResponse, RouterError
from tracing.last_trace import print_last_turn
from tracing.tracer import Tracer


def _config(tmp_path: Path, **overrides: Any) -> Dict[str, Any]:
    cfg = {"tracing": {"enabled": True, "project_name": "vesper-test", "local_dir": str(tmp_path)}}
    cfg["tracing"].update(overrides)
    return cfg


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# =============================================================================
# Local fallback (no LANGSMITH_API_KEY): never breaks, always writes locally
# =============================================================================

def test_no_api_key_falls_back_to_local_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    tracer = Tracer(config=_config(tmp_path))

    assert tracer.remote_enabled is False


def test_tracing_disabled_writes_nothing(tmp_path: Path) -> None:
    tracer = Tracer(config=_config(tmp_path, enabled=False))

    turn = tracer.start_turn("hello")
    it = turn.start_iteration(0, [{"role": "user", "content": "hi"}], "planning")
    it.end(LLMResponse(text="hi", provider="groq", model="m", latency_ms=1.0))
    turn.end(final_reply="hi", aborted=False, total_latency_ms=2.0)

    assert not (tmp_path / "traces.jsonl").exists()


# =============================================================================
# Run hierarchy: turn -> plan_iteration -> tool_execution
# =============================================================================

@pytest.mark.asyncio
async def test_full_hierarchy_written_with_correct_nesting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    tracer = Tracer(config=_config(tmp_path))

    turn = tracer.start_turn("open safari and set volume to 30")
    it = turn.start_iteration(0, [{"role": "user", "content": "open safari"}], "planning")

    tool = it.start_tool("open_app", {"app_name": "Safari"})
    tool.end(guardian_verdict="allow", result="Opened Safari", latency_ms=50.0, success=True)

    it.end(
        LLMResponse(
            text="", tool_calls=[], provider="groq", model="openai/gpt-oss-120b",
            usage={"prompt_tokens": 100, "completion_tokens": 20}, latency_ms=300.0,
        )
    )
    turn.end(final_reply="Done, Sir.", aborted=False, total_latency_ms=500.0)

    records = _read_jsonl(tmp_path / "traces.jsonl")
    assert len(records) == 3

    by_name = {r["name"]: r for r in records}
    turn_record = by_name["turn"]
    iteration_record = by_name["plan_iteration"]
    tool_record = by_name["tool_execution"]

    # Nesting: tool_execution -> plan_iteration -> turn.
    assert iteration_record["parent_run_id"] == turn_record["run_id"]
    assert tool_record["parent_run_id"] == iteration_record["run_id"]
    assert turn_record["parent_run_id"] is None

    # All runs in one turn share a trace_id.
    assert iteration_record["trace_id"] == turn_record["trace_id"]
    assert tool_record["trace_id"] == turn_record["trace_id"]

    # Turn: user input, final reply, total latency.
    assert turn_record["inputs"]["user_text"] == "open safari and set volume to 30"
    assert turn_record["outputs"]["final_reply"] == "Done, Sir."
    assert turn_record["metadata"]["total_latency_ms"] == 500.0

    # plan_iteration: messages in, provider/model, response.
    assert iteration_record["inputs"]["messages_in"] == [{"role": "user", "content": "open safari"}]
    assert iteration_record["metadata"]["provider"] == "groq"
    assert iteration_record["metadata"]["model"] == "openai/gpt-oss-120b"
    assert iteration_record["metadata"]["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}

    # tool_execution: tool, args, guardian verdict, result, latency.
    assert tool_record["inputs"]["tool"] == "open_app"
    assert tool_record["inputs"]["arguments"] == {"app_name": "Safari"}
    assert tool_record["metadata"]["guardian_verdict"] == "allow"
    assert tool_record["outputs"]["result"] == "Opened Safari"
    assert tool_record["metadata"]["latency_ms"] == 50.0


@pytest.mark.asyncio
async def test_router_error_recorded_as_iteration_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    tracer = Tracer(config=_config(tmp_path))

    turn = tracer.start_turn("hi")
    it = turn.start_iteration(0, [{"role": "user", "content": "hi"}], "planning")
    it.end(RouterError(user_message="Sir, I'm having trouble thinking right now.", purpose="planning"))
    turn.end(final_reply="Sir, I'm having trouble thinking right now.", aborted=True, total_latency_ms=10.0)

    records = _read_jsonl(tmp_path / "traces.jsonl")
    iteration_record = next(r for r in records if r["name"] == "plan_iteration")
    assert iteration_record["error"] is not None


# =============================================================================
# observation_injected metadata
# =============================================================================

def test_observation_injected_metadata_present_when_observations_given(tmp_path: Path) -> None:
    tracer = Tracer(config=_config(tmp_path))
    turn = tracer.start_turn("hi")
    turn.mark_observations_injected(["3 apps in 30 minutes."])
    turn.end(final_reply="hi", aborted=False, total_latency_ms=1.0)

    records = _read_jsonl(tmp_path / "traces.jsonl")
    assert records[0]["metadata"]["observation_injected"] is True
    assert records[0]["metadata"]["observations"] == ["3 apps in 30 minutes."]


def test_observation_injected_metadata_absent_when_no_observations(tmp_path: Path) -> None:
    tracer = Tracer(config=_config(tmp_path))
    turn = tracer.start_turn("hi")
    turn.mark_observations_injected([])
    turn.end(final_reply="hi", aborted=False, total_latency_ms=1.0)

    records = _read_jsonl(tmp_path / "traces.jsonl")
    assert "observation_injected" not in records[0]["metadata"]


# =============================================================================
# Defensive: tracing must never break the app
# =============================================================================

def test_persist_failure_is_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Even if the underlying RunTree.end() raises, callers must not see it."""
    tracer = Tracer(config=_config(tmp_path))
    turn = tracer.start_turn("hi")

    def broken_end(self: RunTree, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated tracing backend failure")

    # RunTree is a pydantic model (extra="ignore"), so instance-level
    # setattr of a non-field name is rejected -- patch the class method.
    monkeypatch.setattr(RunTree, "end", broken_end)

    # Must not raise.
    turn.end(final_reply="hi", aborted=False, total_latency_ms=1.0)


def test_local_write_failure_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = Tracer(config=_config(tmp_path))
    turn = tracer.start_turn("hi")

    def broken_write_local(*args: Any, **kwargs: Any) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(tracer, "_write_local", broken_write_local)

    # Must not raise, even though the local sink is broken.
    turn.end(final_reply="hi", aborted=False, total_latency_ms=1.0)


# =============================================================================
# `trace last` helper
# =============================================================================

def test_print_last_turn_renders_tree(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    tracer = Tracer(config=_config(tmp_path))

    turn = tracer.start_turn("open safari")
    it = turn.start_iteration(0, [{"role": "user", "content": "open safari"}], "planning")
    tool = it.start_tool("open_app", {"app_name": "Safari"})
    tool.end(guardian_verdict="allow", result="Opened Safari", latency_ms=50.0, success=True)
    it.end(LLMResponse(text="", provider="groq", model="m", latency_ms=100.0))
    turn.end(final_reply="Opened Safari, Sir.", aborted=False, total_latency_ms=200.0)

    print_last_turn(local_path=tmp_path / "traces.jsonl")
    output = capsys.readouterr().out

    assert "turn (chain)" in output
    assert "open safari" in output
    assert "plan_iteration (chain)" in output
    assert "tool_execution (tool)" in output
    assert "open_app" in output
    assert "Opened Safari, Sir." in output


def test_print_last_turn_picks_the_most_recent_turn(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    tracer = Tracer(config=_config(tmp_path))

    first = tracer.start_turn("first turn")
    first.end(final_reply="first reply", aborted=False, total_latency_ms=1.0)

    second = tracer.start_turn("second turn")
    second.end(final_reply="second reply", aborted=False, total_latency_ms=1.0)

    print_last_turn(local_path=tmp_path / "traces.jsonl")
    output = capsys.readouterr().out

    assert "second turn" in output
    assert "first turn" not in output


def test_print_last_turn_handles_no_traces(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    print_last_turn(local_path=tmp_path / "does_not_exist.jsonl")
    output = capsys.readouterr().out
    assert "No traces recorded" in output
