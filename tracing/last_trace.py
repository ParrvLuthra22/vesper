"""
`trace last` — prints the last turn's run tree to the terminal.

Stub for now: the P08 CLI will expose this as the /trace command. Reads
the local JSONL trace file directly, so it works regardless of whether
LangSmith is configured — it's the same file `Tracer` always writes to.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tracing.tracer import DEFAULT_LOCAL_DIR, LOCAL_TRACE_FILENAME


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _latency_str(run: Dict[str, Any]) -> str:
    start, end = run.get("start_time"), run.get("end_time")
    if not start or not end:
        return "?"
    try:
        delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
        return f"{delta.total_seconds() * 1000:.0f}ms"
    except ValueError:
        return "?"


def _print_run(records: List[Dict[str, Any]], run: Dict[str, Any], indent: int) -> None:
    prefix = "  " * indent
    print(f"{prefix}- {run['name']} ({run['run_type']}) [{_latency_str(run)}]")

    metadata = run.get("metadata", {}) or {}
    inputs = run.get("inputs", {}) or {}
    outputs = run.get("outputs", {}) or {}
    detail_prefix = "  " * (indent + 1)

    if run["name"] == "turn":
        print(f"{detail_prefix}user_text: {inputs.get('user_text', '')!r}")
        print(f"{detail_prefix}final_reply: {outputs.get('final_reply', '')!r}")
        if metadata.get("observation_injected"):
            print(f"{detail_prefix}observation_injected: {metadata.get('observations')}")
    elif run["name"] == "plan_iteration":
        print(
            f"{detail_prefix}provider={metadata.get('provider')} model={metadata.get('model')} "
            f"usage={metadata.get('usage')} tool_calls={outputs.get('tool_call_count')}"
        )
    elif run["name"] == "tool_execution":
        print(f"{detail_prefix}tool={inputs.get('tool')} args={inputs.get('arguments')}")
        print(
            f"{detail_prefix}verdict={metadata.get('guardian_verdict')} "
            f"success={outputs.get('success')} result={outputs.get('result')!r}"
        )

    if run.get("error"):
        print(f"{detail_prefix}error: {run['error']}")

    children = sorted(
        (r for r in records if r.get("parent_run_id") == run["run_id"]),
        key=lambda r: r.get("start_time") or "",
    )
    for child in children:
        _print_run(records, child, indent + 1)


def print_last_turn(local_path: Optional[Path] = None) -> None:
    """Print the most recent completed 'turn' run and its full subtree."""
    path = local_path or (Path(DEFAULT_LOCAL_DIR) / LOCAL_TRACE_FILENAME)
    records = _load_records(path)
    if not records:
        print(f"No traces recorded yet ({path}).")
        return

    turns = [r for r in records if r.get("run_type") == "chain" and r.get("name") == "turn"]
    if not turns:
        print("No completed turns recorded yet.")
        return

    last_turn = max(turns, key=lambda r: r.get("start_time") or "")
    _print_run(records, last_turn, indent=0)
