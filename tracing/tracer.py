"""
Tracer — LangSmith instrumentation over the Planner's tool-calling loop.

Run hierarchy per user turn:
    turn (root)                          — user_text, final_reply, total latency
      └── plan_iteration (child)         — one per Planner loop iteration:
                                            messages in, provider/model, response
            └── tool_execution (child)   — one per tool call: tool, args,
                                            guardian verdict, result, latency

LangSmith is the sink when tracing.enabled and LANGSMITH_API_KEY are both
set. A local JSONL file at <tracing.local_dir>/traces.jsonl is *always*
written too, regardless of LangSmith availability — both as the
degrade-gracefully fallback and as the source `trace last` reads (so it
never depends on the network). Every public method here is defensive:
tracing must never break the assistant, so failures are logged and
swallowed, never raised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langsmith import Client
from langsmith.run_trees import RunTree

from llm.types import LLMResponse, RouterError
from utils.api_keys import get_langsmith_api_key
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PROJECT_NAME = "vesper"
DEFAULT_LOCAL_DIR = "data/traces"
LOCAL_TRACE_FILENAME = "traces.jsonl"


class Tracer:
    """Builds and persists the turn / plan_iteration / tool_execution run tree."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._enabled = bool(self._get_config("tracing.enabled", True))
        self._project_name = str(self._get_config("tracing.project_name", DEFAULT_PROJECT_NAME))
        self._local_path = (
            Path(self._get_config("tracing.local_dir", DEFAULT_LOCAL_DIR)) / LOCAL_TRACE_FILENAME
        )

        self._client: Optional[Client] = None
        if self._enabled:
            api_key = get_langsmith_api_key(self._get_config)
            if api_key:
                try:
                    self._client = Client(api_key=api_key)
                except Exception as exc:
                    logger.warning(f"[Tracer] LangSmith client init failed, using local-only tracing: {exc}")
                    self._client = None
            else:
                logger.info("[Tracer] LANGSMITH_API_KEY not set; tracing to local JSONL only")

    def _get_config(self, key: str, default: Any = None) -> Any:
        value: Any = self._config
        for part in key.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(part)
            if value is None:
                return default
        return value

    @property
    def remote_enabled(self) -> bool:
        """Whether traces are also being sent to LangSmith (not just local)."""
        return self._client is not None

    def start_turn(self, user_text: str) -> "TurnTrace":
        """Start the root 'turn' run for one Planner.run() call."""
        if not self._enabled:
            return TurnTrace(tracer=self, run=None)
        try:
            run = RunTree(
                name="turn",
                run_type="chain",
                project_name=self._project_name,
                client=self._client,
                inputs={"user_text": user_text},
            )
        except Exception as exc:
            logger.debug(f"[Tracer] failed to start turn trace (non-fatal): {exc}")
            run = None
        return TurnTrace(tracer=self, run=run)

    def _persist(self, run: Optional[RunTree]) -> None:
        """Write a completed run to local JSONL, and to LangSmith if configured."""
        if run is None:
            return
        try:
            self._write_local(run)
        except Exception as exc:
            logger.debug(f"[Tracer] local trace write failed (non-fatal): {exc}")

        if self._client is not None:
            try:
                run.post()
            except Exception as exc:
                logger.debug(f"[Tracer] LangSmith post failed (non-fatal): {exc}")

    def _write_local(self, run: RunTree) -> None:
        self._local_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": str(run.id),
            "trace_id": str(run.trace_id),
            "parent_run_id": str(run.parent_run_id) if run.parent_run_id else None,
            "name": run.name,
            "run_type": run.run_type,
            "project_name": run.session_name,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "inputs": run.inputs,
            "outputs": run.outputs,
            "metadata": (run.extra or {}).get("metadata", {}),
            "tags": run.tags or [],
            "error": run.error,
        }
        with self._local_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


class TurnTrace:
    """Root 'turn' run: one per Planner.run() call. `run` is None if tracing is disabled."""

    def __init__(self, tracer: Tracer, run: Optional[RunTree]):
        self._tracer = tracer
        self._run = run

    def mark_observations_injected(self, observations: List[str]) -> None:
        """Attach observation_injected metadata when a call-out was in context."""
        if not observations or self._run is None:
            return
        try:
            self._run.add_metadata({"observation_injected": True, "observations": observations})
        except Exception as exc:
            logger.debug(f"[Tracer] mark_observations_injected failed (non-fatal): {exc}")

    def mark_memories_injected(self, memories: List[str]) -> None:
        """Attach memory_injected metadata when semantic memory matched this turn's input."""
        if not memories or self._run is None:
            return
        try:
            self._run.add_metadata({"memory_injected": True, "memories": memories})
        except Exception as exc:
            logger.debug(f"[Tracer] mark_memories_injected failed (non-fatal): {exc}")

    def start_iteration(
        self, iteration: int, messages_in: List[Dict[str, Any]], purpose: str
    ) -> "IterationTrace":
        if self._run is None:
            return IterationTrace(tracer=self._tracer, run=None)
        try:
            child = self._run.create_child(
                name="plan_iteration",
                run_type="chain",
                inputs={"iteration": iteration, "messages_in": messages_in, "purpose": purpose},
            )
        except Exception as exc:
            logger.debug(f"[Tracer] failed to start plan_iteration trace (non-fatal): {exc}")
            child = None
        return IterationTrace(tracer=self._tracer, run=child)

    def end(self, final_reply: str, aborted: bool, total_latency_ms: float) -> None:
        if self._run is None:
            return
        try:
            self._run.end(
                outputs={"final_reply": final_reply, "aborted": aborted},
                metadata={"total_latency_ms": total_latency_ms},
            )
        except Exception as exc:
            logger.debug(f"[Tracer] failed to end turn trace (non-fatal): {exc}")
            return
        self._tracer._persist(self._run)


class IterationTrace:
    """Child 'plan_iteration' run: one per Planner loop iteration."""

    def __init__(self, tracer: Tracer, run: Optional[RunTree]):
        self._tracer = tracer
        self._run = run

    def start_tool(self, tool_name: str, arguments: Dict[str, Any]) -> "ToolTrace":
        if self._run is None:
            return ToolTrace(tracer=self._tracer, run=None)
        try:
            child = self._run.create_child(
                name="tool_execution",
                run_type="tool",
                inputs={"tool": tool_name, "arguments": arguments},
            )
        except Exception as exc:
            logger.debug(f"[Tracer] failed to start tool_execution trace (non-fatal): {exc}")
            child = None
        return ToolTrace(tracer=self._tracer, run=child)

    def end(self, response: Union[LLMResponse, RouterError]) -> None:
        if self._run is None:
            return
        try:
            if isinstance(response, RouterError):
                self._run.end(outputs={"tool_call_count": 0}, error=response.message)
            else:
                self._run.end(
                    outputs={
                        "response_text": response.text,
                        "tool_call_count": len(response.tool_calls),
                    },
                    metadata={
                        "provider": response.provider,
                        "model": response.model,
                        "usage": response.usage,
                        "latency_ms": response.latency_ms,
                    },
                )
        except Exception as exc:
            logger.debug(f"[Tracer] failed to end plan_iteration trace (non-fatal): {exc}")
            return
        self._tracer._persist(self._run)


class ToolTrace:
    """Child 'tool_execution' run: one per tool call within an iteration."""

    def __init__(self, tracer: Tracer, run: Optional[RunTree]):
        self._tracer = tracer
        self._run = run

    def end(
        self,
        guardian_verdict: str,
        result: str,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        if self._run is None:
            return
        try:
            self._run.end(
                outputs={"result": result, "success": success},
                error=error,
                metadata={"guardian_verdict": guardian_verdict, "latency_ms": latency_ms},
            )
        except Exception as exc:
            logger.debug(f"[Tracer] failed to end tool_execution trace (non-fatal): {exc}")
            return
        self._tracer._persist(self._run)
