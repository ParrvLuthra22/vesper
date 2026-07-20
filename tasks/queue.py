"""
TaskQueue — background execution for slow tool calls.

The Planner routes any tool the registry marks slow=True here instead of
awaiting it inline: submit() schedules the coroutine immediately and
returns a task_id without blocking the current turn. When the job
finishes, the queue publishes TaskCompletedEvent (the raw outcome) and an
ObservationEvent(kind="task_done") — the latter flows into
ConversationContext as a pending observation exactly like any other
call-out (see orchestrator/brain.py), so the Planner mentions it, once,
the next time the user says anything.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Coroutine, Dict, Optional
from uuid import UUID, uuid4

from bus.event_bus import EventBus, get_event_bus
from schemas.events import ObservationEvent, TaskCompletedEvent
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueuedTask:
    task_id: UUID
    description: str
    task: "asyncio.Task[Any]"
    started_at: float = field(default_factory=time.monotonic)


class TaskQueue:
    """Runs submitted coroutines in the background, publishing completion events."""

    def __init__(self, event_bus: Optional[EventBus] = None):
        self._event_bus = event_bus or get_event_bus()
        self._tasks: Dict[UUID, QueuedTask] = {}

    def pending_count(self) -> int:
        return len(self._tasks)

    async def submit(self, coro: "Coroutine[Any, Any, Any]", description: str) -> UUID:
        """Schedule `coro` to run in the background. Returns immediately with a task_id."""
        task_id = uuid4()
        task = asyncio.create_task(self._run(task_id, description, coro))
        self._tasks[task_id] = QueuedTask(task_id=task_id, description=description, task=task)
        return task_id

    async def _run(self, task_id: UUID, description: str, coro: "Coroutine[Any, Any, Any]") -> None:
        success = False
        result: Any = None
        error: Optional[str] = None
        try:
            result = await coro
            success = True
        except Exception as exc:
            error = str(exc)
            logger.error(f"[TaskQueue] background task '{description}' failed: {exc}", exc_info=True)
        finally:
            self._tasks.pop(task_id, None)

        await self._event_bus.emit(
            TaskCompletedEvent(task_id=task_id, success=success, result=result, error=error, source="TaskQueue")
        )

        if success:
            detail = f"Sir, the {description} you asked for is ready."
        else:
            detail = f"Sir, the {description} you asked for failed: {error}"
        await self._event_bus.emit(ObservationEvent(kind="task_done", detail=detail, source="TaskQueue"))

    async def wait_all(self, timeout: Optional[float] = None) -> None:
        """Wait for all currently pending tasks to finish (tests / graceful shutdown)."""
        pending = [qt.task for qt in list(self._tasks.values())]
        if pending:
            await asyncio.wait(pending, timeout=timeout)
