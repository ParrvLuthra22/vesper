"""Tests for tasks/queue.py: background execution + completion events."""

from __future__ import annotations

import asyncio
from typing import List, Tuple

import pytest

from bus.event_bus import EventBus
from schemas.events import ObservationEvent, TaskCompletedEvent
from tasks.queue import TaskQueue


@pytest.fixture(autouse=True)
def _fresh_bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


def _collectors(bus: EventBus) -> Tuple[List[TaskCompletedEvent], List[ObservationEvent]]:
    completed: List[TaskCompletedEvent] = []
    observations: List[ObservationEvent] = []

    async def on_completed(event: TaskCompletedEvent) -> None:
        completed.append(event)

    async def on_observation(event: ObservationEvent) -> None:
        observations.append(event)

    bus.subscribe(TaskCompletedEvent, on_completed)
    bus.subscribe(ObservationEvent, on_observation)
    return completed, observations


@pytest.mark.asyncio
async def test_submit_returns_immediately_and_completes_in_background() -> None:
    bus = EventBus()
    queue = TaskQueue(event_bus=bus)

    async def slow_job():
        await asyncio.sleep(0.05)
        return "the answer"

    task_id = await queue.submit(slow_job(), description="a slow job")
    # submit() itself must not have awaited the job to completion.
    assert queue.pending_count() == 1

    await queue.wait_all()
    assert queue.pending_count() == 0
    assert task_id is not None


@pytest.mark.asyncio
async def test_success_publishes_task_completed_and_observation() -> None:
    bus = EventBus()
    queue = TaskQueue(event_bus=bus)
    completed, observations = _collectors(bus)

    async def job():
        return "email summary"

    task_id = await queue.submit(job(), description="thread summary")
    await queue.wait_all()

    assert len(completed) == 1
    assert completed[0].task_id == task_id
    assert completed[0].success is True
    assert completed[0].result == "email summary"

    assert len(observations) == 1
    assert observations[0].kind == "task_done"
    assert observations[0].detail == "Sir, the thread summary you asked for is ready."


@pytest.mark.asyncio
async def test_failure_publishes_failed_task_completed_and_observation() -> None:
    bus = EventBus()
    queue = TaskQueue(event_bus=bus)
    completed, observations = _collectors(bus)

    async def failing_job():
        raise RuntimeError("network error")

    await queue.submit(failing_job(), description="research task")
    await queue.wait_all()

    assert len(completed) == 1
    assert completed[0].success is False
    assert completed[0].error == "network error"

    assert len(observations) == 1
    assert observations[0].kind == "task_done"
    assert "research task" in observations[0].detail
    assert "network error" in observations[0].detail


@pytest.mark.asyncio
async def test_multiple_tasks_run_concurrently() -> None:
    bus = EventBus()
    queue = TaskQueue(event_bus=bus)

    async def job(n: int) -> int:
        await asyncio.sleep(0.02)
        return n

    for i in range(3):
        await queue.submit(job(i), description=f"job {i}")

    assert queue.pending_count() == 3
    await queue.wait_all()
    assert queue.pending_count() == 0
