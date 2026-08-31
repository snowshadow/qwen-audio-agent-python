from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from qwen_audio_agent.domain.task import now_ms
from qwen_audio_agent.graphs.work import WorkGraph
from qwen_audio_agent.runtime.repository import TaskRepository
from qwen_audio_agent.runtime.task_store import JsonTaskStore


class ScheduledBackend:
    def __init__(self) -> None:
        self.submissions: list[Mapping[str, Any]] = []
        self.cancel_calls: list[str] = []

    def describe(self) -> dict[str, Any]:
        return {"name": "scheduled"}

    async def start(self, context: object = None) -> None:
        return None

    async def health(self) -> dict[str, bool]:
        return {"ok": True}

    async def submit(
        self,
        task: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, str]:
        self.submissions.append(task)
        return {"content": f"done:{task['id']}"}

    async def status(self, task_id: str | None = None, context: object = None) -> Any:
        return {"state": "working"}

    async def cancel(self, task_id: str, context: object = None) -> Any:
        self.cancel_calls.append(task_id)
        return {"state": "cancelled"}

    async def respond_authorization(
        self,
        task_id: str,
        authorization_id: str,
        decision: Any,
        context: object = None,
    ) -> None:
        return None

    def subscribe(self, listener: object) -> Callable[[], None]:
        return lambda: None

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_due_reminder_completes_locally_on_the_same_task_id() -> None:
    backend = ScheduledBackend()
    runtime = WorkGraph(backend=backend)
    events: list[str] = []
    runtime.subscribe(lambda event: events.append(str(event["type"])))

    scheduled = await runtime.submit_scheduled(
        objective="该休息了",
        owner_id="owner",
        at=now_ms() + 10,
        type="reminder",
    )
    assert scheduled["status"] == "scheduled"
    assert scheduled["kind"] == "reminder"

    completed = await runtime.wait(scheduled["id"])
    assert completed is not None
    assert completed["id"] == scheduled["id"]
    assert completed["status"] == "completed"
    assert completed["result"] == "该休息了"
    assert completed["notificationStatus"] == "pending"
    assert backend.submissions == []
    assert events == [
        "task.scheduled",
        "task.scheduled.fired",
        "task.running",
        "task.completed",
    ]
    await runtime.close()


@pytest.mark.asyncio
async def test_due_scheduled_task_enters_existing_work_graph_with_same_id() -> None:
    backend = ScheduledBackend()
    runtime = WorkGraph(backend=backend)
    events: list[str] = []
    runtime.subscribe(lambda event: events.append(str(event["type"])))

    scheduled = await runtime.submit_scheduled(
        objective="检查构建",
        owner_id="owner",
        schedule={"at": now_ms() + 10, "recurrence": "once"},
        type="task",
        timeout_ms=60_000,
    )
    completed = await runtime.wait(scheduled["id"])

    assert completed is not None
    assert completed["id"] == scheduled["id"]
    assert completed["status"] == "completed"
    assert completed["kind"] == "scheduled_task"
    assert completed["timeoutMs"] == 60_000
    assert [submission["id"] for submission in backend.submissions] == [scheduled["id"]]
    assert events == [
        "task.scheduled",
        "task.scheduled.fired",
        "task.running",
        "task.completed",
    ]
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("scheduled_type", ["reminder", "task"])
async def test_restart_preserves_and_rearms_scheduled_work(
    tmp_path: Path,
    scheduled_type: str,
) -> None:
    path = tmp_path / f"{scheduled_type}.json"
    first = WorkGraph(
        backend=ScheduledBackend(),
        repository=TaskRepository(store=JsonTaskStore(path)),
    )
    scheduled = await first.submit_scheduled(
        objective=f"restart:{scheduled_type}",
        owner_id="owner",
        at=now_ms() + 40,
        type=scheduled_type,
    )
    await first.close()

    backend = ScheduledBackend()
    second = WorkGraph(
        backend=backend,
        repository=TaskRepository(store=JsonTaskStore(path)),
    )
    await second.start()
    restored = second.get(scheduled["id"])
    assert restored is not None
    assert restored["status"] == "scheduled"

    completed = await second.wait(scheduled["id"])
    assert completed is not None
    assert completed["id"] == scheduled["id"]
    assert completed["status"] == "completed"
    assert len(backend.submissions) == (1 if scheduled_type == "task" else 0)
    await second.close()


@pytest.mark.asyncio
async def test_scheduled_cancel_is_local_and_close_clears_timers() -> None:
    backend = ScheduledBackend()
    runtime = WorkGraph(backend=backend)
    scheduled = await runtime.submit_scheduled(
        objective="稍后执行",
        owner_id="owner",
        at=now_ms() + 60_000,
        type="task",
    )
    timer = runtime._scheduled_timers[scheduled["id"]]

    cancelled = await runtime.cancel(scheduled["id"], owner_id="owner")
    assert cancelled is not None
    assert cancelled["status"] == "cancelled"
    assert cancelled["notificationStatus"] == "none"
    assert backend.cancel_calls == []
    assert scheduled["id"] not in runtime._scheduled_timers
    await asyncio.sleep(0)
    assert timer.cancelled()

    pending = await runtime.submit_scheduled(
        objective="保持到重启",
        owner_id="owner",
        at=now_ms() + 60_000,
    )
    pending_timer = runtime._scheduled_timers[pending["id"]]
    await runtime.close()
    await asyncio.sleep(0)
    assert runtime._scheduled_timers == {}
    assert pending_timer.cancelled()
    assert runtime.get(pending["id"])["status"] == "scheduled"
