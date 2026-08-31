from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from qwen_audio_agent.domain.task import Task, TaskStatus, persisted_task
from qwen_audio_agent.graphs.work import WorkGraph
from qwen_audio_agent.runtime.repository import TaskRepository
from qwen_audio_agent.runtime.task_store import JsonTaskStore


class RecoveryBackend:
    def __init__(
        self,
        *,
        cancel_error: Exception | None = None,
        cancel_result: Mapping[str, Any] | None = None,
    ) -> None:
        self.start_calls = 0
        self.cancel_calls: list[tuple[str, dict[str, Any]]] = []
        self.cancel_error = cancel_error
        self.cancel_result = dict(
            cancel_result
            if cancel_result is not None
            else {"state": "cancelled", "source": "backend"}
        )

    def describe(self) -> dict[str, Any]:
        return {"name": "recovery"}

    async def start(self, context: object = None) -> None:
        self.start_calls += 1

    async def health(self) -> dict[str, bool]:
        return {"ok": True}

    async def submit(self, task: object, context: object = None) -> None:
        raise AssertionError("recovered terminalization must not resubmit work")

    async def status(self, task_id: str | None = None, context: object = None) -> Any:
        return None

    async def cancel(self, task_id: str, context: object = None) -> Any:
        normalized_context = dict(context) if isinstance(context, Mapping) else {}
        self.cancel_calls.append((task_id, normalized_context))
        if self.cancel_error is not None:
            raise self.cancel_error
        return {"taskId": task_id, **self.cancel_result}

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


class DelegatingBackend(RecoveryBackend):
    def __init__(self) -> None:
        super().__init__()
        self.never_finishes = asyncio.Event()

    async def submit(
        self,
        task: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> None:
        assert context is not None
        context["on_event"](
            {
                "type": "backend.delegated",
                "delegation": {
                    "id": "delegation-private",
                    "sessionId": "session-private",
                    "directory": "/workspace/private",
                    "title": "可恢复项目",
                },
            }
        )
        await self.never_finishes.wait()


class FinalizingBackend(DelegatingBackend):
    async def submit(
        self,
        task: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> None:
        assert context is not None
        delegation = {
            "id": "delegation-private",
            "sessionId": "session-private",
            "directory": "/workspace/private",
            "title": "可恢复项目",
        }
        context["on_event"]({"type": "backend.delegated", "delegation": delegation})
        await asyncio.sleep(0)
        context["on_event"](
            {
                "type": "backend.delegation.completed",
                "delegation": {**delegation, "status": "completed"},
            }
        )
        await self.never_finishes.wait()


class DelegationRecoveryBackend(RecoveryBackend):
    def __init__(self) -> None:
        super().__init__()
        self.can_recover_calls: list[dict[str, Any]] = []
        self.recover_calls: list[dict[str, Any]] = []

    async def can_recover_delegated_work(self, task: Mapping[str, Any]) -> bool:
        projected = deepcopy(dict(task))
        self.can_recover_calls.append(projected)
        delegation = projected.get("delegation")
        return bool(
            isinstance(delegation, Mapping) and delegation.get("id") and delegation.get("sessionId")
        )

    async def recover_delegated_work(
        self,
        task: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.recover_calls.append(deepcopy(dict(task)))
        return {"content": f"recovered:{task['id']}"}


class FinalizingRecoveryBackend(DelegationRecoveryBackend):
    async def recover_delegated_work(
        self,
        task: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert context is not None
        self.recover_calls.append(deepcopy(dict(task)))
        delegation = deepcopy(dict(task["delegation"]))
        context["on_event"]({"type": "backend.delegated", "delegation": delegation})
        await asyncio.sleep(0)
        context["on_event"](
            {
                "type": "backend.delegation.completed",
                "delegation": {**delegation, "status": "completed"},
            }
        )
        await asyncio.sleep(0)
        return {"content": f"recovered:{task['id']}"}


async def eventually(predicate: Callable[[], bool], deadline_seconds: float = 2.0) -> None:
    async with asyncio.timeout(deadline_seconds):
        while not predicate():  # noqa: ASYNC110 - polling an in-memory runtime
            await asyncio.sleep(0)


def seed(path: Path, *tasks: Task) -> JsonTaskStore:
    store = JsonTaskStore(path)
    store.save([persisted_task(task) for task in tasks], {"nextTaskNumber": 9})
    return store


@pytest.mark.asyncio
async def test_restart_fails_unsafe_work_once_and_preserves_cancel_intent(tmp_path: Path) -> None:
    running = Task(
        id="task_1",
        status=TaskStatus.RUNNING,
        objective="unsafe direct work",
        owner_id="owner",
        started_at=1,
    )
    cancelling = Task(
        id="task_2",
        status=TaskStatus.CANCELLING,
        objective="already cancelling",
        owner_id="owner",
        started_at=1,
        cancellation={"status": "requested"},
    )
    store = seed(tmp_path / "tasks.json", running, cancelling)
    events: list[str] = []
    backend = RecoveryBackend()
    runtime = WorkGraph(
        backend=backend,
        repository=TaskRepository(store=store),
    )
    runtime.subscribe(lambda event: events.append(str(event["type"])))

    await runtime.start()

    failed = runtime.get("task_1")
    cancelled = runtime.get("task_2")
    assert failed is not None and failed["status"] == "failed"
    assert failed["notificationStatus"] == "pending"
    assert cancelled is not None and cancelled["status"] == "cancelled"
    assert cancelled["notificationStatus"] == "none"
    assert backend.cancel_calls == [("task_2", {"owner_id": "owner"})]
    internal_cancelled = runtime.repository.get("task_2")
    assert internal_cancelled is not None
    assert internal_cancelled.cancellation == {
        "taskId": "task_2",
        "state": "cancelled",
        "source": "backend",
    }
    assert events == ["task.failed", "task.cancelled"]
    await runtime.close()

    # A second process observes terminal records and emits no duplicate result.
    second_events: list[str] = []
    second = WorkGraph(
        backend=RecoveryBackend(),
        repository=TaskRepository(store=JsonTaskStore(tmp_path / "tasks.json")),
    )
    second.subscribe(lambda event: second_events.append(str(event["type"])))
    await second.start()
    assert second_events == []
    await second.close()


@pytest.mark.asyncio
async def test_restart_marks_cancelling_work_failed_when_backend_cancel_raises(
    tmp_path: Path,
) -> None:
    cancelling = Task(
        id="task_2",
        status=TaskStatus.CANCELLING,
        objective="already cancelling",
        owner_id="owner",
        started_at=1,
        cancellation={"status": "requested"},
    )
    backend = RecoveryBackend(cancel_error=RuntimeError("backend unavailable"))
    runtime = WorkGraph(
        backend=backend,
        repository=TaskRepository(store=seed(tmp_path / "tasks.json", cancelling)),
    )
    events: list[str] = []
    runtime.subscribe(lambda event: events.append(str(event["type"])))

    await runtime.start()

    failed = runtime.get("task_2")
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["notificationStatus"] == "pending"
    assert failed["error"] == "取消失败: backend unavailable"
    assert backend.cancel_calls == [("task_2", {"owner_id": "owner"})]
    internal = runtime.repository.get("task_2")
    assert internal is not None
    assert internal.cancellation == {"status": "requested"}
    assert events == ["task.failed"]
    await runtime.close()


@pytest.mark.asyncio
async def test_restart_marks_cancelling_work_failed_without_explicit_confirmation(
    tmp_path: Path,
) -> None:
    cancelling = Task(
        id="task_2",
        status=TaskStatus.CANCELLING,
        objective="already cancelling",
        owner_id="owner",
        started_at=1,
        cancellation={"status": "requested"},
    )
    backend = RecoveryBackend(cancel_result={"state": "not_found"})
    runtime = WorkGraph(
        backend=backend,
        repository=TaskRepository(store=seed(tmp_path / "tasks.json", cancelling)),
    )
    events: list[str] = []
    runtime.subscribe(lambda event: events.append(str(event["type"])))

    await runtime.start()

    failed = runtime.get("task_2")
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["notificationStatus"] == "pending"
    assert failed["error"] == "取消失败: backend returned cancellation state not_found"
    assert backend.cancel_calls == [("task_2", {"owner_id": "owner"})]
    internal = runtime.repository.get("task_2")
    assert internal is not None
    assert internal.cancellation == {"status": "requested"}
    assert events == ["task.failed"]
    await runtime.close()


@pytest.mark.asyncio
async def test_restart_reattaches_with_private_delegation_without_exposing_it(
    tmp_path: Path,
) -> None:
    task_path = tmp_path / "tasks.json"
    checkpointer = InMemorySaver()
    first = WorkGraph(
        backend=DelegatingBackend(),
        repository=TaskRepository(store=JsonTaskStore(task_path)),
        checkpointer=checkpointer,
    )
    accepted = await first.submit(objective="continue project", owner_id="owner")
    await eventually(lambda: first.get(accepted["id"])["status"] == "delegated")

    projected = first.get(accepted["id"])
    assert projected is not None
    assert projected["delegation"] == {
        "status": "running",
        "title": "可恢复项目",
        "presentation": None,
    }
    await first.close()
    await asyncio.sleep(0)

    backend = DelegationRecoveryBackend()
    recovered = WorkGraph(
        backend=backend,
        repository=TaskRepository(store=JsonTaskStore(task_path)),
        checkpointer=checkpointer,
    )
    await recovered.start()
    completed = await recovered.wait(accepted["id"])

    expected_delegation = {
        "status": "running",
        "title": "可恢复项目",
        "presentation": None,
        "id": "delegation-private",
        "sessionId": "session-private",
        "directory": "/workspace/private",
    }
    assert backend.can_recover_calls[0]["delegation"] == expected_delegation
    assert backend.recover_calls[0]["delegation"] == expected_delegation
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["result"] == f"recovered:{accepted['id']}"
    assert completed["delegation"] == {
        "status": "running",
        "title": "可恢复项目",
        "presentation": None,
    }
    await recovered.close()


@pytest.mark.asyncio
async def test_restart_finalizing_ignores_replayed_delegation_until_runtime_completes(
    tmp_path: Path,
) -> None:
    task_path = tmp_path / "tasks.json"
    checkpointer = InMemorySaver()
    first = WorkGraph(
        backend=FinalizingBackend(),
        repository=TaskRepository(store=JsonTaskStore(task_path)),
        checkpointer=checkpointer,
    )
    accepted = await first.submit(objective="finalize project", owner_id="owner")
    await eventually(lambda: first.get(accepted["id"])["status"] == "finalizing")
    await first.close()
    await asyncio.sleep(0)

    events: list[str] = []
    backend = FinalizingRecoveryBackend()
    recovered = WorkGraph(
        backend=backend,
        repository=TaskRepository(store=JsonTaskStore(task_path)),
        checkpointer=checkpointer,
    )
    recovered.subscribe(lambda event: events.append(str(event["type"])))
    await recovered.start()
    async with asyncio.timeout(2.0):
        completed = await recovered.wait(accepted["id"])

    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["result"] == f"recovered:{accepted['id']}"
    assert completed["delegation"] == {
        "status": "completed",
        "title": "可恢复项目",
        "presentation": None,
    }
    assert events.count("task.delegated") == 1
    assert events.count("task.finalizing") == 1
    assert events[-1] == "task.completed"
    assert len(backend.can_recover_calls) == 1
    assert len(backend.recover_calls) == 1
    await recovered.close()
