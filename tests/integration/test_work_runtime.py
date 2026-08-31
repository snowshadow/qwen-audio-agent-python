from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from qwen_audio_agent.backend.port import BackendEventType
from qwen_audio_agent.graphs.work import WorkGraph
from qwen_audio_agent.runtime.repository import TaskRepository
from qwen_audio_agent.runtime.scheduler import TaskScheduler


async def eventually(predicate: Callable[[], bool], deadline_seconds: float = 2.0) -> None:
    async with asyncio.timeout(deadline_seconds):
        while not predicate():  # noqa: ASYNC110 - polling a synchronous projection
            await asyncio.sleep(0)


class LaneBackend:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.gates: dict[str, asyncio.Event] = {}
        self.delegating: set[str] = set()

    def describe(self) -> dict[str, Any]:
        return {"name": "lane"}

    async def start(self, context: object = None) -> None:
        return None

    async def health(self) -> dict[str, bool]:
        return {"ok": True}

    async def submit(
        self, task: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        task_id = str(task["id"])
        self.started.append(task_id)
        if task_id in self.delegating:
            assert context is not None
            context["on_event"](
                {
                    "type": BackendEventType.DELEGATED,
                    "delegation": {
                        "id": "private-run",
                        "sessionId": "private-session",
                        "title": "已有项目",
                        "directory": "/workspace/project",
                    },
                }
            )
        await self.gates.setdefault(task_id, asyncio.Event()).wait()
        return {"content": f"done:{task_id}"}

    async def status(self, task_id: str | None = None, context: object = None) -> Any:
        return {"state": "working"}

    async def cancel(self, task_id: str, context: object = None) -> Any:
        self.gates.setdefault(task_id, asyncio.Event()).set()
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
        for gate in self.gates.values():
            gate.set()


@pytest.mark.asyncio
async def test_owner_fifo_allows_other_owners_to_run_concurrently() -> None:
    backend = LaneBackend()
    runtime = WorkGraph(
        backend=backend,
        scheduler=TaskScheduler(max_concurrent=3, max_concurrent_per_owner=1),
    )
    first = await runtime.submit(objective="A", owner_id="owner-a")
    second = await runtime.submit(objective="B", owner_id="owner-a")
    other = await runtime.submit(objective="C", owner_id="owner-b")

    await eventually(lambda: len(backend.started) == 2)
    assert backend.started == [first["id"], other["id"]]
    backend.gates[first["id"]].set()
    await eventually(lambda: second["id"] in backend.started)
    assert backend.started == [first["id"], other["id"], second["id"]]

    backend.gates[other["id"]].set()
    backend.gates[second["id"]].set()
    await asyncio.gather(
        runtime.wait(first["id"]),
        runtime.wait(second["id"]),
        runtime.wait(other["id"]),
    )
    await runtime.close()


@pytest.mark.asyncio
async def test_delegated_work_releases_owner_lane_without_exposing_private_ids() -> None:
    backend = LaneBackend()
    runtime = WorkGraph(backend=backend)
    first = await runtime.submit(objective="继续项目", owner_id="owner")
    backend.delegating.add(first["id"])
    second = await runtime.submit(objective="查询状态", owner_id="owner")

    await eventually(lambda: runtime.get(first["id"])["status"] == "delegated")
    await eventually(lambda: second["id"] in backend.started)
    projected = runtime.get(first["id"])
    assert projected is not None
    assert projected["delegation"] == {
        "status": "running",
        "title": "已有项目",
        "presentation": None,
    }
    internal = runtime.repository.get(first["id"])
    assert internal is not None
    assert internal.delegation == {
        "status": "running",
        "title": "已有项目",
        "presentation": None,
        "id": "private-run",
        "sessionId": "private-session",
        "directory": "/workspace/project",
    }

    backend.gates[first["id"]].set()
    backend.gates[second["id"]].set()
    await asyncio.gather(runtime.wait(first["id"]), runtime.wait(second["id"]))
    await runtime.close()


@pytest.mark.asyncio
async def test_repository_submission_key_is_atomic_and_prevents_duplicate_runs() -> None:
    backend = LaneBackend()
    repository = TaskRepository()
    runtime = WorkGraph(backend=backend, repository=repository)

    first, replay = await asyncio.gather(
        runtime.submit(
            objective="检查项目",
            owner_id="owner",
            submission_key="turn-1:call-1",
        ),
        runtime.submit(
            objective="检查项目",
            owner_id="owner",
            submission_key="turn-1:call-1",
        ),
    )
    assert first["id"] == replay["id"]
    assert {first["reused"], replay["reused"]} == {False, True}
    await eventually(lambda: backend.started == [first["id"]])

    backend.gates[first["id"]].set()
    await runtime.wait(first["id"])
    assert len(repository.values()) == 1
    await runtime.close()
