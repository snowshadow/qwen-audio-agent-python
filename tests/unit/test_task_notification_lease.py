from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from qwen_audio_agent.domain.task import Task, TaskStatus
from qwen_audio_agent.runtime.delivery import DeliveryStatus
from qwen_audio_agent.runtime.repository import TaskRepository
from qwen_audio_agent.runtime.result_delivery import ResultDeliveryCoordinator
from qwen_audio_agent.runtime.task_notification_lease import TaskNotificationLeaseAdapter
from qwen_audio_agent.runtime.task_store import JsonTaskStore


def completed_task(
    task_id: str,
    *,
    owner_id: str = "owner",
    session_id: str = "voice-a",
    created_at: int = 1,
    notification_status: str = "pending",
    claimant_id: str | None = None,
    claimed_at: int | None = None,
) -> Task:
    return Task(
        id=task_id,
        status=TaskStatus.COMPLETED,
        objective=f"work {task_id}",
        owner_id=owner_id,
        session_id=session_id,
        created_at=created_at,
        completed_at=created_at + 1,
        result=f"result {task_id}",
        notification_status=notification_status,
        notification_claimant_id=claimant_id,
        notification_claimed_at=claimed_at,
    )


def repository_at(path: Path, *tasks: Task) -> TaskRepository[Task]:
    repository = TaskRepository[Task](store=JsonTaskStore(path))
    for task in tasks:
        repository.add(task)
    if tasks:
        repository.save()
    return repository


def restored_repository(path: Path) -> TaskRepository[Task]:
    repository = TaskRepository[Task](store=JsonTaskStore(path))
    repository.load()
    return repository


def test_claims_task_fields_in_creation_order_and_persists_scope(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    repository = repository_at(
        path,
        completed_task("later", created_at=2),
        completed_task("earlier", created_at=1),
        completed_task("other-session", session_id="voice-b"),
        completed_task("other-owner", owner_id="other"),
        completed_task("silent", notification_status="none"),
    )
    adapter = TaskNotificationLeaseAdapter(repository, clock_ms=lambda: 100)

    claimed = adapter.claim(
        owner_id="owner",
        session_id="voice-a",
        claimant_id="desktop",
    )

    assert [record.task_id for record in claimed] == ["earlier", "later"]
    assert all(record.status is DeliveryStatus.DELIVERING for record in claimed)
    assert repository.get("other-session").notification_status == "pending"  # type: ignore[union-attr]
    assert repository.get("silent").notification_status == "none"  # type: ignore[union-attr]

    restored = restored_repository(path)
    assert restored.get("earlier").notification_claimant_id == "desktop"  # type: ignore[union-attr]
    assert restored.get("later").notification_claimed_at == 100  # type: ignore[union-attr]


def test_cross_session_recovery_requires_explicit_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    repository = repository_at(
        path,
        completed_task("old-result", session_id="old-voice"),
    )
    adapter = TaskNotificationLeaseAdapter(repository, clock_ms=lambda: 100)

    assert (
        adapter.claim(
            owner_id="owner",
            session_id="new-voice",
            claimant_id="desktop",
        )
        == []
    )
    recovered = adapter.claim(
        owner_id="owner",
        session_id="new-voice",
        include_other_sessions=True,
        claimant_id="desktop",
    )
    assert [record.task_id for record in recovered] == ["old-result"]


def test_two_adapters_cannot_claim_the_same_repository_task(tmp_path: Path) -> None:
    repository = repository_at(tmp_path / "tasks.json", completed_task("task_1"))
    first = TaskNotificationLeaseAdapter(repository, clock_ms=lambda: 100)
    second = TaskNotificationLeaseAdapter(repository, clock_ms=lambda: 100)
    ready = Barrier(2)

    def claim(adapter: TaskNotificationLeaseAdapter, claimant_id: str) -> list[str]:
        ready.wait()
        return [
            record.task_id
            for record in adapter.claim(
                owner_id="owner",
                session_id="voice-a",
                claimant_id=claimant_id,
            )
        ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda arguments: claim(*arguments),
                [(first, "first"), (second, "second")],
            )
        )

    assert sorted(len(result) for result in results) == [0, 1]
    assert sum(result == ["task_1"] for result in results) == 1


def test_renew_release_and_delivery_persist_only_for_lease_owner(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    repository = repository_at(
        path,
        completed_task(
            "task_1",
            notification_status="delivering",
            claimant_id="desktop",
            claimed_at=10,
        ),
    )
    now = [100]
    observed_after_save: list[tuple[str, DeliveryStatus]] = []

    def on_delivered(record: object) -> None:
        restored = restored_repository(path).get("task_1")
        assert restored is not None
        observed_after_save.append((restored.notification_status, record.status))  # type: ignore[attr-defined]

    adapter = TaskNotificationLeaseAdapter(
        repository,
        claim_ttl_seconds=10,
        clock_ms=lambda: now[0],
        on_delivered=on_delivered,
    )

    assert adapter.renew(["task_1"], claimant_id="other") == 0
    assert adapter.renew(["task_1"], claimant_id="desktop") == 1
    assert restored_repository(path).get("task_1").notification_claimed_at == 100  # type: ignore[union-attr]

    assert adapter.release(["task_1"], claimant_id="other") == 0
    assert adapter.release(["task_1"], claimant_id="desktop") == 1
    assert restored_repository(path).get("task_1").notification_status == "pending"  # type: ignore[union-attr]

    adapter.claim(owner_id="owner", claimant_id="desktop", task_ids=["task_1"])
    now[0] = 120
    assert adapter.mark_delivered(["task_1"], claimant_id="other") == 0
    assert adapter.mark_delivered(["task_1"], claimant_id="desktop") == 1
    assert adapter.mark_delivered(["task_1"], claimant_id="desktop") == 0
    assert observed_after_save == [("delivered", DeliveryStatus.DELIVERED)]


def test_expired_persisted_claim_is_reclaimed_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    now = [1_000]
    first_repository = repository_at(path, completed_task("task_1"))
    first = TaskNotificationLeaseAdapter(
        first_repository,
        claim_ttl_seconds=0.05,
        clock_ms=lambda: now[0],
    )
    first.claim(owner_id="owner", session_id="voice-a", claimant_id="stale")

    now[0] = 1_060
    second_repository = restored_repository(path)
    second = TaskNotificationLeaseAdapter(
        second_repository,
        claim_ttl_seconds=0.05,
        clock_ms=lambda: now[0],
    )
    reclaimed = second.claim(
        owner_id="owner",
        session_id="voice-a",
        claimant_id="replacement",
    )

    assert [record.task_id for record in reclaimed] == ["task_1"]
    assert second.mark_delivered(["task_1"], claimant_id="stale") == 0
    restored = restored_repository(path).get("task_1")
    assert restored is not None
    assert restored.notification_status == "delivering"
    assert restored.notification_claimant_id == "replacement"


@pytest.mark.asyncio
async def test_adapter_drives_result_coordinator_without_a_second_ledger(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    task = completed_task("task_1")
    repository = repository_at(path, task)
    adapter = TaskNotificationLeaseAdapter(repository, clock_ms=lambda: 100)
    presented = asyncio.Event()

    async def present(
        _text: str,
        _context: dict[str, object],
        _inject_context: bool,
    ) -> dict[str, bool]:
        presented.set()
        return {"completed": True, "contextInjected": True}

    coordinator = ResultDeliveryCoordinator(
        leases=adapter,  # type: ignore[arg-type] - structural DeliveryLeaseQueue interface
        owner_id="owner",
        session_id="voice-a",
        claimant_id="desktop",
        present_result=present,
        is_delivery_blocked=lambda: False,
        batch_window_seconds=0,
        acknowledgement_timeout_seconds=1,
        lease_renew_interval_seconds=1,
    )
    coordinator.completed(task)
    await presented.wait()

    assert repository.get("task_1") is task
    assert task.notification_status == "delivering"
    assert await coordinator.playback_started(["task_1"]) == 1
    assert task.notification_status == "delivered"
    assert restored_repository(path).get("task_1").notification_status == "delivered"  # type: ignore[union-attr]
    await coordinator.close()
