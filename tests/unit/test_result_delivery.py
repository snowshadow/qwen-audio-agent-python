from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from qwen_audio_agent.runtime.delivery import (
    DeliveryLeaseQueue,
    DeliveryRecord,
    DeliveryStatus,
)
from qwen_audio_agent.runtime.result_delivery import ResultDeliveryCoordinator


async def eventually(predicate: Callable[[], bool], deadline_seconds: float = 1.0) -> None:
    async with asyncio.timeout(deadline_seconds):
        while not predicate():  # noqa: ASYNC110 - polling an in-memory coordinator
            await asyncio.sleep(0.001)


def record(
    task_id: str,
    *,
    owner_id: str = "owner",
    session_id: str = "voice-a",
    created_at: float = 1.0,
) -> DeliveryRecord:
    return DeliveryRecord(task_id, owner_id, session_id, created_at=created_at)


@pytest.mark.asyncio
async def test_claim_waits_for_safe_window_and_acknowledges_only_playback_start() -> None:
    records = {"task_1": record("task_1")}
    blocked = [True]
    presentations: list[tuple[str, dict[str, Any], bool]] = []

    async def present(text: str, context: dict[str, Any], inject: bool) -> dict[str, bool]:
        presentations.append((text, context, inject))
        return {"completed": True, "contextInjected": True}

    coordinator = ResultDeliveryCoordinator(
        leases=DeliveryLeaseQueue(records),
        owner_id="owner",
        session_id="voice-a",
        claimant_id="client-a",
        present_result=present,
        is_delivery_blocked=lambda: blocked[0],
        batch_window_seconds=0,
        retry_base_seconds=0.005,
        acknowledgement_timeout_seconds=1,
        lease_renew_interval_seconds=1,
        create_turn_id=lambda: "gateway-result-turn",
    )
    coordinator.completed({"id": "task_1", "objective": "生成报告", "result": "已完成"})
    await asyncio.sleep(0.02)
    assert presentations == []
    assert records["task_1"].status is DeliveryStatus.PENDING

    blocked[0] = False
    await eventually(lambda: len(presentations) == 1)
    assert len(presentations) == 1
    assert presentations[0][1] == {
        "turnId": "gateway-result-turn",
        "taskId": "task_1",
        "taskIds": ["task_1"],
    }
    assert records["task_1"].status is DeliveryStatus.DELIVERING

    # Generation completion is not delivery. The durable acknowledgement is
    # the active output client's playback-start receipt.
    await asyncio.sleep(0)
    assert records["task_1"].status is DeliveryStatus.DELIVERING
    assert await coordinator.playback_started(["task_1"]) == 1
    assert await coordinator.playback_started(["task_1"]) == 0
    assert records["task_1"].status is DeliveryStatus.DELIVERED
    await coordinator.close()


@pytest.mark.asyncio
async def test_failure_retries_then_releases_poison_so_later_result_can_run() -> None:
    records = {
        "poison": record("poison", created_at=1),
        "healthy": record("healthy", created_at=2),
    }
    attempts: list[str] = []

    async def present(text: str, _context: dict[str, Any], _inject: bool) -> dict[str, bool]:
        attempts.append(text)
        return {"completed": "好结果" in text, "contextInjected": True}

    coordinator = ResultDeliveryCoordinator(
        leases=DeliveryLeaseQueue(records),
        owner_id="owner",
        session_id="voice-a",
        claimant_id="client-a",
        present_result=present,
        is_delivery_blocked=lambda: False,
        max_batch_items=1,
        batch_window_seconds=0,
        retry_base_seconds=0.001,
        retry_max_seconds=0.001,
        max_retry_attempts=1,
        acknowledgement_timeout_seconds=1,
        lease_renew_interval_seconds=1,
    )
    coordinator.completed({"id": "poison", "objective": "坏任务", "result": "坏结果"})
    coordinator.completed({"id": "healthy", "objective": "好任务", "result": "好结果"})

    await eventually(lambda: any("好结果" in value for value in attempts))
    assert sum("坏结果" in value for value in attempts) == 2
    assert records["poison"].status is DeliveryStatus.PENDING
    assert records["healthy"].status is DeliveryStatus.DELIVERING
    assert coordinator.active_task_ids == ("healthy",)

    assert await coordinator.playback_started(["healthy"]) == 1
    assert records["healthy"].status is DeliveryStatus.DELIVERED
    await coordinator.close()


@pytest.mark.asyncio
async def test_expired_lease_is_reclaimed_and_stale_client_cannot_acknowledge() -> None:
    now = [10.0]
    records = {"task_1": record("task_1")}
    first_presented = asyncio.Event()
    second_presented = asyncio.Event()

    async def first_present(
        _text: str,
        _context: dict[str, Any],
        _inject: bool,
    ) -> dict[str, bool]:
        first_presented.set()
        return {"completed": True}

    async def second_present(
        _text: str,
        _context: dict[str, Any],
        _inject: bool,
    ) -> dict[str, bool]:
        second_presented.set()
        return {"completed": True}

    first = ResultDeliveryCoordinator(
        leases=DeliveryLeaseQueue(records, claim_ttl_seconds=5, now=lambda: now[0]),
        owner_id="owner",
        session_id="voice-a",
        claimant_id="client-a",
        present_result=first_present,
        is_delivery_blocked=lambda: False,
        batch_window_seconds=0,
        acknowledgement_timeout_seconds=10,
        lease_renew_interval_seconds=10,
    )
    first.completed({"id": "task_1", "objective": "A", "result": "done"})
    await first_presented.wait()
    assert records["task_1"].claimant_id == "client-a"

    now[0] = 16.0
    second = ResultDeliveryCoordinator(
        leases=DeliveryLeaseQueue(records, claim_ttl_seconds=5, now=lambda: now[0]),
        owner_id="owner",
        session_id="voice-a",
        claimant_id="client-b",
        present_result=second_present,
        is_delivery_blocked=lambda: False,
        batch_window_seconds=0,
        acknowledgement_timeout_seconds=10,
        lease_renew_interval_seconds=10,
    )
    second.completed({"id": "task_1", "objective": "A", "result": "done"})
    await second_presented.wait()

    assert records["task_1"].claimant_id == "client-b"
    assert await first.playback_started(["task_1"]) == 0
    assert await second.playback_started(["task_1"]) == 1
    assert records["task_1"].status is DeliveryStatus.DELIVERED
    await first.close()
    await second.close()


@pytest.mark.asyncio
async def test_playback_can_start_before_generation_finishes_without_blocking_next() -> None:
    records = {
        "first": record("first", created_at=1),
        "second": record("second", created_at=2),
    }
    first_started = asyncio.Event()
    finish_first = asyncio.Event()
    second_started = asyncio.Event()

    async def present(text: str, _context: dict[str, Any], _inject: bool) -> dict[str, bool]:
        if "task_id: first" in text:
            first_started.set()
            await finish_first.wait()
        else:
            second_started.set()
        return {"completed": True}

    coordinator = ResultDeliveryCoordinator(
        leases=DeliveryLeaseQueue(records),
        owner_id="owner",
        session_id="voice-a",
        claimant_id="client-a",
        present_result=present,
        is_delivery_blocked=lambda: False,
        max_batch_items=1,
        batch_window_seconds=0,
        acknowledgement_timeout_seconds=1,
        lease_renew_interval_seconds=1,
    )
    coordinator.completed({"id": "first", "result": "one"})
    coordinator.completed({"id": "second", "result": "two"})
    await first_started.wait()

    assert await coordinator.playback_started(["first"]) == 1
    assert records["first"].status is DeliveryStatus.DELIVERED
    assert not second_started.is_set()

    finish_first.set()
    await second_started.wait()
    assert coordinator.active_task_ids == ("second",)
    assert await coordinator.playback_started(["second"]) == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_new_completion_waits_behind_active_unacknowledged_result() -> None:
    records = {
        "first": record("first", created_at=1),
        "second": record("second", created_at=2),
    }
    attempts: list[str] = []

    async def present(text: str, _context: dict[str, Any], _inject: bool) -> dict[str, bool]:
        attempts.append(text)
        return {"completed": True}

    coordinator = ResultDeliveryCoordinator(
        leases=DeliveryLeaseQueue(records),
        owner_id="owner",
        session_id="voice-a",
        claimant_id="client-a",
        present_result=present,
        is_delivery_blocked=lambda: False,
        max_batch_items=1,
        batch_window_seconds=0,
        acknowledgement_timeout_seconds=1,
        lease_renew_interval_seconds=1,
    )
    coordinator.completed({"id": "first", "result": "one"})
    await eventually(lambda: len(attempts) == 1)

    coordinator.completed({"id": "second", "result": "two"})
    await asyncio.sleep(0.02)
    assert len(attempts) == 1
    assert records["second"].status is DeliveryStatus.PENDING

    assert await coordinator.playback_started(["first"]) == 1
    await eventually(lambda: len(attempts) == 2)
    assert "task_id: second" in attempts[1]
    assert await coordinator.playback_started(["second"]) == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_playback_failure_during_generation_requests_one_retry() -> None:
    records = {"task_1": record("task_1")}
    first_started = asyncio.Event()
    finish_first = asyncio.Event()
    attempts = 0

    async def present(
        _text: str,
        _context: dict[str, Any],
        _inject: bool,
    ) -> dict[str, bool]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            first_started.set()
            await finish_first.wait()
        return {"completed": True}

    coordinator = ResultDeliveryCoordinator(
        leases=DeliveryLeaseQueue(records),
        owner_id="owner",
        session_id="voice-a",
        claimant_id="client-a",
        present_result=present,
        is_delivery_blocked=lambda: False,
        batch_window_seconds=0,
        retry_base_seconds=0.001,
        retry_max_seconds=0.001,
        acknowledgement_timeout_seconds=1,
        lease_renew_interval_seconds=1,
    )
    coordinator.completed({"id": "task_1", "result": "done"})
    await first_started.wait()

    await coordinator.playback_failed(["task_1"])
    finish_first.set()
    await eventually(lambda: attempts == 2)
    assert records["task_1"].status is DeliveryStatus.DELIVERING
    assert await coordinator.playback_started(["task_1"]) == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_lease_loss_cancels_stale_attempt_and_continues_with_later_result() -> None:
    now = [10.0]
    records = {
        "stale": record("stale", created_at=1),
        "healthy": record("healthy", created_at=2),
    }
    stale_started = asyncio.Event()
    stale_cancelled = asyncio.Event()
    healthy_started = asyncio.Event()
    never_finishes = asyncio.Event()

    async def present(text: str, _context: dict[str, Any], _inject: bool) -> dict[str, bool]:
        if "task_id: stale" in text:
            stale_started.set()
            try:
                await never_finishes.wait()
            except asyncio.CancelledError:
                stale_cancelled.set()
                raise
        healthy_started.set()
        return {"completed": True}

    lease_queue = DeliveryLeaseQueue(records, claim_ttl_seconds=5, now=lambda: now[0])
    coordinator = ResultDeliveryCoordinator(
        leases=lease_queue,
        owner_id="owner",
        session_id="voice-a",
        claimant_id="client-a",
        present_result=present,
        is_delivery_blocked=lambda: False,
        max_batch_items=1,
        batch_window_seconds=0,
        acknowledgement_timeout_seconds=1,
        lease_renew_interval_seconds=0.005,
    )
    coordinator.completed({"id": "stale", "result": "old"})
    coordinator.completed({"id": "healthy", "result": "new"})
    await stale_started.wait()

    now[0] = 16.0
    replacement_queue = DeliveryLeaseQueue(
        records,
        claim_ttl_seconds=5,
        now=lambda: now[0],
    )
    replacement = replacement_queue.claim(
        owner_id="owner",
        session_id="voice-a",
        claimant_id="client-b",
        task_ids=["stale"],
    )
    assert [item.task_id for item in replacement] == ["stale"]

    await eventually(lambda: stale_cancelled.is_set() and healthy_started.is_set())
    assert records["stale"].claimant_id == "client-b"
    assert records["healthy"].claimant_id == "client-a"
    assert coordinator.active_task_ids == ("healthy",)
    assert await coordinator.playback_started(["stale"]) == 0
    assert await coordinator.playback_started(["healthy"]) == 1

    replacement_queue.release(["stale"], claimant_id="client-b")
    await coordinator.close()


@pytest.mark.asyncio
async def test_lost_lease_retries_after_new_claimant_fails_without_reconnect() -> None:
    now = [10.0]
    records = {
        "transferred": record("transferred", created_at=1),
        "later": record("later", created_at=2),
    }
    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()
    later_started = asyncio.Event()
    reclaimed_started = asyncio.Event()
    new_claimant_started = asyncio.Event()
    fail_new_claimant = asyncio.Event()
    first_attempts = 0

    async def first_present(
        text: str,
        _context: dict[str, Any],
        _inject: bool,
    ) -> dict[str, bool]:
        nonlocal first_attempts
        if "task_id: transferred" in text:
            first_attempts += 1
            if first_attempts == 1:
                first_started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    first_cancelled.set()
                    raise
            reclaimed_started.set()
        else:
            later_started.set()
        return {"completed": True}

    async def second_present(
        _text: str,
        _context: dict[str, Any],
        _inject: bool,
    ) -> dict[str, bool]:
        new_claimant_started.set()
        await fail_new_claimant.wait()
        return {"completed": False}

    first_queue = DeliveryLeaseQueue(records, claim_ttl_seconds=5, now=lambda: now[0])
    first = ResultDeliveryCoordinator(
        leases=first_queue,
        owner_id="owner",
        session_id="voice-a",
        claimant_id="client-a",
        present_result=first_present,
        is_delivery_blocked=lambda: False,
        max_batch_items=1,
        batch_window_seconds=0,
        retry_base_seconds=0.001,
        retry_max_seconds=0.001,
        acknowledgement_timeout_seconds=1,
        lease_renew_interval_seconds=0.005,
    )
    first.completed({"id": "transferred", "result": "old"})
    first.completed({"id": "later", "result": "new"})
    await first_started.wait()

    now[0] = 16.0
    second = ResultDeliveryCoordinator(
        leases=DeliveryLeaseQueue(records, claim_ttl_seconds=5, now=lambda: now[0]),
        owner_id="owner",
        session_id="voice-a",
        claimant_id="client-b",
        present_result=second_present,
        is_delivery_blocked=lambda: False,
        max_batch_items=1,
        batch_window_seconds=0,
        retry_base_seconds=0.001,
        retry_max_seconds=0.001,
        max_retry_attempts=0,
        acknowledgement_timeout_seconds=1,
        lease_renew_interval_seconds=1,
    )
    second.completed({"id": "transferred", "result": "old"})
    await new_claimant_started.wait()
    await eventually(lambda: first_cancelled.is_set() and later_started.is_set())

    assert records["transferred"].claimant_id == "client-b"
    assert await first.playback_started(["later"]) == 1
    await asyncio.sleep(0.08)
    assert first_attempts == 1
    assert "transferred" in first.pending_task_ids

    fail_new_claimant.set()
    await eventually(lambda: records["transferred"].status is DeliveryStatus.PENDING)
    await eventually(lambda: reclaimed_started.is_set())

    assert records["transferred"].claimant_id == "client-a"
    assert first_attempts == 2
    assert await first.playback_started(["transferred"]) == 1
    await first.close()
    await second.close()


@pytest.mark.asyncio
async def test_ack_timeout_never_retries_while_audio_window_is_blocked() -> None:
    records = {"task_1": record("task_1")}
    blocked = [False]
    attempts = 0

    async def present(
        _text: str,
        _context: dict[str, Any],
        _inject: bool,
    ) -> dict[str, bool]:
        nonlocal attempts
        attempts += 1
        blocked[0] = True
        return {"completed": True, "contextInjected": True}

    coordinator = ResultDeliveryCoordinator(
        leases=DeliveryLeaseQueue(records),
        owner_id="owner",
        session_id="voice-a",
        claimant_id="client-a",
        present_result=present,
        is_delivery_blocked=lambda: blocked[0],
        batch_window_seconds=0,
        retry_base_seconds=0.001,
        retry_max_seconds=0.001,
        acknowledgement_timeout_seconds=0.005,
        lease_renew_interval_seconds=1,
    )
    coordinator.completed({"id": "task_1", "objective": "A", "result": "done"})
    await eventually(lambda: attempts == 1)
    await asyncio.sleep(0.02)
    assert attempts == 1

    blocked[0] = False
    await eventually(lambda: attempts == 2)
    assert records["task_1"].status is DeliveryStatus.DELIVERING
    await coordinator.playback_started(["task_1"])
    await coordinator.close()


@pytest.mark.asyncio
async def test_pause_releases_inflight_claim_and_ignores_late_generation() -> None:
    records = {"task_1": record("task_1")}
    release_presentation = asyncio.Event()
    presentation_started = asyncio.Event()

    async def present(
        _text: str,
        _context: dict[str, Any],
        _inject: bool,
    ) -> dict[str, bool]:
        presentation_started.set()
        await release_presentation.wait()
        return {"completed": True}

    coordinator = ResultDeliveryCoordinator(
        leases=DeliveryLeaseQueue(records),
        owner_id="owner",
        session_id="voice-a",
        claimant_id="client-a",
        present_result=present,
        is_delivery_blocked=lambda: False,
        batch_window_seconds=0,
        acknowledgement_timeout_seconds=1,
        lease_renew_interval_seconds=1,
    )
    coordinator.completed({"id": "task_1", "objective": "A", "result": "done"})
    await presentation_started.wait()
    await coordinator.pause()
    release_presentation.set()
    await asyncio.sleep(0)

    assert records["task_1"].status is DeliveryStatus.PENDING
    assert coordinator.active_task_ids == ()
    assert await coordinator.playback_started(["task_1"]) == 0
    await coordinator.close()


@pytest.mark.asyncio
async def test_delivered_rows_from_another_claimant_do_not_block_new_results() -> None:
    records = {
        **{
            f"stale-{index}": record(f"stale-{index}", created_at=float(index))
            for index in range(8)
        },
        "fresh": record("fresh", created_at=9),
    }
    for item in records.values():
        if item.task_id.startswith("stale-"):
            item.status = DeliveryStatus.DELIVERED
    presentations: list[str] = []

    async def present(
        text: str,
        _context: dict[str, Any],
        _inject: bool,
    ) -> dict[str, bool]:
        presentations.append(text)
        return {"completed": True}

    coordinator = ResultDeliveryCoordinator(
        leases=DeliveryLeaseQueue(records),
        owner_id="owner",
        session_id="voice-a",
        claimant_id="loser",
        present_result=present,
        is_delivery_blocked=lambda: False,
        max_batch_items=8,
        batch_window_seconds=0,
        retry_base_seconds=10,
        retry_max_seconds=10,
        acknowledgement_timeout_seconds=1,
        lease_renew_interval_seconds=1,
    )
    stale_ids = [f"stale-{index}" for index in range(8)]
    for task_id in stale_ids:
        coordinator.completed({"id": task_id, "result": "old"})
    coordinator.completed({"id": "fresh", "result": "new"})

    await asyncio.sleep(0.02)
    assert presentations == []
    await coordinator.discard_delivered(stale_ids)
    await eventually(lambda: len(presentations) == 1)

    assert "task_id: fresh" in presentations[0]
    assert all(task_id not in coordinator.pending_task_ids for task_id in stale_ids)
    assert await coordinator.playback_started(["fresh"]) == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_delivered_event_cancels_an_active_stale_batch_and_continues() -> None:
    records = {
        "stale": record("stale", created_at=1),
        "fresh": record("fresh", created_at=2),
    }
    stale_started = asyncio.Event()
    stale_cancelled = asyncio.Event()
    fresh_started = asyncio.Event()

    async def present(
        text: str,
        _context: dict[str, Any],
        _inject: bool,
    ) -> dict[str, bool]:
        if "task_id: stale" in text:
            stale_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                stale_cancelled.set()
                raise
        fresh_started.set()
        return {"completed": True}

    coordinator = ResultDeliveryCoordinator(
        leases=DeliveryLeaseQueue(records),
        owner_id="owner",
        session_id="voice-a",
        claimant_id="loser",
        present_result=present,
        is_delivery_blocked=lambda: False,
        max_batch_items=1,
        batch_window_seconds=0,
        acknowledgement_timeout_seconds=1,
        lease_renew_interval_seconds=1,
    )
    coordinator.completed({"id": "stale", "result": "old"})
    coordinator.completed({"id": "fresh", "result": "new"})
    await stale_started.wait()

    records["stale"].status = DeliveryStatus.DELIVERED
    records["stale"].claimant_id = None
    records["stale"].claimed_at = None
    await coordinator.discard_delivered(["stale"])
    await eventually(lambda: stale_cancelled.is_set() and fresh_started.is_set())

    assert coordinator.active_task_ids == ("fresh",)
    assert await coordinator.playback_started(["fresh"]) == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_artifact_only_completion_uses_bounded_descriptors_without_payloads() -> None:
    records = {"artifact-task": record("artifact-task")}
    presentations: list[str] = []

    async def present(
        text: str,
        _context: dict[str, Any],
        _inject: bool,
    ) -> dict[str, bool]:
        presentations.append(text)
        return {"completed": True}

    coordinator = ResultDeliveryCoordinator(
        leases=DeliveryLeaseQueue(records),
        owner_id="owner",
        session_id="voice-a",
        claimant_id="client-a",
        present_result=present,
        is_delivery_blocked=lambda: False,
        batch_window_seconds=0,
        acknowledgement_timeout_seconds=1,
        lease_renew_interval_seconds=1,
    )
    coordinator.completed(
        {
            "id": "artifact-task",
            "result": "",
            "artifacts": [
                {
                    "artifactId": "report",
                    "name": "分析报告",
                    "parts": [
                        {"mediaType": "application/pdf", "raw": "SECRET_RAW"},
                        {
                            "mediaType": "application/json",
                            "data": {"secret": "SECRET_DATA"},
                        },
                    ],
                }
            ],
        }
    )
    await eventually(lambda: len(presentations) == 1)

    assert "已生成 1 个产物" in presentations[0]
    assert "分析报告 (application/pdf、application/json)" in presentations[0]
    assert "SECRET_RAW" not in presentations[0]
    assert "SECRET_DATA" not in presentations[0]
    assert "结果: \n" not in presentations[0]
    assert await coordinator.playback_started(["artifact-task"]) == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_empty_result_prefers_presentation_speech_over_artifact_summary() -> None:
    records = {"speech-task": record("speech-task")}
    presentations: list[str] = []

    async def present(
        text: str,
        _context: dict[str, Any],
        _inject: bool,
    ) -> dict[str, bool]:
        presentations.append(text)
        return {"completed": True}

    coordinator = ResultDeliveryCoordinator(
        leases=DeliveryLeaseQueue(records),
        owner_id="owner",
        session_id="voice-a",
        claimant_id="client-a",
        present_result=present,
        is_delivery_blocked=lambda: False,
        batch_window_seconds=0,
        acknowledgement_timeout_seconds=1,
        lease_renew_interval_seconds=1,
    )
    coordinator.completed(
        {
            "id": "speech-task",
            "result": "",
            "presentation": {"speech": "报告已生成，可以打开查看。"},
            "artifacts": [
                {
                    "artifactId": "report",
                    "name": "不应该出现的摘要",
                    "parts": [{"mediaType": "application/pdf", "raw": "SECRET"}],
                }
            ],
        }
    )
    await eventually(lambda: len(presentations) == 1)

    assert "结果: 报告已生成，可以打开查看。" in presentations[0]
    assert "不应该出现的摘要" not in presentations[0]
    assert "SECRET" not in presentations[0]
    assert await coordinator.playback_started(["speech-task"]) == 1
    await coordinator.close()
