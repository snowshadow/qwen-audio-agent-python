from qwen_audio_agent.runtime.delivery import (
    DeliveryLeaseQueue,
    DeliveryRecord,
    DeliveryStatus,
)


def test_result_is_claimed_and_acknowledged_exactly_once() -> None:
    now = [10.0]
    events: list[tuple[str, str]] = []
    records = {
        "task_1": DeliveryRecord("task_1", "owner", "voice-a", created_at=1.0),
    }
    queue = DeliveryLeaseQueue(
        records,
        now=lambda: now[0],
        persist=lambda: events.append(("persist", records["task_1"].status)),
        on_delivered=lambda record: events.append(("event", record.status)),
    )

    first = queue.claim(owner_id="owner", session_id="voice-a", claimant_id="client-a")
    second = queue.claim(owner_id="owner", session_id="voice-a", claimant_id="client-b")

    assert [record.task_id for record in first] == ["task_1"]
    assert second == []
    assert queue.mark_delivered(["task_1"], claimant_id="client-b") == 0
    assert queue.mark_delivered(["task_1"], claimant_id="client-a") == 1
    assert queue.mark_delivered(["task_1"], claimant_id="client-a") == 0
    assert events[-2:] == [
        ("persist", DeliveryStatus.DELIVERED),
        ("event", DeliveryStatus.DELIVERED),
    ]


def test_expired_or_released_claim_can_be_recovered_by_same_owner() -> None:
    now = [10.0]
    records = {
        "task_1": DeliveryRecord("task_1", "owner", "voice-a", created_at=1.0),
    }
    queue = DeliveryLeaseQueue(records, claim_ttl_seconds=5.0, now=lambda: now[0])
    queue.claim(owner_id="owner", session_id="voice-a", claimant_id="client-a")

    assert queue.release(["task_1"], claimant_id="not-owner") == 0
    now[0] = 16.0
    recovered = queue.claim(
        owner_id="owner",
        session_id="voice-b",
        include_other_sessions=True,
        claimant_id="client-b",
    )

    assert [record.task_id for record in recovered] == ["task_1"]
    assert records["task_1"].claimant_id == "client-b"


def test_claim_is_owner_and_origin_session_scoped_by_default() -> None:
    records = {
        "task_1": DeliveryRecord("task_1", "owner", "voice-a", created_at=1.0),
        "task_2": DeliveryRecord("task_2", "other", "voice-a", created_at=2.0),
    }
    queue = DeliveryLeaseQueue(records)

    assert queue.claim(owner_id="owner", session_id="voice-b", claimant_id="client") == []
    claimed = queue.claim(owner_id="owner", session_id="voice-a", claimant_id="client")

    assert [record.task_id for record in claimed] == ["task_1"]
    assert records["task_2"].status is DeliveryStatus.PENDING
