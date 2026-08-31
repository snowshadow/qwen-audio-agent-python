from pathlib import Path

from qwen_audio_agent.domain.task import Task, TaskStatus, persisted_task
from qwen_audio_agent.runtime.task_store import JsonTaskStore


def test_task_store_round_trip_preserves_recovery_and_delivery_fields(tmp_path: Path) -> None:
    store = JsonTaskStore(tmp_path / "state" / "tasks.json")
    task = Task(
        id="task_7",
        status=TaskStatus.DELEGATED,
        objective="继续整理项目",
        owner_id="owner",
        submission_key="turn:call",
        lane_key="backend:owner",
        input_parts=[{"text": "evidence"}],
        delegation={"status": "running", "privateResumeKey": "adapter-owned"},
        notification_status="delivering",
        notification_claimant_id="desktop",
        notification_claimed_at=123,
    )

    store.save([persisted_task(task)], {"nextTaskNumber": 8})
    restored = store.load()

    assert store.next_task_number == 8
    assert len(restored) == 1
    assert restored[0].status is TaskStatus.DELEGATED
    assert restored[0].submission_key == "turn:call"
    assert restored[0].input_parts == [{"text": "evidence"}]
    assert "privateResumeKey" not in (restored[0].delegation or {})
    assert restored[0].notification_claimant_id == "desktop"
    assert (tmp_path / "state" / "tasks.json").stat().st_mode & 0o777 == 0o600
