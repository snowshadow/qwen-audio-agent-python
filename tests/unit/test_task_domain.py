from __future__ import annotations

import json

import pytest

from qwen_audio_agent.domain.task import (
    Task,
    TaskScope,
    TaskStatus,
    TaskTransitionError,
    is_task_active,
    is_task_cancellable,
    is_task_terminal,
    persisted_task,
    public_task,
    task_from_persisted,
    transition_task,
)


def task(**overrides: object) -> Task:
    values = {
        "id": "task_1",
        "status": TaskStatus.QUEUED,
        "objective": "生成报告",
        "owner_id": "owner-one",
        "created_at": 10,
    }
    values.update(overrides)
    return Task(**values)  # type: ignore[arg-type]


def test_centralizes_active_cancellable_and_terminal_phases() -> None:
    assert is_task_active(TaskStatus.QUEUED)
    assert is_task_active(TaskStatus.CANCELLING)
    assert is_task_cancellable(TaskStatus.SCHEDULED)
    assert not is_task_cancellable(TaskStatus.CANCELLING)
    assert is_task_terminal(TaskStatus.COMPLETED)
    assert not is_task_terminal(TaskStatus.RUNNING)


def test_accepts_only_monotonic_task_transitions() -> None:
    work = task()
    transition_task(work, TaskStatus.RUNNING)
    transition_task(work, TaskStatus.DELEGATED)
    transition_task(work, TaskStatus.FINALIZING)
    transition_task(work, TaskStatus.COMPLETED)
    assert work.status == TaskStatus.COMPLETED

    with pytest.raises(TaskTransitionError, match="Invalid task transition"):
        transition_task(work, TaskStatus.RUNNING)
    with pytest.raises(TaskTransitionError, match="Unknown task transition"):
        transition_task({"status": "unknown"}, TaskStatus.RUNNING)


def test_public_projection_keeps_internal_phase_separate_from_work_state() -> None:
    work = task(
        status=TaskStatus.RUNNING,
        started_at=40,
        presentation={
            "speech": "报告已完成。",
            "inline": {"title": "报告", "format": "markdown", "content": "# 完成"},
        },
        delegation={
            "id": "private-delegation",
            "sessionId": "private-session",
            "title": "已有项目",
            "status": "running",
        },
    )

    projected = public_task(work, now=100)

    assert projected["workState"] == "working"
    assert projected["status"] == "running"
    assert projected["elapsedMs"] == 60
    assert projected["artifacts"] == [
        {
            "artifactId": "artifact_inline",
            "name": "报告",
            "parts": [{"text": "# 完成", "mediaType": "text/markdown"}],
        }
    ]
    assert projected["delegation"] == {
        "status": "running",
        "title": "已有项目",
        "presentation": None,
    }
    assert "sessionId" not in projected["delegation"]


def test_pending_authorization_projects_auth_required_and_task_identity() -> None:
    work = task(
        status=TaskStatus.RUNNING,
        authorization={
            "id": "auth_1",
            "status": "pending",
            "summary": "执行测试",
            "createdAt": 20,
        },
    )
    projected = public_task(work, now=30)
    assert projected["workState"] == "auth_required"
    assert projected["authorization"]["taskId"] == "task_1"
    assert projected["authorization"]["approvalScope"] == "once"


def test_authorization_is_closed_and_redacted_at_public_and_recovery_boundaries() -> None:
    authorization = {
        "id": "auth_secret",
        "taskId": "forged-task",
        "status": "pending",
        "category": "shell",
        "summary": "CLIENT_SECRET=hunter2 Authorization: Bearer super-secret-token",
        "patterns": [
            "GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz",
            "CREDENTIAL=secret-value",
            "Cookie: sid=abc123; theme=dark",
        ],
        "approvalScope": "persistent",
        "operation": {
            "title": "Run command",
            "kind": "execute",
            "command": (
                "deploy --password hunter2 --token=secret-token https://user:pass@example.com/x"
            ),
            "credential": "must-not-cross",
            "locations": [
                {
                    "path": "/repo?api_key=visible",
                    "line": 7,
                    "providerSessionId": "must-not-cross",
                }
            ],
        },
        "backendSessionId": "must-not-cross",
        "createdAt": 20,
    }
    projected = public_task(task(authorization=authorization), now=30)
    saved = persisted_task(task(authorization=authorization))
    restored = task_from_persisted({**saved, "authorization": authorization})

    for public in (projected["authorization"], saved["authorization"], restored.authorization):
        assert public is not None
        serialized = str(public)
        assert public["taskId"] == "task_1"
        assert public["approvalScope"] == "once"
        assert "super-secret-token" not in serialized
        assert "ghp_abcdefghijklmnopqrstuvwxyz" not in serialized
        assert "hunter2" not in serialized
        assert "secret-token" not in serialized
        assert "secret-value" not in serialized
        assert "sid=abc123" not in serialized
        assert "user:pass" not in serialized
        assert "api_key=visible" not in serialized
        assert "credential" not in serialized
        assert "providerSessionId" not in serialized
        assert "backendSessionId" not in serialized

    huge_numbers = {
        **authorization,
        "createdAt": 10**5_000,
        "operation": {
            **authorization["operation"],
            "locations": [{"path": "/repo/test.py", "line": 10**5_000}],
        },
    }
    huge_projected = public_task(task(authorization=huge_numbers), now=30)["authorization"]
    assert huge_projected is not None
    assert huge_projected["createdAt"] == 30
    assert "line" not in huge_projected["operation"]["locations"][0]
    json.dumps(huge_projected)


def test_public_projection_fail_safe_redacts_historical_activity() -> None:
    projected = public_task(
        task(
            activity=[
                {
                    "id": "old-tool",
                    "kind": "tool",
                    "detail": "GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz --password hunter2",
                    "rawInput": {"private": "must not cross"},
                }
            ]
        )
    )

    assert projected["activity"] == [
        {
            "id": "old-tool",
            "kind": "tool",
            "detail": "GITHUB_TOKEN=[REDACTED] --password [REDACTED]",
        }
    ]


def test_scope_defaults_to_user_and_persisted_projection_keeps_idempotency_key() -> None:
    old_record = {
        "id": "task_8",
        "status": "queued",
        "objective": "继续",
        "createdAt": 1,
    }
    assert public_task(old_record)["scope"] == TaskScope.USER

    work = task(
        submission_key="turn:call",
        delegation={
            "id": "private",
            "sessionId": "session-private",
            "directory": "/workspace/private",
            "title": "恢复项目",
        },
    )
    saved = persisted_task(work)
    assert "workState" not in saved
    assert saved["submissionKey"] == "turn:call"
    assert saved["delegation"] == {
        "status": "running",
        "title": "恢复项目",
        "presentation": None,
        "id": "private",
        "sessionId": "session-private",
        "directory": "/workspace/private",
    }
    restored = task_from_persisted(saved)
    assert restored.delegation == saved["delegation"]
    assert public_task(restored)["delegation"] == {
        "status": "running",
        "title": "恢复项目",
        "presentation": None,
    }


def test_schedule_is_closed_and_bounded_across_public_persistence_and_recovery() -> None:
    malformed_schedule = {
        "type": "at" * 1_000,
        "at": {"nested": "must-not-cross" * 1_000},
        "recurrence": ["must-not-cross" * 1_000],
        "providerState": "must-not-cross" * 1_000,
    }
    work = task(schedule=malformed_schedule)

    projected = public_task(work)
    saved = persisted_task(work)
    restored = task_from_persisted({**saved, "schedule": malformed_schedule})

    assert projected["schedule"] == {"type": ("at" * 1_000)[:40]}
    assert saved["schedule"] == projected["schedule"]
    assert restored.schedule == projected["schedule"]
    assert "must-not-cross" not in str(projected["schedule"])
    assert public_task(task(schedule={"at": 10**1_000}))["schedule"] is None
