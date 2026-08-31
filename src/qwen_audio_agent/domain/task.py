"""Task lifecycle and the stable public Work projection.

The Gateway Task is a delivery receipt.  It deliberately does not mirror a
backend agent's internal graph, session topology, or protocol identifiers.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from qwen_audio_agent.public_safety import (
    normalize_public_authorization,
    redact_public_text,
)


class TaskStatus(StrEnum):
    SCHEDULED = "scheduled"
    QUEUED = "queued"
    RUNNING = "running"
    DELEGATED = "delegated"
    FINALIZING = "finalizing"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskScope(StrEnum):
    USER = "user"
    SYSTEM = "system"


ACTIVE_TASK_STATUSES = frozenset(
    {
        TaskStatus.QUEUED,
        TaskStatus.RUNNING,
        TaskStatus.DELEGATED,
        TaskStatus.FINALIZING,
        TaskStatus.CANCELLING,
    }
)
CANCELLABLE_TASK_STATUSES = frozenset(
    {
        TaskStatus.SCHEDULED,
        TaskStatus.QUEUED,
        TaskStatus.RUNNING,
        TaskStatus.DELEGATED,
        TaskStatus.FINALIZING,
    }
)
TERMINAL_TASK_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED})

_ACTIVITY_KEYS = {
    "id",
    "kind",
    "status",
    "message",
    "label",
    "detail",
    "category",
    "tool",
    "title",
    "updatedAt",
    "mode",
    "completed",
    "total",
}
TASK_TRANSITIONS: Mapping[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.SCHEDULED: frozenset({TaskStatus.QUEUED, TaskStatus.FAILED, TaskStatus.CANCELLED}),
    TaskStatus.QUEUED: frozenset({TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.CANCELLED}),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.DELEGATED,
            TaskStatus.FINALIZING,
            TaskStatus.CANCELLING,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
        }
    ),
    TaskStatus.DELEGATED: frozenset(
        {
            TaskStatus.FINALIZING,
            TaskStatus.CANCELLING,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
        }
    ),
    TaskStatus.FINALIZING: frozenset(
        {TaskStatus.CANCELLING, TaskStatus.COMPLETED, TaskStatus.FAILED}
    ),
    TaskStatus.CANCELLING: frozenset({TaskStatus.CANCELLED, TaskStatus.FAILED}),
}


class TaskTransitionError(ValueError):
    """Raised when a Task phase transition violates the lifecycle."""


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def normalize_task_scope(value: TaskScope | str | None) -> TaskScope:
    return TaskScope.SYSTEM if value == TaskScope.SYSTEM else TaskScope.USER


def normalize_task_schedule(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    schedule: dict[str, Any] = {}
    for key, maximum in (("type", 40), ("recurrence", 80)):
        raw = value.get(key)
        if isinstance(raw, str):
            normalized = raw.replace("\x00", "").strip()[:maximum]
            if normalized:
                schedule[key] = normalized
    at = value.get("at")
    maximum_timestamp = 9_007_199_254_740_991
    if isinstance(at, int) and not isinstance(at, bool):
        if 0 <= at <= maximum_timestamp:
            schedule["at"] = at
    elif isinstance(at, float) and math.isfinite(at) and 0 <= at <= maximum_timestamp:
        schedule["at"] = at
    return schedule or None


def _status(value: TaskStatus | str) -> TaskStatus:
    try:
        return TaskStatus(value)
    except (TypeError, ValueError) as error:
        raise TaskTransitionError(f"Unknown task status: {value}") from error


def is_task_active(value: TaskStatus | str) -> bool:
    try:
        return _status(value) in ACTIVE_TASK_STATUSES
    except TaskTransitionError:
        return False


def is_task_cancellable(value: TaskStatus | str) -> bool:
    try:
        return _status(value) in CANCELLABLE_TASK_STATUSES
    except TaskTransitionError:
        return False


def is_task_terminal(value: TaskStatus | str) -> bool:
    try:
        return _status(value) in TERMINAL_TASK_STATUSES
    except TaskTransitionError:
        return False


@dataclass(slots=True)
class Task:
    """Internal Task record.

    Runtime-only handles never belong here.  This record can therefore be
    safely projected or persisted after bounded normalization.
    """

    id: str
    status: TaskStatus
    objective: str
    owner_id: str
    session_id: str = "main"
    turn_id: str | None = None
    scope: TaskScope = TaskScope.USER
    kind: str = "work"
    parent_task_id: str | None = None
    priority: int = 0
    submission_key: str | None = None
    lane_key: str | None = None
    lane_limit: int = 1
    input_parts: list[dict[str, Any]] = field(default_factory=list)
    input_refs: list[str] = field(default_factory=list)
    schedule: dict[str, Any] | None = None
    timeout_ms: int | None = None
    created_at: int = field(default_factory=now_ms)
    started_at: int | None = None
    completed_at: int | None = None
    elapsed_ms: int = 0
    result: str | None = None
    error: str | None = None
    message: str | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    presentation: dict[str, Any] | None = None
    activity: list[dict[str, Any]] = field(default_factory=list)
    delegation: dict[str, Any] | None = None
    cancellation: dict[str, Any] | None = None
    authorization: dict[str, Any] | None = None
    notification_status: str = "none"
    notification_delivered_at: int | None = None
    notification_claimant_id: str | None = None
    notification_claimed_at: int | None = None

    def transition(self, next_status: TaskStatus | str) -> Task:
        transition_task(self, next_status)
        return self


def _record_value(record: Task | Mapping[str, Any], *names: str, default: Any = None) -> Any:
    if isinstance(record, Task):
        for name in names:
            if hasattr(record, name):
                return getattr(record, name)
        return default
    for name in names:
        if name in record:
            return record[name]
    return default


def transition_task(
    task: Task | MutableMapping[str, Any], next_status: TaskStatus | str
) -> Task | MutableMapping[str, Any]:
    current_value = task.status if isinstance(task, Task) else task.get("status")
    try:
        current = TaskStatus(current_value)
        target = TaskStatus(next_status)
    except (TypeError, ValueError) as error:
        raise TaskTransitionError(
            f"Unknown task transition: {current_value} -> {next_status}"
        ) from error
    if current == target:
        return task
    if target not in TASK_TRANSITIONS.get(current, frozenset()):
        raise TaskTransitionError(f"Invalid task transition: {current} -> {target}")
    if isinstance(task, Task):
        task.status = target
    else:
        task["status"] = target.value
    return task


def public_work_state(task: Task | Mapping[str, Any]) -> str:
    authorization = _record_value(task, "authorization")
    if isinstance(authorization, Mapping) and authorization.get("status") == "pending":
        return "auth_required"
    status = _status(_record_value(task, "status"))
    if status in {TaskStatus.SCHEDULED, TaskStatus.QUEUED}:
        return "submitted"
    if status in ACTIVE_TASK_STATUSES:
        return "working"
    if status in TERMINAL_TASK_STATUSES:
        return status.value
    return "submitted"


def normalize_task_presentation(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    source = value.get("presentation") or (value.get("decision") or {}).get("presentation") or value
    if not isinstance(source, Mapping):
        return None
    speech = _clean(source.get("speech"), 16_000)
    inline_value = source.get("inline")
    inline = None
    if isinstance(inline_value, Mapping):
        content = _clean(inline_value.get("content"), 1_000_000)
        if content:
            format_name = inline_value.get("format")
            inline = {
                "title": _clean(inline_value.get("title"), 120),
                "format": format_name
                if format_name in {"markdown", "code", "link"}
                else "markdown",
                "content": content,
            }
    if not speech and inline is None:
        return None
    return {"speech": speech, "inline": inline}


_MEDIA_TYPE = re.compile(r"^[a-z\d!#$&^_.+-]+/[a-z\d!#$&^_.+-]+$")
_BASE64 = re.compile(r"^[a-z\d+/]*={0,2}$", re.IGNORECASE)
_PUBLIC_URL_SCHEMES = {"data", "http", "https"}


def _clean(value: Any, maximum: int) -> str:
    return str(value or "").replace("\x00", "").strip()[:maximum]


def normalize_public_activity(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, Any] = {}
    for key in _ACTIVITY_KEYS:
        if key not in value:
            continue
        item = value[key]
        if key in {"completed", "total"}:
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                result[key] = item
            continue
        maximum = 160 if key in {"id", "kind", "status", "category", "tool", "mode"} else 600
        text = redact_public_text(item, maximum)
        if text:
            result[key] = text
    if not result.get("kind"):
        return None
    if result.get("kind") == "thinking":
        result.pop("message", None)
        result.pop("detail", None)
    return result


def normalize_public_activities(value: Any) -> list[dict[str, Any]]:
    source = value if isinstance(value, list | tuple) else []
    normalized: list[dict[str, Any]] = []
    for item in source[-20:]:
        activity = normalize_public_activity(item)
        if activity is not None:
            normalized.append(activity)
    return normalized


def _media_type(value: Any, fallback: str) -> str:
    normalized = _clean(value, 160).lower()
    return normalized if _MEDIA_TYPE.fullmatch(normalized) else fallback


def _json_value(value: Any) -> Any:
    try:
        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError):
        return ...
    if len(serialized) > 1_000_000:
        return ...
    return json.loads(serialized)


def _normalize_part(part: Any) -> dict[str, Any] | None:
    if not isinstance(part, Mapping):
        return None
    filename = _clean(part.get("filename"), 240)
    normalized: dict[str, Any]
    if "text" in part:
        text = _clean(part.get("text"), 1_000_000)
        if not text:
            return None
        normalized = {
            "text": text,
            "mediaType": _media_type(part.get("mediaType") or part.get("mimeType"), "text/plain"),
        }
    elif "raw" in part:
        raw = re.sub(r"\s", "", _clean(part.get("raw"), 16_000_000))
        if not raw or not _BASE64.fullmatch(raw):
            return None
        normalized = {
            "raw": raw,
            "mediaType": _media_type(
                part.get("mediaType") or part.get("mimeType"),
                "application/octet-stream",
            ),
        }
    elif "url" in part:
        url = _clean(part.get("url"), 20_000)
        if not url or urlsplit(url).scheme.lower() not in _PUBLIC_URL_SCHEMES:
            return None
        normalized = {
            "url": url,
            "mediaType": _media_type(
                part.get("mediaType") or part.get("mimeType"),
                "application/octet-stream",
            ),
        }
    elif "data" in part:
        data = _json_value(part.get("data"))
        if data is ...:
            return None
        normalized = {
            "data": data,
            "mediaType": _media_type(
                part.get("mediaType") or part.get("mimeType"), "application/json"
            ),
        }
    else:
        return None
    if filename:
        normalized["filename"] = filename
    return normalized


def normalize_artifacts(value: Any) -> list[dict[str, Any]]:
    artifacts = value if isinstance(value, list | tuple) else []
    result: list[dict[str, Any]] = []
    used: set[str] = set()
    remaining = 16_000_000
    for index, artifact in enumerate(artifacts[:32]):
        if not isinstance(artifact, Mapping):
            continue
        artifact_id = _clean(artifact.get("artifactId") or artifact.get("id"), 160)
        artifact_id = artifact_id or f"artifact_{index + 1}"
        if artifact_id in used:
            continue
        parts: list[dict[str, Any]] = []
        raw_parts = artifact.get("parts")
        for raw_part in (raw_parts if isinstance(raw_parts, list | tuple) else [])[:64]:
            part = _normalize_part(raw_part)
            if part is None:
                continue
            payload = next(part[key] for key in ("text", "raw", "url", "data") if key in part)
            size = len(
                payload
                if isinstance(payload, str)
                else json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            )
            if size > remaining:
                continue
            remaining -= size
            parts.append(part)
        if not parts:
            continue
        normalized: dict[str, Any] = {"artifactId": artifact_id, "parts": parts}
        name = _clean(artifact.get("name"), 240)
        description = _clean(artifact.get("description"), 1_000)
        if name:
            normalized["name"] = name
        if description:
            normalized["description"] = description
        used.add(artifact_id)
        result.append(normalized)
    return result


def merge_artifacts(current: Any, incoming: Any) -> list[dict[str, Any]]:
    by_id = {artifact["artifactId"]: artifact for artifact in normalize_artifacts(current)}
    for artifact in normalize_artifacts(incoming):
        by_id[artifact["artifactId"]] = artifact
    return normalize_artifacts(list(by_id.values()))


def artifact_from_inline_presentation(
    presentation: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    inline = presentation.get("inline") if isinstance(presentation, Mapping) else None
    if not isinstance(inline, Mapping):
        return None
    content = _clean(inline.get("content"), 1_000_000)
    if not content:
        return None
    format_name = inline.get("format")
    artifact: dict[str, Any] = {
        "artifactId": "artifact_inline",
        "parts": [
            {
                "text": content,
                "mediaType": "text/uri-list" if format_name == "link" else "text/markdown",
            }
        ],
    }
    title = _clean(inline.get("title"), 240)
    if title:
        artifact["name"] = title
    return artifact


def _normalize_authorization(value: Any, task_id: str, timestamp: int) -> dict[str, Any] | None:
    return normalize_public_authorization(value, task_id=task_id, timestamp=timestamp)


def _persisted_delegation(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    delegation = {
        "status": _clean(value.get("status"), 80) or "running",
        "title": _clean(value.get("title"), 160),
        "presentation": normalize_task_presentation(value.get("presentation")),
    }
    for key, limit in (("id", 160), ("sessionId", 240), ("directory", 2_048)):
        source = value.get(key)
        if key == "sessionId" and source is None:
            source = value.get("session_id")
        normalized = _clean(source, limit)
        if normalized:
            delegation[key] = normalized
    return delegation


def public_task(task: Task | Mapping[str, Any], *, now: int | None = None) -> dict[str, Any]:
    timestamp = now_ms() if now is None else now
    task_id = str(_record_value(task, "id", default=""))
    status = _status(_record_value(task, "status"))
    started_at = _record_value(task, "started_at", "startedAt")
    elapsed_ms = int(_record_value(task, "elapsed_ms", "elapsedMs", default=0) or 0)
    if started_at is not None and status in ACTIVE_TASK_STATUSES:
        elapsed_ms = max(0, timestamp - int(started_at))
    presentation = normalize_task_presentation(
        _record_value(task, "presentation", "result_metadata", "resultMetadata")
    )
    artifacts = normalize_artifacts(_record_value(task, "artifacts", default=[]))
    if not artifacts:
        legacy = artifact_from_inline_presentation(presentation)
        artifacts = [legacy] if legacy else []
    delegation = _record_value(task, "delegation")
    public_delegation = None
    if isinstance(delegation, Mapping):
        delegation_presentation = normalize_task_presentation(delegation.get("presentation"))
        public_delegation = {
            "status": delegation.get("status") or "running",
            "title": str(delegation.get("title") or "")[:160],
            "presentation": delegation_presentation,
        }
    authorization = _normalize_authorization(
        _record_value(task, "authorization"), task_id, timestamp
    )
    return {
        "id": task_id,
        "workState": public_work_state(task),
        "status": status.value,
        "scope": normalize_task_scope(_record_value(task, "scope")).value,
        "kind": str(_record_value(task, "kind", default="work") or "work"),
        "parentTaskId": _record_value(task, "parent_task_id", "parentTaskId"),
        "objective": str(_record_value(task, "objective", default="") or ""),
        "ownerId": str(_record_value(task, "owner_id", "ownerId", default="") or ""),
        "sessionId": str(_record_value(task, "session_id", "sessionId", default="main") or "main"),
        "turnId": _record_value(task, "turn_id", "turnId"),
        "createdAt": int(_record_value(task, "created_at", "createdAt", default=0) or 0),
        "startedAt": started_at,
        "completedAt": _record_value(task, "completed_at", "completedAt"),
        "elapsedMs": elapsed_ms,
        "result": _record_value(task, "result"),
        "error": _record_value(task, "error"),
        "message": _record_value(task, "message"),
        "artifacts": artifacts,
        "presentation": presentation,
        "activity": normalize_public_activities(_record_value(task, "activity", default=[]) or []),
        "delegation": public_delegation,
        "authorization": authorization,
        "notificationStatus": _record_value(
            task, "notification_status", "notificationStatus", default="none"
        ),
        "notificationDeliveredAt": _record_value(
            task, "notification_delivered_at", "notificationDeliveredAt"
        ),
        "schedule": normalize_task_schedule(_record_value(task, "schedule")),
        "timeoutMs": _record_value(task, "timeout_ms", "timeoutMs"),
    }


def persisted_task(task: Task | Mapping[str, Any]) -> dict[str, Any]:
    saved = public_task(task)
    saved.pop("workState", None)
    saved["submissionKey"] = _record_value(task, "submission_key", "submissionKey")
    saved["laneKey"] = _record_value(task, "lane_key", "laneKey")
    saved["laneLimit"] = int(_record_value(task, "lane_limit", "laneLimit", default=1) or 1)
    saved["priority"] = int(_record_value(task, "priority", default=0) or 0)
    saved["inputParts"] = deepcopy(
        _record_value(task, "input_parts", "inputParts", default=[]) or []
    )
    saved["inputRefs"] = [
        _clean(reference, 500)
        for reference in _record_value(task, "input_refs", "inputRefs", default=[]) or []
        if _clean(reference, 500)
    ][:32]
    saved["cancellation"] = deepcopy(_record_value(task, "cancellation"))
    saved["notificationClaimantId"] = _record_value(
        task,
        "notification_claimant_id",
        "notificationClaimantId",
    )
    saved["notificationClaimedAt"] = _record_value(
        task,
        "notification_claimed_at",
        "notificationClaimedAt",
    )
    # Recovery correlation is durable but remains absent from public_task().
    saved["delegation"] = _persisted_delegation(_record_value(task, "delegation"))
    return saved


def task_from_persisted(value: Mapping[str, Any]) -> Task:
    """Rehydrate the durable Gateway record without backend-private state."""

    task_id = _clean(_record_value(value, "id"), 160)
    objective = _clean(_record_value(value, "objective"), 1_000_000)
    owner_id = _clean(_record_value(value, "owner_id", "ownerId"), 240)
    if not task_id or not objective or not owner_id:
        raise ValueError("persisted Task requires id, objective, and ownerId")
    status = TaskStatus(_record_value(value, "status", default=TaskStatus.QUEUED))
    raw_input_parts = _record_value(value, "input_parts", "inputParts", default=[])
    input_parts = [
        deepcopy(dict(part))
        for part in (raw_input_parts if isinstance(raw_input_parts, list | tuple) else [])[:64]
        if isinstance(part, Mapping)
    ]
    created_at = int(_record_value(value, "created_at", "createdAt", default=now_ms()))
    raw_authorization = _record_value(value, "authorization")
    authorization = normalize_public_authorization(
        raw_authorization,
        task_id=task_id,
        timestamp=created_at,
    )
    raw_cancellation = _record_value(value, "cancellation")
    cancellation = (
        deepcopy(dict(raw_cancellation)) if isinstance(raw_cancellation, Mapping) else None
    )
    delegation = _persisted_delegation(_record_value(value, "delegation"))
    notification_status = str(
        _record_value(value, "notification_status", "notificationStatus", default="none") or "none"
    )
    if notification_status not in {"none", "pending", "delivering", "delivered"}:
        notification_status = "none"
    return Task(
        id=task_id,
        status=status,
        objective=objective,
        owner_id=owner_id,
        session_id=_clean(_record_value(value, "session_id", "sessionId", default="main"), 240)
        or "main",
        turn_id=_record_value(value, "turn_id", "turnId"),
        scope=normalize_task_scope(_record_value(value, "scope")),
        kind=_clean(_record_value(value, "kind", default="work"), 120) or "work",
        parent_task_id=_record_value(value, "parent_task_id", "parentTaskId"),
        priority=int(_record_value(value, "priority", default=0) or 0),
        submission_key=_record_value(value, "submission_key", "submissionKey"),
        lane_key=_record_value(value, "lane_key", "laneKey"),
        lane_limit=max(
            1,
            int(_record_value(value, "lane_limit", "laneLimit", default=1) or 1),
        ),
        input_parts=input_parts,
        input_refs=[
            _clean(reference, 500)
            for reference in _record_value(value, "input_refs", "inputRefs", default=[]) or []
            if _clean(reference, 500)
        ][:32],
        schedule=normalize_task_schedule(_record_value(value, "schedule")),
        timeout_ms=_record_value(value, "timeout_ms", "timeoutMs"),
        created_at=created_at,
        started_at=_record_value(value, "started_at", "startedAt"),
        completed_at=_record_value(value, "completed_at", "completedAt"),
        elapsed_ms=int(_record_value(value, "elapsed_ms", "elapsedMs", default=0) or 0),
        result=_record_value(value, "result"),
        error=_record_value(value, "error"),
        message=_record_value(value, "message"),
        artifacts=normalize_artifacts(_record_value(value, "artifacts", default=[])),
        presentation=normalize_task_presentation(_record_value(value, "presentation")),
        activity=normalize_public_activities(_record_value(value, "activity", default=[]) or []),
        delegation=delegation,
        cancellation=cancellation,
        authorization=authorization,
        notification_status=notification_status,
        notification_delivered_at=_record_value(
            value,
            "notification_delivered_at",
            "notificationDeliveredAt",
        ),
        notification_claimant_id=_record_value(
            value,
            "notification_claimant_id",
            "notificationClaimantId",
        ),
        notification_claimed_at=_record_value(
            value,
            "notification_claimed_at",
            "notificationClaimedAt",
        ),
    )


__all__ = [
    "ACTIVE_TASK_STATUSES",
    "CANCELLABLE_TASK_STATUSES",
    "TASK_TRANSITIONS",
    "TERMINAL_TASK_STATUSES",
    "Task",
    "TaskScope",
    "TaskStatus",
    "TaskTransitionError",
    "artifact_from_inline_presentation",
    "is_task_active",
    "is_task_cancellable",
    "is_task_terminal",
    "merge_artifacts",
    "normalize_artifacts",
    "normalize_public_activities",
    "normalize_public_activity",
    "normalize_task_presentation",
    "normalize_task_schedule",
    "normalize_task_scope",
    "now_ms",
    "persisted_task",
    "public_task",
    "public_work_state",
    "redact_public_text",
    "task_from_persisted",
    "transition_task",
]
