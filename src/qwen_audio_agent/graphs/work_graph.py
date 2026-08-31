"""LangGraph Work lifecycle built around the protocol-neutral BackendPort.

Graph nodes reduce normalized backend events into bounded state and checkpoint
before the next wait. Long-running backend I/O lives in the ``WorkGraph``
facade outside this module.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections import deque
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from qwen_audio_agent.backend.port import (
    BackendEventType,
)
from qwen_audio_agent.domain.task import (
    Task,
    TaskScope,
    TaskStatus,
    merge_artifacts,
    normalize_artifacts,
    normalize_public_activity,
    normalize_task_presentation,
    now_ms,
    public_task,
    transition_task,
)
from qwen_audio_agent.public_safety import normalize_public_authorization

TaskListener = Callable[[Mapping[str, Any]], Any]
_MAX_EVENTS = 256
_MAX_PROCESSED_EVENT_IDS = 512


def reduce_work_events(
    current: list[dict[str, Any]] | None,
    incoming: list[dict[str, Any]] | dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Append domain events with bounded, ID-based replay deduplication."""

    result = list(current or [])
    additions = incoming if isinstance(incoming, list) else [incoming] if incoming else []
    known = {event.get("eventId") for event in result if event.get("eventId")}
    for event in additions:
        if not isinstance(event, Mapping):
            continue
        normalized = dict(event)
        event_id = normalized.get("eventId")
        if event_id and event_id in known:
            continue
        result.append(normalized)
        if event_id:
            known.add(event_id)
    return result[-_MAX_EVENTS:]


def reduce_processed_event_ids(
    current: list[str] | None,
    incoming: list[str] | str | None,
) -> list[str]:
    result = list(current or [])
    additions = incoming if isinstance(incoming, list) else [incoming] if incoming else []
    for value in additions:
        normalized = str(value).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result[-_MAX_PROCESSED_EVENT_IDS:]


_COALESCED_EVENT_MARKER = "__runtime.coalesced_backend_event"
_MAX_BACKEND_EVENT_QUEUE = 256
_MAX_BUFFERED_BACKEND_EVENT_BYTES = 20 * 1024 * 1024
_MAX_COALESCED_ACTIVITY_KEYS = 32
_MAX_COALESCED_ARTIFACT_KEYS = 32
_BACKEND_EVENT_TYPES = frozenset(event_type.value for event_type in BackendEventType)
_QUEUED_SIGNAL_TYPES = _BACKEND_EVENT_TYPES | {"runtime.completed", "runtime.failed"}
_BACKEND_EVENT_OVERFLOW_ERROR = "Backend event queue exceeded its bounded capacity"
_BACKEND_OVERFLOW_CANCEL_TIMEOUT_SECONDS = 5.0


def _bounded_text(value: Any, limit: int) -> str:
    try:
        text = str(value or "")
    except (OverflowError, TypeError, ValueError):
        return ""
    return text.replace("\x00", "").strip()[:limit]


def _backend_event_size(event: Mapping[str, Any]) -> int:
    """Measure one normalized signal without creating one giant JSON string."""

    try:
        chunks = json.JSONEncoder(
            ensure_ascii=False,
            separators=(",", ":"),
        ).iterencode(event)
        return sum(len(chunk.encode("utf-8")) for chunk in chunks)
    except (OverflowError, TypeError, ValueError):
        return _MAX_BUFFERED_BACKEND_EVENT_BYTES + 1


def _public_activity(value: Any) -> dict[str, Any] | None:
    return normalize_public_activity(value)


def _backend_event_coalesce_key(
    event: Mapping[str, Any],
    existing: Mapping[str, Any],
) -> str:
    event_type = str(event.get("type") or "")
    if event_type == BackendEventType.MESSAGE:
        return "message"
    if event_type == BackendEventType.ACTIVITY:
        activity = event.get("activity")
        activity_id = str(activity.get("id") or "").strip() if isinstance(activity, Mapping) else ""
        candidate = f"activity:{activity_id}" if activity_id else "activity:latest"
        existing_keys = sum(value.startswith("activity:") for value in existing)
        if candidate not in existing and existing_keys >= _MAX_COALESCED_ACTIVITY_KEYS:
            return "activity:overflow"
        return candidate
    if event_type == BackendEventType.ARTIFACT:
        artifact = event.get("artifact")
        artifact_id = (
            str(artifact.get("artifactId") or artifact.get("id") or "").strip()
            if isinstance(artifact, Mapping)
            else ""
        )
        candidate = f"artifact:{artifact_id}" if artifact_id else "artifact:latest"
        existing_keys = sum(value.startswith("artifact:") for value in existing)
        if candidate not in existing and existing_keys >= _MAX_COALESCED_ARTIFACT_KEYS:
            return "artifact:overflow"
        return candidate
    if event_type == BackendEventType.DELEGATED:
        return "delegation:running"
    if event_type == BackendEventType.DELEGATION_COMPLETED:
        return "delegation:completed"
    return ""


def reduce_activity(
    current: list[dict[str, Any]] | None,
    incoming: list[dict[str, Any]] | dict[str, Any] | None,
) -> list[dict[str, Any]]:
    result = list(current or [])
    additions = incoming if isinstance(incoming, list) else [incoming] if incoming else []
    for raw in additions:
        activity = _public_activity(raw)
        if activity is None:
            continue
        activity_id = activity.get("id")
        if activity_id:
            result = [item for item in result if item.get("id") != activity_id]
        result.append(activity)
    return result[-20:]


def reduce_artifacts(
    current: list[dict[str, Any]] | None,
    incoming: list[dict[str, Any]] | dict[str, Any] | None,
) -> list[dict[str, Any]]:
    additions = incoming if isinstance(incoming, list) else [incoming] if incoming else []
    return merge_artifacts(current or [], additions)


class WorkState(TypedDict, total=False):
    task_id: str
    owner_id: str
    session_id: str
    turn_id: str | None
    scope: str
    kind: str
    objective: str
    submission_key: str | None
    lane_key: str | None
    lane_limit: int
    input_parts: list[dict[str, Any]]
    input_refs: list[str]
    status: str
    created_at: int
    started_at: int | None
    completed_at: int | None
    elapsed_ms: int
    result: str | None
    error: str | None
    message: str | None
    presentation: dict[str, Any] | None
    activity: Annotated[list[dict[str, Any]], reduce_activity]
    artifacts: Annotated[list[dict[str, Any]], reduce_artifacts]
    delegation: dict[str, Any] | None
    cancellation: dict[str, Any] | None
    authorization: dict[str, Any] | None
    notification_status: str
    initial_event_type: str
    backend_started: bool
    pending_signal: dict[str, Any] | None
    processed_event_ids: Annotated[list[str], reduce_processed_event_ids]
    events: Annotated[list[dict[str, Any]], reduce_work_events]


@dataclass(slots=True)
class _WorkHandle:
    task_id: str
    invoke_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cancel_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    event_queue: asyncio.Queue[dict[str, Any]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=_MAX_BACKEND_EVENT_QUEUE)
    )
    bootstrap_task: asyncio.Task[Any] | None = None
    backend_task: asyncio.Task[Any] | None = None
    event_pump_task: asyncio.Task[Any] | None = None
    scheduler_held: bool = False
    published_event_ids: set[str] = field(default_factory=set)
    backend_event_number: int = 0
    coalesced_events: dict[str, dict[str, Any]] = field(default_factory=dict)
    queued_coalesced_keys: set[str] = field(default_factory=set)
    queue_overflowed: bool = False
    cancellation_requested: bool = False
    foreign_event_lock: Any = field(default_factory=Lock)
    foreign_events: deque[dict[str, Any]] = field(default_factory=deque)
    foreign_coalesced_events: dict[str, dict[str, Any]] = field(default_factory=dict)
    foreign_queued_coalesced_keys: set[str] = field(default_factory=set)
    foreign_drain_scheduled: bool = False
    foreign_queue_overflowed: bool = False
    backend_ingress_closed: bool = False
    buffered_event_sizes: dict[str, int] = field(default_factory=dict)
    buffered_event_bytes: int = 0
    overflow_cancel_task: asyncio.Task[Any] | None = None


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _event(event_id: str, event_type: str, **details: Any) -> dict[str, Any]:
    return {"eventId": event_id, "type": event_type, **details}


def _transition(state: WorkState, target: TaskStatus) -> str:
    candidate: dict[str, Any] = {"status": state["status"]}
    transition_task(candidate, target)
    return str(candidate["status"])


def _permission(value: Any, task_id: str) -> dict[str, Any] | None:
    return normalize_public_authorization(
        value,
        task_id=task_id,
        timestamp=now_ms(),
        force_pending=True,
    )


def _authorization_resolution_status(value: Any, authorization_id: str) -> str:
    candidate = value
    if isinstance(value, Mapping) and isinstance(value.get("permission"), Mapping):
        candidate = value["permission"]
    if not isinstance(candidate, Mapping):
        return "cancelled"
    returned_id = _bounded_text(candidate.get("id"), 160)
    if returned_id and returned_id != authorization_id:
        return "cancelled"
    status = _bounded_text(candidate.get("status"), 40).lower()
    if status in {"approved", "denied", "cancelled"}:
        return status
    outcome = candidate.get("outcome")
    if isinstance(outcome, Mapping) and outcome.get("outcome") == "cancelled":
        return "cancelled"
    return "cancelled"


def _delegation(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    delegation = {
        "status": _bounded_text(value.get("status"), 80) or "running",
        "title": _bounded_text(value.get("title"), 160),
        "presentation": normalize_task_presentation(value.get("presentation")),
    }
    for key, limit in (("id", 160), ("sessionId", 240), ("directory", 2_048)):
        normalized = _bounded_text(value.get(key), limit)
        if normalized:
            delegation[key] = normalized
    return delegation


def _normalize_backend_signal(
    value: Any,
    *,
    task_id: str,
    event_id: str,
) -> dict[str, Any] | None:
    """Close and bound an adapter event before any queue retains it."""

    if not isinstance(value, Mapping):
        return None
    event_type = _bounded_text(value.get("type"), 80)
    signal: dict[str, Any] = {"type": event_type, "_event_id": event_id}
    if event_type == BackendEventType.MESSAGE:
        message = _bounded_text(value.get("message"), 4_000)
        if not message:
            return None
        signal["message"] = message
    elif event_type == BackendEventType.ACTIVITY:
        activity = _public_activity(value.get("activity"))
        if activity is None:
            return None
        signal["activity"] = activity
    elif event_type == BackendEventType.ARTIFACT:
        artifacts = normalize_artifacts([value.get("artifact")])
        if not artifacts:
            return None
        signal["artifact"] = artifacts[0]
    elif event_type == BackendEventType.AUTHORIZATION_REQUESTED:
        permission = _permission(value.get("permission"), task_id)
        if permission is None:
            return None
        signal["permission"] = permission
    elif event_type == BackendEventType.AUTHORIZATION_RESOLVED:
        permission = value.get("permission")
        if not isinstance(permission, Mapping):
            return None
        authorization_id = _bounded_text(permission.get("id"), 160)
        status = _bounded_text(permission.get("status"), 40).lower()
        if not authorization_id or status not in {"approved", "denied", "cancelled"}:
            return None
        signal["permission"] = {"id": authorization_id, "status": status}
    elif event_type in {
        BackendEventType.DELEGATED,
        BackendEventType.DELEGATION_COMPLETED,
    }:
        delegation = _delegation(value.get("delegation"))
        if delegation is None:
            return None
        signal["delegation"] = delegation
    else:
        return None
    return signal


def _normalize_backend_outcome(value: Any) -> dict[str, Any]:
    """Bound a terminal backend result before the event queue retains it."""

    source = value if isinstance(value, Mapping) else {}
    metadata = source.get("metadata") if isinstance(source.get("metadata"), Mapping) else {}
    artifacts = normalize_artifacts(source.get("artifacts") or metadata.get("artifacts") or [])
    presentation = normalize_task_presentation(source.get("presentation") or source.get("metadata"))
    content_source = source.get("content") if source else value
    return {
        "content": _bounded_text(content_source, 1_000_000),
        "artifacts": artifacts,
        "presentation": presentation,
    }


def _backend_recovery_task(task: Task) -> dict[str, Any]:
    projection = public_task(task)
    projection["delegation"] = deepcopy(task.delegation)
    return projection


def _cancel_confirmation_error(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return "backend did not explicitly confirm cancellation"
    for key in ("state", "status"):
        state = value.get(key)
        if not isinstance(state, str):
            continue
        normalized = state.strip().lower()
        if normalized in {"cancelled", "canceled"}:
            return None
        return f"backend returned cancellation {key} {normalized or 'unknown'}"
    if value.get("cancelled") is True:
        return None
    return "backend did not explicitly confirm cancellation"


def _initialize(state: WorkState) -> Command:
    event_type = state.get("initial_event_type") or "task.accepted"
    event_suffix = "scheduled.fired" if event_type == "task.scheduled.fired" else "accepted"
    return Command(
        update={"events": [_event(f"{state['task_id']}:{event_suffix}", event_type)]},
        goto="wait_for_lane",
    )


def _wait_for_lane(state: WorkState) -> Command:
    signal = interrupt(
        {
            "kind": "scheduler.lane",
            "taskId": state["task_id"],
            "ownerId": state["owner_id"],
            "laneKey": state.get("lane_key"),
        }
    )
    if isinstance(signal, Mapping) and signal.get("type") == "control.cancel":
        timestamp = now_ms()
        return Command(
            update={
                "status": _transition(state, TaskStatus.CANCELLED),
                "completed_at": timestamp,
                "elapsed_ms": 0,
                "notification_status": "none",
                "events": [_event(f"{state['task_id']}:cancelled", "task.cancelled")],
            },
            goto=END,
        )
    if not isinstance(signal, Mapping) or signal.get("type") != "scheduler.acquired":
        raise ValueError("WorkGraph lane wait requires scheduler.acquired or control.cancel")
    timestamp = now_ms()
    return Command(
        update={
            "status": _transition(state, TaskStatus.RUNNING),
            "started_at": timestamp,
            "backend_started": True,
            "events": [_event(f"{state['task_id']}:running", "task.running")],
        },
        goto="backend_lifecycle",
    )


def _wait_backend_event(state: WorkState) -> Command:
    signal = interrupt(
        {
            "kind": "backend.event",
            "taskId": state["task_id"],
            "status": state["status"],
        }
    )
    if not isinstance(signal, Mapping):
        raise ValueError("WorkGraph backend wait requires a normalized event")
    return Command(update={"pending_signal": dict(signal)}, goto="reduce_backend_event")


def _domain_event_id(signal: Mapping[str, Any], suffix: str) -> str:
    event_id = _bounded_text(signal.get("_event_id") or signal.get("eventId"), 240)
    return f"{event_id or 'backend'}:{suffix}"


def _processed_update(signal: Mapping[str, Any]) -> dict[str, Any]:
    event_id = _bounded_text(signal.get("_event_id") or signal.get("eventId"), 240)
    return {"processed_event_ids": [event_id]} if event_id else {}


def _reduce_backend_event(state: WorkState) -> Command:
    signal = state.get("pending_signal") or {}
    signal_id = _bounded_text(signal.get("_event_id") or signal.get("eventId"), 240)
    if signal_id and signal_id in state.get("processed_event_ids", []):
        return Command(update={"pending_signal": None}, goto="wait_backend_event")
    event_type = signal.get("type")
    common = {"pending_signal": None, **_processed_update(signal)}

    if state.get("status") == TaskStatus.FINALIZING and event_type in {
        BackendEventType.DELEGATED,
        BackendEventType.DELEGATION_COMPLETED,
    }:
        # A recovering adapter may replay its lifecycle before returning the
        # final outcome.  FINALIZING already includes both facts, so consume
        # only these duplicates and keep waiting for runtime.completed.
        return Command(update=common, goto="wait_backend_event")

    if event_type == "control.cancel":
        return Command(
            update={
                **common,
                "status": _transition(state, TaskStatus.CANCELLING),
                "authorization": None,
                "events": [
                    _event(
                        f"{state['task_id']}:cancelling",
                        "task.cancelling",
                    )
                ],
            },
            goto="wait_cancel_confirmation",
        )

    if event_type == BackendEventType.ACTIVITY:
        activity = _public_activity(signal.get("activity"))
        update = common
        if activity is not None:
            update = {
                **common,
                "activity": [activity],
                "events": [_event(_domain_event_id(signal, "progress"), "task.progress")],
            }
        return Command(update=update, goto="wait_backend_event")

    if event_type == BackendEventType.MESSAGE:
        message = _bounded_text(signal.get("message"), 4_000)
        update = common
        if message:
            update = {
                **common,
                "message": message,
                "events": [
                    _event(
                        _domain_event_id(signal, "updated"),
                        "task.updated",
                        message=message,
                    )
                ],
            }
        return Command(update=update, goto="wait_backend_event")

    if event_type == BackendEventType.ARTIFACT:
        artifacts = normalize_artifacts([signal.get("artifact")])
        update = common
        if artifacts:
            update = {
                **common,
                "artifacts": artifacts,
                "events": [_event(_domain_event_id(signal, "artifact"), "task.updated")],
            }
        return Command(update=update, goto="wait_backend_event")

    if event_type == BackendEventType.AUTHORIZATION_REQUESTED:
        authorization = _permission(signal.get("permission"), state["task_id"])
        update = common
        if authorization is not None:
            update = {
                **common,
                "authorization": authorization,
                "events": [
                    _event(
                        _domain_event_id(signal, "permission.requested"),
                        "task.permission.requested",
                        permission=authorization,
                    )
                ],
            }
        return Command(update=update, goto="wait_backend_event")

    if event_type == BackendEventType.AUTHORIZATION_RESOLVED:
        permission = state.get("authorization")
        signal_permission = signal.get("permission")
        resolved_id = (
            _bounded_text(signal_permission.get("id"), 160)
            if isinstance(signal_permission, Mapping)
            else ""
        )
        if (
            not isinstance(permission, Mapping)
            or not resolved_id
            or resolved_id != _bounded_text(permission.get("id"), 160)
        ):
            return Command(update=common, goto="wait_backend_event")
        status = _bounded_text(signal_permission.get("status"), 40).lower()
        if status not in {"approved", "denied", "cancelled"}:
            status = "cancelled"
        resolved = {
            **permission,
            "status": status,
            "resolvedAt": now_ms(),
        }
        return Command(
            update={
                **common,
                "authorization": None,
                "events": [
                    _event(
                        _domain_event_id(signal, "permission.resolved"),
                        "task.permission.resolved",
                        **({"permission": resolved} if resolved else {}),
                    )
                ],
            },
            goto="wait_backend_event",
        )

    if event_type == BackendEventType.DELEGATED:
        delegation = _delegation(signal.get("delegation"))
        if delegation is None:
            return Command(update=common, goto="wait_backend_event")
        return Command(
            update={
                **common,
                "status": _transition(state, TaskStatus.DELEGATED),
                "delegation": delegation,
                "events": [_event(_domain_event_id(signal, "delegated"), "task.delegated")],
            },
            goto="wait_backend_event",
        )

    if event_type == BackendEventType.DELEGATION_COMPLETED:
        delegation = _delegation(signal.get("delegation")) or state.get("delegation") or {}
        return Command(
            update={
                **common,
                "status": _transition(state, TaskStatus.FINALIZING),
                "delegation": {**delegation, "status": "completed"},
                "events": [_event(_domain_event_id(signal, "finalizing"), "task.finalizing")],
            },
            goto="wait_backend_event",
        )

    if event_type == "runtime.completed":
        timestamp = now_ms()
        outcome = signal.get("outcome")
        outcome_map = outcome if isinstance(outcome, Mapping) else {}
        presentation = normalize_task_presentation(
            outcome_map.get("presentation") or outcome_map.get("metadata")
        )
        artifacts = normalize_artifacts(
            outcome_map.get("artifacts") or (outcome_map.get("metadata") or {}).get("artifacts", [])
            if isinstance(outcome_map.get("metadata"), Mapping)
            else outcome_map.get("artifacts", [])
        )
        result = _bounded_text(outcome_map.get("content", outcome), 1_000_000)
        return Command(
            update={
                **common,
                "status": _transition(state, TaskStatus.COMPLETED),
                "completed_at": timestamp,
                "elapsed_ms": max(0, timestamp - int(state.get("started_at") or timestamp)),
                "result": result,
                "error": None,
                "presentation": presentation,
                "artifacts": artifacts,
                "authorization": None,
                "notification_status": "pending"
                if state.get("scope") == TaskScope.USER
                else "none",
                "events": [_event(_domain_event_id(signal, "completed"), "task.completed")],
            },
            goto=END,
        )

    if event_type == "runtime.failed":
        timestamp = now_ms()
        return Command(
            update={
                **common,
                "status": _transition(state, TaskStatus.FAILED),
                "completed_at": timestamp,
                "elapsed_ms": max(0, timestamp - int(state.get("started_at") or timestamp)),
                "error": _bounded_text(signal.get("error"), 4_000) or "Backend work failed",
                "authorization": None,
                "notification_status": "pending"
                if state.get("scope") == TaskScope.USER
                else "none",
                "events": [_event(_domain_event_id(signal, "failed"), "task.failed")],
            },
            goto=END,
        )

    return Command(update=common, goto="wait_backend_event")


def _wait_cancel_confirmation(state: WorkState) -> Command:
    signal = interrupt(
        {
            "kind": "backend.cancel.confirmation",
            "taskId": state["task_id"],
        }
    )
    if not isinstance(signal, Mapping):
        raise ValueError("WorkGraph cancellation requires backend confirmation")
    return Command(update={"pending_signal": dict(signal)}, goto="reduce_cancel_confirmation")


def _reduce_cancel_confirmation(state: WorkState) -> Command:
    signal = state.get("pending_signal") or {}
    timestamp = now_ms()
    if signal.get("type") == "control.cancel.confirmed":
        return Command(
            update={
                "pending_signal": None,
                "status": _transition(state, TaskStatus.CANCELLED),
                "completed_at": timestamp,
                "elapsed_ms": max(0, timestamp - int(state.get("started_at") or timestamp)),
                "result": None,
                "error": None,
                "authorization": None,
                "cancellation": deepcopy(signal.get("confirmation"))
                if isinstance(signal.get("confirmation"), Mapping)
                else None,
                "notification_status": "none",
                "events": [_event(f"{state['task_id']}:cancelled", "task.cancelled")],
            },
            goto=END,
        )
    error = _bounded_text(signal.get("error"), 3_900) or "backend did not confirm cancellation"
    return Command(
        update={
            "pending_signal": None,
            "status": _transition(state, TaskStatus.FAILED),
            "completed_at": timestamp,
            "elapsed_ms": max(0, timestamp - int(state.get("started_at") or timestamp)),
            "error": f"取消失败: {error}"[:4_000],
            "authorization": None,
            "notification_status": "pending" if state.get("scope") == TaskScope.USER else "none",
            "events": [_event(f"{state['task_id']}:cancel.failed", "task.failed")],
        },
        goto=END,
    )


def build_work_graph(*, checkpointer: BaseCheckpointSaver[Any] | None = None) -> CompiledStateGraph:
    """Build the WorkGraph and its protocol-neutral backend subgraph."""

    backend_builder = StateGraph(WorkState)
    backend_builder.add_node("wait_backend_event", _wait_backend_event)
    backend_builder.add_node("reduce_backend_event", _reduce_backend_event)
    backend_builder.add_node("wait_cancel_confirmation", _wait_cancel_confirmation)
    backend_builder.add_node("reduce_cancel_confirmation", _reduce_cancel_confirmation)
    backend_builder.add_edge(START, "wait_backend_event")
    backend_lifecycle = backend_builder.compile(name="backend_lifecycle")

    builder = StateGraph(WorkState)
    builder.add_node("initialize", _initialize)
    builder.add_node("wait_for_lane", _wait_for_lane)
    builder.add_node("backend_lifecycle", backend_lifecycle)
    builder.add_edge(START, "initialize")
    builder.add_edge("backend_lifecycle", END)
    return builder.compile(
        checkpointer=checkpointer or InMemorySaver(),
        name="work_graph",
    )
