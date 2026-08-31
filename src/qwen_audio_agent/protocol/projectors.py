"""Public Task and AG-UI projections.

Projection is intentionally stricter than parsing: an adapter may attach
private scheduling or provider fields to its internal record, but only the
declared Gateway Task surface crosses this boundary.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError

from qwen_audio_agent.public_safety import normalize_public_authorization

from .constants import AGUI_TASK_ACTIVITY_TYPE, AguiEventType, GatewayTaskEvent
from .models import (
    AguiActivitySnapshotEvent,
    GatewayArtifact,
    GatewayArtifactPart,
    GatewayAuthorization,
    GatewayTask,
    GatewayTaskEventMessage,
)

_MEDIA_TYPE = re.compile(r"^[a-z\d!#$&^_.+-]+/[a-z\d!#$&^_.+-]+$")
_PAYLOAD_NAMES = ("text", "raw", "url", "data")
_PAYLOAD_DEFAULT_MEDIA_TYPE = {
    "text": "text/plain",
    "raw": "application/octet-stream",
    "url": "application/octet-stream",
    "data": "application/json",
}


def wire_dict(model: BaseModel, *, exclude_unset: bool = True) -> dict[str, Any]:
    """Serialize a wire model with JavaScript-compatible field names."""

    return model.model_dump(
        by_alias=True,
        exclude_unset=exclude_unset,
        mode="json",
    )


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, BaseModel):
        return value.model_dump(round_trip=True)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    try:
        return dict(vars(value))
    except (TypeError, ValueError):
        return {}


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _declared_payload(model_type: type[BaseModel], source: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name, field in model_type.model_fields.items():
        alias = field.alias or field_name
        if alias in source:
            payload[alias] = source[alias]
        elif field_name in source:
            payload[field_name] = source[field_name]
    return payload


def _clean(value: Any, maximum: int) -> str:
    return str(value or "").replace("\x00", "").strip()[:maximum]


def _media_type(value: Any, fallback: str) -> str:
    normalized = _clean(value, 160).lower()
    return normalized if _MEDIA_TYPE.fullmatch(normalized) else fallback


def _payload_size(part: Mapping[str, Any]) -> int:
    name = next(name for name in _PAYLOAD_NAMES if name in part)
    value = part[name]
    if isinstance(value, str):
        return len(value)
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _project_artifacts(value: Any) -> list[dict[str, Any]]:
    """Normalize adapter artifacts into the bounded public v4 representation."""

    if not isinstance(value, list | tuple):
        return []
    projected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    remaining = 16_000_000
    for index, raw_artifact in enumerate(value[:32]):
        artifact = _mapping(raw_artifact)
        if not artifact:
            continue
        artifact_id = _clean(artifact.get("artifactId") or artifact.get("id"), 160)
        artifact_id = artifact_id or f"artifact_{index + 1}"
        if artifact_id in used_ids:
            continue
        raw_parts = artifact.get("parts")
        if not isinstance(raw_parts, list | tuple):
            continue
        parts: list[dict[str, Any]] = []
        for raw_part in raw_parts[:64]:
            source = _mapping(raw_part)
            payload_name = next((name for name in _PAYLOAD_NAMES if name in source), None)
            if payload_name is None:
                continue
            payload: dict[str, Any] = {
                payload_name: source[payload_name],
                "mediaType": _media_type(
                    source.get("mediaType") or source.get("media_type") or source.get("mimeType"),
                    _PAYLOAD_DEFAULT_MEDIA_TYPE[payload_name],
                ),
            }
            filename = _clean(source.get("filename"), 240)
            if filename:
                payload["filename"] = filename
            try:
                part = wire_dict(GatewayArtifactPart.model_validate(payload))
                size = _payload_size(part)
            except (TypeError, ValueError, ValidationError):
                continue
            if size > remaining:
                continue
            remaining -= size
            parts.append(part)
        if not parts:
            continue
        public_artifact: dict[str, Any] = {
            "artifactId": artifact_id,
            "parts": parts,
        }
        name = _clean(artifact.get("name"), 240)
        description = _clean(artifact.get("description"), 1_000)
        if name:
            public_artifact["name"] = name
        if description:
            public_artifact["description"] = description
        try:
            parsed = GatewayArtifact.model_validate(public_artifact)
        except ValidationError:
            continue
        projected.append(wire_dict(parsed))
        used_ids.add(artifact_id)
    return projected


def _project_authorization(value: Any, *, task_id: str) -> dict[str, Any] | None:
    """Strip adapter-private authorization fields and bind the public Task ID."""

    normalized = normalize_public_authorization(_mapping(value), task_id=task_id)
    if normalized is None:
        return None
    try:
        return wire_dict(GatewayAuthorization.model_validate(normalized))
    except ValidationError:
        return None


def _public_work_state(task: Mapping[str, Any]) -> str:
    authorization = task.get("authorization")
    if isinstance(authorization, Mapping) and authorization.get("status") == "pending":
        return "auth_required"
    status = str(task.get("status") or "queued")
    if status in {"scheduled", "queued"}:
        return "submitted"
    if status in {"running", "delegated", "finalizing", "cancelling"}:
        return "working"
    if status in {"completed", "failed", "cancelled"}:
        return status
    return "submitted"


def project_gateway_task(task: Any) -> dict[str, Any] | None:
    """Project an internal Task-like object into the public Task schema."""

    source = _mapping(task)
    if not source:
        return None
    scope = _enum_value(source.get("scope", "user"))
    if scope == "system":
        return None

    payload = _declared_payload(GatewayTask, source)
    payload.setdefault("scope", scope or "user")
    payload.setdefault("kind", "work")
    payload.setdefault("workState", _public_work_state(source))
    payload.setdefault("elapsedMs", 0)
    if "artifacts" in source:
        payload["artifacts"] = _project_artifacts(source.get("artifacts"))
    if "authorization" in source:
        payload["authorization"] = _project_authorization(
            source.get("authorization"),
            task_id=str(source.get("id") or ""),
        )
    try:
        parsed = GatewayTask.model_validate(payload)
    except ValidationError:
        return None

    # Deliberately enumerate declared fields again. GatewayTask accepts extras
    # when parsing forward-compatible client data, but this projector must not
    # expose adapter-private fields accidentally.
    parsed_wire = wire_dict(parsed)
    aliases = {field.alias or field_name for field_name, field in GatewayTask.model_fields.items()}
    return {key: value for key, value in parsed_wire.items() if key in aliases}


def project_gateway_task_snapshot(task: Any) -> dict[str, Any] | None:
    """Create the first native SSE event for a Task."""

    public_task = project_gateway_task(task)
    if public_task is None:
        return None
    return wire_dict(
        GatewayTaskEventMessage(
            type=GatewayTaskEvent.SNAPSHOT,
            task=public_task,
        )
    )


def project_gateway_task_event(event: Any) -> dict[str, Any] | None:
    """Project a domain Task event without leaking routing metadata."""

    source = _mapping(event)
    event_type = _enum_value(source.get("type"))
    domain_event_types = {
        item.value for item in GatewayTaskEvent if item != GatewayTaskEvent.SNAPSHOT
    }
    if event_type not in domain_event_types:
        return None
    public_task = project_gateway_task(source.get("task"))
    if public_task is None:
        return None

    payload: dict[str, Any] = {"type": event_type, "task": public_task}
    permission = source.get("permission")
    if isinstance(permission, (Mapping, BaseModel)):
        public_permission = _project_authorization(
            permission,
            task_id=public_task["id"],
        )
        if public_permission is None:
            return None
        payload["permission"] = public_permission
    if isinstance(source.get("message"), str):
        payload["message"] = source["message"]
    try:
        parsed = GatewayTaskEventMessage.model_validate(payload)
    except ValidationError:
        return None
    return wire_dict(parsed)


def project_gateway_task_event_to_agui(event: Any) -> dict[str, Any] | None:
    """Project one native Task event to an AG-UI activity snapshot."""

    source = _mapping(event)
    public_task = project_gateway_task(source.get("task"))
    if public_task is None:
        return None
    payload: dict[str, Any] = {
        "type": _enum_value(source.get("type")),
        "task": public_task,
    }
    permission = source.get("permission")
    if isinstance(permission, (Mapping, BaseModel)):
        public_permission = _project_authorization(
            permission,
            task_id=public_task["id"],
        )
        if public_permission is None:
            return None
        payload["permission"] = public_permission
    if isinstance(source.get("message"), str):
        payload["message"] = source["message"]
    try:
        parsed = GatewayTaskEventMessage.model_validate(payload)
    except (TypeError, ValidationError):
        return None
    native = wire_dict(parsed)
    task = native["task"]
    content: dict[str, Any] = {
        "schema": "qwen.audio.task/v1",
        "eventType": native["type"],
        "task": task,
    }
    if "permission" in native and native["permission"] is not None:
        content["permission"] = native["permission"]
    if "message" in native and native["message"] is not None:
        content["message"] = native["message"]
    return wire_dict(
        AguiActivitySnapshotEvent(
            type=AguiEventType.ACTIVITY_SNAPSHOT,
            message_id=f"{AGUI_TASK_ACTIVITY_TYPE}:{task['id']}",
            activity_type=AGUI_TASK_ACTIVITY_TYPE,
            content=content,
            replace=True,
        )
    )


def project_gateway_task_event_for_format(
    event: dict[str, Any],
    format_name: str | None,
) -> dict[str, Any] | None:
    """Keep native Task events unless AG-UI was explicitly requested."""

    if format_name == "ag-ui":
        return project_gateway_task_event_to_agui(event)
    return event
