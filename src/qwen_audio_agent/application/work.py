"""Narrow application adapter between FrontendTurnGraph and WorkGraph."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from qwen_audio_agent.domain.frontend import (
    FrontendExecutionContext,
    FrontendToolCall,
    SubmissionReceipt,
    WorkSubmissionRequest,
)
from qwen_audio_agent.domain.task import normalize_public_activities, normalize_task_schedule
from qwen_audio_agent.graphs.frontend import FrontendPorts
from qwen_audio_agent.graphs.work import WorkGraph
from qwen_audio_agent.public_safety import normalize_public_authorization


def _task_id(call: FrontendToolCall) -> str:
    return str(call.arguments.get("task_id") or call.arguments.get("taskId") or "").strip()


_ACTIVE_STATUSES = {"queued", "running", "delegated", "finalizing", "cancelling"}


def _bounded_text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)]}…"


def _artifact_descriptors(value: Any) -> list[dict[str, Any]]:
    artifacts = value if isinstance(value, list) else []
    descriptors: list[dict[str, Any]] = []
    for artifact in artifacts[:32]:
        if not isinstance(artifact, dict):
            continue
        descriptor: dict[str, Any] = {
            "artifactId": str(artifact.get("artifactId") or "")[:160],
        }
        for key, limit in (("name", 240), ("description", 1_000)):
            normalized = _bounded_text(artifact.get(key), limit)
            if normalized:
                descriptor[key] = normalized
        parts: list[dict[str, Any]] = []
        raw_parts = artifact.get("parts")
        for part in (raw_parts if isinstance(raw_parts, list) else [])[:64]:
            if not isinstance(part, dict):
                continue
            payload_key = next(
                (key for key in ("text", "raw", "url", "data") if key in part),
                "",
            )
            payload = part.get(payload_key) if payload_key else None
            if isinstance(payload, str):
                size = len(payload.encode("utf-8"))
            elif payload is None:
                size = 0
            else:
                size = len(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                )
            part_descriptor: dict[str, Any] = {
                "type": payload_key or "unknown",
                "mediaType": str(part.get("mediaType") or "application/octet-stream")[:160],
                "size": size,
            }
            filename = _bounded_text(part.get("filename"), 240)
            if filename:
                part_descriptor["filename"] = filename
            parts.append(part_descriptor)
        descriptor["parts"] = parts
        descriptors.append(descriptor)
    return descriptors


def _status_projection(task: dict[str, Any], *, compact: bool = False) -> dict[str, Any]:
    if compact:
        keys = (
            "id",
            "workState",
            "status",
            "kind",
            "createdAt",
            "startedAt",
            "completedAt",
            "elapsedMs",
            "notificationStatus",
        )
        projected = {key: task.get(key) for key in keys}
        projected["objective"] = _bounded_text(task.get("objective"), 300)
        projected["message"] = _bounded_text(task.get("message"), 500)
        return projected

    keys = (
        "id",
        "workState",
        "status",
        "scope",
        "kind",
        "parentTaskId",
        "sessionId",
        "turnId",
        "createdAt",
        "startedAt",
        "completedAt",
        "elapsedMs",
        "notificationStatus",
        "notificationDeliveredAt",
        "timeoutMs",
    )
    projected = {key: task.get(key) for key in keys}
    projected["objective"] = _bounded_text(task.get("objective"), 2_000)
    projected["message"] = _bounded_text(task.get("message"), 4_000)
    projected["result"] = _bounded_text(task.get("result"), 12_000)
    projected["error"] = _bounded_text(task.get("error"), 4_000)
    projected["artifacts"] = _artifact_descriptors(task.get("artifacts"))
    projected["activity"] = normalize_public_activities(task.get("activity"))
    delegation = task.get("delegation")
    if isinstance(delegation, Mapping):
        projected["delegation"] = {
            key: normalized
            for key, limit in (("status", 80), ("title", 160))
            if (normalized := _bounded_text(delegation.get(key), limit))
        } or None
    authorization = normalize_public_authorization(
        task.get("authorization"),
        task_id=str(task.get("id") or ""),
        timestamp=int(task.get("createdAt") or 0),
    )
    projected["authorization"] = authorization
    projected["schedule"] = normalize_task_schedule(task.get("schedule"))
    return projected


@dataclass(slots=True)
class WorkApplication:
    """Expose Work intake and controls without exposing BackendPort to frontend."""

    runtime: WorkGraph

    async def submit(self, request: WorkSubmissionRequest) -> SubmissionReceipt:
        accepted = await self.runtime.submit(
            objective=request.objective,
            owner_id=request.owner_id,
            session_id=request.session_id,
            turn_id=request.turn_id,
            submission_key=request.submission_key,
            input_parts=[dict(value) for value in request.input_parts],
            input_refs=list(request.input_refs),
        )
        return SubmissionReceipt(
            task_id=str(accepted["id"]),
            reused=bool(accepted.get("reused")),
        )

    async def status(
        self,
        call: FrontendToolCall,
        context: FrontendExecutionContext,
    ) -> dict[str, Any]:
        values = [
            task
            for task in self.runtime.list(owner_id=context.owner_id)
            if task.get("sessionId") == context.session_id
        ]
        if call.arguments.get("list_all") is True or call.arguments.get("listAll") is True:
            return {
                "status": "ok",
                "tasks": [_status_projection(task, compact=True) for task in values[-20:]],
            }
        task_id = _task_id(call)
        task = self.runtime.get(task_id, owner_id=context.owner_id) if task_id else None
        if task is None and not task_id:
            task = values[-1] if values else None
        if task is None:
            return {
                "status": "not_found",
                "error": True,
                "error_code": "task_not_found",
            }
        return {"status": "ok", "task": _status_projection(task)}

    async def cancel(
        self,
        call: FrontendToolCall,
        context: FrontendExecutionContext,
    ) -> dict[str, Any]:
        values = [
            task
            for task in self.runtime.list(owner_id=context.owner_id)
            if task.get("sessionId") == context.session_id
            and task.get("status") in _ACTIVE_STATUSES
        ]
        if call.arguments.get("all") is True:
            cancelled = []
            for value in values:
                result = await self.runtime.cancel(str(value["id"]), owner_id=context.owner_id)
                if result is not None:
                    cancelled.append(_status_projection(result))
            return {"status": "ok", "tasks": cancelled}
        task_id = _task_id(call) or (str(values[-1]["id"]) if values else "")
        task = await self.runtime.cancel(task_id, owner_id=context.owner_id) if task_id else None
        if task is None:
            return {
                "status": "not_found",
                "error": True,
                "error_code": "task_not_found",
            }
        return {"status": "ok", "task": _status_projection(task)}

    async def respond_authorization(
        self,
        call: FrontendToolCall,
        context: FrontendExecutionContext,
    ) -> dict[str, Any]:
        has_current_user_input = any(
            part.get("type") == "text" and str(part.get("text") or "").strip()
            for part in context.input_parts
        )
        if not has_current_user_input:
            return {
                "status": "failed",
                "error": True,
                "error_code": "invalid_permission_response",
                "user_message": "没有当前轮次的用户决定，不能回复权限请求。",
            }
        task_id = _task_id(call)
        authorization_id = str(
            call.arguments.get("authorization_id") or call.arguments.get("authorizationId") or ""
        ).strip()
        decision = call.arguments.get("decision")
        if not task_id and authorization_id:
            task_id = next(
                (
                    str(task["id"])
                    for task in reversed(self.runtime.list(owner_id=context.owner_id))
                    if isinstance(task.get("authorization"), dict)
                    and task["authorization"].get("id") == authorization_id
                    and task["authorization"].get("status") == "pending"
                ),
                "",
            )
        task = (
            await self.runtime.respond_authorization(
                task_id,
                authorization_id,
                decision,
                owner_id=context.owner_id,
            )
            if task_id and authorization_id
            else None
        )
        if task is None:
            return {
                "status": "not_found",
                "error": True,
                "error_code": "authorization_not_pending",
            }
        return {"status": "ok", "task": _status_projection(task)}

    def frontend_ports(self) -> FrontendPorts:
        return FrontendPorts(
            control={
                "get_agent_task_status": self.status,
                "cancel_agent_task": self.cancel,
                "respond_agent_permission": self.respond_authorization,
            },
            submission=self,
        )
