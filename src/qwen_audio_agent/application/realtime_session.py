"""Connection-local helpers and state for one realtime WebSocket session."""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from qwen_audio_agent.providers.realtime import RealtimeProviderPort, RealtimeProviderSession
from qwen_audio_agent.runtime.announcement_window import AnnouncementWindow
from qwen_audio_agent.runtime.result_delivery import (
    ResultDeliveryCoordinator,
)
from qwen_audio_agent.runtime.turns import RealtimeTurnState, TurnContext

ProviderFactory = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    RealtimeProviderPort | Awaitable[RealtimeProviderPort],
]
EventEmitter = Callable[[Mapping[str, Any]], Any]
ContextProvider = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any] | Awaitable[Mapping[str, Any]],
]


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _response_id(event: Mapping[str, Any]) -> str:
    response = event.get("response")
    item = event.get("item")
    return str(
        event.get("response_id")
        or event.get("responseId")
        or (response.get("id") if isinstance(response, Mapping) else "")
        or (item.get("response_id") if isinstance(item, Mapping) else "")
        or ""
    )


def _item_id(event: Mapping[str, Any]) -> str:
    item = event.get("item")
    return str(
        event.get("item_id")
        or event.get("itemId")
        or (item.get("id") if isinstance(item, Mapping) else "")
        or ""
    )


def _response_correlation_id(event: Mapping[str, Any]) -> str:
    response = event.get("response")
    metadata = response.get("metadata") if isinstance(response, Mapping) else None
    if not isinstance(metadata, Mapping):
        return ""
    return str(metadata.get("qwen_audio_request_id") or "")


def _tool_identity(event: Mapping[str, Any]) -> tuple[str, str]:
    item = event.get("item")
    call_id = event.get("call_id") or event.get("callId")
    name = event.get("name")
    if isinstance(item, Mapping):
        call_id = call_id or item.get("call_id") or item.get("callId")
        name = name or item.get("name")
    return str(call_id or ""), str(name or "")


def _owner_id(value: Mapping[str, Any]) -> str:
    return str(value.get("ownerId", value.get("owner_id")) or "").strip()


def _task_event_belongs_to_owner(
    event: Mapping[str, Any],
    task: Mapping[str, Any],
    owner_id: str,
) -> bool:
    task_owner = _owner_id(task)
    if not task_owner or task_owner != owner_id:
        return False
    source_has_owner = "ownerId" in event or "owner_id" in event
    return not source_has_owner or _owner_id(event) == task_owner


def _event_context(event: Mapping[str, Any], fallback: TurnContext) -> TurnContext:
    raw = event.get("__voiceContext")
    if not isinstance(raw, Mapping):
        return fallback
    turn_id = str(raw.get("turnId") or raw.get("turn_id") or fallback.turn_id)
    generation = raw.get("turnGeneration", raw.get("turn_generation"))
    return TurnContext(
        turn_id=turn_id,
        turn_generation=generation if isinstance(generation, int) else fallback.turn_generation,
    )


def _normalize_input_parts(event: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_parts = event.get("parts")
    source = raw_parts if isinstance(raw_parts, list) else []
    normalized: list[dict[str, Any]] = []
    for value in source[:16]:
        if not isinstance(value, Mapping):
            continue
        part_type = str(value.get("type") or "")
        if part_type == "text":
            text = str(value.get("text") or "").replace("\x00", "").strip()[:100_000]
            if text:
                normalized.append({"type": "text", "text": text})
        elif part_type == "file":
            mime = str(value.get("mime") or value.get("mimeType") or "").strip()[:160]
            url = str(value.get("url") or "").strip()
            if "/" not in mime or not url.startswith(("data:", "http://", "https://")):
                continue
            part = {"type": "file", "mime": mime, "url": url}
            filename = str(value.get("filename") or "").replace("\n", " ").strip()[:240]
            if filename:
                part["filename"] = filename
            source_value = value.get("source")
            if isinstance(source_value, Mapping):
                part["source"] = dict(source_value)
            meta = value.get("_meta")
            if isinstance(meta, Mapping):
                part["_meta"] = dict(meta)
            normalized.append(part)
    if not any(part.get("type") == "text" for part in normalized):
        text = str(event.get("text") or "").replace("\x00", "").strip()[:100_000]
        if text:
            normalized.insert(0, {"type": "text", "text": text})
    return tuple(normalized)


def _display_input(parts: tuple[dict[str, Any], ...]) -> str:
    text = "\n".join(
        str(part.get("text") or "").strip()
        for part in parts
        if part.get("type") == "text" and str(part.get("text") or "").strip()
    )
    if text:
        return text
    return " ".join(
        str(part.get("filename") or "[附件]") for part in parts if part.get("type") == "file"
    )


def _completed_event() -> asyncio.Event:
    event = asyncio.Event()
    event.set()
    return event


@dataclass(slots=True)
class _PendingResponse:
    context: TurnContext
    origin: str = "model"
    task_ids: tuple[str, ...] = ()
    authorization_id: str = ""
    correlation_id: str = ""
    completion: asyncio.Future[bool] | None = None
    cancelled: bool = False
    provider_epoch: int = 0


@dataclass(slots=True)
class _RealtimeConnectionState:
    connection_id: str
    owner_id: str
    session_id: str
    emit: EventEmitter
    provider: RealtimeProviderSession
    turns: RealtimeTurnState = field(default_factory=RealtimeTurnState)
    announcement_window: AnnouncementWindow = field(default_factory=AnnouncementWindow)
    response_contexts: dict[str, TurnContext] = field(default_factory=dict)
    response_text: dict[str, str] = field(default_factory=dict)
    response_origins: dict[str, str] = field(default_factory=dict)
    response_task_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    response_authorizations: dict[str, str] = field(default_factory=dict)
    response_completions: dict[str, asyncio.Future[bool]] = field(default_factory=dict)
    response_has_audio: set[str] = field(default_factory=set)
    response_has_function_call: set[str] = field(default_factory=set)
    playback_started: set[str] = field(default_factory=set)
    pending_response_contexts: deque[_PendingResponse] = field(default_factory=deque)
    pending_permissions: dict[str, dict[str, Any]] = field(default_factory=dict)
    announced_permissions: set[str] = field(default_factory=set)
    announcing_permission: str = ""
    permission_task: asyncio.Task[Any] | None = None
    active_responses: set[str] = field(default_factory=set)
    interrupted_responses: set[str] = field(default_factory=set)
    input_parts: tuple[dict[str, Any], ...] = ()
    frontend_capabilities: tuple[str, ...] = ()
    client_states: tuple[str, ...] = ()
    client_context: dict[str, Any] = field(default_factory=dict)
    input_enabled: bool = False
    output_enabled: bool = False
    text_only: bool = False
    sleeping: bool = False
    closed: bool = False
    manual_sequence: int = 0
    connect_task: asyncio.Task[None] | None = None
    unsubscribe: Callable[[], None] | None = None
    unsubscribe_tasks: Callable[[], None] | None = None
    delivery: ResultDeliveryCoordinator | None = None
    is_voice_active: Callable[[], bool] = field(default_factory=lambda: lambda: False)
    presentation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    frontend_tool_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tool_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    frontend_tool_tasks: dict[asyncio.Task[Any], str] = field(default_factory=dict)
    tool_calls_pending: dict[str, int] = field(default_factory=dict)
    tool_response_done: set[str] = field(default_factory=set)
    tool_followup_created: set[str] = field(default_factory=set)
    tool_followup_suppressed: set[str] = field(default_factory=set)
    tool_contexts: dict[str, TurnContext] = field(default_factory=dict)
    provider_epoch: int = 0
    provider_recovering: bool = False
    provider_recovered: asyncio.Event = field(default_factory=_completed_event)
