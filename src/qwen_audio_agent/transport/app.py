"""FastAPI composition root for the Python Gateway transport."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import ValidationError
from starlette.requests import HTTPConnection

from qwen_audio_agent.protocol import (
    GATEWAY_CAPABILITIES,
    GATEWAY_PROTOCOL_VERSION,
    GatewayHealth,
    GatewayInputStatus,
    GatewayVoiceClients,
    InputResumeRequest,
    InputSuspendRequest,
    project_gateway_task,
    project_gateway_task_event,
    project_gateway_task_event_for_format,
    project_gateway_task_snapshot,
    wire_dict,
)

from .ports import (
    EmptyTaskSource,
    TaskEventBroker,
    call_with_supported_kwargs,
    get_task,
    list_tasks,
    mapping,
    maybe_await,
)
from .realtime import ClientDisconnectHandler, ClientEventHandler, RealtimeGateway
from .security import IdentityManager, is_allowed_origin

OwnerResolver = Callable[[HTTPConnection], str | Awaitable[str]]
HealthProvider = Callable[[], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]

_ACTIVE_TASK_STATUSES = {"queued", "running", "delegated", "finalizing", "cancelling"}


def _default_owner_resolver(connection: HTTPConnection) -> str:
    del connection
    return "personal"


async def _owner_id(resolver: OwnerResolver, connection: HTTPConnection) -> str:
    resolved = str(getattr(connection.state, "owner_id", "") or "").strip()
    value = resolved or str(await maybe_await(resolver(connection))).strip()
    if not value:
        raise HTTPException(status_code=401, detail="identity required")
    return value


def _belongs_to_owner(task: Mapping[str, Any], owner_id: str) -> bool:
    task_owner = task.get("ownerId", task.get("owner_id"))
    return bool(str(task_owner or "").strip()) and str(task_owner) == owner_id


def _event_owner_matches_task(event: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    if "ownerId" not in event and "owner_id" not in event:
        return True
    source_owner = str(event.get("ownerId", event.get("owner_id")) or "").strip()
    task_owner = str(task.get("ownerId", task.get("owner_id")) or "").strip()
    return bool(source_owner) and source_owner == task_owner


def _matches_task_query(
    task: Mapping[str, Any],
    *,
    session_id: str | None,
    active: bool,
) -> bool:
    task_session = task.get("sessionId", task.get("session_id"))
    if session_id is not None and task_session != session_id:
        return False
    if active and task.get("status") not in _ACTIVE_TASK_STATUSES:
        return False
    return True


def _sse(event: Mapping[str, Any]) -> bytes:
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return f"data: {payload}\n\n".encode()


def create_gateway_app(
    *,
    input_arbitration: Any,
    task_source: Any | None = None,
    conversation_source: Any | None = None,
    owner_resolver: OwnerResolver = _default_owner_resolver,
    identity_manager: IdentityManager | None = None,
    allowed_origins: Iterable[str] = (),
    health_provider: HealthProvider | None = None,
    on_client_event: ClientEventHandler | None = None,
    on_client_disconnect: ClientDisconnectHandler | None = None,
    capabilities: tuple[str, ...] | list[str] = GATEWAY_CAPABILITIES,
    gateway_instance_id: str | None = None,
    gateway_started_at: str | int | float | None = None,
    web_root: str | Path | None = None,
    skins_root: str | Path | None = None,
    close_task_source_on_shutdown: bool = False,
    close_input_on_shutdown: bool = True,
) -> FastAPI:
    """Compose the protocol transport around injected runtime ports.

    ``task_source`` needs only synchronous or asynchronous ``list``, ``get``
    and ``subscribe`` methods. ``on_client_event`` is the seam for the graph
    and realtime-provider runtime; this module never imports either.
    """

    tasks = task_source or EmptyTaskSource()
    trusted_origins = tuple(str(value) for value in allowed_origins)
    task_events = TaskEventBroker(tasks)
    realtime = RealtimeGateway(
        input_arbitration=input_arbitration,
        task_events=task_events,
        on_client_event=on_client_event,
        on_client_disconnect=on_client_disconnect,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        await task_events.start()
        start_tasks = getattr(tasks, "start", None)
        if callable(start_tasks):
            await maybe_await(start_tasks())
        await realtime.start()
        try:
            yield
        finally:
            await realtime.close()
            await task_events.close()
            close_tasks = getattr(tasks, "close", None)
            if close_task_source_on_shutdown and callable(close_tasks):
                await maybe_await(close_tasks())
            close = getattr(input_arbitration, "close", None)
            if close_input_on_shutdown and callable(close):
                await maybe_await(close())

    app = FastAPI(
        title="qwen-audio-agent Gateway",
        version=GATEWAY_PROTOCOL_VERSION,
        lifespan=lifespan,
    )
    app.state.input_arbitration = input_arbitration
    app.state.task_source = tasks
    app.state.conversation_source = conversation_source
    app.state.task_events = task_events
    app.state.realtime_gateway = realtime
    app.state.identity_manager = identity_manager
    app.state.allowed_origins = trusted_origins
    app.state.web_root = Path(web_root).resolve() if web_root is not None else None
    app.state.skins_root = Path(skins_root).resolve() if skins_root is not None else None

    if identity_manager is not None:

        @app.middleware("http")
        async def secure_http(request: Request, call_next: Callable[..., Any]) -> Response:
            if not is_allowed_origin(request.headers, allowed_origins=trusted_origins):
                return JSONResponse({"error": "origin not allowed"}, status_code=403)
            identity = identity_manager.resolve_http(
                request.headers,
                encrypted=request.url.scheme == "https",
            )
            request.state.owner_id = identity.owner_id
            response = await call_next(request)
            if identity.set_cookie:
                response.headers.append("set-cookie", identity.set_cookie)
            return response

    @app.get("/livez")
    async def livez() -> dict[str, Any]:
        return {"ok": True, "status": "live"}

    @app.get("/readyz")
    async def readyz() -> dict[str, Any]:
        return {"ok": True, "status": "ready"}

    @app.get("/api/health")
    async def health() -> JSONResponse:
        extensions = dict(await maybe_await(health_provider())) if health_provider else {}
        backend = extensions.pop(
            "backend",
            {
                "enabled": False,
                "configured": False,
                "ok": False,
                "status": "disabled",
                "capabilities": {},
            },
        )
        status = GatewayInputStatus.model_validate(input_arbitration.status())
        voice_clients = GatewayVoiceClients.model_validate(realtime.status())
        payload: dict[str, Any] = {
            **extensions,
            "ok": True,
            "status": "ready",
            "protocolVersion": GATEWAY_PROTOCOL_VERSION,
            "capabilities": list(capabilities),
            "gatewayInstanceId": gateway_instance_id
            if gateway_instance_id is not None
            else os.getenv("QWEN_AUDIO_GATEWAY_INSTANCE_ID"),
            "gatewayStartedAt": gateway_started_at
            if gateway_started_at is not None
            else os.getenv("QWEN_AUDIO_GATEWAY_STARTED_AT"),
            "inputSuspension": wire_dict(status, exclude_unset=False),
            "voiceConfigured": bool(extensions.get("voiceConfigured", False)),
            "voiceClients": wire_dict(voice_clients, exclude_unset=False),
            "backend": backend,
        }
        parsed = GatewayHealth.model_validate(payload)
        return JSONResponse(wire_dict(parsed, exclude_unset=False))

    @app.post("/api/input/suspend")
    async def suspend_input(request: InputSuspendRequest) -> JSONResponse:
        suspend = input_arbitration.suspend
        try:
            status = await maybe_await(
                call_with_supported_kwargs(
                    suspend,
                    owner=request.owner or "",
                    reason=request.reason or "",
                    ttl_ms=request.ttl_ms,
                    ttlMs=request.ttl_ms,
                )
            )
        except (ValueError, TypeError) as error:
            if getattr(error, "code", None) == "QWAUDIO_INPUT_OWNER_REQUIRED":
                return JSONResponse(
                    {"error": str(error), "code": error.code},
                    status_code=400,
                )
            raise
        parsed = GatewayInputStatus.model_validate(status)
        return JSONResponse(wire_dict(parsed, exclude_unset=False))

    @app.post("/api/input/resume")
    async def resume_input(request: InputResumeRequest) -> JSONResponse:
        resume = input_arbitration.resume
        status = await maybe_await(call_with_supported_kwargs(resume, owner=request.owner or ""))
        parsed = GatewayInputStatus.model_validate(status)
        return JSONResponse(wire_dict(parsed, exclude_unset=False))

    @app.get("/api/input")
    async def input_status() -> JSONResponse:
        parsed = GatewayInputStatus.model_validate(input_arbitration.status())
        return JSONResponse(wire_dict(parsed, exclude_unset=False))

    @app.get("/api/tasks")
    async def task_list(
        request: Request,
        session_id: str | None = Query(default=None, alias="sessionId"),
        active: str | None = Query(default=None),
    ) -> JSONResponse:
        owner_id = await _owner_id(owner_resolver, request)
        projected: list[dict[str, Any]] = []
        for value in await list_tasks(tasks, owner_id=owner_id):
            public = project_gateway_task(value)
            if public is None or not _belongs_to_owner(public, owner_id):
                continue
            if _matches_task_query(
                public,
                session_id=session_id,
                active=active == "true",
            ):
                projected.append(public)
        return JSONResponse({"tasks": projected})

    @app.get("/api/timeline")
    async def timeline(
        request: Request,
        session_id: str | None = Query(default=None, alias="sessionId"),
    ) -> JSONResponse:
        owner_id = await _owner_id(owner_resolver, request)
        items: list[dict[str, Any]] = []
        for value in await list_tasks(tasks, owner_id=owner_id):
            public = project_gateway_task(value)
            if public is None or not _belongs_to_owner(public, owner_id):
                continue
            if not _matches_task_query(public, session_id=session_id, active=False):
                continue
            presentation = public.get("presentation")
            inline = presentation.get("inline") if isinstance(presentation, Mapping) else None
            if not isinstance(inline, Mapping) or not inline.get("content"):
                continue
            items.append(
                {
                    "id": f"inline_{public['id']}",
                    "taskId": public["id"],
                    "createdAt": public.get("completedAt") or public.get("createdAt") or 0,
                    **dict(inline),
                }
            )
        items.sort(key=lambda item: int(item.get("createdAt") or 0))
        return JSONResponse({"items": items})

    @app.get("/api/conversations/{session_id}/messages")
    async def conversation_messages(session_id: str, request: Request) -> JSONResponse:
        owner_id = await _owner_id(owner_resolver, request)
        method = getattr(conversation_source, "messages", None)
        if not callable(method):
            return JSONResponse({"messages": []})
        result = await maybe_await(
            call_with_supported_kwargs(
                method,
                owner_id=owner_id,
                session_id=session_id,
                limit=200,
            )
        )
        messages = [dict(value) for value in (result or []) if isinstance(value, Mapping)]
        return JSONResponse({"messages": messages[-200:]})

    @app.get("/api/tasks/{task_id}")
    async def task_get(task_id: str, request: Request) -> JSONResponse:
        owner_id = await _owner_id(owner_resolver, request)
        value = await get_task(tasks, task_id, owner_id=owner_id)
        public = project_gateway_task(value)
        if public is None or not _belongs_to_owner(public, owner_id):
            return JSONResponse({"error": "task not found"}, status_code=404)
        return JSONResponse(public)

    @app.delete("/api/tasks/{task_id}")
    async def task_cancel(task_id: str, request: Request) -> JSONResponse:
        owner_id = await _owner_id(owner_resolver, request)
        existing_value = await get_task(tasks, task_id, owner_id=owner_id)
        existing = project_gateway_task(existing_value)
        if existing is None or not _belongs_to_owner(existing, owner_id):
            return JSONResponse({"error": "task not found"}, status_code=404)
        method = getattr(tasks, "cancel", None)
        if not callable(method):
            return JSONResponse(
                {"error": "task is no longer active", "task": existing},
                status_code=409,
            )
        result = await maybe_await(call_with_supported_kwargs(method, task_id, owner_id=owner_id))
        public = project_gateway_task(result)
        if public is None or not _belongs_to_owner(public, owner_id):
            return JSONResponse(
                {"error": "task is no longer active", "task": existing},
                status_code=409,
            )
        return JSONResponse(public)

    @app.post("/api/permissions/{authorization_id}")
    async def respond_permission(
        authorization_id: str,
        request: Request,
    ) -> JSONResponse:
        owner_id = await _owner_id(owner_resolver, request)
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            body = {}
        decision = str(body.get("decision") or "") if isinstance(body, Mapping) else ""
        if decision not in {"always", "reject"}:
            return JSONResponse(
                {"error": "decision must be always or reject"},
                status_code=400,
            )
        permission_task: dict[str, Any] | None = None
        for value in await list_tasks(tasks, owner_id=owner_id):
            public = project_gateway_task(value)
            authorization = public.get("authorization") if public else None
            if (
                public is not None
                and _belongs_to_owner(public, owner_id)
                and isinstance(authorization, Mapping)
                and authorization.get("id") == authorization_id
                and authorization.get("status") == "pending"
            ):
                permission_task = public
                break
        if permission_task is None:
            return JSONResponse({"error": "permission request not found"}, status_code=404)
        method = getattr(tasks, "respond_authorization", None)
        if not callable(method):
            return JSONResponse({"error": "permission request not found"}, status_code=404)
        result = await maybe_await(
            call_with_supported_kwargs(
                method,
                str(permission_task["id"]),
                authorization_id,
                decision,
                owner_id=owner_id,
            )
        )
        if result is None:
            return JSONResponse({"error": "permission request not found"}, status_code=404)
        refreshed_value = await get_task(
            tasks,
            str(permission_task["id"]),
            owner_id=owner_id,
        )
        public = project_gateway_task(refreshed_value)
        if (
            public is None
            or not _belongs_to_owner(public, owner_id)
            or public.get("id") != permission_task.get("id")
        ):
            return JSONResponse({"error": "permission request not found"}, status_code=404)
        return JSONResponse(public)

    @app.get("/api/tasks/{task_id}/events", response_model=None)
    async def task_stream(
        task_id: str,
        request: Request,
        format_name: str | None = Query(default=None, alias="format"),
    ) -> StreamingResponse | JSONResponse:
        owner_id = await _owner_id(owner_resolver, request)
        value = await get_task(tasks, task_id, owner_id=owner_id)
        public = project_gateway_task(value)
        if public is None or not _belongs_to_owner(public, owner_id):
            return JSONResponse({"error": "task not found"}, status_code=404)
        snapshot = project_gateway_task_snapshot(public)
        if snapshot is None:
            return JSONResponse({"error": "task projection failed"}, status_code=500)
        initial = project_gateway_task_event_for_format(snapshot, format_name)
        if initial is None:
            return JSONResponse({"error": "task projection failed"}, status_code=500)

        async def events():
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
            loop = asyncio.get_running_loop()

            def receive(raw_event: Any) -> None:
                source = mapping(raw_event)
                native = project_gateway_task_event(source)
                if native is None or native["task"]["id"] != task_id:
                    return
                if not _belongs_to_owner(native["task"], owner_id) or not (
                    _event_owner_matches_task(source, native["task"])
                ):
                    return
                projected_event = project_gateway_task_event_for_format(native, format_name)
                if projected_event is None:
                    return

                def enqueue() -> None:
                    if queue.full():
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    queue.put_nowait(projected_event)

                loop.call_soon_threadsafe(enqueue)

            unsubscribe = task_events.subscribe(receive)
            try:
                yield _sse(initial)
                while True:
                    yield _sse(await queue.get())
            finally:
                unsubscribe()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.websocket("/api/realtime")
    async def realtime_socket(websocket: WebSocket) -> None:
        if identity_manager is not None:
            if not is_allowed_origin(websocket.headers, allowed_origins=trusted_origins):
                await websocket.close(code=1008, reason="origin not allowed")
                return
            identity = identity_manager.resolve_upgrade(websocket.headers)
            if identity is None:
                await websocket.close(code=1008, reason="identity required")
                return
            websocket.state.owner_id = identity.owner_id
        try:
            owner_id = await _owner_id(owner_resolver, websocket)
        except HTTPException:
            await websocket.close(code=1008, reason="identity required")
            return
        session_id = websocket.query_params.get("sessionId") or "main"
        await realtime.serve(websocket, owner_id=owner_id, session_id=session_id)

    @app.exception_handler(ValidationError)
    async def protocol_validation_error(
        request: Request,
        error: ValidationError,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            {
                "error": "gateway protocol validation failed",
                "details": error.errors(include_context=False),
            },
            status_code=500,
        )

    def static_file(root: Path | None, relative_path: str) -> Path | None:
        if root is None or any(part.startswith(".") for part in Path(relative_path).parts):
            return None
        candidate = (root / relative_path).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            return None
        return candidate

    # Imported skins are immutable client assets but imports/removals must be
    # visible on the next reload.  Missing skin files never fall through to the
    # SPA index, which would turn a missing image into a misleading 200 HTML.
    @app.get("/skins/{asset_path:path}", include_in_schema=False)
    async def skin_asset(asset_path: str) -> Response:
        asset = static_file(app.state.skins_root, asset_path)
        if asset is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(asset, headers={"cache-control": "no-store"})

    # Register the SPA fallback last so every API/WebSocket route remains
    # authoritative.  Exact files are served first; client-side routes receive
    # index.html only when a built web distribution is present.
    @app.get("/{asset_path:path}", include_in_schema=False)
    async def web_asset(asset_path: str) -> Response:
        root = app.state.web_root
        asset = static_file(root, asset_path or "index.html")
        if asset is not None:
            return FileResponse(asset)
        index = static_file(root, "index.html")
        if index is not None:
            return FileResponse(index)
        return JSONResponse({"error": "not found"}, status_code=404)

    return app
