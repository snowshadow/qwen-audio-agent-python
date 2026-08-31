"""Protocol-compatible `/api/realtime` WebSocket skeleton.

Realtime media/provider logic is injected through ``on_client_event``. This
module owns only the stable Gateway envelope, connection bookkeeping, Task
event fan-out, voice ownership hints, and host input-suspension control plane.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastapi import WebSocket
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect, WebSocketState

from qwen_audio_agent.protocol import (
    GatewayClientEvent,
    GatewayServerEvent,
    parse_gateway_client_message,
    parse_gateway_server_message,
    project_gateway_task_event,
    wire_dict,
)

from .ports import TaskEventBroker, mapping, maybe_await

ClientEventHandler = Callable[[dict[str, Any], dict[str, Any]], Any | Awaitable[Any]]
ClientDisconnectHandler = Callable[[dict[str, Any]], Any | Awaitable[Any]]


def _status_value(status: Mapping[str, Any], camel: str, snake: str) -> Any:
    return status.get(camel, status.get(snake))


def _owner_id(value: Mapping[str, Any]) -> str:
    return str(value.get("ownerId", value.get("owner_id")) or "").strip()


@dataclass(eq=False, slots=True)
class RealtimeConnection:
    websocket: WebSocket
    owner_id: str
    session_id: str
    loop: asyncio.AbstractEventLoop
    connection_id: str = field(default_factory=lambda: uuid4().hex)
    descriptor: dict[str, Any] = field(default_factory=lambda: {"type": "web", "instanceId": None})
    outgoing: asyncio.Queue[dict[str, Any] | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=256)
    )
    closed: bool = False

    def enqueue(self, event: dict[str, Any]) -> None:
        if self.closed or self.loop.is_closed():
            return

        def put() -> None:
            if self.closed:
                return
            if self.outgoing.full():
                try:
                    self.outgoing.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self.outgoing.put_nowait(event)

        self.loop.call_soon_threadsafe(put)


class RealtimeGateway:
    """Small transport hub shared by FastAPI routes and runtime ports."""

    def __init__(
        self,
        *,
        input_arbitration: Any,
        task_events: TaskEventBroker,
        on_client_event: ClientEventHandler | None = None,
        on_client_disconnect: ClientDisconnectHandler | None = None,
    ) -> None:
        self._input = input_arbitration
        self._task_events = task_events
        self._on_client_event = on_client_event
        self._on_client_disconnect = on_client_disconnect
        self._connections: set[RealtimeConnection] = set()
        self._active_by_owner: dict[str, RealtimeConnection] = {}
        self._unsubscribe_input: Callable[[], None] | None = None
        self._unsubscribe_tasks: Callable[[], None] | None = None

    async def start(self) -> None:
        if self._unsubscribe_tasks is None:
            self._unsubscribe_tasks = self._task_events.subscribe(self._on_task_event)
        subscribe = getattr(self._input, "subscribe", None)
        if self._unsubscribe_input is None and callable(subscribe):
            result = await maybe_await(subscribe(self._on_input_change))
            self._unsubscribe_input = result if callable(result) else lambda: None

    async def close(self) -> None:
        for connection in tuple(self._connections):
            connection.enqueue({"type": GatewayServerEvent.GATEWAY_DISCONNECTED.value})
            if not connection.outgoing.full():
                connection.outgoing.put_nowait(None)
            connection.closed = True
            try:
                await connection.websocket.close(code=1001)
            except RuntimeError:
                pass
        for unsubscribe_name in ("_unsubscribe_input", "_unsubscribe_tasks"):
            unsubscribe = getattr(self, unsubscribe_name)
            setattr(self, unsubscribe_name, None)
            if callable(unsubscribe):
                await maybe_await(unsubscribe())
        self._connections.clear()
        self._active_by_owner.clear()

    def status(self) -> dict[str, Any]:
        by_type = Counter(
            connection.descriptor.get("type", "web") for connection in self._connections
        )
        return {
            "connected": len(self._connections),
            "activeOwners": len(self._active_by_owner),
            "byType": dict(by_type),
        }

    async def serve(self, websocket: WebSocket, *, owner_id: str, session_id: str) -> None:
        await websocket.accept()
        connection = RealtimeConnection(
            websocket=websocket,
            owner_id=owner_id,
            session_id=session_id,
            loop=asyncio.get_running_loop(),
        )
        self._connections.add(connection)
        sender = asyncio.create_task(self._send_loop(connection))
        connection.enqueue({"type": GatewayServerEvent.VOICE_STATE.value, "state": "idle"})
        self._send_current_suspension(connection)
        try:
            while True:
                incoming = await websocket.receive()
                if incoming["type"] == "websocket.disconnect":
                    break
                raw = incoming.get("text")
                if raw is None and incoming.get("bytes") is not None:
                    raw = incoming["bytes"].decode(errors="replace")
                try:
                    value = json.loads(raw or "")
                except (TypeError, ValueError):
                    continue
                try:
                    parsed = parse_gateway_client_message(value)
                except ValidationError:
                    # Matches the existing Gateway: malformed JSON envelopes
                    # and unknown event names are ignored, not reflected.
                    continue
                event = wire_dict(parsed)
                await self._handle_event(connection, event)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if self._on_client_disconnect is not None:
                await maybe_await(self._on_client_disconnect(self._context(connection)))
            self._release(connection)
            connection.closed = True
            if not connection.outgoing.full():
                connection.outgoing.put_nowait(None)
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

    async def _send_loop(self, connection: RealtimeConnection) -> None:
        while True:
            event = await connection.outgoing.get()
            if event is None:
                return
            if connection.websocket.client_state != WebSocketState.CONNECTED:
                return
            await connection.websocket.send_json(event)

    def _send_current_suspension(self, connection: RealtimeConnection) -> None:
        status = mapping(self._input.status())
        if status.get("suspended") is not True:
            return
        connection.enqueue(
            {
                "type": GatewayServerEvent.INPUT_SUSPEND.value,
                "owner": status.get("owner"),
                "reason": status.get("reason", ""),
                "expiresAt": _status_value(status, "expiresAt", "expires_at"),
            }
        )

    async def _handle_event(
        self,
        connection: RealtimeConnection,
        event: dict[str, Any],
    ) -> None:
        event_type = event["type"]
        deactivated: RealtimeConnection | None = None
        if event_type == GatewayClientEvent.CONNECT.value:
            client_type = event.get("clientType", "web")
            if client_type not in {"desktop", "cli", "web"}:
                client_type = "web"
            label = str(event.get("clientLabel") or "").strip()[:40]
            connection.descriptor = {
                "type": client_type,
                **({"label": label} if label else {}),
                "instanceId": str(event.get("clientInstanceId") or "").strip()[:80] or None,
            }
            can_output = (
                event.get("outputEnabled") is True
                if "outputEnabled" in event
                else event.get("voiceEnabled") is True
            )
            if can_output and event.get("textOnly") is not True:
                previous = self._active_by_owner.get(connection.owner_id)
                activated = self._activate(
                    connection,
                    takeover=event.get("takeover") is True,
                )
                if activated and previous is not connection:
                    deactivated = previous
        elif event_type in {
            GatewayClientEvent.UNMUTE.value,
            GatewayClientEvent.INPUT_UNMUTE.value,
        }:
            previous = self._active_by_owner.get(connection.owner_id)
            activated = self._activate(
                connection,
                takeover=event.get("takeover") is True,
            )
            if activated and previous is not connection:
                deactivated = previous
        elif event_type == GatewayClientEvent.MUTE.value:
            self._release_voice(connection)

        if event_type == GatewayClientEvent.AUDIO_APPEND.value:
            input_status = mapping(self._input.status())
            if (
                input_status.get("suspended") is True
                or self._active_by_owner.get(connection.owner_id) is not connection
            ):
                return

        if self._on_client_event is None:
            return
        if deactivated is not None:
            await maybe_await(
                self._on_client_event(
                    self._context(deactivated),
                    {"type": "__voice.deactivated"},
                )
            )
        context = self._context(connection)
        result = await maybe_await(self._on_client_event(context, event))
        if result is None:
            return
        events = result if isinstance(result, list | tuple) else [result]
        for candidate in events:
            self._emit(connection, candidate)

    def _context(self, connection: RealtimeConnection) -> dict[str, Any]:
        return {
            "ownerId": connection.owner_id,
            "sessionId": connection.session_id,
            "connectionId": connection.connection_id,
            "client": dict(connection.descriptor),
            "voiceActive": self._active_by_owner.get(connection.owner_id) is connection,
            "isVoiceActive": lambda: self._active_by_owner.get(connection.owner_id) is connection,
            "inputSuspended": mapping(self._input.status()).get("suspended") is True,
            # Runtime-only callback. It never crosses the public protocol.
            "emit": lambda candidate: self._emit(connection, candidate),
        }

    def _emit(self, connection: RealtimeConnection, candidate: Any) -> bool:
        try:
            validated = parse_gateway_server_message(candidate)
        except ValidationError:
            return False
        connection.enqueue(wire_dict(validated))
        return True

    def _activate(self, connection: RealtimeConnection, *, takeover: bool) -> bool:
        previous = self._active_by_owner.get(connection.owner_id)
        if previous is not None and previous is not connection and not takeover:
            self._broadcast_ownership(connection.owner_id)
            return False
        if previous is not None and previous is not connection:
            previous.enqueue({"type": GatewayServerEvent.PLAYBACK_CLEAR.value})
            previous.enqueue(
                {
                    "type": GatewayServerEvent.VOICE_DEACTIVATED.value,
                    "holder": dict(connection.descriptor),
                }
            )
        self._active_by_owner[connection.owner_id] = connection
        self._broadcast_ownership(connection.owner_id)
        return True

    def _release_voice(self, connection: RealtimeConnection) -> None:
        if self._active_by_owner.get(connection.owner_id) is not connection:
            return
        self._active_by_owner.pop(connection.owner_id, None)
        self._broadcast_ownership(connection.owner_id)

    def _release(self, connection: RealtimeConnection) -> None:
        self._connections.discard(connection)
        self._release_voice(connection)

    def _broadcast_ownership(self, owner_id: str) -> None:
        active = self._active_by_owner.get(owner_id)
        holder = dict(active.descriptor) if active else None
        for connection in tuple(self._connections):
            if connection.owner_id != owner_id:
                continue
            connection.enqueue(
                {
                    "type": GatewayServerEvent.VOICE_OWNERSHIP.value,
                    "state": (
                        "active" if active is connection else "busy" if holder else "available"
                    ),
                    "holder": holder,
                }
            )

    def _on_input_change(self, status_value: Any, change_value: Any) -> None:
        status = mapping(status_value)
        change = mapping(change_value)
        if change.get("state") == "suspended":
            events = (
                {
                    "type": GatewayServerEvent.PLAYBACK_CLEAR.value,
                    "reason": "input_suspended",
                },
                {
                    "type": GatewayServerEvent.INPUT_SUSPEND.value,
                    "owner": status.get("owner"),
                    "reason": status.get("reason", ""),
                    "expiresAt": _status_value(status, "expiresAt", "expires_at"),
                },
            )
        elif change.get("state") == "resumed":
            events = ({"type": GatewayServerEvent.INPUT_RESUME.value},)
        else:
            return
        for connection in tuple(self._connections):
            for event in events:
                connection.enqueue(event)

    def _on_task_event(self, value: Any) -> None:
        source = mapping(value)
        event = project_gateway_task_event(source)
        if event is None:
            return
        task = event["task"]
        task_owner = _owner_id(task)
        source_has_owner = "ownerId" in source or "owner_id" in source
        if not task_owner or (source_has_owner and _owner_id(source) != task_owner):
            return
        event_session = task.get("sessionId")
        for connection in tuple(self._connections):
            if task_owner != connection.owner_id:
                continue
            if event_session and event_session != connection.session_id:
                continue
            connection.enqueue(event)
