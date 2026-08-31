"""RealtimeProviderPort and its bounded, graph-outside session owner."""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from typing import Any, Protocol, runtime_checkable

MaybeAwaitable = Any | Awaitable[Any]
RealtimeListener = Callable[[Mapping[str, Any]], Any]

REALTIME_PROVIDER_METHODS = (
    "connect",
    "update_session",
    "send_audio",
    "send_input",
    "send_tool_result",
    "create_response",
    "cancel_response",
    "close",
    "capabilities",
    "subscribe",
)


@runtime_checkable
class RealtimeProviderPort(Protocol):
    def connect(self) -> MaybeAwaitable: ...

    def update_session(self, context: Mapping[str, Any]) -> MaybeAwaitable: ...

    def send_audio(self, audio: bytes | str) -> MaybeAwaitable: ...

    def send_input(self, value: Mapping[str, Any]) -> MaybeAwaitable: ...

    def send_tool_result(self, value: Mapping[str, Any]) -> MaybeAwaitable: ...

    def create_response(self, context: Mapping[str, Any] | None = None) -> MaybeAwaitable: ...

    def cancel_response(self, response_id: str | None = None) -> MaybeAwaitable: ...

    def close(self) -> MaybeAwaitable: ...

    def capabilities(self) -> Mapping[str, Any]: ...

    def subscribe(self, listener: RealtimeListener) -> Callable[[], None]: ...


class RealtimeProviderContractError(TypeError):
    code = "INVALID_REALTIME_PROVIDER"

    def __init__(self, missing: list[str]) -> None:
        super().__init__(f"RealtimeProviderPort is missing: {', '.join(missing)}")
        self.missing = tuple(missing)


def assert_realtime_provider(value: Any) -> RealtimeProviderPort:
    missing = [
        name for name in REALTIME_PROVIDER_METHODS if not callable(getattr(value, name, None))
    ]
    if missing:
        raise RealtimeProviderContractError(missing)
    return value


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class RealtimeProviderSession:
    """Serialize connect/reconnect and bound audio while the provider is down."""

    def __init__(
        self,
        provider: RealtimeProviderPort,
        *,
        max_pending_audio_chunks: int = 30,
        should_reconnect: Callable[[], bool] = lambda: True,
    ) -> None:
        if max_pending_audio_chunks <= 0:
            raise ValueError("max_pending_audio_chunks must be positive")
        self.provider = assert_realtime_provider(provider)
        self.max_pending_audio_chunks = max_pending_audio_chunks
        self.should_reconnect = should_reconnect
        self.pending_audio: deque[bytes | str] = deque(maxlen=max_pending_audio_chunks)
        self.ready = False
        self.blocked_error = ""
        self._session_context: dict[str, Any] | None = None
        self._closed = False
        self._connect_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self._reset_lock = asyncio.Lock()
        self._resetting = False
        self._listeners: set[RealtimeListener] = set()
        self._unsubscribe = self.provider.subscribe(self._on_provider_event)

    @property
    def connecting(self) -> bool:
        return self._connect_task is not None and not self._connect_task.done()

    def subscribe(self, listener: RealtimeListener) -> Callable[[], None]:
        self._listeners.add(listener)
        return lambda: self._listeners.discard(listener)

    def _publish(self, event: Mapping[str, Any]) -> None:
        for listener in tuple(self._listeners):
            with suppress(Exception):
                listener(dict(event))

    def _on_provider_event(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type in {"provider.error", "connection.error"} and event.get("fatal"):
            self.blocked_error = str(event.get("message") or "realtime provider unavailable")
            self.ready = False
            self.pending_audio.clear()
        elif event_type in {"provider.closed", "connection.closed"}:
            self.ready = False
            if self._resetting:
                return
            if not self.blocked_error and self.should_reconnect() and not self._closed:
                self._schedule_reconnect()
        self._publish(event)

    async def ensure_connected(self) -> None:
        if self._closed:
            raise RuntimeError("Realtime provider session is closed")
        if self.blocked_error:
            raise RuntimeError(self.blocked_error)
        if self.ready:
            return
        task = self._connect_task
        if task is None or task.done():
            task = asyncio.create_task(self._connect(), name="realtime-provider-connect")
            self._connect_task = task
        await asyncio.shield(task)

    async def _connect(self) -> None:
        try:
            await _maybe_await(self.provider.connect())
            if self._closed:
                return
            if self._session_context is not None:
                await _maybe_await(self.provider.update_session(self._session_context))
            self.ready = True
            while self.pending_audio:
                await _maybe_await(self.provider.send_audio(self.pending_audio.popleft()))
        except Exception as error:
            self.ready = False
            classification = getattr(error, "classification", "")
            if classification == "fatal" or getattr(error, "fatal", False):
                self.blocked_error = str(error)
                self.pending_audio.clear()
            raise
        finally:
            if self._connect_task is asyncio.current_task():
                self._connect_task = None

    def _schedule_reconnect(self) -> None:
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return

        async def reconnect() -> None:
            await asyncio.sleep(0)
            with suppress(Exception):
                await self.ensure_connected()

        self._reconnect_task = asyncio.create_task(
            reconnect(),
            name="realtime-provider-reconnect",
        )

    async def send_audio(self, audio: bytes | str) -> None:
        if self.ready:
            await _maybe_await(self.provider.send_audio(audio))
            return
        self.pending_audio.append(audio)
        await self.ensure_connected()

    def clear_pending_audio(self) -> None:
        """Drop audio that no longer belongs to the active input generation."""

        self.pending_audio.clear()

    async def update_session(self, context: Mapping[str, Any]) -> Any:
        await self.ensure_connected()
        result = await _maybe_await(self.provider.update_session(context))
        self._session_context = dict(context)
        return result

    async def send_input(self, value: Mapping[str, Any]) -> Any:
        await self.ensure_connected()
        return await _maybe_await(self.provider.send_input(value))

    async def send_tool_result(self, value: Mapping[str, Any]) -> Any:
        await self.ensure_connected()
        return await _maybe_await(self.provider.send_tool_result(value))

    async def create_response(self, context: Mapping[str, Any] | None = None) -> Any:
        await self.ensure_connected()
        return await _maybe_await(self.provider.create_response(context))

    async def cancel_response(self, response_id: str | None = None) -> Any:
        if not self.ready:
            return None
        return await _maybe_await(self.provider.cancel_response(response_id))

    async def reset_connection(self) -> None:
        """Replace the provider socket after an uncorrelated cancellation.

        Some providers do not echo request metadata. If a cancelled response
        never emits ``response.created``, FIFO correlation cannot distinguish
        its tombstone from the next response. Closing the socket is the only
        safe boundary: no late response from the old socket can cross it.
        """

        async with self._reset_lock:
            if self._closed:
                raise RuntimeError("Realtime provider session is closed")
            self.ready = False
            self.pending_audio.clear()
            for task in (self._connect_task, self._reconnect_task):
                if task is not None and task is not asyncio.current_task() and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(
                    task
                    for task in (self._connect_task, self._reconnect_task)
                    if task is not None and task is not asyncio.current_task()
                ),
                return_exceptions=True,
            )
            self._connect_task = None
            self._reconnect_task = None
            self._resetting = True
            try:
                await _maybe_await(self.provider.close())
            finally:
                self._resetting = False
                # Advance consumers only after the old socket is gone. Any
                # event observed while close is in progress still belongs to
                # the old epoch and can be fenced by this boundary.
                self._publish({"type": "provider.reset"})
            await self.ensure_connected()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.ready = False
        self.pending_audio.clear()
        for task in (self._connect_task, self._reconnect_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (self._connect_task, self._reconnect_task) if task),
            return_exceptions=True,
        )
        self._unsubscribe()
        self._listeners.clear()
        await _maybe_await(self.provider.close())
