"""Duck-typed transport ports and adapters.

Transport depends on these shapes, never on a concrete LangGraph runtime,
Task repository, or backend adapter.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, Protocol


class TaskSource(Protocol):
    """Minimum Task read/event surface consumed by HTTP and WebSocket."""

    def get(self, task_id: str, *, owner_id: str | None = None) -> Any | None: ...

    def list(self, *, owner_id: str | None = None) -> list[Any]: ...

    def subscribe(self, listener: Callable[[Any], None]) -> Callable[[], None]: ...


class InputArbitrationPort(Protocol):
    """Host microphone arbitration surface consumed by transport."""

    def status(self) -> Mapping[str, Any]: ...

    def suspend(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def resume(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def subscribe(
        self,
        listener: Callable[[Mapping[str, Any], Mapping[str, Any]], None],
    ) -> Callable[[], None]: ...


class EmptyTaskSource:
    """Read-only null object used when no background runtime is configured."""

    def get(self, task_id: str, *, owner_id: str | None = None) -> None:
        del task_id, owner_id
        return None

    def list(self, *, owner_id: str | None = None) -> list[Any]:
        del owner_id
        return []

    def subscribe(self, listener: Callable[[Any], None]) -> Callable[[], None]:
        del listener
        return lambda: None


def mapping(value: Any) -> dict[str, Any]:
    """Convert a mapping/model/dataclass-like object without importing domain."""

    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump(by_alias=True))
    try:
        return dict(vars(value))
    except (TypeError, ValueError):
        return {}


async def maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def call_with_supported_kwargs(
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call a duck-typed port while filtering optional query parameters."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    accepts_any = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_any:
        return function(*args, **kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(*args, **supported)


async def list_tasks(source: Any, *, owner_id: str) -> list[Any]:
    method = getattr(source, "list_tasks", None) or getattr(source, "list", None)
    if not callable(method):
        return []
    result = await maybe_await(call_with_supported_kwargs(method, owner_id=owner_id))
    if isinstance(result, Mapping):
        result = result.get("tasks", [])
    return list(result or [])


async def get_task(source: Any, task_id: str, *, owner_id: str) -> Any | None:
    method = getattr(source, "get_task", None) or getattr(source, "get", None)
    if not callable(method):
        return None
    return await maybe_await(call_with_supported_kwargs(method, task_id, owner_id=owner_id))


class TaskEventBroker:
    """Fan one runtime subscription out to SSE and Realtime consumers."""

    def __init__(self, source: Any) -> None:
        self._source = source
        self._listeners: set[Callable[[Any], None]] = set()
        self._unsubscribe_source: Callable[[], None] | None = None

    async def start(self) -> None:
        if self._unsubscribe_source is not None:
            return
        subscribe = getattr(self._source, "subscribe", None)
        if not callable(subscribe):
            self._unsubscribe_source = lambda: None
            return
        result = await maybe_await(call_with_supported_kwargs(subscribe, self.publish))
        self._unsubscribe_source = result if callable(result) else lambda: None

    def subscribe(self, listener: Callable[[Any], None]) -> Callable[[], None]:
        self._listeners.add(listener)

        def unsubscribe() -> None:
            self._listeners.discard(listener)

        return unsubscribe

    def publish(self, event: Any) -> None:
        for listener in tuple(self._listeners):
            try:
                listener(event)
            except Exception:
                # One disconnected transport consumer must not break the Task
                # runtime or prevent delivery to other clients.
                continue

    async def close(self) -> None:
        unsubscribe, self._unsubscribe_source = self._unsubscribe_source, None
        if callable(unsubscribe):
            await maybe_await(unsubscribe())
        self._listeners.clear()
