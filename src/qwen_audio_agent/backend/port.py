"""Protocol-neutral boundary between the Gateway and one backend adapter."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class BackendEventType(StrEnum):
    ACTIVITY = "backend.activity"
    MESSAGE = "backend.message"
    ARTIFACT = "backend.artifact"
    AUTHORIZATION_REQUESTED = "backend.permission.requested"
    AUTHORIZATION_RESOLVED = "backend.permission.resolved"
    DELEGATED = "backend.delegated"
    DELEGATION_COMPLETED = "backend.delegation.completed"


type BackendEvent = Mapping[str, Any]
type BackendEventListener = Callable[[BackendEvent], None]
type Unsubscribe = Callable[[], Any]
type MaybeAwaitable = Any | Awaitable[Any]


@runtime_checkable
class BackendPort(Protocol):
    """Complete nine-method backend surface.

    Python uses ``respond_authorization``.  The retained JavaScript host maps
    that name to the v4 SDK spelling ``respondAuthorization`` in its thin
    compatibility adapter.
    """

    def describe(self) -> Mapping[str, Any]: ...

    def start(self, context: Mapping[str, Any] | None = None) -> MaybeAwaitable: ...

    def health(self) -> MaybeAwaitable: ...

    def submit(
        self,
        task: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> MaybeAwaitable: ...

    def status(
        self,
        task_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> MaybeAwaitable: ...

    def cancel(
        self,
        task_id: str,
        context: Mapping[str, Any] | None = None,
    ) -> MaybeAwaitable: ...

    def respond_authorization(
        self,
        task_id: str,
        authorization_id: str,
        decision: Any,
        context: Mapping[str, Any] | None = None,
    ) -> MaybeAwaitable: ...

    def subscribe(self, listener: BackendEventListener) -> Unsubscribe: ...

    def close(self) -> MaybeAwaitable: ...


BACKEND_PORT_METHODS: tuple[str, ...] = (
    "describe",
    "start",
    "health",
    "submit",
    "status",
    "cancel",
    "respond_authorization",
    "subscribe",
    "close",
)


class BackendPortContractError(TypeError):
    code = "INVALID_BACKEND_PORT"

    def __init__(self, message: str, *, missing: tuple[str, ...] | list[str] = ()) -> None:
        super().__init__(message)
        self.missing = tuple(missing)


def assert_backend_port(value: Any, *, name: str = "BackendPort") -> BackendPort:
    """Validate the whole port at the composition boundary.

    Optional capabilities belong in ``describe()``; a missing method is never
    interpreted as lack of support.
    """

    if value is None or isinstance(value, type):
        raise BackendPortContractError(
            f"{name} must be an object",
            missing=BACKEND_PORT_METHODS,
        )
    missing = tuple(
        method for method in BACKEND_PORT_METHODS if not callable(getattr(value, method, None))
    )
    if missing:
        raise BackendPortContractError(
            f"{name} is missing required methods: {', '.join(missing)}",
            missing=missing,
        )
    return value


def backend_event(event_type: BackendEventType | str, **details: Any) -> dict[str, Any]:
    try:
        normalized = BackendEventType(event_type)
    except (TypeError, ValueError) as error:
        raise TypeError(f"Unknown BackendPort event: {event_type}") from error
    return {"type": normalized.value, **details}


__all__ = [
    "BACKEND_PORT_METHODS",
    "BackendEvent",
    "BackendEventListener",
    "BackendEventType",
    "BackendPort",
    "BackendPortContractError",
    "Unsubscribe",
    "assert_backend_port",
    "backend_event",
]
