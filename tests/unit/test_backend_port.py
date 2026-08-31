from __future__ import annotations

from typing import Any

import pytest

from qwen_audio_agent.backend.port import (
    BACKEND_PORT_METHODS,
    BackendEventType,
    BackendPortContractError,
    assert_backend_port,
    backend_event,
)


class CompleteBackend:
    def describe(self) -> dict[str, Any]:
        return {}

    async def start(self, context: object = None) -> None:
        return None

    async def health(self) -> dict[str, bool]:
        return {"ok": True}

    async def submit(self, task: object, context: object = None) -> dict[str, str]:
        return {"content": "完成"}

    async def status(self, task_id: str | None = None, context: object = None) -> dict[str, str]:
        return {"state": "working"}

    async def cancel(self, task_id: str, context: object = None) -> dict[str, str]:
        return {"state": "cancelled"}

    async def respond_authorization(
        self,
        task_id: str,
        authorization_id: str,
        decision: object,
        context: object = None,
    ) -> None:
        return None

    def subscribe(self, listener: object) -> Any:
        return lambda: None

    async def close(self) -> None:
        return None


def test_defines_the_complete_protocol_neutral_backend_boundary() -> None:
    assert BACKEND_PORT_METHODS == (
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
    backend = CompleteBackend()
    assert assert_backend_port(backend) is backend


def test_rejects_all_missing_or_non_callable_methods_in_one_error() -> None:
    backend = CompleteBackend()
    backend.submit = None  # type: ignore[method-assign]
    backend.respond_authorization = True  # type: ignore[method-assign]

    with pytest.raises(BackendPortContractError) as caught:
        assert_backend_port(backend, name="Test adapter")

    assert caught.value.code == "INVALID_BACKEND_PORT"
    assert caught.value.missing == ("submit", "respond_authorization")
    assert "Test adapter" in str(caught.value)


def test_normalizes_only_known_backend_events() -> None:
    assert backend_event(BackendEventType.MESSAGE, message="处理中") == {
        "type": "backend.message",
        "message": "处理中",
    }
    with pytest.raises(TypeError, match="Unknown BackendPort event"):
        backend_event("acp.private.event")
