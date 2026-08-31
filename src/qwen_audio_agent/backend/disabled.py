"""BackendPort used when only the complete lightweight frontend is enabled."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


class DisabledBackend:
    enabled = False

    def describe(self) -> Mapping[str, Any]:
        return {
            "enabled": False,
            "configured": False,
            "protocol": "none",
            "label": "No backend Agent",
            "capabilities": {},
        }

    async def start(self, context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        del context
        return {"ok": True, "status": "not_configured"}

    async def health(self) -> Mapping[str, Any]:
        return {"ok": True, "status": "not_configured"}

    async def submit(
        self,
        task: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        del task, context
        raise RuntimeError("No backend Agent is configured")

    async def status(
        self,
        task_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        del task_id, context
        return {"ok": False, "status": "not_configured"}

    async def cancel(
        self,
        task_id: str,
        context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        del context
        return {"taskId": task_id, "state": "not_found"}

    async def respond_authorization(
        self,
        task_id: str,
        authorization_id: str,
        decision: Any,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        del task_id, authorization_id, decision, context
        raise RuntimeError("No backend Agent is configured")

    def subscribe(self, listener: Callable[[Mapping[str, Any]], Any]) -> Callable[[], None]:
        del listener
        return lambda: None

    async def close(self) -> None:
        return None


__all__ = ["DisabledBackend"]
