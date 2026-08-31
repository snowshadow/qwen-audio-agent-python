from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from qwen_audio_agent.application import WorkApplication
from qwen_audio_agent.backend.port import BackendEventType
from qwen_audio_agent.domain.frontend import (
    BASE_FRONTEND_TOOLS,
    FrontendExecutionContext,
    FrontendToolCall,
)
from qwen_audio_agent.graphs.frontend import FrontendTurnGraph
from qwen_audio_agent.graphs.work import WorkGraph


class ImmediateBackend:
    def __init__(self, result: Mapping[str, Any] | None = None) -> None:
        self.received: list[dict[str, Any]] = []
        self.result = dict(result or {"content": "done"})

    def describe(self) -> dict[str, Any]:
        return {"name": "immediate"}

    async def start(self, context: object = None) -> None:
        return None

    async def health(self) -> dict[str, bool]:
        return {"ok": True}

    async def submit(
        self,
        task: Mapping[str, Any],
        context: object = None,
    ) -> dict[str, Any]:
        self.received.append(dict(task))
        return dict(self.result)

    async def status(self, task_id: str | None = None, context: object = None) -> Any:
        return None

    async def cancel(self, task_id: str, context: object = None) -> Any:
        return {"state": "cancelled"}

    async def respond_authorization(
        self,
        task_id: str,
        authorization_id: str,
        decision: Any,
        context: object = None,
    ) -> None:
        return None

    def subscribe(self, listener: object) -> Callable[[], None]:
        return lambda: None

    async def close(self) -> None:
        return None


class LargePresentationBackend(ImmediateBackend):
    async def submit(
        self,
        task: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.received.append(dict(task))
        assert context is not None
        context["on_event"](
            {
                "type": BackendEventType.DELEGATED,
                "delegation": {
                    "id": "private-delegation",
                    "sessionId": "private-session",
                    "directory": "/private/project",
                    "status": "running",
                    "title": "Large delegated work",
                    "presentation": {
                        "speech": "delegated speech",
                        "inline": {
                            "title": "Delegated output",
                            "format": "markdown",
                            "content": "delegated-private-marker" * 60_000,
                        },
                    },
                },
            }
        )
        return {
            "content": "done",
            "presentation": {
                "speech": "direct speech",
                "inline": {
                    "title": "Direct output",
                    "format": "markdown",
                    "content": "direct-private-marker" * 60_000,
                },
            },
        }


@pytest.mark.asyncio
async def test_spawn_handoff_reaches_work_without_frontend_history_or_backend_wait() -> None:
    backend = ImmediateBackend()
    work = WorkGraph(backend=backend)
    application = WorkApplication(work)
    frontend = FrontendTurnGraph(BASE_FRONTEND_TOOLS, application.frontend_ports())
    context = FrontendExecutionContext(
        owner_id="owner",
        session_id="voice",
        turn_id="turn-1",
        turn_generation=1,
        input_parts=({"text": "current attachment", "mediaType": "text/plain"},),
    )

    accepted = await frontend.ainvoke(
        FrontendToolCall(
            "call-1",
            "spawn_thinking",
            {"objective": "整理附件", "input_refs": ["asset:older"]},
        ),
        context,
    )

    assert accepted["result"]["status"] == "accepted"
    task_id = str(accepted["result"]["task_id"])
    completed = await work.wait(task_id)
    assert completed is not None and completed["status"] == "completed"
    assert backend.received == [
        {
            "id": task_id,
            "ownerId": "owner",
            "objective": "整理附件",
            "instruction": "整理附件",
            "inputParts": [{"text": "current attachment", "mediaType": "text/plain"}],
            "inputRefs": ["asset:older"],
        }
    ]
    assert "history" not in backend.received[0]
    assert "memory" not in backend.received[0]

    before = len(work.list(owner_id="owner"))
    status = await frontend.ainvoke(
        FrontendToolCall("call-2", "get_agent_task_status", {"task_id": task_id}),
        context,
    )
    assert status["result"]["task"]["status"] == "completed"
    assert len(work.list(owner_id="owner")) == before
    await work.close()


@pytest.mark.asyncio
async def test_status_returns_bounded_artifact_descriptors_instead_of_payloads() -> None:
    backend = ImmediateBackend(
        {
            "content": "x" * 100_000,
            "artifacts": [
                {
                    "artifactId": "artifact-large",
                    "name": "large.bin",
                    "parts": [
                        {
                            "raw": "YQ" * 50_000,
                            "mediaType": "application/octet-stream",
                            "filename": "large.bin",
                        }
                    ],
                }
            ],
        }
    )
    work = WorkGraph(backend=backend)
    application = WorkApplication(work)
    frontend = FrontendTurnGraph(BASE_FRONTEND_TOOLS, application.frontend_ports())
    context = FrontendExecutionContext(
        owner_id="owner",
        session_id="voice",
        turn_id="turn-large",
        turn_generation=1,
    )
    accepted = await work.submit(
        objective="generate a large artifact",
        owner_id="owner",
        session_id="voice",
        turn_id="turn-large",
    )
    completed = await work.wait(str(accepted["id"]))
    assert completed is not None
    stored = work.repository.get(str(accepted["id"]))
    assert stored is not None
    stored.schedule = {
        "type": "at" * 50_000,
        "at": {"private": "schedule-private-marker" * 50_000},
        "recurrence": ["schedule-private-marker" * 50_000],
    }
    work.repository.set(stored.id, stored)

    status = await frontend.ainvoke(
        FrontendToolCall(
            "call-status-large",
            "get_agent_task_status",
            {"task_id": accepted["id"]},
        ),
        context,
    )

    projected = status["result"]["task"]
    assert len(projected["result"]) <= 12_000
    part = projected["artifacts"][0]["parts"][0]
    assert part == {
        "type": "raw",
        "mediaType": "application/octet-stream",
        "size": 100_000,
        "filename": "large.bin",
    }
    assert "raw" not in part
    serialized = json.dumps(projected, ensure_ascii=False)
    assert "schedule-private-marker" not in serialized
    assert projected["schedule"] == {"type": ("at" * 50_000)[:40]}
    await work.close()


@pytest.mark.asyncio
async def test_status_omits_direct_and_delegated_inline_presentation_payloads() -> None:
    work = WorkGraph(backend=LargePresentationBackend())
    application = WorkApplication(work)
    frontend = FrontendTurnGraph(BASE_FRONTEND_TOOLS, application.frontend_ports())
    context = FrontendExecutionContext(
        owner_id="owner",
        session_id="voice",
        turn_id="turn-presentation",
        turn_generation=1,
    )
    accepted = await work.submit(
        objective="generate large presentations",
        owner_id="owner",
        session_id="voice",
        turn_id="turn-presentation",
    )
    completed = await work.wait(str(accepted["id"]))
    assert completed is not None

    status = await frontend.ainvoke(
        FrontendToolCall(
            "call-status-presentation",
            "get_agent_task_status",
            {"task_id": accepted["id"]},
        ),
        context,
    )

    projected = status["result"]["task"]
    serialized = json.dumps(projected, ensure_ascii=False)
    assert len(serialized) < 20_000
    assert "direct-private-marker" not in serialized
    assert "delegated-private-marker" not in serialized
    assert "presentation" not in projected
    assert projected["delegation"] == {
        "status": "running",
        "title": "Large delegated work",
    }
    await work.close()
