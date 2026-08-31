from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from threading import Barrier, Thread
from typing import Any

import pytest

from qwen_audio_agent.backend.port import BackendEventType
from qwen_audio_agent.graphs.work import WorkGraph, reduce_activity, reduce_work_events
from qwen_audio_agent.runtime.task_notification_lease import TaskNotificationLeaseAdapter


async def eventually(predicate: Callable[[], bool], deadline_seconds: float = 2.0) -> None:
    async with asyncio.timeout(deadline_seconds):
        while not predicate():  # noqa: ASYNC110 - polling a synchronous projection
            await asyncio.sleep(0)


class BackendStub:
    def __init__(self) -> None:
        self.submissions: list[Mapping[str, Any]] = []
        self.cancel_calls: list[str] = []
        self.authorization_calls: list[tuple[str, str, Any]] = []

    def describe(self) -> dict[str, Any]:
        return {"name": "stub", "capabilities": {}}

    async def start(self, context: object = None) -> None:
        return None

    async def health(self) -> dict[str, bool]:
        return {"ok": True}

    async def submit(
        self, task: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        self.submissions.append(task)
        return {"content": "完成", "artifacts": []}

    async def status(self, task_id: str | None = None, context: object = None) -> dict[str, str]:
        return {"state": "working"}

    async def cancel(self, task_id: str, context: object = None) -> dict[str, str]:
        self.cancel_calls.append(task_id)
        return {"taskId": task_id, "state": "cancelled"}

    async def respond_authorization(
        self,
        task_id: str,
        authorization_id: str,
        decision: Any,
        context: object = None,
    ) -> None:
        self.authorization_calls.append((task_id, authorization_id, decision))

    def subscribe(self, listener: object) -> Callable[[], None]:
        return lambda: None

    async def close(self) -> None:
        return None


class PermissionBackend(BackendStub):
    def __init__(self, *, resolution_status: str = "approved") -> None:
        super().__init__()
        self.permission_answered = asyncio.Event()
        self.resolution_status = resolution_status

    async def submit(
        self, task: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        self.submissions.append(task)
        assert context is not None
        context["on_event"](
            {
                "type": BackendEventType.AUTHORIZATION_REQUESTED,
                "permission": {
                    "id": "auth_1",
                    "status": "pending",
                    "summary": "运行测试",
                    "category": "bash",
                },
            }
        )
        await self.permission_answered.wait()
        return {"content": "授权后完成"}

    async def respond_authorization(
        self,
        task_id: str,
        authorization_id: str,
        decision: Any,
        context: object = None,
    ) -> Mapping[str, Any]:
        await super().respond_authorization(task_id, authorization_id, decision, context)
        self.permission_answered.set()
        return {"id": authorization_id, "status": self.resolution_status}


class MultiplePermissionBackend(BackendStub):
    def __init__(self) -> None:
        super().__init__()
        self.answers = {"auth_1": asyncio.Event(), "auth_2": asyncio.Event()}

    async def submit(
        self, task: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        assert context is not None
        for authorization_id in ("auth_1", "auth_2"):
            context["on_event"](
                {
                    "type": BackendEventType.AUTHORIZATION_REQUESTED,
                    "permission": {
                        "id": authorization_id,
                        "status": "pending",
                        "summary": f"运行 {authorization_id}",
                        "category": "bash",
                    },
                }
            )
            await self.answers[authorization_id].wait()
        return {"content": "两次授权后完成"}

    async def respond_authorization(
        self,
        task_id: str,
        authorization_id: str,
        decision: Any,
        context: object = None,
    ) -> Mapping[str, Any]:
        await super().respond_authorization(task_id, authorization_id, decision, context)
        self.answers[authorization_id].set()
        return {"id": authorization_id, "status": "approved"}


class ChattyBackend(BackendStub):
    async def submit(
        self, task: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        self.submissions.append(task)
        assert context is not None
        for index in range(10_000):
            context["on_event"](
                {
                    "type": BackendEventType.MESSAGE,
                    "message": f"stream update {index}",
                }
            )
        return {"content": "complete"}


class BurstBackend(BackendStub):
    def __init__(self, event_type: str) -> None:
        super().__init__()
        self.event_type = event_type

    async def submit(
        self, task: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        assert context is not None
        for index in range(10_000):
            event: dict[str, Any] = {"type": self.event_type}
            if self.event_type == BackendEventType.ARTIFACT:
                event["artifact"] = {
                    "artifactId": f"artifact_{index}",
                    "parts": [{"text": str(index), "mediaType": "text/plain"}],
                }
            elif self.event_type == BackendEventType.AUTHORIZATION_REQUESTED:
                event["permission"] = {
                    "id": f"authorization_{index}",
                    "status": "pending",
                    "summary": f"authorize {index}",
                    "category": "shell",
                }
            context["on_event"](event)
        return {"content": "burst complete"}


class ThreadBurstBackend(BurstBackend):
    async def submit(
        self, task: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        assert context is not None

        def emit_all() -> None:
            for index in range(10_000):
                event: dict[str, Any] = {"type": self.event_type}
                if self.event_type == BackendEventType.ARTIFACT:
                    event["artifact"] = {
                        "artifactId": f"artifact_{index}",
                        "parts": [{"text": str(index), "mediaType": "text/plain"}],
                    }
                elif self.event_type == BackendEventType.AUTHORIZATION_REQUESTED:
                    event["permission"] = {
                        "id": f"authorization_{index}",
                        "status": "pending",
                        "summary": f"authorize {index}",
                        "category": "shell",
                    }
                context["on_event"](event)

        await asyncio.to_thread(emit_all)
        return {"content": "thread burst complete"}


class ConcurrentThreadBackend(BackendStub):
    async def submit(
        self, task: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        assert context is not None

        def emit_concurrently() -> None:
            barrier = Barrier(3)

            def emit(worker: int) -> None:
                barrier.wait()
                for index in range(100):
                    context["on_event"](
                        {
                            "type": BackendEventType.AUTHORIZATION_REQUESTED,
                            "permission": {
                                "id": f"authorization_{worker}_{index}",
                                "status": "pending",
                                "summary": f"authorize {worker} {index}",
                                "category": "shell",
                            },
                        }
                    )

            workers = [Thread(target=emit, args=(worker,)) for worker in range(2)]
            for worker in workers:
                worker.start()
            barrier.wait()
            for worker in workers:
                worker.join()

        await asyncio.to_thread(emit_concurrently)
        return {"content": "concurrent complete"}


class LargeArtifactBurstBackend(BackendStub):
    async def submit(
        self, task: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        assert context is not None
        payload = "x" * 750_000
        for index in range(40):
            context["on_event"](
                {
                    "type": BackendEventType.ARTIFACT,
                    "artifact": {
                        "artifactId": f"large_{index}",
                        "parts": [{"text": payload, "mediaType": "text/plain"}],
                    },
                }
            )
        return {"content": "must not complete"}


class LateQueuedEventWorkGraph(WorkGraph):
    def _enqueue_backend_event(
        self,
        handle: Any,
        event: dict[str, Any],
        *,
        already_reserved: bool = False,
    ) -> None:
        super()._enqueue_backend_event(
            handle,
            event,
            already_reserved=already_reserved,
        )
        if event.get("type") == "runtime.completed":
            super()._enqueue_backend_event(
                handle,
                {
                    "type": BackendEventType.ARTIFACT,
                    "artifact": {
                        "artifactId": "late-artifact",
                        "parts": [{"text": "x" * 1_000_000, "mediaType": "text/plain"}],
                    },
                    "_event_id": f"{handle.task_id}:late-artifact",
                },
            )


class LateForeignBurstBackend(BackendStub):
    def __init__(self) -> None:
        super().__init__()
        self.on_event: Callable[[Mapping[str, Any]], None] | None = None

    async def submit(
        self, task: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        assert context is not None
        self.on_event = context["on_event"]
        return {"content": "done"}


class LateForeignBurstWorkGraph(WorkGraph):
    def __init__(self, *, backend: LateForeignBurstBackend) -> None:
        super().__init__(backend=backend)
        self.flooded = False

    def _enqueue_backend_event(
        self,
        handle: Any,
        event: dict[str, Any],
        *,
        already_reserved: bool = False,
    ) -> None:
        super()._enqueue_backend_event(
            handle,
            event,
            already_reserved=already_reserved,
        )
        if event.get("type") != "runtime.completed" or self.flooded:
            return
        self.flooded = True
        callback = self.backend.on_event
        assert callback is not None

        def flood() -> None:
            for index in range(300):
                callback(
                    {
                        "type": BackendEventType.AUTHORIZATION_REQUESTED,
                        "permission": {
                            "id": f"late-{index}",
                            "summary": "late permission",
                            "status": "pending",
                        },
                    }
                )

        worker = Thread(target=flood)
        worker.start()
        worker.join()


class CancelForeignBurstBackend(BackendStub):
    def __init__(self) -> None:
        super().__init__()
        self.running = asyncio.Event()
        self.submit_done = asyncio.Event()
        self.on_event: Callable[[Mapping[str, Any]], None] | None = None

    async def submit(
        self, task: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        assert context is not None
        self.on_event = context["on_event"]
        self.running.set()
        await self.submit_done.wait()
        return {"content": "late completion"}

    async def cancel(self, task_id: str, context: object = None) -> dict[str, Any]:
        self.cancel_calls.append(task_id)
        callback = self.on_event
        assert callback is not None

        def flood() -> None:
            for index in range(300):
                callback(
                    {
                        "type": BackendEventType.AUTHORIZATION_REQUESTED,
                        "permission": {
                            "id": f"cancel-late-{index}",
                            "summary": "cancel late permission",
                            "status": "pending",
                        },
                    }
                )

        worker = Thread(target=flood)
        worker.start()
        worker.join()
        return {"taskId": task_id, "state": "cancelled"}


class SecretPermissionBackend(PermissionBackend):
    async def submit(
        self, task: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        assert context is not None
        context["on_event"](
            {
                "type": BackendEventType.AUTHORIZATION_REQUESTED,
                "permission": {
                    "id": "auth_1",
                    "status": "pending",
                    "summary": ("CLIENT_SECRET=hunter2 Authorization: Bearer super-secret-token"),
                    "patterns": [
                        "GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz",
                        "CREDENTIAL=secret-value",
                        "Cookie: sid=abc123; theme=dark",
                    ],
                    "category": "shell",
                    "operation": {
                        "title": "Run command",
                        "kind": "execute",
                        "command": (
                            "deploy --password hunter2 --token=secret-token "
                            "https://user:pass@example.com/x"
                        ),
                        "credential": "must-not-cross",
                    },
                },
            }
        )
        await self.permission_answered.wait()
        return {"content": "authorization complete"}


class ConfirmingCancelBackend(BackendStub):
    def __init__(self, confirmation: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.running = asyncio.Event()
        self.cancel_started = asyncio.Event()
        self.confirm_cancel = asyncio.Event()
        self.submit_done = asyncio.Event()
        self.confirmation = dict(confirmation or {"taskId": "unused", "state": "cancelled"})

    async def submit(
        self, task: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        self.submissions.append(task)
        self.running.set()
        await self.submit_done.wait()
        return {"content": "late completion"}

    async def cancel(self, task_id: str, context: object = None) -> dict[str, Any]:
        self.cancel_calls.append(task_id)
        self.cancel_started.set()
        await self.confirm_cancel.wait()
        self.submit_done.set()
        return {**self.confirmation, "taskId": task_id}


def test_event_reducers_are_bounded_and_replay_safe() -> None:
    events = reduce_work_events([], {"eventId": "one", "type": "task.progress"})
    events = reduce_work_events(events, {"eventId": "one", "type": "task.progress"})
    assert len(events) == 1

    activity = reduce_activity(
        [],
        [
            {"id": "thinking", "kind": "thinking", "detail": "secret thought"},
            {"id": "tool", "kind": "tool", "tool": "read"},
            {"id": "thinking", "kind": "thinking", "status": "done"},
        ],
    )
    assert [item["id"] for item in activity] == ["tool", "thinking"]
    assert "detail" not in activity[-1]


def test_public_activity_redacts_common_command_and_environment_secrets() -> None:
    activity = reduce_activity(
        [],
        {
            "id": "tool-secret",
            "kind": "tool",
            "label": "Authorization: Bearer super-secret-token",
            "detail": (
                "GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz "
                "--password hunter2 "
                "https://example.test/?api_key=visible"
            ),
        },
    )

    serialized = str(activity)
    assert "super-secret-token" not in serialized
    assert "ghp_abcdefghijklmnopqrstuvwxyz" not in serialized
    assert "hunter2" not in serialized
    assert "api_key=visible" not in serialized
    assert serialized.count("[REDACTED]") >= 4


@pytest.mark.asyncio
async def test_work_graph_checkpoints_each_resume_and_completes() -> None:
    backend = BackendStub()
    runtime = WorkGraph(backend=backend)
    events: list[Mapping[str, Any]] = []
    runtime.subscribe(events.append)

    accepted = await runtime.submit(objective="检查项目", owner_id="owner")
    assert accepted["status"] == "queued"
    completed = await runtime.wait(accepted["id"])

    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["result"] == "完成"
    assert backend.submissions == [
        {
            "id": accepted["id"],
            "ownerId": "owner",
            "objective": "检查项目",
            "instruction": "检查项目",
            "inputParts": [],
        }
    ]
    assert [event["type"] for event in events] == [
        "task.accepted",
        "task.running",
        "task.completed",
    ]
    checkpoint = await runtime.checkpoint(accepted["id"])
    assert checkpoint.values["status"] == "completed"
    await runtime.close()


@pytest.mark.asyncio
async def test_streaming_messages_are_coalesced_without_delaying_completion() -> None:
    runtime = WorkGraph(backend=ChattyBackend())
    accepted = await runtime.submit(objective="chatty", owner_id="owner")

    completed = await asyncio.wait_for(runtime.wait(accepted["id"]), timeout=2)

    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["result"] == "complete"
    assert completed["message"] == "stream update 9999"
    await runtime.close()


@pytest.mark.asyncio
async def test_unknown_backend_event_burst_is_dropped_at_ingress() -> None:
    runtime = WorkGraph(backend=BurstBackend("backend.private.unknown"))
    accepted = await runtime.submit(objective="unknown burst", owner_id="owner")

    completed = await asyncio.wait_for(runtime.wait(accepted["id"]), timeout=2)

    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["result"] == "burst complete"
    assert runtime._handles[accepted["id"]].event_queue.maxsize == 256
    await runtime.close()


@pytest.mark.asyncio
async def test_artifact_burst_is_coalesced_into_bounded_state() -> None:
    runtime = WorkGraph(backend=BurstBackend(BackendEventType.ARTIFACT))
    accepted = await runtime.submit(objective="artifact burst", owner_id="owner")

    completed = await asyncio.wait_for(runtime.wait(accepted["id"]), timeout=2)

    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["result"] == "burst complete"
    assert len(completed["artifacts"]) <= 32
    assert runtime._handles[accepted["id"]].queue_overflowed is False
    await runtime.close()


@pytest.mark.asyncio
async def test_non_coalescible_backend_event_burst_fails_deterministically() -> None:
    backend = BurstBackend(BackendEventType.AUTHORIZATION_REQUESTED)
    runtime = WorkGraph(backend=backend)
    accepted = await runtime.submit(objective="permission burst", owner_id="owner")

    failed = await asyncio.wait_for(runtime.wait(accepted["id"]), timeout=2)

    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["error"] == "Backend event queue exceeded its bounded capacity"
    assert runtime._handles[accepted["id"]].queue_overflowed is True
    await eventually(lambda: backend.cancel_calls == [accepted["id"]])
    await runtime.close()


@pytest.mark.asyncio
async def test_backend_event_byte_budget_fails_and_cancels_large_artifact_burst() -> None:
    backend = LargeArtifactBurstBackend()
    runtime = WorkGraph(backend=backend)
    accepted = await runtime.submit(objective="large artifact burst", owner_id="owner")

    failed = await asyncio.wait_for(runtime.wait(accepted["id"]), timeout=5)

    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["error"] == "Backend event queue exceeded its bounded capacity"
    await eventually(lambda: backend.cancel_calls == [accepted["id"]])
    handle = runtime._handles[accepted["id"]]
    assert handle.queue_overflowed is True
    assert handle.buffered_event_bytes == 0
    await runtime.close()


@pytest.mark.asyncio
async def test_terminal_transition_discards_events_queued_behind_completion() -> None:
    runtime = LateQueuedEventWorkGraph(backend=BackendStub())
    accepted = await runtime.submit(objective="complete before late event", owner_id="owner")

    completed = await asyncio.wait_for(runtime.wait(accepted["id"]), timeout=2)

    assert completed is not None and completed["status"] == "completed"
    handle = runtime._handles[accepted["id"]]
    assert handle.event_queue.empty()
    assert handle.coalesced_events == {}
    assert handle.buffered_event_bytes == 0
    assert handle.buffered_event_sizes == {}
    await runtime.close()


@pytest.mark.asyncio
async def test_terminal_signal_closes_foreign_ingress_before_late_callback_burst() -> None:
    backend = LateForeignBurstBackend()
    runtime = LateForeignBurstWorkGraph(backend=backend)
    accepted = await runtime.submit(objective="complete before foreign burst", owner_id="owner")

    completed = await asyncio.wait_for(runtime.wait(accepted["id"]), timeout=2)
    await asyncio.sleep(0.05)

    assert completed is not None and completed["status"] == "completed"
    assert backend.cancel_calls == []
    handle = runtime._handles[accepted["id"]]
    assert handle.queue_overflowed is False
    assert handle.foreign_queue_overflowed is False
    assert handle.buffered_event_bytes == 0
    await runtime.close()


@pytest.mark.asyncio
async def test_foreign_thread_artifact_burst_uses_one_bounded_ingress_coalescer() -> None:
    runtime = WorkGraph(backend=ThreadBurstBackend(BackendEventType.ARTIFACT))
    accepted = await runtime.submit(objective="thread artifact burst", owner_id="owner")

    completed = await asyncio.wait_for(runtime.wait(accepted["id"]), timeout=2)

    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["result"] == "thread burst complete"
    assert len(completed["artifacts"]) <= 32
    handle = runtime._handles[accepted["id"]]
    assert handle.queue_overflowed is False
    assert len(handle.foreign_events) <= 256
    await runtime.close()


@pytest.mark.asyncio
async def test_foreign_thread_permission_burst_fails_and_cancels_backend() -> None:
    backend = ThreadBurstBackend(BackendEventType.AUTHORIZATION_REQUESTED)
    runtime = WorkGraph(backend=backend)
    accepted = await runtime.submit(objective="thread permission burst", owner_id="owner")

    failed = await asyncio.wait_for(runtime.wait(accepted["id"]), timeout=2)

    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["error"] == "Backend event queue exceeded its bounded capacity"
    await eventually(lambda: backend.cancel_calls == [accepted["id"]])
    handle = runtime._handles[accepted["id"]]
    assert handle.queue_overflowed is True
    assert len(handle.foreign_events) <= 256
    await runtime.close()


@pytest.mark.asyncio
async def test_concurrent_backend_threads_receive_unique_gateway_event_ids() -> None:
    runtime = WorkGraph(backend=ConcurrentThreadBackend())
    events: list[Mapping[str, Any]] = []
    runtime.subscribe(events.append)
    accepted = await runtime.submit(objective="concurrent permissions", owner_id="owner")

    completed = await asyncio.wait_for(runtime.wait(accepted["id"]), timeout=2)

    assert completed is not None and completed["status"] == "completed"
    permission_events = [
        event for event in events if event.get("type") == "task.permission.requested"
    ]
    assert len(permission_events) == 200
    checkpoint = await runtime.checkpoint(accepted["id"])
    processed = checkpoint.values.get("processed_event_ids", [])
    assert len(processed) == len(set(processed))
    await runtime.close()


@pytest.mark.asyncio
async def test_notification_delivery_callback_publishes_the_durable_task() -> None:
    runtime = WorkGraph(backend=BackendStub())
    events: list[Mapping[str, Any]] = []
    runtime.subscribe(events.append)
    accepted = await runtime.submit(
        objective="检查项目",
        owner_id="owner",
        session_id="voice",
    )
    completed = await runtime.wait(accepted["id"])
    assert completed is not None
    leases = TaskNotificationLeaseAdapter(
        runtime.repository,
        on_delivered=runtime.notification_delivered,
    )
    claimed = leases.claim(
        owner_id="owner",
        session_id="voice",
        claimant_id="voice:one",
    )
    assert [record.task_id for record in claimed] == [accepted["id"]]

    events.clear()
    assert leases.mark_delivered([accepted["id"]], claimant_id="voice:one") == 1

    assert len(events) == 1
    assert events[0]["type"] == "task.notification.delivered"
    assert events[0]["task"]["notificationStatus"] == "delivered"
    await runtime.close()


@pytest.mark.asyncio
async def test_authorization_is_checkpointed_and_relayed_by_task_id() -> None:
    backend = PermissionBackend()
    runtime = WorkGraph(backend=backend)
    accepted = await runtime.submit(objective="运行测试", owner_id="owner")

    await eventually(lambda: runtime.get(accepted["id"])["workState"] == "auth_required")
    pending = runtime.get(accepted["id"])
    assert pending is not None
    assert pending["authorization"]["taskId"] == accepted["id"]
    assert pending["authorization"]["approvalScope"] == "once"

    await runtime.respond_authorization(accepted["id"], "auth_1", "always", owner_id="owner")
    completed = await runtime.wait(accepted["id"])
    assert completed is not None
    assert completed["status"] == "completed"
    assert backend.authorization_calls == [(accepted["id"], "auth_1", "always")]
    await runtime.close()


@pytest.mark.asyncio
async def test_authorization_operation_is_redacted_before_checkpointing() -> None:
    backend = SecretPermissionBackend()
    runtime = WorkGraph(backend=backend)
    accepted = await runtime.submit(objective="secret authorization", owner_id="owner")
    await eventually(lambda: runtime.get(accepted["id"])["workState"] == "auth_required")

    pending = runtime.get(accepted["id"])
    assert pending is not None
    authorization = pending["authorization"]
    serialized = str(authorization)
    assert authorization["taskId"] == accepted["id"]
    assert authorization["approvalScope"] == "once"
    assert "super-secret-token" not in serialized
    assert "ghp_abcdefghijklmnopqrstuvwxyz" not in serialized
    assert "hunter2" not in serialized
    assert "secret-token" not in serialized
    assert "secret-value" not in serialized
    assert "sid=abc123" not in serialized
    assert "user:pass" not in serialized
    assert "credential" not in serialized

    await runtime.respond_authorization(accepted["id"], "auth_1", "once", owner_id="owner")
    completed = await runtime.wait(accepted["id"])
    assert completed is not None and completed["status"] == "completed"
    await runtime.close()


@pytest.mark.asyncio
async def test_authorization_resolution_uses_backend_outcome_not_requested_decision() -> None:
    backend = PermissionBackend(resolution_status="cancelled")
    runtime = WorkGraph(backend=backend)
    events: list[Mapping[str, Any]] = []
    runtime.subscribe(events.append)
    accepted = await runtime.submit(objective="运行测试", owner_id="owner")
    await eventually(lambda: runtime.get(accepted["id"])["workState"] == "auth_required")

    await runtime.respond_authorization(
        accepted["id"],
        "auth_1",
        "always",
        owner_id="owner",
    )
    await runtime.wait(accepted["id"])

    resolved = next(event for event in events if event["type"] == "task.permission.resolved")
    assert resolved["permission"]["status"] == "cancelled"
    assert not any(
        event.get("permission", {}).get("status") == "approved"
        for event in events
        if isinstance(event.get("permission"), Mapping)
    )
    await runtime.close()


@pytest.mark.asyncio
async def test_stale_resolution_cannot_clear_the_next_permission() -> None:
    backend = MultiplePermissionBackend()
    runtime = WorkGraph(backend=backend)
    accepted = await runtime.submit(objective="运行两次授权测试", owner_id="owner")
    await eventually(
        lambda: (
            ((runtime.get(accepted["id"]) or {}).get("authorization") or {}).get("id") == "auth_1"
        )
    )

    await runtime.respond_authorization(
        accepted["id"],
        "auth_1",
        "always",
        owner_id="owner",
    )
    await eventually(
        lambda: (
            ((runtime.get(accepted["id"]) or {}).get("authorization") or {}).get("id") == "auth_2"
        )
    )
    await asyncio.sleep(0)
    assert runtime.get(accepted["id"])["authorization"]["id"] == "auth_2"

    await runtime.respond_authorization(
        accepted["id"],
        "auth_2",
        "always",
        owner_id="owner",
    )
    completed = await runtime.wait(accepted["id"])
    assert completed is not None and completed["status"] == "completed"
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "confirmation",
    [
        {"state": "cancelled"},
        {"state": "canceled"},
        {"cancelled": True},
    ],
)
async def test_active_cancel_stays_cancelling_until_backend_confirms(
    confirmation: Mapping[str, Any],
) -> None:
    backend = ConfirmingCancelBackend(confirmation)
    runtime = WorkGraph(backend=backend)
    accepted = await runtime.submit(objective="长任务", owner_id="owner")
    await backend.running.wait()

    cancellation = asyncio.create_task(runtime.cancel(accepted["id"], owner_id="owner"))
    await backend.cancel_started.wait()
    assert runtime.get(accepted["id"])["status"] == "cancelling"
    assert not cancellation.done()

    backend.confirm_cancel.set()
    cancelled = await cancellation
    assert cancelled is not None
    assert cancelled["status"] == "cancelled"
    assert backend.cancel_calls == [accepted["id"]]
    await runtime.close()


@pytest.mark.asyncio
async def test_active_cancel_closes_foreign_ingress_before_backend_confirmation() -> None:
    backend = CancelForeignBurstBackend()
    runtime = WorkGraph(backend=backend)
    accepted = await runtime.submit(objective="cancel before foreign burst", owner_id="owner")
    await backend.running.wait()

    cancelled = await runtime.cancel(accepted["id"], owner_id="owner")
    await asyncio.sleep(0.05)

    assert cancelled is not None and cancelled["status"] == "cancelled"
    assert backend.cancel_calls == [accepted["id"]]
    handle = runtime._handles[accepted["id"]]
    assert handle.queue_overflowed is False
    assert handle.foreign_queue_overflowed is False
    assert handle.buffered_event_bytes == 0
    await runtime.close()


@pytest.mark.asyncio
async def test_cancel_closes_ingress_before_waiting_for_the_graph_lock() -> None:
    backend = CancelForeignBurstBackend()
    runtime = WorkGraph(backend=backend)
    accepted = await runtime.submit(objective="cancel lock race", owner_id="owner")
    await backend.running.wait()
    handle = runtime._handles[accepted["id"]]
    await handle.invoke_lock.acquire()
    cancellation = asyncio.create_task(runtime.cancel(accepted["id"], owner_id="owner"))
    await eventually(lambda: handle.cancellation_requested)

    callback = backend.on_event
    assert callback is not None
    callback({"type": BackendEventType.MESSAGE, "message": "must be dropped"})
    handle.invoke_lock.release()
    cancelled = await asyncio.wait_for(cancellation, timeout=2)

    assert cancelled is not None and cancelled["status"] == "cancelled"
    assert cancelled["message"] is None
    assert backend.cancel_calls == [accepted["id"]]
    assert handle.queue_overflowed is False
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["failed", "unknown", "not_found"])
async def test_active_cancel_fails_without_explicit_backend_confirmation(state: str) -> None:
    backend = ConfirmingCancelBackend({"state": state})
    runtime = WorkGraph(backend=backend)
    events: list[Mapping[str, Any]] = []
    runtime.subscribe(events.append)
    accepted = await runtime.submit(objective="长任务", owner_id="owner")
    await backend.running.wait()

    cancellation = asyncio.create_task(runtime.cancel(accepted["id"], owner_id="owner"))
    await backend.cancel_started.wait()
    backend.confirm_cancel.set()
    failed = await cancellation

    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["notificationStatus"] == "pending"
    assert failed["error"] == f"取消失败: backend returned cancellation state {state}"
    assert backend.cancel_calls == [accepted["id"]]
    assert events[-1]["type"] == "task.failed"
    await runtime.close()
