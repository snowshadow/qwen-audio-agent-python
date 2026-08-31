from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from qwen_audio_agent.application import FrontendRealtimeApplication
from qwen_audio_agent.protocol import GatewayClientEvent, GatewayServerEvent
from qwen_audio_agent.runtime.delivery import (
    DeliveryLeaseQueue,
    DeliveryRecord,
    DeliveryStatus,
)


async def eventually(predicate: Callable[[], bool], deadline_seconds: float = 2.0) -> None:
    async with asyncio.timeout(deadline_seconds):
        while not predicate():  # noqa: ASYNC110 - polling an in-memory application
            await asyncio.sleep(0.001)


class FakeProvider:
    key = "fake"

    def __init__(self, *, metadata_correlation: bool = True) -> None:
        self.metadata_correlation = metadata_correlation
        self.connected = False
        self.closed = False
        self.connect_calls = 0
        self.reconnect_gate: asyncio.Event | None = None
        self.emit_ready_on_reconnect = False
        self.inputs: list[dict[str, Any]] = []
        self.tool_results: list[dict[str, Any]] = []
        self.response_requests: list[dict[str, Any]] = []
        self.cancel_calls = 0
        self.listeners: set[Callable[[Mapping[str, Any]], None]] = set()

    async def connect(self) -> None:
        self.connect_calls += 1
        if self.connect_calls > 1 and self.reconnect_gate is not None:
            await self.reconnect_gate.wait()
        self.connected = True
        if self.connect_calls > 1 and self.emit_ready_on_reconnect:
            self.emit({"type": "connection.ready"})

    async def update_session(self, context: Mapping[str, Any]) -> None:
        return None

    async def send_audio(self, audio: bytes | str) -> None:
        return None

    async def send_input(self, value: Mapping[str, Any]) -> None:
        self.inputs.append(dict(value))

    async def send_tool_result(self, value: Mapping[str, Any]) -> None:
        self.tool_results.append(dict(value))

    async def create_response(self, context: Mapping[str, Any] | None = None) -> None:
        self.response_requests.append(dict(context or {}))

    async def cancel_response(self, response_id: str | None = None) -> None:
        self.cancel_calls += 1

    async def close(self) -> None:
        self.closed = True

    def capabilities(self) -> dict[str, Any]:
        return {
            "provider": self.key,
            "responseMetadataCorrelation": self.metadata_correlation,
            "frontendCapabilities": [],
            "inputSampleRate": 16_000,
            "outputSampleRate": 24_000,
        }

    def subscribe(
        self,
        listener: Callable[[Mapping[str, Any]], None],
    ) -> Callable[[], None]:
        self.listeners.add(listener)
        return lambda: self.listeners.discard(listener)

    def emit(self, event: Mapping[str, Any]) -> None:
        for listener in tuple(self.listeners):
            listener(dict(event))


class FakeFrontendGraph:
    def __init__(self, result: Mapping[str, Any] | None = None) -> None:
        self.result = dict(result) if result is not None else None
        self.calls: list[tuple[Any, Any]] = []

    async def ainvoke(self, call: Any, context: Any) -> dict[str, Any]:
        self.calls.append((call, context))
        if self.result is None:
            raise AssertionError("these tests do not execute frontend tools")
        return {"result": dict(self.result)}


class ConcurrentFrontendGraph(FakeFrontendGraph):
    def __init__(self) -> None:
        super().__init__({"status": "ok"})
        self.active_calls = 0
        self.max_active_calls = 0

    async def ainvoke(self, call: Any, context: Any) -> dict[str, Any]:
        self.calls.append((call, context))
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        await asyncio.sleep(0.01)
        self.active_calls -= 1
        return {"result": dict(self.result or {})}


class BlockingFrontendGraph(FakeFrontendGraph):
    def __init__(self) -> None:
        super().__init__({"status": "ok"})
        self.started = asyncio.Event()
        self.cancelled = False

    async def ainvoke(self, call: Any, context: Any) -> dict[str, Any]:
        self.calls.append((call, context))
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class FakeTaskSource:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, Any]] = {}
        self.listeners: set[Callable[[Mapping[str, Any]], None]] = set()

    def list(self, *, owner_id: str | None = None) -> list[dict[str, Any]]:
        return [
            value
            for value in self.values.values()
            if owner_id is None or value.get("ownerId") == owner_id
        ]

    def subscribe(
        self,
        listener: Callable[[Mapping[str, Any]], None],
    ) -> Callable[[], None]:
        self.listeners.add(listener)
        return lambda: self.listeners.discard(listener)

    def publish(
        self,
        event_type: str,
        task: Mapping[str, Any],
        *,
        permission: Mapping[str, Any] | None = None,
    ) -> None:
        value = dict(task)
        self.values[str(value["id"])] = value
        event = {
            "type": event_type,
            "ownerId": value.get("ownerId"),
            "task": value,
        }
        if permission is not None:
            event["permission"] = dict(permission)
        self.publish_event(event)

    def publish_event(self, event: Mapping[str, Any]) -> None:
        for listener in tuple(self.listeners):
            listener(dict(event))


class UnscopedTaskSource(FakeTaskSource):
    def __init__(self, *, asynchronous: bool) -> None:
        super().__init__()
        self.asynchronous = asynchronous

    def list(self, *, owner_id: str | None = None) -> Any:
        del owner_id
        values = [dict(value) for value in self.values.values()]
        if not self.asynchronous:
            return values

        async def load() -> list[dict[str, Any]]:
            return values

        return load()


@dataclass
class RealtimeHarness:
    application: FrontendRealtimeApplication
    provider: FakeProvider
    tasks: FakeTaskSource
    records: dict[str, DeliveryRecord]
    emitted: list[dict[str, Any]]
    active_voice: list[bool]
    context: dict[str, Any]

    async def close(self) -> None:
        await self.application.close()


async def open_harness(
    *,
    task_ids: tuple[str, ...] = (),
    active_voice: bool = True,
    with_delivery: bool = True,
    metadata_correlation: bool = True,
    text_only: bool = False,
    frontend_graph: FakeFrontendGraph | None = None,
    client_states: tuple[str, ...] = (),
    task_source: FakeTaskSource | None = None,
) -> RealtimeHarness:
    provider = FakeProvider(metadata_correlation=metadata_correlation)
    tasks = task_source or FakeTaskSource()
    records = {
        task_id: DeliveryRecord(
            task_id=task_id,
            owner_id="owner",
            session_id="voice",
            created_at=float(index),
        )
        for index, task_id in enumerate(task_ids, start=1)
    }
    emitted: list[dict[str, Any]] = []
    active = [active_voice]
    context: dict[str, Any] = {
        "connectionId": "connection",
        "ownerId": "owner",
        "sessionId": "voice",
        "emit": lambda event: emitted.append(dict(event)),
        "isVoiceActive": lambda: active[0],
    }
    application = FrontendRealtimeApplication(
        provider_factory=lambda _context, _event: provider,
        frontend_graph=frontend_graph or FakeFrontendGraph(),  # type: ignore[arg-type]
        task_source=tasks,
        delivery_leases=DeliveryLeaseQueue(records) if with_delivery else None,
        presentation_timeout_seconds=1,
    )
    await application(
        context,
        {
            "type": GatewayClientEvent.CONNECT.value,
            "voiceEnabled": True,
            "textOnly": text_only,
            "clientStates": list(client_states),
        },
    )
    await eventually(lambda: provider.connected)
    return RealtimeHarness(
        application=application,
        provider=provider,
        tasks=tasks,
        records=records,
        emitted=emitted,
        active_voice=active,
        context=context,
    )


def completed_task(task_id: str, *, inline: bool = False) -> dict[str, Any]:
    presentation: dict[str, Any] = {"speech": f"{task_id} 已完成", "inline": None}
    if inline:
        presentation["inline"] = {
            "title": "任务结果",
            "format": "code",
            "content": "print('done')",
        }
    return {
        "id": task_id,
        "ownerId": "owner",
        "sessionId": "voice",
        "turnId": "turn-original",
        "status": "completed",
        "objective": "生成结果",
        "result": f"{task_id} result",
        "notificationStatus": "pending",
        "presentation": presentation,
    }


def response_created(request: Mapping[str, Any], response_id: str) -> dict[str, Any]:
    return {
        "type": "response.created",
        "response": {
            "id": response_id,
            "metadata": dict(request.get("metadata") or {}),
        },
    }


def response_done(response_id: str) -> dict[str, Any]:
    return {
        "type": "response.done",
        "response": {"id": response_id, "status": "completed"},
    }


def event_count(harness: RealtimeHarness, event_type: str) -> int:
    return sum(event.get("type") == event_type for event in harness.emitted)


@pytest.mark.asyncio
async def test_task_events_require_matching_nonempty_task_and_source_owners() -> None:
    task_ids = (
        "ownerless-permission",
        "ownerless-completed",
        "source-conflict-permission",
        "source-conflict-completed",
        "task-conflict-permission",
        "task-conflict-completed",
    )
    harness = await open_harness(task_ids=task_ids, text_only=True)
    try:

        def permission(task_id: str, owner_id: str | None) -> dict[str, Any]:
            task = {
                "id": task_id,
                "sessionId": "voice",
                "status": "running",
                "authorization": {
                    "id": f"authorization-{task_id}",
                    "status": "pending",
                    "summary": "must not cross",
                },
            }
            if owner_id is not None:
                task["ownerId"] = owner_id
            return task

        def completed(task_id: str, owner_id: str | None) -> dict[str, Any]:
            task = completed_task(task_id, inline=True)
            if owner_id is None:
                task.pop("ownerId")
            else:
                task["ownerId"] = owner_id
            return task

        for event_type, task, source_owner in (
            (
                "task.permission.requested",
                permission("ownerless-permission", None),
                None,
            ),
            ("task.completed", completed("ownerless-completed", None), None),
            (
                "task.permission.requested",
                permission("source-conflict-permission", "owner"),
                "attacker",
            ),
            (
                "task.completed",
                completed("source-conflict-completed", "owner"),
                "attacker",
            ),
            (
                "task.permission.requested",
                permission("task-conflict-permission", "attacker"),
                "owner",
            ),
            (
                "task.completed",
                completed("task-conflict-completed", "attacker"),
                "owner",
            ),
        ):
            event: dict[str, Any] = {"type": event_type, "task": task}
            if source_owner is not None:
                event["ownerId"] = source_owner
            harness.tasks.publish_event(event)

        await asyncio.sleep(0.05)
        state = harness.application._connections["connection"]
        assert state.pending_permissions == {}
        assert harness.provider.inputs == []
        assert harness.provider.response_requests == []
        assert event_count(harness, GatewayServerEvent.TIMELINE_INLINE.value) == 0
        assert all(record.status is DeliveryStatus.PENDING for record in harness.records.values())
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_notification_backfill_rechecks_owner_when_source_ignores_filter(
    asynchronous: bool,
) -> None:
    tasks = UnscopedTaskSource(asynchronous=asynchronous)
    tasks.values = {
        "ownerless-result": {
            **completed_task("ownerless-result"),
            "ownerId": "",
        },
        "foreign-result": {
            **completed_task("foreign-result"),
            "ownerId": "attacker",
        },
        "ownerless-permission": {
            "id": "ownerless-permission",
            "sessionId": "voice",
            "status": "running",
            "authorization": {
                "id": "authorization-ownerless-backfill",
                "status": "pending",
                "summary": "must not cross",
            },
        },
    }
    harness = await open_harness(
        task_ids=tuple(tasks.values),
        text_only=True,
        task_source=tasks,
    )
    try:
        await asyncio.sleep(0.05)
        state = harness.application._connections["connection"]
        assert state.pending_permissions == {}
        assert harness.provider.inputs == []
        assert harness.provider.response_requests == []
        assert all(record.status is DeliveryStatus.PENDING for record in harness.records.values())
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_completed_result_is_claimed_generated_and_acked_on_playback_start() -> None:
    harness = await open_harness(task_ids=("task-one",))
    try:
        task = completed_task("task-one", inline=True)
        harness.tasks.publish("task.completed", task)
        harness.tasks.publish("task.notification.pending", task)

        await eventually(lambda: len(harness.provider.response_requests) == 1)
        assert harness.records["task-one"].status is DeliveryStatus.DELIVERING
        assert len(harness.provider.inputs) == 1
        assert "task-one result" in harness.provider.inputs[0]["text"]
        assert event_count(harness, GatewayServerEvent.TIMELINE_INLINE.value) == 1

        request = harness.provider.response_requests[0]
        harness.provider.emit(response_created(request, "response-one"))
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
                and event.get("responseId") == "response-one"
                and event.get("taskId") == "task-one"
                for event in harness.emitted
            )
        )
        harness.provider.emit(
            {
                "type": "response.audio.delta",
                "response_id": "response-one",
                "delta": "audio",
            }
        )
        harness.provider.emit(response_done("response-one"))
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.AUDIO_DONE.value
                and event.get("responseId") == "response-one"
                for event in harness.emitted
            )
        )

        # Model generation is not the durable delivery acknowledgement.
        assert harness.records["task-one"].status is DeliveryStatus.DELIVERING
        started = await harness.application(
            harness.context,
            {
                "type": GatewayClientEvent.PLAYBACK_STARTED.value,
                "responseId": "response-one",
            },
        )

        assert started["state"] == "speaking"
        assert harness.records["task-one"].status is DeliveryStatus.DELIVERED
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_playback_cancelled_before_start_retries_without_reinjecting_context() -> None:
    harness = await open_harness(task_ids=("task-retry",))
    try:
        harness.tasks.publish("task.completed", completed_task("task-retry"))
        await eventually(lambda: len(harness.provider.response_requests) == 1)

        first_request = harness.provider.response_requests[0]
        harness.provider.emit(response_created(first_request, "response-first"))
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
                and event.get("responseId") == "response-first"
                for event in harness.emitted
            )
        )
        harness.provider.emit(
            {
                "type": "response.audio.delta",
                "response_id": "response-first",
                "delta": "audio",
            }
        )
        harness.provider.emit(response_done("response-first"))
        state = harness.application._connections["connection"]
        assert state.delivery is not None
        await eventually(lambda: not state.delivery._delivering)

        cancelled = asyncio.create_task(
            harness.application(
                harness.context,
                {
                    "type": GatewayClientEvent.PLAYBACK_CANCELLED.value,
                    "responseId": "response-first",
                },
            )
        )
        await eventually(lambda: len(harness.provider.response_requests) == 2)
        assert harness.records["task-retry"].status is DeliveryStatus.DELIVERING
        assert len(harness.provider.inputs) == 1

        second_request = harness.provider.response_requests[1]
        harness.provider.emit(response_created(second_request, "response-second"))
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
                and event.get("responseId") == "response-second"
                for event in harness.emitted
            )
        )
        harness.provider.emit(
            {
                "type": "response.audio.delta",
                "response_id": "response-second",
                "delta": "audio",
            }
        )
        harness.provider.emit(response_done("response-second"))
        cancelled_result = await asyncio.wait_for(cancelled, timeout=1)

        assert cancelled_result["state"] == "idle"
        assert harness.records["task-retry"].status is DeliveryStatus.DELIVERING
        await harness.application(
            harness.context,
            {
                "type": GatewayClientEvent.PLAYBACK_STARTED.value,
                "responseId": "response-second",
            },
        )
        assert harness.records["task-retry"].status is DeliveryStatus.DELIVERED
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_restart_backfill_reclaims_an_expired_delivery_claim() -> None:
    harness = await open_harness(task_ids=("task-stale-claim",))
    try:
        record = harness.records["task-stale-claim"]
        record.status = DeliveryStatus.DELIVERING
        record.claimant_id = "dead-client"
        record.claimed_at = 0
        task = {
            **completed_task("task-stale-claim"),
            "notificationStatus": "delivering",
        }

        state = harness.application._connections["connection"]
        harness.application._queue_restored_task(state, task)

        await eventually(lambda: len(harness.provider.response_requests) == 1)
        assert record.status is DeliveryStatus.DELIVERING
        assert record.claimant_id == "voice:connection"
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_notification_pending_does_not_repeat_completed_inline_projection() -> None:
    harness = await open_harness(with_delivery=False)
    try:
        task = completed_task("task-inline", inline=True)
        harness.tasks.publish("task.completed", task)
        harness.tasks.publish("task.notification.pending", task)
        await eventually(
            lambda: event_count(harness, GatewayServerEvent.TIMELINE_INLINE.value) == 1
        )
        await asyncio.sleep(0.01)

        inline = [
            event
            for event in harness.emitted
            if event.get("type") == GatewayServerEvent.TIMELINE_INLINE.value
        ]
        assert inline == [
            {
                "type": GatewayServerEvent.TIMELINE_INLINE.value,
                "item": {
                    "id": "inline_task-inline",
                    "taskId": "task-inline",
                    "title": "任务结果",
                    "format": "code",
                    "content": "print('done')",
                },
            }
        ]
    finally:
        await harness.close()


@pytest.mark.parametrize(
    ("blocked_by", "resume_event"),
    [
        ("mute", GatewayClientEvent.UNMUTE.value),
        ("sleep", GatewayClientEvent.WAKE.value),
        ("inactive-voice", GatewayClientEvent.UNMUTE.value),
    ],
)
@pytest.mark.asyncio
async def test_result_delivery_waits_while_output_is_blocked(
    blocked_by: str,
    resume_event: str,
) -> None:
    harness = await open_harness(
        task_ids=(f"task-{blocked_by}",),
        active_voice=blocked_by != "inactive-voice",
    )
    try:
        if blocked_by in {"mute", "sleep"}:
            await harness.application(
                harness.context,
                {"type": blocked_by},
            )

        task_id = f"task-{blocked_by}"
        harness.tasks.publish("task.completed", completed_task(task_id))
        await asyncio.sleep(0.2)
        assert harness.provider.response_requests == []
        assert harness.records[task_id].status is DeliveryStatus.PENDING

        harness.active_voice[0] = True
        resumed = asyncio.create_task(harness.application(harness.context, {"type": resume_event}))
        await eventually(lambda: len(harness.provider.response_requests) == 1)
        assert harness.records[task_id].status is DeliveryStatus.DELIVERING

        request = harness.provider.response_requests[0]
        response_id = f"response-{blocked_by}"
        harness.provider.emit(response_created(request, response_id))
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
                and event.get("responseId") == response_id
                for event in harness.emitted
            )
        )
        harness.provider.emit(
            {
                "type": "response.audio.delta",
                "response_id": response_id,
                "delta": "audio",
            }
        )
        harness.provider.emit(response_done(response_id))
        await asyncio.wait_for(resumed, timeout=1)
        await harness.application(
            harness.context,
            {
                "type": GatewayClientEvent.PLAYBACK_STARTED.value,
                "responseId": response_id,
            },
        )
        assert harness.records[task_id].status is DeliveryStatus.DELIVERED
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_response_metadata_correlates_out_of_order_provider_responses() -> None:
    harness = await open_harness(with_delivery=False, text_only=True)
    try:
        state = harness.application._connections["connection"]
        first = asyncio.create_task(
            harness.application._present_result(
                state,
                "first",
                {"turnId": "turn-first", "taskIds": ["task-first"]},
                True,
            )
        )
        second = asyncio.create_task(
            harness.application._present_result(
                state,
                "second",
                {"turnId": "turn-second", "taskIds": ["task-second"]},
                True,
            )
        )
        await eventually(lambda: len(harness.provider.response_requests) == 2)
        first_request, second_request = harness.provider.response_requests
        first_id = first_request["metadata"]["qwen_audio_request_id"]
        second_id = second_request["metadata"]["qwen_audio_request_id"]
        assert first_id and second_id and first_id != second_id

        harness.provider.emit(response_created(second_request, "response-second"))
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
                and event.get("responseId") == "response-second"
                and event.get("taskId") == "task-second"
                and event.get("turnId") == "turn-second"
                for event in harness.emitted
            )
        )
        harness.provider.emit(response_created(first_request, "response-first"))
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
                and event.get("responseId") == "response-first"
                and event.get("taskId") == "task-first"
                and event.get("turnId") == "turn-first"
                for event in harness.emitted
            )
        )
        harness.provider.emit(response_done("response-second"))
        harness.provider.emit(response_done("response-first"))

        outcomes = await asyncio.wait_for(
            asyncio.gather(first, second),
            timeout=1,
        )
        assert all(outcome.completed and outcome.context_injected for outcome in outcomes)
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_permission_request_is_injected_and_resolution_cleans_up_context() -> None:
    harness = await open_harness(with_delivery=False, text_only=True)
    try:
        task = {
            "id": "task-permission",
            "ownerId": "owner",
            "sessionId": "voice",
            "status": "running",
            "authorization": {
                "id": "authorization-one",
                "status": "pending",
                "category": "filesystem",
                "summary": "允许写入报告",
                "operation": {
                    "title": "写入文件",
                    "description": "保存最终报告",
                    "path": "/workspace/report.md",
                },
            },
        }
        harness.tasks.publish("task.permission.requested", task)

        await eventually(lambda: len(harness.provider.response_requests) == 1)
        assert len(harness.provider.inputs) == 1
        permission_input = harness.provider.inputs[0]["text"]
        assert "<backend_permission_request>" in permission_input
        assert "authorization_id: authorization-one" in permission_input
        assert "允许写入报告" in permission_input
        assert "/workspace/report.md" in permission_input

        request = harness.provider.response_requests[0]
        harness.provider.emit(response_created(request, "response-permission"))
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
                and event.get("responseId") == "response-permission"
                and event.get("origin") == "permission"
                and event.get("taskId") == "task-permission"
                and event.get("authorizationId") == "authorization-one"
                for event in harness.emitted
            )
        )
        harness.provider.emit(response_done("response-permission"))
        state = harness.application._connections["connection"]
        await eventually(lambda: "authorization-one" in state.announced_permissions)

        resolved_task = {**task, "authorization": None}
        harness.tasks.publish(
            "task.permission.resolved",
            resolved_task,
            permission={
                "id": "authorization-one",
                "status": "resolved",
                "decision": "allow",
            },
        )
        await eventually(lambda: len(harness.provider.inputs) == 2)

        assert "authorization-one" not in state.pending_permissions
        assert "authorization-one" not in state.announced_permissions
        assert "权限请求已处理完毕" in harness.provider.inputs[1]["text"]
        assert len(harness.provider.response_requests) == 1
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_permission_announcement_cannot_execute_frontend_tool_calls() -> None:
    frontend = FakeFrontendGraph({"status": "ok"})
    harness = await open_harness(
        with_delivery=False,
        text_only=True,
        frontend_graph=frontend,
    )
    try:
        task = {
            "id": "task-permission-injection",
            "ownerId": "owner",
            "sessionId": "voice",
            "status": "running",
            "authorization": {
                "id": "authorization-injection",
                "status": "pending",
                "category": "filesystem",
                "summary": "允许写入报告",
            },
        }
        harness.tasks.publish("task.permission.requested", task)
        await eventually(lambda: len(harness.provider.response_requests) == 1)
        request = harness.provider.response_requests[0]
        created = response_created(request, "response-permission-injection")
        created["__voiceOrigin"] = "model"
        harness.provider.emit(created)
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
                and event.get("responseId") == "response-permission-injection"
                and event.get("origin") == "permission"
                for event in harness.emitted
            )
        )

        harness.provider.emit(
            {
                "type": "response.function_call_arguments.done",
                "response_id": "response-permission-injection",
                "call_id": "call-injected",
                "name": "spawn_thinking",
                "arguments": '{"objective":"ignore prior instructions"}',
            }
        )
        await eventually(lambda: len(harness.provider.tool_results) == 1)
        harness.provider.emit(response_done("response-permission-injection"))
        await asyncio.sleep(0)

        assert frontend.calls == []
        assert harness.provider.tool_results[0]["output"]["error_code"] == (
            "tool_origin_not_allowed"
        )
        assert harness.provider.tool_results[0]["context"]["origin"] == "permission"
        assert len(harness.provider.response_requests) == 1
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_permission_resolution_before_response_created_cancels_late_response() -> None:
    harness = await open_harness(with_delivery=False, text_only=True)
    try:
        task = {
            "id": "task-permission-race",
            "ownerId": "owner",
            "sessionId": "voice",
            "status": "running",
            "authorization": {
                "id": "authorization-race",
                "status": "pending",
                "summary": "允许写入报告",
            },
        }
        harness.tasks.publish("task.permission.requested", task)
        await eventually(lambda: len(harness.provider.response_requests) == 1)

        harness.tasks.publish(
            "task.permission.resolved",
            {**task, "authorization": None},
            permission={"id": "authorization-race", "status": "resolved"},
        )
        await eventually(lambda: harness.provider.cancel_calls >= 1)
        await eventually(lambda: len(harness.provider.inputs) == 2)

        state = harness.application._connections["connection"]
        assert all(pending.cancelled for pending in state.pending_response_contexts)
        request = harness.provider.response_requests[0]
        harness.provider.emit(response_created(request, "response-late-permission"))
        await eventually(lambda: harness.provider.cancel_calls >= 2)

        assert not any(
            event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
            and event.get("responseId") == "response-late-permission"
            for event in harness.emitted
        )
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_enter_sleep_tool_emits_client_state_without_follow_up_response() -> None:
    frontend = FakeFrontendGraph({"status": "sleeping"})
    harness = await open_harness(
        with_delivery=False,
        frontend_graph=frontend,
        client_states=("sleeping",),
    )
    try:
        harness.provider.emit(
            {
                "type": "response.created",
                "response": {"id": "response-model"},
                "__voiceContext": {
                    "turnId": "turn-one",
                    "turnGeneration": 1,
                },
            }
        )
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
                and event.get("responseId") == "response-model"
                for event in harness.emitted
            )
        )
        harness.provider.emit(
            {
                "type": "response.function_call_arguments.done",
                "response_id": "response-model",
                "call_id": "call-sleep",
                "name": "enter_sleep",
                "arguments": "{}",
            }
        )
        await eventually(lambda: len(harness.provider.tool_results) == 1)
        await eventually(lambda: event_count(harness, GatewayServerEvent.VOICE_SLEEP.value) == 1)

        call, execution = frontend.calls[0]
        assert call.name == "enter_sleep"
        assert execution.client_states == ("sleeping",)
        assert harness.provider.tool_results == [
            {
                "callId": "call-sleep",
                "output": {"status": "sleeping"},
                "context": {
                    "turnId": "turn-one",
                    "turnGeneration": 1,
                    "responseId": "response-model",
                    "origin": "model",
                },
            }
        ]
        assert harness.provider.response_requests == []
        assert {
            "type": GatewayServerEvent.CLIENT_STATE.value,
            "state": "sleeping",
        } in harness.emitted
        assert {
            "type": GatewayServerEvent.VOICE_SLEEP.value,
            "state": "sleeping",
        } in harness.emitted
        state = harness.application._connections["connection"]
        assert state.sleeping is True
        assert state.input_enabled is False
        assert state.output_enabled is False
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_sleep_rejects_manual_input_until_wake() -> None:
    harness = await open_harness(with_delivery=False)
    try:
        await harness.application(
            harness.context,
            {"type": GatewayClientEvent.SLEEP.value},
        )
        rejected = await harness.application(
            harness.context,
            {"type": GatewayClientEvent.TEXT_MESSAGE.value, "text": "还在吗"},
        )

        assert rejected == {
            "type": GatewayServerEvent.ERROR.value,
            "message": "已休眠，请先唤醒。",
        }
        assert harness.provider.inputs == []
        assert harness.provider.response_requests == []

        await harness.application(
            harness.context,
            {"type": GatewayClientEvent.WAKE.value},
        )
        accepted = await harness.application(
            harness.context,
            {"type": GatewayClientEvent.TEXT_MESSAGE.value, "text": "还在吗"},
        )

        assert any(event.get("type") == GatewayServerEvent.TURN_STARTED.value for event in accepted)
        assert harness.provider.inputs[-1]["text"] == "还在吗"
        assert len(harness.provider.response_requests) == 1
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_reconnect_drops_uncorrelated_pending_responses_from_old_socket() -> None:
    harness = await open_harness(
        with_delivery=False,
        metadata_correlation=False,
    )
    try:
        await harness.application(
            harness.context,
            {"type": GatewayClientEvent.TEXT_MESSAGE.value, "text": "请求 A"},
        )
        assert len(harness.provider.response_requests) == 1

        harness.provider.emit({"type": "connection.closed"})
        state = harness.application._connections["connection"]
        await eventually(lambda: state.provider.ready)
        await eventually(lambda: not state.pending_response_contexts)

        await harness.application(
            harness.context,
            {"type": GatewayClientEvent.TEXT_MESSAGE.value, "text": "请求 B"},
        )
        request_b = harness.provider.response_requests[-1]
        harness.provider.emit(response_created(request_b, "response-b"))

        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
                and event.get("responseId") == "response-b"
                for event in harness.emitted
            )
        )
        assert harness.provider.cancel_calls >= 1  # A was cleared on disconnect.
        assert not state.pending_response_contexts
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_interrupt_resets_uncorrelated_socket_before_accepting_next_response() -> None:
    harness = await open_harness(
        with_delivery=False,
        metadata_correlation=False,
    )
    try:
        await harness.application(
            harness.context,
            {"type": GatewayClientEvent.TEXT_MESSAGE.value, "text": "请求 A"},
        )
        assert len(harness.provider.response_requests) == 1

        await harness.application(
            harness.context,
            {"type": GatewayClientEvent.INTERRUPT.value},
        )
        state = harness.application._connections["connection"]
        assert harness.provider.connect_calls == 2
        assert state.provider.ready is True
        assert not state.pending_response_contexts

        await harness.application(
            harness.context,
            {"type": GatewayClientEvent.TEXT_MESSAGE.value, "text": "请求 B"},
        )
        request_b = harness.provider.response_requests[-1]
        harness.provider.emit(response_created(request_b, "response-b-after-reset"))
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
                and event.get("responseId") == "response-b-after-reset"
                for event in harness.emitted
            )
        )

        assert not state.pending_response_contexts
        assert "response-b-after-reset" in state.active_responses
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_frontend_calls_are_serialized_and_voice_turn_drops_old_attachments() -> None:
    frontend = ConcurrentFrontendGraph()
    harness = await open_harness(
        with_delivery=False,
        frontend_graph=frontend,
    )
    try:
        await harness.application(
            harness.context,
            {
                "type": GatewayClientEvent.TEXT_MESSAGE.value,
                "parts": [
                    {"type": "text", "text": "整理这张图"},
                    {
                        "type": "file",
                        "mime": "image/png",
                        "url": "data:image/png;base64,AA==",
                        "filename": "old.png",
                    },
                ],
            },
        )
        manual_request = harness.provider.response_requests[0]
        harness.provider.emit(response_created(manual_request, "response-manual-attachment"))
        harness.provider.emit(response_done("response-manual-attachment"))
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.AUDIO_DONE.value
                and event.get("responseId") == "response-manual-attachment"
                for event in harness.emitted
            )
        )
        harness.provider.emit(
            {"type": "input_audio_buffer.speech_started", "item_id": "voice-item"}
        )
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.TURN_STARTED.value
                for event in harness.emitted
            )
        )

        for index in range(2):
            harness.provider.emit(
                {
                    "type": "response.function_call_arguments.done",
                    "response_id": "response-model",
                    "call_id": f"call-{index}",
                    "name": "get_current_time",
                    "arguments": "{}",
                }
            )

        await eventually(lambda: len(frontend.calls) == 2)
        await eventually(lambda: len(harness.provider.tool_results) == 2)
        assert frontend.max_active_calls == 1
        assert all(execution.input_parts == () for _call, execution in frontend.calls)
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_multiple_tool_calls_create_one_follow_up_after_response_done() -> None:
    frontend = ConcurrentFrontendGraph()
    harness = await open_harness(
        with_delivery=False,
        text_only=True,
        frontend_graph=frontend,
    )
    try:
        await harness.application(
            harness.context,
            {"type": GatewayClientEvent.TEXT_MESSAGE.value, "text": "查时间和清单"},
        )
        request = harness.provider.response_requests[0]
        harness.provider.emit(response_created(request, "response-tools"))
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
                and event.get("responseId") == "response-tools"
                for event in harness.emitted
            )
        )
        for index, name in enumerate(("get_current_time", "notes")):
            harness.provider.emit(
                {
                    "type": "response.function_call_arguments.done",
                    "response_id": "response-tools",
                    "call_id": f"call-batch-{index}",
                    "name": name,
                    "arguments": "{}",
                }
            )
        harness.provider.emit(response_done("response-tools"))

        await eventually(lambda: len(harness.provider.tool_results) == 2)
        await eventually(lambda: len(harness.provider.response_requests) == 2)
        await asyncio.sleep(0)
        assert frontend.max_active_calls == 1
        assert len(harness.provider.response_requests) == 2
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_interruption_drops_late_and_in_flight_tool_calls() -> None:
    late_frontend = FakeFrontendGraph({"status": "ok"})
    late = await open_harness(
        with_delivery=False,
        frontend_graph=late_frontend,
    )
    try:
        await late.application(
            late.context,
            {"type": GatewayClientEvent.TEXT_MESSAGE.value, "text": "先处理"},
        )
        request = late.provider.response_requests[0]
        late.provider.emit(response_created(request, "response-late-tool"))
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
                and event.get("responseId") == "response-late-tool"
                for event in late.emitted
            )
        )
        await late.application(
            late.context,
            {"type": GatewayClientEvent.INTERRUPT.value},
        )
        late.provider.emit(
            {
                "type": "response.function_call_arguments.done",
                "response_id": "response-late-tool",
                "call_id": "call-late",
                "name": "notes",
                "arguments": "{}",
            }
        )
        await asyncio.sleep(0.02)
        assert late_frontend.calls == []
    finally:
        await late.close()

    blocking_frontend = BlockingFrontendGraph()
    active = await open_harness(
        with_delivery=False,
        frontend_graph=blocking_frontend,
    )
    try:
        await active.application(
            active.context,
            {"type": GatewayClientEvent.TEXT_MESSAGE.value, "text": "开始处理"},
        )
        request = active.provider.response_requests[0]
        active.provider.emit(response_created(request, "response-active-tool"))
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
                and event.get("responseId") == "response-active-tool"
                for event in active.emitted
            )
        )
        active.provider.emit(
            {
                "type": "response.function_call_arguments.done",
                "response_id": "response-active-tool",
                "call_id": "call-active",
                "name": "notes",
                "arguments": "{}",
            }
        )
        await asyncio.wait_for(blocking_frontend.started.wait(), timeout=1)
        active.provider.emit(response_done("response-active-tool"))
        state = active.application._connections["connection"]
        await eventually(lambda: "response-active-tool" in state.tool_response_done)
        await active.application(
            active.context,
            {"type": GatewayClientEvent.INTERRUPT.value},
        )

        assert blocking_frontend.cancelled is True
        assert active.provider.tool_results == []
        assert "response-active-tool" not in state.tool_calls_pending
        assert "response-active-tool" not in state.tool_response_done
        assert "response-active-tool" not in state.tool_followup_created
        assert "response-active-tool" not in state.tool_followup_suppressed
        assert "response-active-tool" not in state.tool_contexts
        assert not state.frontend_tool_tasks
    finally:
        await active.close()


@pytest.mark.asyncio
async def test_delivered_events_unblock_losing_frontends_after_a_full_stale_batch() -> None:
    task_ids = tuple(f"stale-{index}" for index in range(8))
    records = {
        **{
            task_id: DeliveryRecord(
                task_id=task_id,
                owner_id="owner",
                session_id="old-session",
                created_at=float(index),
            )
            for index, task_id in enumerate(task_ids)
        },
        "fresh": DeliveryRecord(
            task_id="fresh",
            owner_id="owner",
            session_id="old-session",
            created_at=9,
        ),
    }
    leases = DeliveryLeaseQueue(records)
    tasks = FakeTaskSource()
    providers = {"one": FakeProvider(), "two": FakeProvider()}
    emitted: dict[str, list[dict[str, Any]]] = {"one": [], "two": []}
    application = FrontendRealtimeApplication(
        provider_factory=lambda context, _event: providers[str(context["connectionId"])],
        frontend_graph=FakeFrontendGraph(),  # type: ignore[arg-type]
        task_source=tasks,
        delivery_leases=leases,
        presentation_timeout_seconds=1,
    )
    contexts = {
        connection_id: {
            "connectionId": connection_id,
            "ownerId": "owner",
            "sessionId": connection_id,
            "emit": lambda event, connection_id=connection_id: emitted[connection_id].append(
                dict(event)
            ),
            "isVoiceActive": lambda: True,
        }
        for connection_id in providers
    }
    try:
        for context in contexts.values():
            await application(
                context,
                {"type": GatewayClientEvent.CONNECT.value, "voiceEnabled": True},
            )
        await eventually(lambda: all(provider.connected for provider in providers.values()))

        for task_id in task_ids:
            tasks.publish(
                "task.completed",
                {**completed_task(task_id), "sessionId": "old-session"},
            )
        claimed = leases.claim(
            owner_id="owner",
            session_id="external",
            include_other_sessions=True,
            claimant_id="external-frontend",
            task_ids=task_ids,
        )
        assert {record.task_id for record in claimed} == set(task_ids)
        await eventually(
            lambda: all(
                set(state.delivery.pending_task_ids) == set(task_ids)  # type: ignore[union-attr]
                for state in application._connections.values()
            )
        )
        assert leases.mark_delivered(task_ids, claimant_id="external-frontend") == 8
        for task_id in task_ids:
            tasks.publish(
                "task.notification.delivered",
                {
                    **completed_task(task_id),
                    "sessionId": "old-session",
                    "notificationStatus": "delivered",
                },
            )
        await eventually(
            lambda: all(
                state.delivery is not None and not state.delivery.pending_task_ids
                for state in application._connections.values()
            )
        )

        tasks.publish(
            "task.completed",
            {**completed_task("fresh"), "sessionId": "old-session"},
        )
        await eventually(
            lambda: sum(len(provider.response_requests) for provider in providers.values()) == 1
        )
        assert records["fresh"].status is DeliveryStatus.DELIVERING
    finally:
        await application.close()


@pytest.mark.asyncio
async def test_old_session_result_is_announced_without_projecting_inline_into_current_ui() -> None:
    harness = await open_harness(task_ids=("old-task",))
    try:
        harness.records["old-task"].session_id = "old-session"
        harness.tasks.publish(
            "task.completed",
            {
                **completed_task("old-task", inline=True),
                "sessionId": "old-session",
            },
        )

        await eventually(lambda: len(harness.provider.response_requests) == 1)
        assert event_count(harness, GatewayServerEvent.TIMELINE_INLINE.value) == 0
        assert harness.records["old-task"].claimant_id == "voice:connection"
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_reconnect_reinjects_active_delivery_context_on_the_new_socket() -> None:
    harness = await open_harness(task_ids=("reconnect-result",))
    try:
        gate = asyncio.Event()
        harness.provider.reconnect_gate = gate
        harness.provider.emit_ready_on_reconnect = True
        harness.tasks.publish("task.completed", completed_task("reconnect-result"))
        await eventually(lambda: len(harness.provider.response_requests) == 1)
        first_request = harness.provider.response_requests[0]
        harness.provider.emit(response_created(first_request, "response-before-disconnect"))
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
                and event.get("responseId") == "response-before-disconnect"
                for event in harness.emitted
            )
        )

        harness.provider.emit({"type": "connection.closed"})
        state = harness.application._connections["connection"]
        await eventually(lambda: harness.provider.connect_calls == 2 and not state.provider.ready)
        assert len(harness.provider.inputs) == 1

        gate.set()
        await eventually(lambda: state.provider.ready)
        await eventually(lambda: len(harness.provider.response_requests) == 2)
        assert len(harness.provider.inputs) == 2
        assert harness.provider.inputs[1] == harness.provider.inputs[0]
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_connection_ready_flushes_permission_queued_while_provider_is_down() -> None:
    harness = await open_harness(with_delivery=False, text_only=True)
    try:
        gate = asyncio.Event()
        harness.provider.reconnect_gate = gate
        harness.provider.emit_ready_on_reconnect = True
        harness.provider.emit({"type": "connection.closed"})
        state = harness.application._connections["connection"]
        await eventually(lambda: harness.provider.connect_calls == 2 and not state.provider.ready)

        task = {
            "id": "permission-during-reconnect",
            "ownerId": "owner",
            "sessionId": "voice",
            "status": "running",
            "authorization": {
                "id": "authorization-during-reconnect",
                "status": "pending",
                "summary": "允许写入报告",
            },
        }
        harness.tasks.publish("task.permission.requested", task)
        await eventually(lambda: "authorization-during-reconnect" in state.pending_permissions)
        assert harness.provider.response_requests == []

        gate.set()
        await eventually(lambda: state.provider.ready)
        await eventually(lambda: len(harness.provider.response_requests) == 1)
        assert "authorization-during-reconnect" in harness.provider.inputs[0]["text"]
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_reconnect_restarts_an_inflight_permission_on_the_new_socket() -> None:
    harness = await open_harness(with_delivery=False, text_only=True)
    try:
        task = {
            "id": "permission-inflight",
            "ownerId": "owner",
            "sessionId": "voice",
            "status": "running",
            "authorization": {
                "id": "authorization-inflight",
                "status": "pending",
                "summary": "允许写入报告",
            },
        }
        harness.tasks.publish("task.permission.requested", task)
        await eventually(lambda: len(harness.provider.response_requests) == 1)
        first_request = harness.provider.response_requests[0]
        harness.provider.emit(response_created(first_request, "permission-old-socket"))
        await eventually(
            lambda: any(
                event.get("type") == GatewayServerEvent.RESPONSE_STARTED.value
                and event.get("responseId") == "permission-old-socket"
                for event in harness.emitted
            )
        )

        gate = asyncio.Event()
        harness.provider.reconnect_gate = gate
        harness.provider.emit_ready_on_reconnect = True
        harness.provider.emit({"type": "connection.closed"})
        state = harness.application._connections["connection"]
        await eventually(lambda: harness.provider.connect_calls == 2 and not state.provider.ready)
        assert "authorization-inflight" in state.pending_permissions

        gate.set()
        await eventually(lambda: state.provider.ready)
        await eventually(lambda: len(harness.provider.response_requests) == 2)
        assert len(harness.provider.inputs) == 2
        assert "authorization-inflight" in harness.provider.inputs[1]["text"]
    finally:
        await harness.close()
