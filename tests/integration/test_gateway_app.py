from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from typing import Any

from fastapi.testclient import TestClient

from qwen_audio_agent.protocol import GATEWAY_CAPABILITIES
from qwen_audio_agent.runtime.input_arbitration import InputArbitration
from qwen_audio_agent.transport import create_gateway_app


def task(task_id: str = "task_1", **overrides: object) -> dict[str, object]:
    return {
        "id": task_id,
        "workState": "working",
        "status": "running",
        "scope": "user",
        "kind": "work",
        "objective": "inspect the system",
        "ownerId": "owner-a",
        "sessionId": "main",
        "turnId": "turn_1",
        "createdAt": 1,
        "elapsedMs": 0,
        "result": None,
        "error": None,
        **overrides,
    }


class FakeTasks:
    def __init__(self, values: list[dict[str, object]]) -> None:
        self.values = {str(value["id"]): value for value in values}
        self.listeners: set[Callable[[dict[str, Any]], None]] = set()
        self.cancel_calls: list[tuple[str, str | None]] = []
        self.authorization_calls: list[tuple[str, str, str, str | None]] = []

    def list(self, *, owner_id: str | None = None) -> list[dict[str, object]]:
        return [
            value
            for value in self.values.values()
            if owner_id is None or value.get("ownerId") == owner_id
        ]

    def get(
        self,
        task_id: str,
        *,
        owner_id: str | None = None,
    ) -> dict[str, object] | None:
        value = self.values.get(task_id)
        if value is None or (owner_id is not None and value.get("ownerId") != owner_id):
            return None
        return value

    def subscribe(
        self,
        listener: Callable[[dict[str, Any]], None],
    ) -> Callable[[], None]:
        self.listeners.add(listener)
        return lambda: self.listeners.discard(listener)

    def publish(self, event: dict[str, Any]) -> None:
        self.values[event["task"]["id"]] = event["task"]
        for listener in tuple(self.listeners):
            listener(event)

    async def cancel(
        self,
        task_id: str,
        *,
        owner_id: str | None = None,
    ) -> dict[str, object] | None:
        self.cancel_calls.append((task_id, owner_id))
        value = self.get(task_id, owner_id=owner_id)
        if value is None or value.get("status") in {"completed", "failed", "cancelled"}:
            return None
        cancelled = {**value, "status": "cancelled", "workState": "cancelled"}
        self.values[task_id] = cancelled
        return cancelled

    async def respond_authorization(
        self,
        task_id: str,
        authorization_id: str,
        decision: str,
        *,
        owner_id: str | None = None,
    ) -> dict[str, object] | None:
        self.authorization_calls.append((task_id, authorization_id, decision, owner_id))
        value = self.get(task_id, owner_id=owner_id)
        authorization = value.get("authorization") if value else None
        if (
            not isinstance(authorization, Mapping)
            or authorization.get("id") != authorization_id
            or authorization.get("status") != "pending"
        ):
            return None
        return {
            "ok": True,
            "taskId": task_id,
            "authorizationId": authorization_id,
            "decision": decision,
        }


class UnscopedFakeTasks(FakeTasks):
    """Deliberately ignores owner hints to verify the HTTP boundary itself."""

    def list(self, *, owner_id: str | None = None) -> list[dict[str, object]]:
        del owner_id
        return list(self.values.values())

    def get(
        self,
        task_id: str,
        *,
        owner_id: str | None = None,
    ) -> dict[str, object] | None:
        del owner_id
        return self.values.get(task_id)


class UntrustedPermissionResultTasks(UnscopedFakeTasks):
    """Returns adapter-private results and can change ownership after the action."""

    def __init__(
        self,
        values: list[dict[str, object]],
        *,
        result: dict[str, object],
        replacements: Mapping[str, dict[str, object]] | None = None,
    ) -> None:
        super().__init__(values)
        self.result = result
        self.replacements = dict(replacements or {})

    async def respond_authorization(
        self,
        task_id: str,
        authorization_id: str,
        decision: str,
        *,
        owner_id: str | None = None,
    ) -> dict[str, object] | None:
        accepted = await super().respond_authorization(
            task_id,
            authorization_id,
            decision,
            owner_id=owner_id,
        )
        if accepted is None:
            return None
        replacement = self.replacements.get(task_id)
        if replacement is not None:
            self.values[task_id] = replacement
        return self.result


class FakeConversations:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self.values: dict[tuple[str, str], list[dict[str, object]]] = {
            ("owner-a", "main"): [{"id": "a-main", "role": "user", "content": "A main"}],
            ("owner-a", "other"): [{"id": "a-other", "role": "assistant", "content": "A other"}],
            ("owner-b", "main"): [{"id": "b-main", "role": "user", "content": "B main"}],
        }

    def messages(
        self,
        *,
        owner_id: str,
        session_id: str,
        limit: int,
    ) -> list[dict[str, object]]:
        self.calls.append((owner_id, session_id, limit))
        return self.values.get((owner_id, session_id), [])


def owner(connection: Any) -> str:
    return connection.headers.get("x-owner", "owner-a")


def make_app(
    *,
    tasks: FakeTasks | None = None,
    conversations: FakeConversations | None = None,
    arbitration: InputArbitration | None = None,
    web_root: object = None,
    skins_root: object = None,
):
    input_arbitration = arbitration or InputArbitration()
    return create_gateway_app(
        input_arbitration=input_arbitration,
        task_source=tasks,
        conversation_source=conversations,
        owner_resolver=owner,
        gateway_instance_id="gateway-test",
        gateway_started_at="2026-08-28T00:00:00Z",
        web_root=web_root,
        skins_root=skins_root,
        health_provider=lambda: {
            "voiceConfigured": True,
            "realtimeProvider": "fake",
            "runtimeExtension": {"graph": "ready"},
            "backend": {"enabled": True, "ok": True, "status": "ready"},
        },
    )


def test_health_and_input_http_contract() -> None:
    app = make_app()
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        body = health.json()
        assert body["protocolVersion"] == "4.0.0"
        assert body["capabilities"] == list(GATEWAY_CAPABILITIES)
        assert body["gatewayInstanceId"] == "gateway-test"
        assert body["voiceConfigured"] is True
        assert body["voiceClients"] == {
            "connected": 0,
            "activeOwners": 0,
            "byType": {},
        }
        assert body["runtimeExtension"] == {"graph": "ready"}

        missing = client.post("/api/input/suspend", json={})
        assert missing.status_code == 400
        assert missing.json()["code"] == "QWAUDIO_INPUT_OWNER_REQUIRED"

        suspended = client.post(
            "/api/input/suspend",
            json={"owner": "dictation", "reason": "host recording", "ttlMs": 60_000},
        )
        assert suspended.status_code == 200
        assert suspended.json()["suspended"] is True
        assert suspended.json()["holders"][0]["ttlMs"] == 60_000
        assert client.get("/api/input").json()["owner"] == "dictation"

        resumed = client.post("/api/input/resume", json={"owner": "dictation"})
        assert resumed.json()["suspended"] is False


def test_websocket_input_suspension_and_extension_envelopes() -> None:
    app = make_app()
    with (
        TestClient(app) as client,
        client.websocket_connect(
            "/api/realtime?sessionId=main",
            headers={"x-owner": "owner-a"},
        ) as websocket,
    ):
        assert websocket.receive_json() == {"type": "voice.state", "state": "idle"}
        websocket.send_json(
            {
                "type": "connect",
                "voiceEnabled": True,
                "clientType": "web",
                "clientExtension": {"version": 1},
            }
        )
        ownership = websocket.receive_json()
        assert ownership["type"] == "voice.ownership"
        assert ownership["state"] == "active"

        response = client.post(
            "/api/input/suspend",
            json={"owner": "dictation", "reason": "host recording", "ttlMs": 60_000},
        )
        assert response.status_code == 200
        assert websocket.receive_json() == {
            "type": "playback.clear",
            "reason": "input_suspended",
        }
        suspend = websocket.receive_json()
        assert suspend["type"] == "input.suspend"
        assert suspend["owner"] == "dictation"
        assert suspend["reason"] == "host recording"

        websocket.send_json({"type": "input.suspend.ack", "owner": "dictation"})
        assert client.post("/api/input/resume", json={"owner": "dictation"}).status_code == 200
        assert websocket.receive_json() == {"type": "input.resume"}


def test_late_websocket_connection_learns_current_suspension() -> None:
    arbitration = InputArbitration()
    arbitration.suspend(owner="host-app", ttl_ms=60_000)
    app = make_app(arbitration=arbitration)
    with (
        TestClient(app) as client,
        client.websocket_connect("/api/realtime?sessionId=main") as websocket,
    ):
        assert websocket.receive_json()["type"] == "voice.state"
        suspension = websocket.receive_json()
        assert suspension["type"] == "input.suspend"
        assert suspension["owner"] == "host-app"


def test_task_queries_are_public_and_owner_scoped() -> None:
    tasks = FakeTasks(
        [
            task(),
            task("task_2", ownerId="owner-b"),
            task("task_3", status="completed", workState="completed"),
            task("task_system", scope="system"),
        ]
    )
    app = make_app(tasks=tasks)
    with TestClient(app) as client:
        listed = client.get("/api/tasks", headers={"x-owner": "owner-a"}).json()["tasks"]
        assert [value["id"] for value in listed] == ["task_1", "task_3"]
        active = client.get(
            "/api/tasks?active=true&sessionId=main",
            headers={"x-owner": "owner-a"},
        ).json()["tasks"]
        assert [value["id"] for value in active] == ["task_1"]
        assert client.get("/api/tasks/task_1", headers={"x-owner": "owner-a"}).status_code == 200
        assert client.get("/api/tasks/task_2", headers={"x-owner": "owner-a"}).status_code == 404
        assert client.get("/api/tasks/missing", headers={"x-owner": "owner-a"}).json() == {
            "error": "task not found"
        }


def test_ownerless_tasks_fail_closed_even_when_the_source_ignores_owner_hints() -> None:
    pending = {
        "id": "auth_ownerless",
        "taskId": "task_ownerless",
        "status": "pending",
        "category": "shell",
        "summary": "run command",
        "patterns": [],
        "approvalScope": "once",
        "operation": None,
        "createdAt": 1,
        "resolvedAt": None,
    }
    tasks = UnscopedFakeTasks(
        [
            task(
                "task_ownerless",
                ownerId=None,
                workState="auth_required",
                authorization=pending,
                presentation={
                    "speech": "private",
                    "inline": {
                        "title": "Private",
                        "format": "markdown",
                        "content": "must not cross the owner boundary",
                    },
                },
            ),
            task("task_owned"),
        ]
    )
    app = make_app(tasks=tasks)
    with TestClient(app) as client:
        listed = client.get("/api/tasks", headers={"x-owner": "owner-a"}).json()["tasks"]
        timeline = client.get("/api/timeline", headers={"x-owner": "owner-a"}).json()["items"]
        fetched = client.get(
            "/api/tasks/task_ownerless",
            headers={"x-owner": "owner-a"},
        )
        cancelled = client.delete(
            "/api/tasks/task_ownerless",
            headers={"x-owner": "owner-a"},
        )
        permission = client.post(
            "/api/permissions/auth_ownerless",
            headers={"x-owner": "owner-a"},
            json={"decision": "always"},
        )

    assert [value["id"] for value in listed] == ["task_owned"]
    assert timeline == []
    assert fetched.status_code == 404
    assert cancelled.status_code == 404
    assert permission.status_code == 404
    assert tasks.cancel_calls == []
    assert tasks.authorization_calls == []


def test_conversation_messages_are_owner_and_session_scoped() -> None:
    conversations = FakeConversations()
    app = make_app(conversations=conversations)
    with TestClient(app) as client:
        owner_a_main = client.get(
            "/api/conversations/main/messages",
            headers={"x-owner": "owner-a"},
        )
        owner_a_other = client.get(
            "/api/conversations/other/messages",
            headers={"x-owner": "owner-a"},
        )
        owner_b_main = client.get(
            "/api/conversations/main/messages",
            headers={"x-owner": "owner-b"},
        )

    assert owner_a_main.json() == {
        "messages": [{"id": "a-main", "role": "user", "content": "A main"}]
    }
    assert owner_a_other.json() == {
        "messages": [{"id": "a-other", "role": "assistant", "content": "A other"}]
    }
    assert owner_b_main.json() == {
        "messages": [{"id": "b-main", "role": "user", "content": "B main"}]
    }
    assert conversations.calls == [
        ("owner-a", "main", 200),
        ("owner-a", "other", 200),
        ("owner-b", "main", 200),
    ]


def test_timeline_projects_only_scoped_inline_presentations_in_time_order() -> None:
    tasks = FakeTasks(
        [
            task(
                "task_late",
                completedAt=30,
                presentation={
                    "speech": "late",
                    "inline": {
                        "title": "Late",
                        "format": "markdown",
                        "content": "# Late",
                    },
                },
            ),
            task(
                "task_early",
                completedAt=20,
                presentation={
                    "speech": "early",
                    "inline": {
                        "title": "Early",
                        "format": "code",
                        "content": "print('early')",
                    },
                },
            ),
            task(
                "task_no_inline",
                completedAt=10,
                presentation={"speech": "speech only", "inline": None},
            ),
            task(
                "task_other_session",
                sessionId="other",
                presentation={
                    "speech": "other",
                    "inline": {
                        "title": "Other",
                        "format": "markdown",
                        "content": "other session",
                    },
                },
            ),
            task(
                "task_other_owner",
                ownerId="owner-b",
                presentation={
                    "speech": "private",
                    "inline": {
                        "title": "Private",
                        "format": "markdown",
                        "content": "owner b",
                    },
                },
            ),
        ]
    )
    app = make_app(tasks=tasks)
    with TestClient(app) as client:
        response = client.get(
            "/api/timeline?sessionId=main",
            headers={"x-owner": "owner-a"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {
                "id": "inline_task_early",
                "taskId": "task_early",
                "createdAt": 20,
                "title": "Early",
                "format": "code",
                "content": "print('early')",
            },
            {
                "id": "inline_task_late",
                "taskId": "task_late",
                "createdAt": 30,
                "title": "Late",
                "format": "markdown",
                "content": "# Late",
            },
        ]
    }


def test_delete_task_returns_200_404_and_409_without_cross_owner_calls() -> None:
    tasks = FakeTasks(
        [
            task("task_active"),
            task("task_terminal", status="completed", workState="completed"),
            task("task_private", ownerId="owner-b"),
        ]
    )
    app = make_app(tasks=tasks)
    with TestClient(app) as client:
        cancelled = client.delete(
            "/api/tasks/task_active",
            headers={"x-owner": "owner-a"},
        )
        terminal = client.delete(
            "/api/tasks/task_terminal",
            headers={"x-owner": "owner-a"},
        )
        missing = client.delete(
            "/api/tasks/missing",
            headers={"x-owner": "owner-a"},
        )
        private = client.delete(
            "/api/tasks/task_private",
            headers={"x-owner": "owner-a"},
        )

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert terminal.status_code == 409
    assert terminal.json()["task"]["id"] == "task_terminal"
    assert missing.status_code == 404
    assert private.status_code == 404
    assert tasks.cancel_calls == [
        ("task_active", "owner-a"),
        ("task_terminal", "owner-a"),
    ]


def test_permission_response_validates_decision_owner_and_pending_request() -> None:
    pending = {
        "id": "auth_1",
        "taskId": "task_auth",
        "status": "pending",
        "category": "bash",
        "summary": "Run tests",
        "patterns": [],
        "approvalScope": "session",
        "operation": None,
        "createdAt": 10,
        "resolvedAt": None,
    }
    tasks = FakeTasks(
        [
            task("task_auth", authorization=pending, workState="auth_required"),
            task(
                "task_private_auth",
                ownerId="owner-b",
                authorization={**pending, "id": "auth_private", "taskId": "task_private_auth"},
                workState="auth_required",
            ),
        ]
    )
    app = make_app(tasks=tasks)
    with TestClient(app) as client:
        invalid = client.post(
            "/api/permissions/auth_1",
            headers={"x-owner": "owner-a"},
            json={"decision": "once"},
        )
        private = client.post(
            "/api/permissions/auth_private",
            headers={"x-owner": "owner-a"},
            json={"decision": "always"},
        )
        accepted = client.post(
            "/api/permissions/auth_1",
            headers={"x-owner": "owner-a"},
            json={"decision": "reject"},
        )

    assert invalid.status_code == 400
    assert invalid.json() == {"error": "decision must be always or reject"}
    assert private.status_code == 404
    assert accepted.status_code == 200
    assert accepted.json()["id"] == "task_auth"
    assert accepted.json()["ownerId"] == "owner-a"
    assert accepted.json()["authorization"]["id"] == "auth_1"
    assert "ok" not in accepted.json()
    assert tasks.authorization_calls == [("task_auth", "auth_1", "reject", "owner-a")]


def test_permission_response_ignores_private_result_and_rechecks_owner() -> None:
    pending = {
        "id": "auth_safe",
        "taskId": "task_safe",
        "status": "pending",
        "category": "bash",
        "summary": "Run tests",
        "patterns": [],
        "approvalScope": "session",
        "operation": None,
        "createdAt": 10,
        "resolvedAt": None,
    }
    private_marker = "adapter-private-do-not-leak"
    tasks = UntrustedPermissionResultTasks(
        [
            task(
                "task_safe",
                authorization=pending,
                workState="auth_required",
                adapterPrivate={"secret": private_marker},
            ),
            task(
                "task_cross_owner",
                authorization={
                    **pending,
                    "id": "auth_cross_owner",
                    "taskId": "task_cross_owner",
                },
                workState="auth_required",
            ),
            task(
                "task_ownerless_after_response",
                authorization={
                    **pending,
                    "id": "auth_ownerless_after_response",
                    "taskId": "task_ownerless_after_response",
                },
                workState="auth_required",
            ),
        ],
        result={
            "ok": True,
            "private": {
                "secret": private_marker,
                "oversized": "x" * 1_000_000,
            },
            "taskId": "forged_task",
            "ownerId": "owner-b",
        },
        replacements={
            "task_cross_owner": task(
                "task_cross_owner",
                ownerId="owner-b",
                authorization={
                    **pending,
                    "id": "auth_cross_owner",
                    "taskId": "task_cross_owner",
                },
                workState="auth_required",
            ),
            "task_ownerless_after_response": task(
                "task_ownerless_after_response",
                ownerId=None,
                authorization={
                    **pending,
                    "id": "auth_ownerless_after_response",
                    "taskId": "task_ownerless_after_response",
                },
                workState="auth_required",
            ),
        },
    )
    app = make_app(tasks=tasks)
    with TestClient(app) as client:
        safe = client.post(
            "/api/permissions/auth_safe",
            headers={"x-owner": "owner-a"},
            json={"decision": "always"},
        )
        cross_owner = client.post(
            "/api/permissions/auth_cross_owner",
            headers={"x-owner": "owner-a"},
            json={"decision": "always"},
        )
        ownerless = client.post(
            "/api/permissions/auth_ownerless_after_response",
            headers={"x-owner": "owner-a"},
            json={"decision": "reject"},
        )

    assert safe.status_code == 200
    assert safe.json()["id"] == "task_safe"
    assert safe.json()["ownerId"] == "owner-a"
    assert "adapterPrivate" not in safe.json()
    assert private_marker not in safe.text
    assert len(safe.content) < 10_000
    assert cross_owner.status_code == 404
    assert cross_owner.json() == {"error": "permission request not found"}
    assert ownerless.status_code == 404
    assert ownerless.json() == {"error": "permission request not found"}
    assert private_marker not in cross_owner.text
    assert private_marker not in ownerless.text
    assert tasks.authorization_calls == [
        ("task_safe", "auth_safe", "always", "owner-a"),
        ("task_cross_owner", "auth_cross_owner", "always", "owner-a"),
        (
            "task_ownerless_after_response",
            "auth_ownerless_after_response",
            "reject",
            "owner-a",
        ),
    ]


def test_task_events_are_pushed_over_the_existing_websocket() -> None:
    tasks = FakeTasks([task()])
    app = make_app(tasks=tasks)
    with (
        TestClient(app) as client,
        client.websocket_connect(
            "/api/realtime?sessionId=main",
            headers={"x-owner": "owner-a"},
        ) as websocket,
    ):
        assert websocket.receive_json()["type"] == "voice.state"
        tasks.publish(
            {
                "type": "task.updated",
                "task": task("task_ownerless", ownerId=None, message="ownerless"),
                "message": "ownerless",
            }
        )
        tasks.publish(
            {
                "type": "task.updated",
                "ownerId": "owner-a",
                "task": task("task_forged", ownerId="owner-b", message="forged"),
                "message": "forged",
            }
        )
        tasks.publish(
            {
                "type": "task.updated",
                "owner_id": "owner-b",
                "task": task("task_snake_forged", message="snake forged"),
                "message": "snake forged",
            }
        )
        tasks.publish(
            {
                "type": "task.updated",
                "ownerId": "owner-a",
                "task": task(message="halfway"),
                "message": "halfway",
            }
        )
        event = websocket.receive_json()
        assert event["type"] == "task.updated"
        assert event["task"]["id"] == "task_1"
        assert event["message"] == "halfway"


async def _first_stream_body(app: Any, path: str) -> tuple[dict[str, Any], bytes]:
    sent: list[dict[str, Any]] = []
    disconnect = asyncio.Event()
    request_delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal request_delivered
        if not request_delivered:
            request_delivered = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            disconnect.set()

    query = path.partition("?")[2].encode()
    clean_path = path.partition("?")[0]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": clean_path,
        "raw_path": clean_path.encode(),
        "query_string": query,
        "headers": [(b"x-owner", b"owner-a")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "root_path": "",
    }
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app(scope, receive, send), timeout=1)
    start = next(message for message in sent if message["type"] == "http.response.start")
    body = next(
        message["body"]
        for message in sent
        if message["type"] == "http.response.body" and message.get("body")
    )
    return start, body


async def _first_stream_update(
    app: Any,
    path: str,
    publish: Callable[[], None],
) -> bytes:
    bodies: list[bytes] = []
    disconnect = asyncio.Event()
    request_delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal request_delivered
        if not request_delivered:
            request_delivered = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] != "http.response.body" or not message.get("body"):
            return
        bodies.append(message["body"])
        if len(bodies) == 1:
            publish()
        elif len(bodies) == 2:
            disconnect.set()

    query = path.partition("?")[2].encode()
    clean_path = path.partition("?")[0]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": clean_path,
        "raw_path": clean_path.encode(),
        "query_string": query,
        "headers": [(b"x-owner", b"owner-a")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "root_path": "",
    }
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app(scope, receive, send), timeout=1)
    return bodies[1]


def test_task_sse_native_and_agui_initial_snapshots() -> None:
    native_app = make_app(tasks=FakeTasks([task()]))
    start, body = asyncio.run(_first_stream_body(native_app, "/api/tasks/task_1/events"))
    assert start["status"] == 200
    assert dict(start["headers"])[b"content-type"].startswith(b"text/event-stream")
    native = json.loads(body.decode().removeprefix("data: ").strip())
    assert native["type"] == "task.snapshot"
    assert native["task"]["id"] == "task_1"

    agui_app = make_app(tasks=FakeTasks([task()]))
    _, body = asyncio.run(_first_stream_body(agui_app, "/api/tasks/task_1/events?format=ag-ui"))
    agui = json.loads(body.decode().removeprefix("data: ").strip())
    assert agui["type"] == "ACTIVITY_SNAPSHOT"
    assert agui["messageId"] == "qwen.audio.task:task_1"
    assert agui["content"]["eventType"] == "task.snapshot"


def test_ownerless_task_sse_fails_closed_with_an_unscoped_source() -> None:
    app = make_app(tasks=UnscopedFakeTasks([task("task_ownerless", ownerId=None)]))
    start, body = asyncio.run(_first_stream_body(app, "/api/tasks/task_ownerless/events"))

    assert start["status"] == 404
    assert json.loads(body) == {"error": "task not found"}


def test_task_sse_rejects_snake_case_source_owner_conflicts() -> None:
    tasks = FakeTasks([task()])
    app = make_app(tasks=tasks)

    def publish() -> None:
        tasks.publish(
            {
                "type": "task.updated",
                "owner_id": "owner-b",
                "task": task(message="snake forged"),
                "message": "snake forged",
            }
        )
        tasks.publish(
            {
                "type": "task.updated",
                "ownerId": "owner-a",
                "task": task(message="valid"),
                "message": "valid",
            }
        )

    body = asyncio.run(_first_stream_update(app, "/api/tasks/task_1/events", publish))
    event = json.loads(body.decode().removeprefix("data: ").strip())
    assert event["type"] == "task.updated"
    assert event["message"] == "valid"


def test_same_origin_web_and_skin_assets_do_not_cross_fallbacks(tmp_path: Any) -> None:
    web = tmp_path / "web"
    skins = tmp_path / "skins"
    (web / "assets").mkdir(parents=True)
    (skins / "probe").mkdir(parents=True)
    (web / "index.html").write_text("<main>gateway web</main>", encoding="utf-8")
    (web / "assets" / "app.js").write_text("export const ready = true", encoding="utf-8")
    (skins / "probe" / "pet.json").write_text('{"id":"probe"}', encoding="utf-8")

    app = make_app(web_root=web, skins_root=skins)
    with TestClient(app) as client:
        assert client.get("/").text == "<main>gateway web</main>"
        assert client.get("/conversation/main").text == "<main>gateway web</main>"
        assert "ready = true" in client.get("/assets/app.js").text
        manifest = client.get("/skins/probe/pet.json")
        assert manifest.json() == {"id": "probe"}
        assert manifest.headers["cache-control"] == "no-store"
        assert client.get("/skins/probe/missing.webp").status_code == 404
        assert client.get("/skins/.secret").status_code == 404
