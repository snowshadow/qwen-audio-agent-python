from __future__ import annotations

from pathlib import Path

import pytest

from qwen_audio_agent.backend.node_bridge import BackendBridgeError, NodeBackendBridge


@pytest.mark.asyncio
async def test_node_bridge_preserves_backend_port_and_event_boundary() -> None:
    root = Path(__file__).parents[2]
    backend = NodeBackendBridge(
        root=root,
        command=("node", str(root / "tests" / "fixtures" / "backend_bridge.mjs")),
    )
    events: list[dict] = []

    await backend.start()
    assert backend.describe()["protocol"] == "fixture"
    result = await backend.submit(
        {"id": "task_1", "ownerId": "owner", "instruction": "inspect"},
        {"on_event": events.append, "private": "kept out of callbacks"},
    )

    assert result == {"content": "done:inspect"}
    assert events == [
        {
            "type": "backend.activity",
            "taskId": "task_1",
            "activity": {"kind": "status", "message": "fixture started"},
        }
    ]
    assert await backend.cancel(
        "task_1",
        {
            "owner_id": "owner-a",
            "session_id": "voice-a",
            "turn_id": "turn-a",
        },
    ) == {
        "taskId": "task_1",
        "state": "cancelled",
        "context": {
            "ownerId": "owner-a",
            "sessionId": "voice-a",
            "turnId": "turn-a",
        },
    }
    assert await backend.respond_authorization(
        "task_1",
        "auth_1",
        "always",
        {
            "owner_id": "owner-a",
            "session_id": "voice-a",
            "permission_scope_id": "scope-a",
        },
    ) == {
        "ok": True,
        "context": {
            "ownerId": "owner-a",
            "sessionId": "voice-a",
            "permissionScopeId": "scope-a",
        },
    }

    large = await backend.submit(
        {"id": "task_2", "ownerId": "owner", "instruction": "large"},
    )
    assert len(large["content"]) == 100_000
    await backend.close()


@pytest.mark.asyncio
async def test_control_rpc_timeout_restarts_a_live_but_unresponsive_bridge() -> None:
    root = Path(__file__).parents[2]
    backend = NodeBackendBridge(
        root=root,
        command=("node", str(root / "tests" / "fixtures" / "backend_bridge.mjs")),
        env={"QWEN_AUDIO_TEST_HANG_HEALTH": "1"},
        control_timeout_seconds=0.05,
    )
    try:
        await backend.start()
        with pytest.raises(BackendBridgeError) as captured:
            await backend.health()

        assert captured.value.code == "BACKEND_RPC_TIMEOUT"
        assert backend.process is None
    finally:
        await backend.close()
