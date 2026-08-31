from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from qwen_audio_agent.protocol import (
    GATEWAY_CAPABILITIES,
    GATEWAY_PROTOCOL_VERSION,
    GatewayClientEvent,
    GatewayServerEvent,
    GatewayTaskEvent,
    parse_agui_gateway_event,
    parse_gateway_client_message,
    parse_gateway_server_message,
    project_gateway_task_event,
    project_gateway_task_event_to_agui,
    project_gateway_task_snapshot,
    wire_dict,
)


def task(**overrides: object) -> dict[str, object]:
    return {
        "id": "task_1",
        "workState": "working",
        "status": "running",
        "kind": "work",
        "objective": "inspect the system",
        "createdAt": 1,
        "elapsedMs": 0,
        "result": None,
        "error": None,
        **overrides,
    }


def authorization() -> dict[str, object]:
    return {
        "id": "permission_1",
        "taskId": "task_1",
        "status": "pending",
        "category": "shell",
        "summary": "run command",
        "patterns": [],
        "createdAt": 1,
        "resolvedAt": None,
    }


def test_gateway_4_contract_constants_match_the_js_baseline() -> None:
    assert GATEWAY_PROTOCOL_VERSION == "4.0.0"
    assert re.fullmatch(r"\d+\.\d+\.\d+", GATEWAY_PROTOCOL_VERSION)
    assert len(GATEWAY_CAPABILITIES) == 20
    assert len(set(GATEWAY_CAPABILITIES)) == len(GATEWAY_CAPABILITIES)
    assert "input.suspend-protocol" in GATEWAY_CAPABILITIES
    assert "tasks.unified-id-updates" in GATEWAY_CAPABILITIES
    assert "messages.citations" in GATEWAY_CAPABILITIES


def test_event_names_match_existing_clients() -> None:
    assert GatewayClientEvent.INPUT_SUSPEND_ACK == "input.suspend.ack"
    assert GatewayServerEvent.INPUT_SUSPEND == "input.suspend"
    assert GatewayServerEvent.INPUT_RESUME == "input.resume"
    assert GatewayTaskEvent.UPDATED == "task.updated"


def test_client_and_voice_parsers_preserve_unknown_extension_fields() -> None:
    client = parse_gateway_client_message(
        {
            "type": "connect",
            "provider": "dashscope",
            "clientExtension": {"inputSampleRate": 16_000},
        }
    )
    assert wire_dict(client)["clientExtension"] == {"inputSampleRate": 16_000}

    voice = parse_gateway_server_message(
        {
            "type": "voice.ready",
            "inputSampleRate": 16_000,
            "providerExtension": {"codec": "pcm16"},
        }
    )
    assert wire_dict(voice)["providerExtension"] == {"codec": "pcm16"}

    with pytest.raises(ValidationError):
        parse_gateway_client_message({"type": "voice.ready"})
    with pytest.raises(ValidationError):
        parse_gateway_server_message({"type": "connect"})


def test_citations_are_bounded_to_final_assistant_transcripts() -> None:
    citation = {
        "id": "source_1",
        "title": "Example source",
        "url": "https://example.com/page",
        "snippet": "A bounded factual excerpt.",
    }
    parsed = parse_gateway_server_message(
        {
            "type": "transcript.final",
            "role": "assistant",
            "content": "Answer with a source.",
            "citations": [citation],
        }
    )
    assert wire_dict(parsed)["citations"] == [citation]

    with pytest.raises(ValidationError):
        parse_gateway_server_message(
            {
                "type": "transcript.delta",
                "role": "assistant",
                "citations": [citation],
            }
        )
    with pytest.raises(ValidationError):
        parse_gateway_server_message(
            {
                "type": "transcript.final",
                "role": "assistant",
                "citations": [{**citation, "url": "https://user:secret@example.com/private"}],
            }
        )


def test_task_projector_redacts_private_fields_but_keeps_activity_extensions() -> None:
    projected = project_gateway_task_event(
        {
            "type": "task.permission.requested",
            "ownerId": "internal-routing-key",
            "task": task(
                internalSchedulerLane="coordinator:owner",
                activity=[
                    {
                        "id": "progress_1",
                        "kind": "custom-progress",
                        "providerHint": "safe extension",
                    }
                ],
            ),
            "permission": authorization(),
            "message": "permission needed",
            "privateReason": "adapter-only",
        }
    )
    assert projected is not None
    assert projected["permission"] == {
        **authorization(),
        "approvalScope": "once",
        "operation": None,
    }
    assert projected["message"] == "permission needed"
    assert "ownerId" not in projected
    assert "privateReason" not in projected
    assert "internalSchedulerLane" not in projected["task"]
    assert projected["task"]["activity"][0]["providerHint"] == "safe extension"


def test_task_snapshot_and_agui_projection_use_one_unified_task_id() -> None:
    snapshot = project_gateway_task_snapshot(task(privateStoreVersion=2))
    assert snapshot is not None
    assert snapshot["type"] == "task.snapshot"
    assert snapshot["task"]["id"] == "task_1"
    assert "privateStoreVersion" not in snapshot["task"]

    agui = project_gateway_task_event_to_agui(snapshot)
    assert agui is not None
    assert agui["type"] == "ACTIVITY_SNAPSHOT"
    assert agui["messageId"] == "qwen.audio.task:task_1"
    assert agui["activityType"] == "qwen.audio.task"
    assert agui["replace"] is True
    assert agui["content"]["eventType"] == "task.snapshot"
    parsed = parse_agui_gateway_event(agui)
    assert wire_dict(parsed) == agui


def test_system_and_malformed_tasks_never_cross_the_public_boundary() -> None:
    assert project_gateway_task_snapshot(task(scope="system")) is None
    assert project_gateway_task_event({"type": "backend.activity"}) is None
    assert project_gateway_task_event_to_agui({"type": "voice.ready"}) is None
