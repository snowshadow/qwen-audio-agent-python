from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from qwen_audio_agent.protocol import (
    GatewayArtifact,
    GatewayArtifactPart,
    GatewayAuthorization,
    project_gateway_task_event,
    wire_dict,
)


def task(**overrides: object) -> dict[str, object]:
    return {
        "id": "task_public",
        "workState": "working",
        "status": "running",
        "kind": "work",
        "objective": "inspect the system",
        "createdAt": 1,
        "elapsedMs": 0,
        **overrides,
    }


def authorization(**overrides: object) -> dict[str, object]:
    return {
        "id": "auth_1",
        "taskId": "wrong-private-task-id",
        "status": "pending",
        "category": "shell",
        "summary": "run tests",
        "patterns": ["pytest"],
        "approvalScope": "session",
        "operation": {
            "title": "Run tests",
            "kind": "execute",
            "command": "pytest -q",
        },
        "createdAt": 1,
        "resolvedAt": None,
        **overrides,
    }


def test_artifact_part_has_exactly_one_payload_and_closed_fields() -> None:
    parsed = GatewayArtifactPart.model_validate(
        {
            "text": " report ",
            "mediaType": "Text/Markdown",
            "filename": "report.md",
            "providerToken": "must-not-cross",
        }
    )
    assert wire_dict(parsed) == {
        "text": "report",
        "mediaType": "text/markdown",
        "filename": "report.md",
    }

    with pytest.raises(ValidationError, match="exactly one"):
        GatewayArtifactPart.model_validate(
            {
                "text": "safe text",
                "url": "https://example.com/second-payload",
                "mediaType": "text/plain",
            }
        )
    with pytest.raises(ValidationError, match="exactly one"):
        GatewayArtifactPart.model_validate({"mediaType": "text/plain"})


@pytest.mark.parametrize(
    "part",
    [
        {"text": "safe", "mediaType": "not-a-mime"},
        {"raw": "not base64!", "mediaType": "application/octet-stream"},
        {"url": "javascript:alert(1)", "mediaType": "text/html"},
        {"url": "file:///etc/passwd", "mediaType": "text/plain"},
        {
            "url": "https://user:secret@example.com/private",
            "mediaType": "application/octet-stream",
        },
        {"url": "https://example.com/white space", "mediaType": "text/plain"},
        {"data": object(), "mediaType": "application/json"},
    ],
)
def test_artifact_part_rejects_unsafe_mime_payload_and_urls(part: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        GatewayArtifactPart.model_validate(part)


def test_artifact_part_accepts_all_four_safe_payload_variants() -> None:
    safe_parts = [
        {"text": "hello", "mediaType": "text/plain"},
        {"raw": "YQ==", "mediaType": "application/octet-stream"},
        {"url": "https://example.com/report.pdf", "mediaType": "application/pdf"},
        {"data": None, "mediaType": "application/json"},
    ]
    for part in safe_parts:
        assert wire_dict(GatewayArtifactPart.model_validate(part)) == part


def test_artifact_and_authorization_nested_objects_strip_private_fields() -> None:
    artifact = GatewayArtifact.model_validate(
        {
            "artifactId": "report",
            "parts": [{"text": "done", "mediaType": "text/plain"}],
            "backendSessionId": "private-session",
        }
    )
    assert "backendSessionId" not in wire_dict(artifact)

    parsed = GatewayAuthorization.model_validate(
        authorization(
            backendSessionId="private-session",
            credential="secret",
            operation={
                "title": "Run tests",
                "kind": "execute",
                "command": "pytest -q",
                "credential": "secret",
                "locations": [{"path": "/repo/test.py", "line": 7, "rawProviderId": "private"}],
            },
        )
    )
    public = wire_dict(parsed)
    assert "backendSessionId" not in public
    assert "credential" not in public
    assert "credential" not in public["operation"]
    assert "rawProviderId" not in public["operation"]["locations"][0]


def test_authorization_status_and_bounds_match_normalized_js_public_values() -> None:
    with pytest.raises(ValidationError, match="pending authorization"):
        GatewayAuthorization.model_validate(authorization(resolvedAt=2))
    with pytest.raises(ValidationError, match="requires resolvedAt"):
        GatewayAuthorization.model_validate(authorization(status="approved"))
    with pytest.raises(ValidationError):
        GatewayAuthorization.model_validate(authorization(patterns=["x" * 301]))
    with pytest.raises(ValidationError):
        GatewayAuthorization.model_validate(authorization(createdAt=True))


def test_public_projector_sanitizes_artifacts_and_binds_authorization_to_task() -> None:
    projected = project_gateway_task_event(
        {
            "type": "task.updated",
            "task": task(
                artifacts=[
                    {
                        "artifactId": "mixed",
                        "privateAdapterId": "private",
                        "parts": [
                            {
                                "text": "first payload wins",
                                "url": "https://example.com/ignored",
                                "mediaType": "Text/Plain",
                                "providerToken": "private",
                            },
                            {
                                "url": "https://user:secret@example.com/private",
                                "mediaType": "application/octet-stream",
                            },
                            {
                                "data": {"ok": True},
                                "mediaType": "application/json",
                                "providerTrace": "private",
                            },
                        ],
                    }
                ],
                authorization=authorization(
                    credential="private",
                    summary=("CLIENT_SECRET=hunter2 Authorization: Bearer super-secret-token"),
                    patterns=[
                        "GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz",
                        "CREDENTIAL=secret-value",
                        "Cookie: sid=abc123; theme=dark",
                    ],
                    operation={
                        "title": "Run tests",
                        "kind": "execute",
                        "command": (
                            "pytest -q --password hunter2 --token=secret-token "
                            "https://user:pass@example.com/x"
                        ),
                        "credential": "private",
                    },
                ),
            ),
        }
    )
    assert projected is not None
    assert projected["task"]["artifacts"] == [
        {
            "artifactId": "mixed",
            "parts": [
                {"text": "first payload wins", "mediaType": "text/plain"},
                {"data": {"ok": True}, "mediaType": "application/json"},
            ],
        }
    ]
    public_authorization = projected["task"]["authorization"]
    assert public_authorization["taskId"] == "task_public"
    assert "credential" not in public_authorization
    serialized = str(public_authorization)
    assert "super-secret-token" not in serialized
    assert "ghp_abcdefghijklmnopqrstuvwxyz" not in serialized
    assert "hunter2" not in serialized
    assert "secret-token" not in serialized
    assert "secret-value" not in serialized
    assert "sid=abc123" not in serialized
    assert "user:pass" not in serialized
    assert serialized.count("[REDACTED]") >= 8


def _js_v4_parse(event: dict[str, object]) -> dict[str, object]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the cross-language v4 schema oracle")
    local_root = Path(__file__).resolve().parents[2]
    canonical_root = Path("/Users/alfred/Code/qwen-audio-agent")
    oracle_root = canonical_root if canonical_root.exists() else local_root
    script = """
import { readFileSync } from 'node:fs'
import { GatewayServerMessageSchema } from './shared/protocol/gateway-events.mjs'
const value = JSON.parse(readFileSync(0, 'utf8'))
const parsed = GatewayServerMessageSchema.safeParse(value)
process.stdout.write(JSON.stringify(parsed.success
  ? { success: true, data: parsed.data }
  : { success: false, issues: parsed.error.issues }))
"""
    result = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        cwd=oracle_root,
        input=json.dumps(event),
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    return json.loads(result.stdout)


def test_python_public_artifact_and_authorization_pass_the_original_js_v4_oracle() -> None:
    projected = project_gateway_task_event(
        {
            "type": "task.permission.requested",
            "task": task(
                artifacts=[
                    {
                        "artifactId": "report",
                        "parts": [
                            {
                                "url": "https://example.com/report.pdf",
                                "mediaType": "application/pdf",
                            }
                        ],
                    }
                ],
                authorization=authorization(),
            ),
            "permission": authorization(),
        }
    )
    assert projected is not None
    oracle = _js_v4_parse(projected)
    assert oracle["success"] is True, oracle
    parsed = oracle["data"]
    assert parsed["task"]["artifacts"][0]["parts"] == [
        {"url": "https://example.com/report.pdf", "mediaType": "application/pdf"}
    ]
    assert parsed["permission"]["taskId"] == "task_public"
