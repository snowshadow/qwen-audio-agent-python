from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from qwen_audio_agent.runtime.input_arbitration import InputArbitration
from qwen_audio_agent.transport import IdentityManager, create_gateway_app

_SECRET = "test-secret-that-is-longer-than-32-characters"


def test_identity_and_origin_security_are_applied_to_http_and_websocket() -> None:
    manager = IdentityManager(secret=_SECRET, mode="browser")
    observed: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def on_client_event(
        context: dict[str, Any],
        event: dict[str, Any],
    ) -> None:
        observed.append((context, event))

    app = create_gateway_app(
        input_arbitration=InputArbitration(),
        identity_manager=manager,
        on_client_event=on_client_event,
    )

    with TestClient(app, base_url="http://localhost") as unauthenticated:
        with pytest.raises(WebSocketDisconnect) as rejected:
            with unauthenticated.websocket_connect(
                "/api/realtime",
                headers={"host": "localhost", "origin": "http://localhost"},
            ):
                pass
        assert rejected.value.code == 1008

    with TestClient(app, base_url="http://localhost") as client:
        issued = client.get("/livez")
        assert issued.status_code == 200
        cookie_header = issued.headers["set-cookie"].partition(";")[0]
        identity = manager.resolve_upgrade({"cookie": cookie_header})
        assert identity is not None

        denied = client.get(
            "/livez",
            headers={"origin": "https://attacker.example"},
        )
        assert denied.status_code == 403
        assert denied.json() == {"error": "origin not allowed"}

        with client.websocket_connect(
            "/api/realtime",
            headers={
                "host": "localhost",
                "origin": "http://localhost",
                "cookie": cookie_header,
            },
        ) as websocket:
            assert websocket.receive_json()["type"] == "voice.state"
            websocket.send_json(
                {
                    "type": "connect",
                    "voiceEnabled": True,
                    "clientType": "web",
                }
            )
            assert websocket.receive_json()["type"] == "voice.ownership"

    assert observed
    assert observed[0][0]["ownerId"] == identity.owner_id
