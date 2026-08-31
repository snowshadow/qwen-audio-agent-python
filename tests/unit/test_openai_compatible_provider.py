from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import pytest

from qwen_audio_agent.providers.openai_compatible import (
    DashScopeRealtimeProvider,
    GARealtimeProtocol,
    RealtimeProviderError,
    SpeechToSpeechRealtimeProvider,
    build_speech_to_speech_session,
    classify_dashscope_error,
    classify_speech_to_speech_error,
)
from qwen_audio_agent.providers.realtime import assert_realtime_provider


class FakeRealtimeSocket:
    def __init__(self, *, acknowledge_session_updates: bool = True) -> None:
        self.incoming: asyncio.Queue[str | bytes | BaseException] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self.acknowledge_session_updates = acknowledge_session_updates

    async def send(self, message: str) -> None:
        event = json.loads(message)
        self.sent.append(event)
        if event["type"] == "session.update" and self.acknowledge_session_updates:
            self.feed({"type": "session.updated", "session": event["session"]})

    async def recv(self) -> str | bytes:
        value = await self.incoming.get()
        if isinstance(value, BaseException):
            raise value
        return value

    async def close(self) -> None:
        self.closed = True

    def feed(self, event: Mapping[str, Any]) -> None:
        self.incoming.put_nowait(json.dumps(event))


class FakeConnector:
    def __init__(self, socket: FakeRealtimeSocket) -> None:
        self.socket = socket
        self.calls: list[tuple[str, dict[str, str], float]] = []

    async def __call__(
        self,
        url: str,
        headers: Mapping[str, str],
        open_timeout: float,
    ) -> FakeRealtimeSocket:
        self.calls.append((url, dict(headers), open_timeout))
        self.socket.feed({"type": "session.created", "session": {"id": "session_1"}})
        return self.socket


@pytest.mark.asyncio
async def test_dashscope_connect_performs_the_required_session_handshake() -> None:
    socket = FakeRealtimeSocket()
    connector = FakeConnector(socket)
    events: list[dict[str, Any]] = []
    provider = DashScopeRealtimeProvider(
        api_key="secret",
        model="qwen-audio-3.0-realtime-plus",
        voice="custom-voice",
        instructions="be helpful",
        tools=[{"type": "function", "function": {"name": "clock"}}],
        connector=connector,
    )
    provider.subscribe(lambda event: events.append(dict(event)))

    assert assert_realtime_provider(provider) is provider
    await provider.connect()

    url, headers, timeout = connector.calls[0]
    assert url == (
        "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen-audio-3.0-realtime-plus"
    )
    assert headers == {"Authorization": "Bearer secret"}
    assert timeout == 25.0
    assert [event["type"] for event in events] == [
        "session.created",
        "session.updated",
        "connection.ready",
    ]

    update = socket.sent[0]
    assert update["type"] == "session.update"
    assert update["event_id"].startswith("event_")
    assert update["session"] == {
        "instructions": "be helpful",
        "tools": [{"type": "function", "function": {"name": "clock"}}],
        "modalities": ["text", "audio"],
        "voice": "custom-voice",
        "output_audio_format": "pcm",
        "input_audio_format": "pcm",
        "turn_detection": {"type": "smart_turn"},
    }
    assert provider.capabilities()["conversationItemIdEcho"] is True
    await provider.close()


@pytest.mark.asyncio
async def test_outgoing_audio_text_tool_result_response_and_cancel_match_wire() -> None:
    socket = FakeRealtimeSocket()
    provider = DashScopeRealtimeProvider(
        api_key="secret",
        model="qwen3.5-omni-flash-realtime",
        connector=FakeConnector(socket),
    )
    await provider.connect()

    await provider.send_audio(b"\x00\xff")
    await provider.send_input({"text": "\u4f60\u597d"})
    await provider.send_tool_result({"callId": "call_1", "output": {"ok": True}})
    await provider.create_response({"modalities": ["text", "audio"]})
    await provider.cancel_response("response_ignored_by_beta_dialect")

    audio, user_input, tool_output, response, cancel = socket.sent[1:]
    assert audio == {
        "event_id": audio["event_id"],
        "type": "input_audio_buffer.append",
        "audio": "AP8=",
    }
    assert user_input["type"] == "conversation.item.create"
    assert user_input["item"]["id"].startswith("item_")
    assert user_input["item"]["content"] == [{"type": "input_text", "text": "\u4f60\u597d"}]
    assert tool_output["type"] == "conversation.item.create"
    assert tool_output["item"]["type"] == "function_call_output"
    assert tool_output["item"]["call_id"] == "call_1"
    assert json.loads(tool_output["item"]["output"]) == {"ok": True}
    assert response["type"] == "response.create"
    assert response["response"] == {"modalities": ["text", "audio"]}
    assert cancel["type"] == "response.cancel"
    assert "response_id" not in cancel
    assert provider.capabilities()["conversationItemIdEcho"] is False
    await provider.close()


@pytest.mark.asyncio
async def test_incoming_events_stay_openai_compatible_and_fatal_errors_are_classified() -> None:
    socket = FakeRealtimeSocket()
    provider = DashScopeRealtimeProvider(
        api_key="secret",
        connector=FakeConnector(socket),
    )
    events: list[dict[str, Any]] = []
    provider.subscribe(lambda event: events.append(dict(event)))
    await provider.connect()
    events.clear()

    socket.feed({"type": "response.created", "response": {"id": "response_1"}})
    socket.feed({"type": "response.audio.delta", "response_id": "response_1", "delta": "abc"})
    socket.feed(
        {
            "type": "response.function_call_arguments.done",
            "response_id": "response_1",
            "call_id": "call_1",
            "arguments": "{}",
        }
    )
    socket.feed({"type": "response.done", "response": {"id": "response_1"}})
    await asyncio.sleep(0)

    assert [event["type"] for event in events] == [
        "response.created",
        "response.audio.delta",
        "response.function_call_arguments.done",
        "response.done",
    ]

    socket.feed(
        {
            "type": "error",
            "error": {"code": "InvalidApiKey", "message": "Invalid API-key provided."},
        }
    )
    await asyncio.sleep(0)
    assert events[-2]["type"] == "error"
    assert events[-2]["classification"] == "fatal"
    assert events[-2]["fatal"] is True
    assert events[-1]["type"] == "provider.error"
    assert events[-1]["fatal"] is True
    await provider.close()


@pytest.mark.asyncio
async def test_connect_fails_closed_when_provider_errors_before_session_ready() -> None:
    socket = FakeRealtimeSocket()

    async def connector(
        _url: str,
        _headers: Mapping[str, str],
        _timeout: float,
    ) -> FakeRealtimeSocket:
        socket.feed(
            {
                "type": "error",
                "error": {"code": "InvalidApiKey", "message": "authentication failed"},
            }
        )
        return socket

    provider = DashScopeRealtimeProvider(api_key="secret", connector=connector)
    with pytest.raises(RealtimeProviderError) as raised:
        await provider.connect()

    assert raised.value.classification == "fatal"
    assert raised.value.fatal is True
    assert socket.closed is True


def test_dashscope_error_classification_matches_original_provider() -> None:
    assert classify_dashscope_error("InvalidApiKey: invalid API key") == "fatal"
    assert (
        classify_dashscope_error(
            "Session was closed because no response was generated for 120 seconds"
        )
        == "inactivity"
    )
    assert classify_dashscope_error("User is speaking") == "input_busy"
    assert classify_dashscope_error("No active response") == "no_active_response"
    assert classify_dashscope_error("temporary upstream failure") == "other"


def test_ga_protocol_rewrites_responses_and_namespaces_item_ids() -> None:
    protocol = GARealtimeProtocol()

    response = protocol.response_create(
        {
            "modalities": ["text", "audio"],
            "tool_choice": "none",
        }
    )
    assert response == {
        "type": "response.create",
        "response": {
            "tool_choice": "none",
            "output_modalities": ["text", "audio"],
        },
    }
    assert protocol.response_create() == {"type": "response.create"}

    prefixes = {
        "message": "msg_",
        "function_call": "fc_",
        "function_call_output": "fco_",
        "unknown": "item_",
    }
    for item_type, prefix in prefixes.items():
        item = protocol.conversation_item_create({"type": item_type})["item"]
        assert item["id"].startswith(prefix)
        assert len(item["id"]) == len(prefix) + 32

    explicit = protocol.conversation_item_create({"id": "msg_supplied", "type": "message"})
    assert explicit["item"]["id"] == "msg_supplied"


def test_ga_protocol_normalizes_text_events_and_correlates_responses() -> None:
    protocol = GARealtimeProtocol()

    assert protocol.normalize_incoming({"type": "response.output_text.delta", "delta": "你好"}) == {
        "type": "response.text.delta",
        "delta": "你好",
    }
    assert protocol.normalize_incoming({"type": "response.output_text.done", "text": "你好"}) == {
        "type": "response.text.done",
        "text": "你好",
    }
    untouched = {"type": "response.audio.delta", "delta": "abc", "item_id": "item_1"}
    assert protocol.normalize_incoming(untouched) == untouched

    correlated = protocol.correlate_response_create(
        {
            "type": "response.create",
            "response": {"metadata": {"trace": "trace_1"}},
        },
        "request_1",
    )
    assert correlated["response"]["metadata"] == {
        "trace": "trace_1",
        "qwen_audio_request_id": "request_1",
    }
    assert (
        protocol.response_correlation_id(
            {"response": {"metadata": correlated["response"]["metadata"]}}
        )
        == "request_1"
    )
    assert protocol.response_correlation_id({"response": {}}) == ""


def test_speech_to_speech_session_uses_the_ga_shape_and_audio_rates() -> None:
    session = build_speech_to_speech_session(
        instructions="be helpful",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "clock",
                    "description": "Read the clock",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert session == {
        "type": "realtime",
        "instructions": "be helpful",
        "tools": [
            {
                "type": "function",
                "name": "clock",
                "description": "Read the clock",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "output_modalities": ["audio"],
        "audio": {
            "input": {"turn_detection": {"type": "server_vad", "interrupt_response": True}},
            "output": {"format": {"type": "audio/pcm", "rate": 24_000}},
        },
    }
    assert "format" not in session["audio"]["input"]
    assert "voice" not in session["audio"]["output"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auth_token", "expected_headers"),
    [
        (None, {}),
        (" local-secret ", {"Authorization": "Bearer local-secret"}),
    ],
)
async def test_speech_to_speech_connects_without_session_updated_ack(
    auth_token: str | None,
    expected_headers: dict[str, str],
) -> None:
    socket = FakeRealtimeSocket(acknowledge_session_updates=False)
    connector = FakeConnector(socket)
    events: list[dict[str, Any]] = []
    provider = SpeechToSpeechRealtimeProvider(
        auth_token=auth_token,
        instructions="be helpful",
        connector=connector,
    )
    provider.subscribe(lambda event: events.append(dict(event)))

    await asyncio.wait_for(provider.connect(), timeout=0.2)

    assert connector.calls[0][0] == "ws://127.0.0.1:8765/v1/realtime"
    assert connector.calls[0][1] == expected_headers
    assert [event["type"] for event in events] == ["session.created", "connection.ready"]
    assert [event["type"] for event in socket.sent] == ["session.update"]
    assert socket.sent[0]["session"]["output_modalities"] == ["audio"]
    assert socket.sent[0]["session"]["audio"]["output"]["format"]["rate"] == 24_000

    capabilities = provider.capabilities()
    assert capabilities == {
        "inputSampleRate": 16_000,
        "outputSampleRate": 24_000,
        "acknowledgesSessionUpdate": False,
        "singleResponseSlot": True,
        "responseMetadataCorrelation": True,
        "perResponseInstructions": True,
        "conversationItemIdEcho": True,
    }
    assert provider.response_start_timeout_ms == 60_000
    await provider.close()


@pytest.mark.parametrize(
    ("message", "classification"),
    [
        ("session_limit_reached", "capacity_busy"),
        ("All 1 session slots are in use", "capacity_busy"),
        ("The session slot is in use", "capacity_busy"),
        ("Cannot create response while another response is in progress", "response_slot_busy"),
        ("No active response", "no_active_response"),
        ("temporary upstream failure", "other"),
    ],
)
def test_speech_to_speech_error_classification_matches_original_provider(
    message: str,
    classification: str,
) -> None:
    assert classify_speech_to_speech_error(message) == classification


@pytest.mark.asyncio
async def test_speech_to_speech_capacity_error_rejects_connect_before_session_created() -> None:
    socket = FakeRealtimeSocket(acknowledge_session_updates=False)

    async def connector(
        _url: str,
        _headers: Mapping[str, str],
        _timeout: float,
    ) -> FakeRealtimeSocket:
        socket.feed(
            {
                "type": "error",
                "error": {
                    "type": "session_limit_reached",
                    "message": "All 1 session slots are in use",
                },
            }
        )
        return socket

    provider = SpeechToSpeechRealtimeProvider(connector=connector)
    with pytest.raises(RealtimeProviderError) as raised:
        await asyncio.wait_for(provider.connect(), timeout=0.2)

    assert raised.value.classification == "capacity_busy"
    assert raised.value.fatal is False
    assert socket.closed is True
