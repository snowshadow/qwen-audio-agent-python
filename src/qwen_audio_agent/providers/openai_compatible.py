"""OpenAI-compatible Realtime wire adapters for supported voice providers.

The chosen boundary mirrors the original Gateway: the provider owns the
upstream WebSocket and wire events, while turn arbitration, playback, tool
execution, and LangGraph orchestration remain outside this module.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from .realtime import RealtimeListener
from .realtime_catalog import (
    DASHSCOPE_AUDIO_REALTIME_MODELS as _DASHSCOPE_AUDIO_MODELS,
)
from .realtime_catalog import (
    DASHSCOPE_OMNI_REALTIME_MODELS as _DASHSCOPE_OMNI_MODELS,
)
from .realtime_catalog import (
    DEFAULT_DASHSCOPE_REALTIME_MODEL,
    DEFAULT_DASHSCOPE_REALTIME_URL,
    DEFAULT_DASHSCOPE_REALTIME_VOICE,
    DEFAULT_SPEECH_TO_SPEECH_REALTIME_URL,
)

SPEECH_TO_SPEECH_INPUT_SAMPLE_RATE = 16_000
SPEECH_TO_SPEECH_OUTPUT_SAMPLE_RATE = 24_000
SPEECH_TO_SPEECH_RESPONSE_START_TIMEOUT_MS = 60_000

_FATAL_ERROR = re.compile(
    r"invalid[_ -]?api[_ -]?key|incorrect api key|authentication failed|unauthorized"
    r"|unexpected server response: (?:401|403)|\barrearage\b"
    r"|account is not in good standing|allocationquota\.freetieronly"
    r"|free allocated quota exceeded|free tier .* exhausted"
    r"|model(?:\.|_)?accessdenied|model[_ -]?not[_ -]?found",
    re.IGNORECASE,
)
_GA_ITEM_ID_PREFIXES = {
    "message": "msg",
    "function_call": "fc",
    "function_call_output": "fco",
}
_GA_RESPONSE_CORRELATION_KEY = "qwen_audio_request_id"


class RealtimeSocket(Protocol):
    async def send(self, message: str) -> Any: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> Any: ...


RealtimeConnector = Callable[
    [str, Mapping[str, str], float],
    Awaitable[RealtimeSocket],
]


class RealtimeProviderError(RuntimeError):
    """Provider failure with the classification consumed by session owner."""

    def __init__(self, message: str, *, classification: str = "other") -> None:
        super().__init__(message)
        self.classification = classification
        self.fatal = classification == "fatal"


def _event_id() -> str:
    return f"event_{uuid4().hex}"


def _item_id() -> str:
    return f"item_{uuid4().hex}"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _with_model(base_url: str, model: str) -> str:
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["model"] = model
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _error_message(event: Mapping[str, Any]) -> str:
    error = event.get("error")
    details: list[str] = []
    if isinstance(error, Mapping):
        details.extend(str(error.get(key) or "").strip() for key in ("code", "type", "message"))
    details.append(str(event.get("message") or "").strip())
    return (
        ": ".join(dict.fromkeys(value for value in details if value)) or "Realtime provider error"
    )


def classify_dashscope_error(message: str) -> str:
    if re.search(
        r"session was closed because no response was generated for \d+ seconds",
        message,
        re.IGNORECASE,
    ):
        return "inactivity"
    if re.search(r"user is speaking", message, re.IGNORECASE):
        return "input_busy"
    if re.search(r"no active response", message, re.IGNORECASE):
        return "no_active_response"
    if _FATAL_ERROR.search(message):
        return "fatal"
    return "other"


def classify_speech_to_speech_error(message: str) -> str:
    if re.search(
        r"session_limit_reached|session slots? (?:are|is) in use",
        message,
        re.IGNORECASE,
    ):
        return "capacity_busy"
    if re.search(r"another response is in progress", message, re.IGNORECASE):
        return "response_slot_busy"
    if re.search(r"no active response", message, re.IGNORECASE):
        return "no_active_response"
    return "other"


class OpenAICompatibleRealtimeProtocol:
    """Pure mapping for the beta OpenAI-compatible Realtime dialect."""

    @staticmethod
    def encode_outgoing(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {"event_id": _event_id(), **dict(payload)}

    @staticmethod
    def normalize_incoming(event: Mapping[str, Any]) -> dict[str, Any]:
        return dict(event)

    @staticmethod
    def session_update(session: Mapping[str, Any]) -> dict[str, Any]:
        return {"type": "session.update", "session": dict(session)}

    @staticmethod
    def audio_append(audio: str) -> dict[str, Any]:
        return {"type": "input_audio_buffer.append", "audio": audio}

    @staticmethod
    def conversation_item_create(item: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(item)
        value.setdefault("id", _item_id())
        return {"type": "conversation.item.create", "item": value}

    @staticmethod
    def response_create(response: Mapping[str, Any] | None = None) -> dict[str, Any]:
        event: dict[str, Any] = {"type": "response.create"}
        if response:
            event["response"] = dict(response)
        return event

    @staticmethod
    def response_cancel() -> dict[str, Any]:
        return {"type": "response.cancel"}

    @staticmethod
    def user_text_item(text: str) -> dict[str, Any]:
        return {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }

    @staticmethod
    def function_output_item(call_id: str, output: Any) -> dict[str, Any]:
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": _json(output),
        }


class GARealtimeProtocol(OpenAICompatibleRealtimeProtocol):
    """OpenAI Realtime GA dialect used by Hugging Face speech-to-speech."""

    @staticmethod
    def normalize_incoming(event: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(event)
        event_type = value.get("type")
        if event_type == "response.output_text.delta":
            value["type"] = "response.text.delta"
        elif event_type == "response.output_text.done":
            value["type"] = "response.text.done"
        return value

    @staticmethod
    def conversation_item_id(item: Mapping[str, Any]) -> str:
        prefix = _GA_ITEM_ID_PREFIXES.get(str(item.get("type") or ""), "item")
        return f"{prefix}_{uuid4().hex}"

    @classmethod
    def conversation_item_create(cls, item: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(item)
        value.setdefault("id", cls.conversation_item_id(value))
        return {"type": "conversation.item.create", "item": value}

    @staticmethod
    def response_create(response: Mapping[str, Any] | None = None) -> dict[str, Any]:
        event: dict[str, Any] = {"type": "response.create"}
        if response is None:
            return event
        body = dict(response)
        modalities = body.pop("modalities", None)
        if modalities is not None:
            body["output_modalities"] = modalities
        event["response"] = body
        return event

    @staticmethod
    def correlate_response_create(
        payload: Mapping[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        value = dict(payload)
        source_response = value.get("response")
        response = dict(source_response) if isinstance(source_response, Mapping) else {}
        source_metadata = response.get("metadata")
        metadata = dict(source_metadata) if isinstance(source_metadata, Mapping) else {}
        metadata[_GA_RESPONSE_CORRELATION_KEY] = request_id
        response["metadata"] = metadata
        value["response"] = response
        return value

    @staticmethod
    def response_correlation_id(event: Mapping[str, Any]) -> str:
        response = event.get("response")
        if not isinstance(response, Mapping):
            return ""
        metadata = response.get("metadata")
        if not isinstance(metadata, Mapping):
            return ""
        return str(metadata.get(_GA_RESPONSE_CORRELATION_KEY) or "")


async def _default_connector(
    url: str,
    headers: Mapping[str, str],
    open_timeout: float,
) -> RealtimeSocket:
    # Imported lazily so pure wire mapping remains usable without the optional
    # transport package. Production packaging must declare websockets directly.
    from websockets.asyncio.client import connect as websocket_connect

    return await websocket_connect(
        url,
        additional_headers=dict(headers),
        open_timeout=open_timeout,
    )


class OpenAICompatibleRealtimeProvider:
    """Async WebSocket implementation of ``RealtimeProviderPort``."""

    def __init__(
        self,
        *,
        key: str,
        label: str,
        url: str,
        headers: Mapping[str, str],
        initial_session: Mapping[str, Any],
        capabilities: Mapping[str, Any] | None = None,
        classify_error: Callable[[str], str] = lambda _message: "other",
        connector: RealtimeConnector = _default_connector,
        connect_timeout: float = 25.0,
        acknowledges_session_update: bool = True,
        protocol: OpenAICompatibleRealtimeProtocol | None = None,
    ) -> None:
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be positive")
        self.key = key
        self.label = label
        self.url = url
        self.headers = dict(headers)
        self.initial_session = dict(initial_session)
        self._capabilities = dict(capabilities or {})
        self._classify_error = classify_error
        self._connector = connector
        self._connect_timeout = connect_timeout
        self._acknowledges_session_update = acknowledges_session_update
        self.protocol = protocol or OpenAICompatibleRealtimeProtocol()

        self._listeners: set[RealtimeListener] = set()
        self._socket: RealtimeSocket | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._connect_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._session_created = asyncio.Event()
        self._session_updated = asyncio.Event()
        self._terminal = asyncio.Event()
        self._terminal_error: RealtimeProviderError | None = None
        self._ready = False
        self._closing = False
        self._closed_published = False

    def capabilities(self) -> Mapping[str, Any]:
        return dict(self._capabilities)

    def subscribe(self, listener: RealtimeListener) -> Callable[[], None]:
        self._listeners.add(listener)
        return lambda: self._listeners.discard(listener)

    def _publish(self, event: Mapping[str, Any]) -> None:
        for listener in tuple(self._listeners):
            with suppress(Exception):
                listener(dict(event))

    def _publish_closed(self) -> None:
        if self._closed_published:
            return
        self._closed_published = True
        self._publish({"type": "connection.closed", "provider": self.key})

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._ready:
                return
            self._reset_connection_state()
            try:
                async with asyncio.timeout(self._connect_timeout):
                    self._socket = await self._connector(
                        self.url,
                        self.headers,
                        self._connect_timeout,
                    )
                    self._reader_task = asyncio.create_task(
                        self._read_events(),
                        name=f"{self.key}-realtime-reader",
                    )
                    await self._wait_for(self._session_created, "session.created")
                    await self._send_event(
                        self.protocol.session_update(self.initial_session),
                        require_ready=False,
                    )
                    if self._acknowledges_session_update:
                        await self._wait_for(self._session_updated, "session.updated")
                    self._ready = True
            except TimeoutError as error:
                failure = RealtimeProviderError(
                    f"{self.label} connection timed out",
                    classification="other",
                )
                await self._shutdown(publish_closed=False)
                raise failure from error
            except Exception as error:
                failure = (
                    error if isinstance(error, RealtimeProviderError) else self._failure(error)
                )
                await self._shutdown(publish_closed=False)
                raise failure from error
            self._publish({"type": "connection.ready", "provider": self.key})

    def _reset_connection_state(self) -> None:
        self._session_created = asyncio.Event()
        self._session_updated = asyncio.Event()
        self._terminal = asyncio.Event()
        self._terminal_error = None
        self._ready = False
        self._closing = False
        self._closed_published = False

    async def _wait_for(self, signal: asyncio.Event, name: str) -> None:
        signal_task = asyncio.create_task(signal.wait())
        terminal_task = asyncio.create_task(self._terminal.wait())
        done, pending = await asyncio.wait(
            {signal_task, terminal_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if signal_task in done and signal.is_set():
            return
        if self._terminal_error is not None:
            raise self._terminal_error
        raise RealtimeProviderError(f"{self.label} closed before {name}")

    async def _read_events(self) -> None:
        socket = self._socket
        if socket is None:
            return
        try:
            while True:
                raw = await socket.recv()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                try:
                    value = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if not isinstance(value, Mapping):
                    continue
                event = self.protocol.normalize_incoming(value)
                event_type = str(event.get("type") or "")
                if event_type == "session.created":
                    self._session_created.set()
                elif event_type == "session.updated":
                    self._session_updated.set()
                elif event_type == "error":
                    message = _error_message(event)
                    classification = self._classify_error(message)
                    event["classification"] = classification
                    event["fatal"] = classification == "fatal"
                    failure = RealtimeProviderError(message, classification=classification)
                    if not self._ready or failure.fatal:
                        self._terminal_error = failure
                        self._terminal.set()
                    self._publish(event)
                    if failure.fatal:
                        self._publish(
                            {
                                "type": "provider.error",
                                "provider": self.key,
                                "message": message,
                                "classification": classification,
                                "fatal": True,
                            }
                        )
                    continue
                self._publish(event)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self._closing:
                failure = self._failure(error)
                self._terminal_error = failure
                self._publish(
                    {
                        "type": "connection.error",
                        "provider": self.key,
                        "message": str(failure),
                        "classification": failure.classification,
                        "fatal": failure.fatal,
                    }
                )
        finally:
            self._ready = False
            self._terminal.set()
            if not self._closing:
                self._publish_closed()

    def _failure(self, error: BaseException) -> RealtimeProviderError:
        message = str(error) or type(error).__name__
        return RealtimeProviderError(
            message,
            classification=self._classify_error(message),
        )

    async def _send_event(
        self,
        payload: Mapping[str, Any],
        *,
        require_ready: bool = True,
    ) -> None:
        socket = self._socket
        if socket is None or (require_ready and not self._ready):
            raise RealtimeProviderError(f"{self.label} is not connected")
        message = _json(self.protocol.encode_outgoing(payload))
        async with self._send_lock:
            await socket.send(message)

    async def update_session(self, context: Mapping[str, Any]) -> None:
        await self._send_event(self.protocol.session_update(context))

    async def send_audio(self, audio: bytes | str) -> None:
        encoded = base64.b64encode(audio).decode("ascii") if isinstance(audio, bytes) else audio
        await self._send_event(self.protocol.audio_append(encoded))

    async def send_input(self, value: Mapping[str, Any]) -> None:
        source = dict(value)
        supplied_item = source.get("item")
        if isinstance(supplied_item, Mapping):
            item = dict(supplied_item)
        elif "text" in source:
            item = self.protocol.user_text_item(str(source.get("text") or ""))
        else:
            item = source
        await self._send_event(self.protocol.conversation_item_create(item))

    async def send_tool_result(self, value: Mapping[str, Any]) -> None:
        call_id = str(value.get("call_id") or value.get("callId") or "").strip()
        if not call_id:
            raise ValueError("tool result requires call_id")
        item = self.protocol.function_output_item(call_id, value.get("output"))
        await self._send_event(self.protocol.conversation_item_create(item))

    async def create_response(self, context: Mapping[str, Any] | None = None) -> None:
        await self._send_event(self.protocol.response_create(context))

    async def cancel_response(self, response_id: str | None = None) -> None:
        del response_id  # Both supported dialects cancel the active response slot.
        await self._send_event(self.protocol.response_cancel())

    async def close(self) -> None:
        await self._shutdown(publish_closed=True)

    async def _shutdown(self, *, publish_closed: bool) -> None:
        self._closing = True
        self._ready = False
        socket, self._socket = self._socket, None
        reader, self._reader_task = self._reader_task, None
        if socket is not None:
            with suppress(Exception):
                await socket.close()
        if reader is not None and reader is not asyncio.current_task() and not reader.done():
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        self._terminal.set()
        if publish_closed:
            self._publish_closed()


def build_dashscope_session(
    *,
    model: str,
    voice: str | None = None,
    instructions: str = "",
    tools: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if model in _DASHSCOPE_AUDIO_MODELS:
        selected_voice = voice or DEFAULT_DASHSCOPE_REALTIME_VOICE
        turn_detection = {"type": "smart_turn"}
    elif model in _DASHSCOPE_OMNI_MODELS:
        selected_voice = voice or "Ethan"
        turn_detection = {"type": "semantic_vad"}
    else:
        raise ValueError(f"unsupported DashScope Realtime model: {model}")
    return {
        "instructions": instructions,
        "tools": [dict(tool) for tool in tools],
        "modalities": ["text", "audio"],
        "voice": selected_voice,
        "output_audio_format": "pcm",
        "input_audio_format": "pcm",
        "turn_detection": turn_detection,
    }


def _flatten_ga_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(tool)
    function = value.get("function")
    if not isinstance(function, Mapping):
        return value
    flattened: dict[str, Any] = {"type": "function"}
    for key in ("name", "description", "parameters"):
        if key in function:
            flattened[key] = function[key]
    return flattened


def build_speech_to_speech_session(
    *,
    instructions: str = "",
    tools: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the GA session shape accepted by speech-to-speech."""

    return {
        "type": "realtime",
        "instructions": instructions,
        "tools": [_flatten_ga_tool(tool) for tool in tools],
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                # GA only permits an explicit PCM input format at 24 kHz. An
                # omitted format keeps speech-to-speech's native 16 kHz input.
                "turn_detection": {"type": "server_vad", "interrupt_response": True},
            },
            "output": {
                "format": {
                    "type": "audio/pcm",
                    "rate": SPEECH_TO_SPEECH_OUTPUT_SAMPLE_RATE,
                }
            },
        },
    }


class DashScopeRealtimeProvider(OpenAICompatibleRealtimeProvider):
    """OpenAI-compatible adapter configured for the supported Qwen models."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_DASHSCOPE_REALTIME_MODEL,
        base_url: str = DEFAULT_DASHSCOPE_REALTIME_URL,
        voice: str | None = None,
        instructions: str = "",
        tools: Sequence[Mapping[str, Any]] = (),
        connector: RealtimeConnector = _default_connector,
        connect_timeout: float = 25.0,
    ) -> None:
        key = api_key.strip()
        if not key:
            raise ValueError("DashScope API key is required")
        family = "omni" if model in _DASHSCOPE_OMNI_MODELS else "audio"
        session = build_dashscope_session(
            model=model,
            voice=voice,
            instructions=instructions,
            tools=tools,
        )
        super().__init__(
            key="dashscope",
            label="Qwen-Audio-Realtime",
            url=_with_model(base_url.rstrip("?"), model),
            headers={"Authorization": f"Bearer {key}"},
            initial_session=session,
            capabilities={
                "inputSampleRate": 16000,
                "outputSampleRate": 24000,
                "acknowledgesSessionUpdate": True,
                "singleResponseSlot": False,
                "responseMetadataCorrelation": False,
                "perResponseInstructions": True,
                "conversationItemIdEcho": family != "omni",
            },
            classify_error=classify_dashscope_error,
            connector=connector,
            connect_timeout=connect_timeout,
        )


class SpeechToSpeechRealtimeProvider(OpenAICompatibleRealtimeProvider):
    """GA Realtime profile for a user-managed Hugging Face endpoint."""

    aliases = ("s2s",)
    input_sample_rate = SPEECH_TO_SPEECH_INPUT_SAMPLE_RATE
    output_sample_rate = SPEECH_TO_SPEECH_OUTPUT_SAMPLE_RATE
    response_start_timeout_ms = SPEECH_TO_SPEECH_RESPONSE_START_TIMEOUT_MS

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_SPEECH_TO_SPEECH_REALTIME_URL,
        auth_token: str | None = None,
        instructions: str = "",
        tools: Sequence[Mapping[str, Any]] = (),
        connector: RealtimeConnector = _default_connector,
        connect_timeout: float = 25.0,
    ) -> None:
        url = base_url.strip()
        if not url:
            raise ValueError("speech-to-speech Realtime URL is required")
        token = (auth_token or "").strip()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        super().__init__(
            key="speech-to-speech",
            label="Hugging Face Speech-to-Speech",
            url=url,
            headers=headers,
            initial_session=build_speech_to_speech_session(
                instructions=instructions,
                tools=tools,
            ),
            capabilities={
                "inputSampleRate": SPEECH_TO_SPEECH_INPUT_SAMPLE_RATE,
                "outputSampleRate": SPEECH_TO_SPEECH_OUTPUT_SAMPLE_RATE,
                "acknowledgesSessionUpdate": False,
                "singleResponseSlot": True,
                "responseMetadataCorrelation": True,
                "perResponseInstructions": True,
                "conversationItemIdEcho": True,
            },
            classify_error=classify_speech_to_speech_error,
            connector=connector,
            connect_timeout=connect_timeout,
            acknowledges_session_update=False,
            protocol=GARealtimeProtocol(),
        )
