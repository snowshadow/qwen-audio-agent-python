from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from qwen_audio_agent.providers import RealtimeProviderSession


class FakeProvider:
    def __init__(self) -> None:
        self.connect_calls = 0
        self.connect_gate = asyncio.Event()
        self.audio: list[bytes | str] = []
        self.session_updates: list[dict[str, Any]] = []
        self.listeners: set[Callable[[Mapping[str, Any]], None]] = set()
        self.closed = False

    async def connect(self) -> None:
        self.connect_calls += 1
        await self.connect_gate.wait()

    async def update_session(self, context: Mapping[str, Any]) -> None:
        self.session_updates.append(dict(context))

    async def send_audio(self, audio: bytes | str) -> None:
        self.audio.append(audio)

    async def send_input(self, value: Mapping[str, Any]) -> None:
        return None

    async def send_tool_result(self, value: Mapping[str, Any]) -> None:
        return None

    async def create_response(self, context: Mapping[str, Any] | None = None) -> None:
        return None

    async def cancel_response(self, response_id: str | None = None) -> None:
        return None

    def capabilities(self) -> dict[str, bool]:
        return {"audio": True}

    def subscribe(
        self,
        listener: Callable[[Mapping[str, Any]], None],
    ) -> Callable[[], None]:
        self.listeners.add(listener)
        return lambda: self.listeners.discard(listener)

    def emit(self, event: Mapping[str, Any]) -> None:
        for listener in tuple(self.listeners):
            listener(event)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_concurrent_connect_is_coalesced_and_audio_buffer_is_bounded() -> None:
    provider = FakeProvider()
    session = RealtimeProviderSession(provider, max_pending_audio_chunks=2)

    calls = [asyncio.create_task(session.send_audio(value)) for value in (b"a", b"b", b"c")]
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert provider.connect_calls == 1
    provider.connect_gate.set()
    await asyncio.gather(*calls)

    assert provider.audio == [b"b", b"c"]
    await session.close()


@pytest.mark.asyncio
async def test_close_event_schedules_only_one_reconnect_and_fatal_clears_audio() -> None:
    provider = FakeProvider()
    provider.connect_gate.set()
    session = RealtimeProviderSession(provider)
    await session.ensure_connected()

    provider.emit({"type": "connection.closed"})
    provider.emit({"type": "connection.closed"})
    await asyncio.sleep(0.01)
    assert provider.connect_calls == 2

    session.ready = False
    session.pending_audio.append(b"queued")
    provider.emit({"type": "provider.error", "fatal": True, "message": "bad key"})
    assert session.blocked_error == "bad key"
    assert list(session.pending_audio) == []
    with pytest.raises(RuntimeError, match="bad key"):
        await session.ensure_connected()
    await session.close()


@pytest.mark.asyncio
async def test_safe_reset_replaces_socket_and_publishes_one_epoch_boundary() -> None:
    provider = FakeProvider()
    provider.connect_gate.set()
    session = RealtimeProviderSession(provider)
    events: list[dict[str, Any]] = []
    session.subscribe(lambda event: events.append(dict(event)))
    await session.ensure_connected()
    await session.update_session({"instructions": "remember this"})

    await session.reset_connection()

    assert provider.connect_calls == 2
    assert session.ready is True
    assert provider.session_updates == [
        {"instructions": "remember this"},
        {"instructions": "remember this"},
    ]
    assert [event["type"] for event in events] == ["provider.reset"]
    await session.close()
