"""Graph-outside realtime orchestration for one Gateway WebSocket connection.

The realtime provider owns model wire details.  This runtime owns the
latency-sensitive connection timeline: turn correlation, barge-in fencing,
public event projection, and delivery of normalized function calls into the
checkpointed :class:`FrontendTurnGraph`.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Awaitable, Mapping
from typing import Any
from uuid import uuid4

from qwen_audio_agent.application.realtime_session import (
    ContextProvider,
    ProviderFactory,
    _display_input,
    _event_context,
    _item_id,
    _maybe_await,
    _normalize_input_parts,
    _owner_id,
    _PendingResponse,
    _RealtimeConnectionState,
    _response_correlation_id,
    _response_id,
    _task_event_belongs_to_owner,
    _tool_identity,
)
from qwen_audio_agent.domain.frontend import FrontendExecutionContext, FrontendToolCall
from qwen_audio_agent.graphs.frontend import FrontendTurnGraph
from qwen_audio_agent.protocol import GatewayClientEvent, GatewayServerEvent
from qwen_audio_agent.providers.realtime import RealtimeProviderSession
from qwen_audio_agent.runtime.conversation import ConversationStore
from qwen_audio_agent.runtime.result_delivery import (
    PresentationOutcome,
    ResultDeliveryCoordinator,
)
from qwen_audio_agent.runtime.turns import TurnContext


class FrontendRealtimeApplication:
    """Bind Gateway client events to a provider and FrontendTurnGraph.

    The object is intentionally callable so it can be injected directly as
    ``create_gateway_app(on_client_event=...)``.  Provider events are pushed
    later through the transport-supplied private ``emit`` callback.
    """

    def __init__(
        self,
        *,
        provider_factory: ProviderFactory,
        frontend_graph: FrontendTurnGraph,
        context_provider: ContextProvider | None = None,
        conversation_store: ConversationStore | None = None,
        task_source: Any | None = None,
        delivery_leases: Any | None = None,
        presentation_timeout_seconds: float = 90.0,
    ) -> None:
        self.provider_factory = provider_factory
        self.frontend_graph = frontend_graph
        self.context_provider = context_provider
        self.conversation_store = conversation_store
        self.task_source = task_source
        self.delivery_leases = delivery_leases
        self.presentation_timeout_seconds = max(1.0, presentation_timeout_seconds)
        self._connections: dict[str, _RealtimeConnectionState] = {}

    async def __call__(
        self,
        context: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> list[dict[str, Any]] | dict[str, Any] | None:
        event_type = str(event.get("type") or "")
        connection_id = str(context.get("connectionId") or "")
        if not connection_id:
            return None
        if event_type == GatewayClientEvent.CONNECT.value:
            return await self._open(context, event)
        state = self._connections.get(connection_id)
        if state is None or state.closed:
            return None

        if event_type == "__voice.deactivated":
            state.input_enabled = False
            state.output_enabled = False
            if state.delivery is not None:
                await state.delivery.pause()
            for interrupted in await self._clear_responses(
                state,
                reason="voice_deactivated",
            ):
                state.emit(interrupted)
            return None

        if event_type in {
            GatewayClientEvent.UNMUTE.value,
            GatewayClientEvent.INPUT_UNMUTE.value,
        }:
            state.sleeping = False
            state.output_enabled = True
            if event_type == GatewayClientEvent.INPUT_UNMUTE.value:
                state.input_enabled = True
            self._ensure_connect_task(state)
            await self._flush_deferred(state)
        elif event_type == GatewayClientEvent.INPUT_MUTE.value:
            state.input_enabled = False
            state.provider.clear_pending_audio()
        elif event_type == GatewayClientEvent.MUTE.value:
            state.input_enabled = False
            state.output_enabled = False
            state.provider.clear_pending_audio()
            await state.provider.cancel_response()
            state.announcement_window.reset()
            if state.delivery is not None:
                await state.delivery.pause()
        elif event_type == GatewayClientEvent.SLEEP.value:
            state.sleeping = True
            state.input_enabled = False
            state.output_enabled = False
            state.provider.clear_pending_audio()
            await state.provider.cancel_response()
            state.announcement_window.reset()
            if state.delivery is not None:
                await state.delivery.pause()
            return [
                {"type": GatewayServerEvent.PLAYBACK_CLEAR.value, "reason": "sleep"},
                {"type": GatewayServerEvent.VOICE_SLEEP.value, "state": "sleeping"},
            ]
        elif event_type == GatewayClientEvent.WAKE.value:
            state.sleeping = False
            state.input_enabled = True
            state.output_enabled = True
            self._ensure_connect_task(state)
            await self._flush_deferred(state)
            return {
                "type": GatewayServerEvent.VOICE_STATE.value,
                "state": "idle",
            }
        elif event_type == GatewayClientEvent.AUDIO_APPEND.value:
            if state.input_enabled and not state.sleeping and not context.get("inputSuspended"):
                audio = event.get("audio")
                if isinstance(audio, (str, bytes)) and 0 < len(audio) <= 8 * 1024 * 1024:
                    await state.provider.send_audio(audio)
        elif event_type in {
            GatewayClientEvent.TEXT_MESSAGE.value,
            GatewayClientEvent.INPUT_MESSAGE.value,
        }:
            if state.sleeping:
                return {
                    "type": GatewayServerEvent.ERROR.value,
                    "message": "已休眠，请先唤醒。",
                }
            return await self._manual_input(state, event)
        elif event_type == GatewayClientEvent.INTERRUPT.value:
            return await self._interrupt(state, reason="user_interruption")
        elif event_type == GatewayClientEvent.PLAYBACK_STARTED.value:
            response_id = str(event.get("responseId") or event.get("response_id") or "")
            response_context = state.response_contexts.get(response_id, state.turns.current())
            if response_id and state.output_enabled and state.is_voice_active():
                state.playback_started.add(response_id)
                state.announcement_window.start_playback(response_id)
                task_ids = state.response_task_ids.get(response_id, ())
                if task_ids and state.delivery is not None:
                    await state.delivery.playback_started(task_ids)
                return {
                    "type": GatewayServerEvent.VOICE_STATE.value,
                    "state": "speaking",
                    "responseId": response_id,
                    "turnId": response_context.turn_id,
                    "origin": state.response_origins.get(response_id, "model"),
                }
        elif event_type in {
            GatewayClientEvent.PLAYBACK_ENDED.value,
            GatewayClientEvent.PLAYBACK_CANCELLED.value,
        }:
            response_id = str(event.get("responseId") or event.get("response_id") or "")
            response_context = state.response_contexts.get(response_id, state.turns.current())
            if response_id and state.output_enabled and state.is_voice_active():
                task_ids = state.response_task_ids.get(response_id, ())
                started = response_id in state.playback_started
                origin = state.response_origins.get(response_id, "model")
                state.announcement_window.finish_playback(
                    response_id,
                    has_function_call=response_id in state.response_has_function_call,
                )
                if (
                    event_type == GatewayClientEvent.PLAYBACK_CANCELLED.value
                    and task_ids
                    and not started
                    and state.delivery is not None
                ):
                    await state.delivery.playback_failed(task_ids)
                state.active_responses.discard(response_id)
                self._forget_response(state, response_id)
                if state.delivery is not None:
                    await state.delivery.flush()
                self._flush_permissions(state)
                return {
                    "type": GatewayServerEvent.VOICE_STATE.value,
                    "state": "listening" if state.turns.user_speaking else "idle",
                    "responseId": response_id,
                    "turnId": response_context.turn_id,
                    "origin": origin,
                }
        return None

    async def disconnect(self, context: Mapping[str, Any]) -> None:
        connection_id = str(context.get("connectionId") or "")
        state = self._connections.pop(connection_id, None)
        if state is None:
            return
        state.closed = True
        if state.unsubscribe:
            state.unsubscribe()
        if state.unsubscribe_tasks:
            state.unsubscribe_tasks()
        if state.delivery is not None:
            await state.delivery.close()
        for future in state.response_completions.values():
            if not future.done():
                future.set_result(False)
        tasks = [task for task in [state.connect_task, *state.tool_tasks] if task]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        state.turns.close()
        await state.provider.close()

    async def close(self) -> None:
        for connection_id in tuple(self._connections):
            await self.disconnect({"connectionId": connection_id})

    async def _open(
        self,
        context: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        connection_id = str(context["connectionId"])
        if connection_id in self._connections:
            await self.disconnect({"connectionId": connection_id})
        provider = await _maybe_await(self.provider_factory(context, event))
        session = RealtimeProviderSession(provider)
        emitter = context.get("emit")
        if not callable(emitter):
            raise TypeError("realtime context requires an emit callback")
        voice_enabled = event.get("voiceEnabled") is True
        active_probe = context.get("isVoiceActive")
        if not callable(active_probe):
            initially_active = context.get("voiceActive") is True

            def active_probe() -> bool:
                return initially_active

        state = _RealtimeConnectionState(
            connection_id=connection_id,
            owner_id=str(context.get("ownerId") or "personal"),
            session_id=str(context.get("sessionId") or "main"),
            emit=emitter,
            provider=session,
            input_enabled=event.get("inputEnabled") is True or voice_enabled,
            output_enabled=event.get("outputEnabled") is True or voice_enabled,
            text_only=event.get("textOnly") is True,
            client_states=tuple(str(value) for value in event.get("clientStates", ())[:16])
            if isinstance(event.get("clientStates"), list)
            else (),
            client_context={
                "timeZone": str(event.get("timeZone") or "")[:80],
                "locale": str(event.get("locale") or "")[:40],
                "workingDirectory": str(event.get("workingDirectory") or "")[:2_048],
            },
            is_voice_active=active_probe,
        )
        loop = asyncio.get_running_loop()

        if self.delivery_leases is not None:
            state.delivery = ResultDeliveryCoordinator(
                leases=self.delivery_leases,
                owner_id=state.owner_id,
                session_id=state.session_id,
                claimant_id=f"voice:{state.connection_id}",
                present_result=lambda text, result_context, inject: self._present_result(
                    state,
                    text,
                    result_context,
                    inject,
                ),
                is_delivery_blocked=lambda: self._delivery_blocked(state),
                include_other_sessions=True,
                on_error=lambda error: state.emit(
                    {
                        "type": GatewayServerEvent.ERROR.value,
                        "message": f"异步结果暂时无法播报: {error}",
                    }
                ),
            )

        def on_provider_event(value: Mapping[str, Any]) -> None:
            normalized = dict(value)
            if str(normalized.get("type") or "") in {
                "provider.closed",
                "connection.closed",
                "provider.reset",
            }:
                # Advance synchronously, before queued event handlers run, so
                # late events already emitted by the old socket can be fenced.
                state.provider_epoch += 1
                state.provider_recovering = True
                state.provider_recovered.clear()
            normalized["__providerEpoch"] = state.provider_epoch

            def schedule() -> None:
                if state.closed:
                    return
                task = asyncio.create_task(self._provider_event(state, normalized))
                state.tool_tasks.add(task)
                task.add_done_callback(state.tool_tasks.discard)

            loop.call_soon_threadsafe(schedule)

        state.unsubscribe = session.subscribe(on_provider_event)
        subscribe_tasks = getattr(self.task_source, "subscribe", None)
        if callable(subscribe_tasks):

            def on_task_event(value: Mapping[str, Any]) -> None:
                loop.call_soon_threadsafe(self._task_event, state, dict(value))

            unsubscribe = subscribe_tasks(on_task_event)
            state.unsubscribe_tasks = unsubscribe if callable(unsubscribe) else None
        self._connections[connection_id] = state
        self._queue_pending_notifications(state)
        self._ensure_connect_task(state, context=context, event=event)
        return [
            {
                "type": GatewayServerEvent.VOICE_CONNECTION.value,
                "state": "connecting",
                "provider": str(event.get("provider") or ""),
            }
        ]

    def _ensure_connect_task(
        self,
        state: _RealtimeConnectionState,
        *,
        context: Mapping[str, Any] | None = None,
        event: Mapping[str, Any] | None = None,
    ) -> None:
        if state.connect_task is not None and not state.connect_task.done():
            return
        state.connect_task = asyncio.create_task(
            self._connect(state, context or {}, event or {}),
            name=f"realtime-connect:{state.connection_id}",
        )

    async def _connect(
        self,
        state: _RealtimeConnectionState,
        context: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> None:
        try:
            if self.context_provider is not None:
                agent_context = dict(await _maybe_await(self.context_provider(context, event)))
                await state.provider.update_session(agent_context)
            else:
                await state.provider.ensure_connected()
            if state.closed:
                return
            capabilities = dict(state.provider.provider.capabilities())
            state.frontend_capabilities = tuple(
                str(value) for value in capabilities.get("frontendCapabilities", ())
            )
            provider_name = str(
                capabilities.get("provider")
                or getattr(state.provider.provider, "key", "")
                or event.get("provider")
                or ""
            )
            state.emit(
                {
                    "type": GatewayServerEvent.VOICE_CONNECTION.value,
                    "state": "connected",
                    "provider": provider_name,
                }
            )
            state.emit(
                {
                    "type": GatewayServerEvent.VOICE_READY.value,
                    "provider": provider_name,
                    "inputSampleRate": int(capabilities.get("inputSampleRate") or 16_000),
                    "outputSampleRate": int(capabilities.get("outputSampleRate") or 24_000),
                }
            )
            await self._flush_deferred(state)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not state.closed:
                state.emit(
                    {
                        "type": GatewayServerEvent.VOICE_CONNECTION.value,
                        "state": "unavailable",
                        "message": str(error),
                    }
                )
                state.emit({"type": GatewayServerEvent.ERROR.value, "message": str(error)})

    async def _manual_input(
        self,
        state: _RealtimeConnectionState,
        event: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        parts = _normalize_input_parts(event)
        if not parts:
            return [{"type": GatewayServerEvent.ERROR.value, "message": "输入内容不能为空"}]
        state.manual_sequence += 1
        turn_id = f"text_{int(time.time() * 1000)}_{state.manual_sequence}"
        begun = state.turns.begin_manual(turn_id)
        state.input_parts = parts
        await self._clear_responses(state, reason="user_interruption")
        context = begun.context
        state.announcement_window.begin_turn(context.turn_id)
        state.announcement_window.end_speech()
        self._record(
            state,
            {
                "id": f"text:user:{context.turn_id}",
                "role": "user",
                "content": _display_input(parts),
                "source": "text-user",
                "turnId": context.turn_id,
            },
        )
        await state.provider.send_input({"text": self._provider_input(parts)})
        await self._request_response(
            state,
            _PendingResponse(context=context),
            {"modalities": ["text"] if state.text_only else ["text", "audio"]},
        )
        return [
            {"type": GatewayServerEvent.PLAYBACK_CLEAR.value, "reason": "user_interruption"},
            {"type": GatewayServerEvent.TURN_STARTED.value, "turnId": context.turn_id},
            {
                "type": GatewayServerEvent.VOICE_STATE.value,
                "state": "processing",
                "turnId": context.turn_id,
                "origin": "model",
            },
            {
                "type": GatewayServerEvent.TRANSCRIPT_FINAL.value,
                "role": "user",
                "content": _display_input(parts),
                "turnId": context.turn_id,
            },
        ]

    async def _flush_deferred(self, state: _RealtimeConnectionState) -> None:
        """Retry deferred permissions and result delivery on a ready socket."""

        self._queue_pending_notifications(state)
        self._flush_permissions(state)
        # Let a permission request reserve its pending response before result
        # delivery checks the shared presentation window.
        if state.permission_task is not None:
            await asyncio.sleep(0)
        if state.delivery is not None:
            await state.delivery.flush()

    async def _interrupt(
        self,
        state: _RealtimeConnectionState,
        *,
        reason: str,
    ) -> list[dict[str, Any]]:
        state.turns.advance_boundary()
        interrupted = await self._clear_responses(state, reason=reason)
        return [
            {"type": GatewayServerEvent.PLAYBACK_CLEAR.value, "reason": reason},
            *interrupted,
        ]

    async def _provider_event(
        self,
        state: _RealtimeConnectionState,
        event: Mapping[str, Any],
    ) -> None:
        if state.closed:
            return
        event_type = str(event.get("type") or "")
        event_epoch = event.get("__providerEpoch")
        if isinstance(event_epoch, int) and event_epoch != state.provider_epoch:
            return
        if event_type in {"provider.closed", "connection.closed", "provider.reset"}:
            try:
                permission_task = state.permission_task
                if permission_task is not None and not permission_task.done():
                    permission_task.cancel()
                    await asyncio.gather(permission_task, return_exceptions=True)
                state.permission_task = None
                state.announcing_permission = ""
                for interrupted in await self._clear_responses(
                    state,
                    reason="provider_reconnect",
                ):
                    state.emit(interrupted)
                # A closed socket cannot produce the missing response.created
                # events. Its cancellation tombstones must not consume responses
                # created by the replacement socket.
                state.pending_response_contexts.clear()
                if state.delivery is not None:
                    await state.delivery.provider_disconnected()
                # Cancellation can race with request creation, so clear once
                # more after presentation and permission jobs are fenced.
                state.pending_response_contexts.clear()
            finally:
                if event_epoch == state.provider_epoch:
                    state.provider_recovering = False
                    state.provider_recovered.set()
            if event_epoch == state.provider_epoch and state.provider.ready:
                await self._flush_deferred(state)
            return
        if event_type in {"provider.ready", "connection.ready"}:
            try:
                # Provider implementations publish ready just before the
                # session wrapper finishes its connect task. Waiting here
                # makes deferred work observe the replacement socket as ready.
                await state.provider.ensure_connected()
            except Exception:
                return
            if event_epoch != state.provider_epoch or state.closed or state.provider_recovering:
                return
            await self._flush_deferred(state)
            return
        if event_type in {"provider.error", "connection.error", "error"}:
            classification = str(event.get("classification") or "")
            if classification == "no_active_response" or event.get("__voiceRetried"):
                return
            message = str(
                event.get("message")
                or (
                    event.get("error", {}).get("message")
                    if isinstance(event.get("error"), Mapping)
                    else ""
                )
                or "实时语音服务错误"
            )
            state.emit({"type": GatewayServerEvent.ERROR.value, "message": message})
            return
        item_id = _item_id(event)
        if event_type == "input_audio_buffer.speech_started":
            begun = state.turns.begin_voice(item_id)
            if not begun.accepted or begun.context is None:
                return
            state.input_parts = ()
            interrupted = await self._clear_responses(
                state,
                reason="user_interruption",
            )
            state.announcement_window.begin_turn(begun.context.turn_id)
            state.emit(
                {
                    "type": GatewayServerEvent.PLAYBACK_CLEAR.value,
                    "reason": "user_interruption",
                }
            )
            for value in interrupted:
                state.emit(value)
            state.emit(
                {"type": GatewayServerEvent.TURN_STARTED.value, "turnId": begun.context.turn_id}
            )
            state.emit(
                {
                    "type": GatewayServerEvent.VOICE_STATE.value,
                    "state": "listening",
                    "turnId": begun.context.turn_id,
                    "origin": "model",
                }
            )
            return
        if event_type in {
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.committed",
        }:
            state.turns.end_speech()
            state.announcement_window.end_speech()
            context = state.turns.resolve_input(item_id)
            if context.turn_id and not state.turns.is_stale(context):
                state.emit(
                    {
                        "type": GatewayServerEvent.VOICE_STATE.value,
                        "state": "processing",
                        "turnId": context.turn_id,
                        "origin": "model",
                    }
                )
            return
        if event_type in {
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.text",
        }:
            context = state.turns.resolve_input(item_id)
            content = str(event.get("delta") or event.get("text") or "")
            if content and context.turn_id and not state.turns.is_stale(context):
                state.emit(
                    {
                        "type": GatewayServerEvent.TRANSCRIPT_DELTA.value,
                        "role": "user",
                        "content": content,
                        "turnId": context.turn_id,
                        "replace": True,
                    }
                )
            return
        if event_type == "conversation.item.input_audio_transcription.completed":
            completed = state.turns.complete_input(item_id)
            content = str(event.get("transcript") or event.get("text") or "").strip()
            if (
                content
                and not completed.invalid
                and not completed.duplicate
                and not state.turns.is_stale(completed.context)
            ):
                state.turns.commit(completed.context)
                state.input_parts = ({"type": "text", "text": content},)
                self._record(
                    state,
                    {
                        "id": f"voice:user:{completed.context.turn_id}",
                        "role": "user",
                        "content": content,
                        "source": "voice-user",
                        "turnId": completed.context.turn_id,
                    },
                )
                state.emit(
                    {
                        "type": GatewayServerEvent.TRANSCRIPT_FINAL.value,
                        "role": "user",
                        "content": content,
                        "turnId": completed.context.turn_id,
                    }
                )
            return

        response_id = _response_id(event)
        if event_type == "response.created":
            pending = self._take_pending_response(state, event)
            if pending is not None and pending.cancelled:
                if response_id:
                    state.interrupted_responses.add(response_id)
                try:
                    await state.provider.cancel_response(response_id or None)
                except Exception:
                    pass
                return
            fallback = state.turns.committed()
            if not fallback.turn_id:
                fallback = state.turns.current()
            if pending is not None:
                fallback = pending.context
            context = _event_context(event, fallback)
            state.response_contexts[response_id] = context
            origin = str(
                pending.origin if pending is not None else event.get("__voiceOrigin") or "model"
            )
            state.response_origins[response_id] = origin
            if pending is not None:
                if pending.task_ids:
                    state.response_task_ids[response_id] = pending.task_ids
                if pending.authorization_id:
                    state.response_authorizations[response_id] = pending.authorization_id
                if pending.completion is not None:
                    state.response_completions[response_id] = pending.completion
            state.active_responses.add(response_id)
            if response_id and not state.turns.is_stale(context):
                state.emit(
                    {
                        "type": GatewayServerEvent.RESPONSE_STARTED.value,
                        "responseId": response_id,
                        "turnId": context.turn_id,
                        "turnGeneration": context.turn_generation,
                        **self._response_public_context(state, response_id),
                    }
                )
            return

        context = state.response_contexts.get(response_id, state.turns.committed())
        stale = bool(context.turn_id and state.turns.is_stale(context))
        if event_type == "response.function_call_arguments.done":
            if not stale and response_id not in state.interrupted_responses:
                state.response_has_function_call.add(response_id)
                origin = state.response_origins.get(response_id, "model")
                if origin != "model":
                    state.tool_followup_suppressed.add(response_id)
                    self._spawn(
                        state,
                        self._reject_non_model_tool_call(
                            state,
                            event,
                            context,
                            response_id,
                            origin,
                        ),
                        name=f"reject-frontend-tool:{state.connection_id}",
                    )
                else:
                    self._spawn_tool_call(state, event, context, response_id)
            return
        if stale or response_id in state.interrupted_responses:
            if event_type == "response.done":
                state.active_responses.discard(response_id)
                completion = state.response_completions.get(response_id)
                if completion is not None and not completion.done():
                    completion.set_result(False)
                self._forget_response(state, response_id)
            return
        if event_type in {"response.audio.delta", "response.output_audio.delta"}:
            state.response_has_audio.add(response_id)
            state.announcement_window.queue_audio(
                response_id,
                turn_id=context.turn_id,
                origin=state.response_origins.get(response_id, "model"),
            )
            state.emit(
                {
                    "type": GatewayServerEvent.AUDIO_DELTA.value,
                    "audio": str(event.get("delta") or ""),
                    "sampleRate": int(event.get("sampleRate") or 24_000),
                    "responseId": response_id,
                    "turnId": context.turn_id,
                    **self._response_public_context(state, response_id),
                }
            )
        elif event_type in {
            "response.audio_transcript.delta",
            "response.output_audio_transcript.delta",
            "response.text.delta",
        }:
            delta = str(event.get("delta") or event.get("transcript") or event.get("text") or "")
            if delta:
                state.response_text[response_id] = (
                    f"{state.response_text.get(response_id, '')}{delta}"
                )[:100_000]
            self._record(
                state,
                {
                    "id": f"voice:assistant:{response_id}",
                    "role": "assistant",
                    "content": state.response_text.get(response_id, ""),
                    "source": "realtime-direct",
                    "turnId": context.turn_id,
                    "responseId": response_id,
                },
            )
            state.emit(
                {
                    "type": GatewayServerEvent.TRANSCRIPT_DELTA.value,
                    "role": "assistant",
                    "content": delta,
                    "responseId": response_id,
                    "turnId": context.turn_id,
                    **self._response_public_context(state, response_id),
                }
            )
        elif event_type in {
            "response.audio_transcript.done",
            "response.output_audio_transcript.done",
            "response.text.done",
        }:
            content = str(
                event.get("transcript")
                or event.get("text")
                or state.response_text.get(response_id, "")
            )
            if content:
                state.response_text[response_id] = content[:100_000]
                self._record(
                    state,
                    {
                        "id": f"voice:assistant:{response_id}",
                        "role": "assistant",
                        "content": content,
                        "source": "realtime-direct",
                        "turnId": context.turn_id,
                        "responseId": response_id,
                    },
                )
            state.emit(
                {
                    "type": GatewayServerEvent.TRANSCRIPT_FINAL.value,
                    "role": "assistant",
                    "content": content,
                    "responseId": response_id,
                    "turnId": context.turn_id,
                    **self._response_public_context(state, response_id),
                }
            )
        elif event_type == "response.done":
            state.active_responses.discard(response_id)
            response = event.get("response")
            response_status = (
                str(response.get("status") or "") if isinstance(response, Mapping) else ""
            )
            failed = response_status in {"failed", "cancelled", "incomplete"}
            origin = state.response_origins.get(response_id, "model")
            has_audio = response_id in state.response_has_audio
            has_function_call = response_id in state.response_has_function_call
            state.announcement_window.response_done(
                turn_id=context.turn_id,
                origin=origin,
                has_audio=has_audio,
                has_function_call=has_function_call,
                failed=failed,
            )
            completion = state.response_completions.get(response_id)
            if completion is not None and not completion.done():
                completion.set_result(not failed)
            state.emit(
                {
                    "type": GatewayServerEvent.AUDIO_DONE.value,
                    "responseId": response_id,
                    "turnId": context.turn_id,
                    **self._response_public_context(state, response_id),
                }
            )
            if not has_audio:
                state.emit(
                    {
                        "type": GatewayServerEvent.VOICE_STATE.value,
                        "state": "idle",
                        "turnId": context.turn_id,
                        "origin": origin,
                    }
                )
                task_ids = state.response_task_ids.get(response_id, ())
                if task_ids and state.delivery is not None and state.text_only and not failed:
                    await state.delivery.playback_started(task_ids)
                self._forget_response(state, response_id)
                if state.delivery is not None:
                    await state.delivery.flush()
            if origin == "model":
                state.turns.finish_manual_response(context)
            if has_function_call:
                state.tool_response_done.add(response_id)
                await self._maybe_create_tool_followup(state, response_id)
            self._flush_permissions(state)

    async def _clear_responses(
        self,
        state: _RealtimeConnectionState,
        *,
        reason: str,
    ) -> list[dict[str, Any]]:
        capabilities = dict(state.provider.provider.capabilities())
        requires_safe_reset = bool(
            not capabilities.get("responseMetadataCorrelation")
            and not state.provider_recovering
            and any(not pending.cancelled for pending in state.pending_response_contexts)
        )
        try:
            await state.provider.cancel_response()
        except Exception:
            pass
        for pending in tuple(state.pending_response_contexts):
            pending.cancelled = True
            if pending.completion is not None and not pending.completion.done():
                pending.completion.set_result(False)
        events: list[dict[str, Any]] = []
        response_ids = set(state.response_contexts) | set(state.active_responses)
        cancelled_tool_tasks = list(state.frontend_tool_tasks)
        tool_response_ids = (
            response_ids
            | set(state.frontend_tool_tasks.values())
            | set(state.tool_calls_pending)
            | set(state.tool_response_done)
            | set(state.tool_contexts)
        )
        state.tool_followup_suppressed.update(tool_response_ids)
        for task in cancelled_tool_tasks:
            task.cancel()
        if cancelled_tool_tasks:
            await asyncio.gather(*cancelled_tool_tasks, return_exceptions=True)
            for task in cancelled_tool_tasks:
                state.frontend_tool_tasks.pop(task, None)
                state.tool_tasks.discard(task)
        for response_id in response_ids:
            state.tool_followup_suppressed.add(response_id)
            context = state.response_contexts.get(response_id, state.turns.current())
            if response_id not in state.interrupted_responses:
                state.interrupted_responses.add(response_id)
                events.append(
                    {
                        "type": GatewayServerEvent.RESPONSE_INTERRUPTED.value,
                        "responseId": response_id,
                        "turnId": context.turn_id,
                        "reason": reason,
                        **self._response_public_context(state, response_id),
                    }
                )
            completion = state.response_completions.get(response_id)
            if completion is not None and not completion.done():
                completion.set_result(False)
            task_ids = state.response_task_ids.get(response_id, ())
            if (
                task_ids
                and response_id not in state.playback_started
                and state.delivery is not None
            ):
                await state.delivery.playback_failed(task_ids)
            state.announcement_window.finish_playback(
                response_id,
                has_function_call=response_id in state.response_has_function_call,
            )
            state.active_responses.discard(response_id)
            self._forget_response(state, response_id)
        for response_id in tool_response_ids:
            state.tool_calls_pending.pop(response_id, None)
            state.tool_response_done.discard(response_id)
            state.tool_followup_created.discard(response_id)
            state.tool_followup_suppressed.discard(response_id)
            state.tool_contexts.pop(response_id, None)
        state.announcement_window.reset()
        while len(state.interrupted_responses) > 256:
            state.interrupted_responses.pop()
        if requires_safe_reset and not state.closed:
            if not state.provider_recovering:
                await state.provider.reset_connection()
            state.pending_response_contexts.clear()
            if state.provider_recovering and not state.closed:
                await state.provider_recovered.wait()
        return events

    async def _request_response(
        self,
        state: _RealtimeConnectionState,
        pending: _PendingResponse,
        response: Mapping[str, Any],
    ) -> None:
        body = dict(response)
        capabilities = dict(state.provider.provider.capabilities())
        if capabilities.get("responseMetadataCorrelation"):
            pending.correlation_id = pending.correlation_id or uuid4().hex
            metadata = body.get("metadata")
            body["metadata"] = {
                **(dict(metadata) if isinstance(metadata, Mapping) else {}),
                "qwen_audio_request_id": pending.correlation_id,
            }
        state.pending_response_contexts.append(pending)
        pending.provider_epoch = state.provider_epoch
        try:
            await state.provider.create_response(body)
        except Exception:
            try:
                state.pending_response_contexts.remove(pending)
            except ValueError:
                pass
            if pending.completion is not None and not pending.completion.done():
                pending.completion.set_result(False)
            raise

    @staticmethod
    def _take_pending_response(
        state: _RealtimeConnectionState,
        event: Mapping[str, Any],
    ) -> _PendingResponse | None:
        while (
            state.pending_response_contexts
            and state.pending_response_contexts[0].provider_epoch != state.provider_epoch
        ):
            stale = state.pending_response_contexts.popleft()
            if stale.completion is not None and not stale.completion.done():
                stale.completion.set_result(False)
        correlation_id = _response_correlation_id(event)
        if correlation_id:
            for pending in state.pending_response_contexts:
                if pending.correlation_id == correlation_id:
                    state.pending_response_contexts.remove(pending)
                    return pending
        if state.pending_response_contexts:
            return state.pending_response_contexts.popleft()
        return None

    @staticmethod
    def _response_public_context(
        state: _RealtimeConnectionState,
        response_id: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"origin": state.response_origins.get(response_id, "model")}
        task_ids = state.response_task_ids.get(response_id, ())
        if task_ids:
            result["taskIds"] = list(task_ids)
            if len(task_ids) == 1:
                result["taskId"] = task_ids[0]
        authorization_id = state.response_authorizations.get(response_id)
        if authorization_id:
            result["authorizationId"] = authorization_id
        return result

    @staticmethod
    def _forget_response(state: _RealtimeConnectionState, response_id: str) -> None:
        state.response_contexts.pop(response_id, None)
        state.response_text.pop(response_id, None)
        state.response_origins.pop(response_id, None)
        state.response_task_ids.pop(response_id, None)
        state.response_authorizations.pop(response_id, None)
        state.response_completions.pop(response_id, None)
        state.response_has_audio.discard(response_id)
        state.response_has_function_call.discard(response_id)
        state.playback_started.discard(response_id)

    def _delivery_blocked(self, state: _RealtimeConnectionState) -> bool:
        return bool(
            state.closed
            or state.provider_recovering
            or state.sleeping
            or not state.output_enabled
            or not state.provider.ready
            or (not state.text_only and not state.is_voice_active())
            or state.announcement_window.blocked
            or state.active_responses
            or any(
                not pending.cancelled and pending.origin != "announcement"
                for pending in state.pending_response_contexts
            )
        )

    async def _present_result(
        self,
        state: _RealtimeConnectionState,
        text: str,
        delivery_context: Mapping[str, Any],
        inject_context: bool,
    ) -> PresentationOutcome:
        injected = False
        completion: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        pending: _PendingResponse | None = None
        async with state.presentation_lock:
            if self._delivery_blocked(state):
                return PresentationOutcome(completed=False, context_injected=False)
            turn_id = str(delivery_context.get("turnId") or f"gateway_{uuid4().hex}")
            task_ids = tuple(
                str(value) for value in (delivery_context.get("taskIds") or []) if str(value)
            )
            context = TurnContext(
                turn_id=turn_id,
                turn_generation=max(
                    state.turns.current().turn_generation,
                    state.turns.committed().turn_generation,
                ),
            )
            if inject_context:
                try:
                    await state.provider.send_input({"text": text})
                    injected = True
                except Exception:
                    return PresentationOutcome(completed=False, context_injected=False)
            pending = _PendingResponse(
                context=context,
                origin="announcement",
                task_ids=task_ids,
                completion=completion,
            )
            try:
                await self._request_response(
                    state,
                    pending,
                    {
                        "modalities": ["text"] if state.text_only else ["text", "audio"],
                        "tool_choice": "none",
                        "instructions": (
                            "这是先前工作的最终结果, 不是用户的新请求。"
                            "直接自然说明实际结果或阻塞, 不要提后台 Agent、协议、Task 或 ID; "
                            "不要调用工具, 也不要添加结果中没有的事实。"
                        ),
                    },
                )
            except Exception:
                return PresentationOutcome(completed=False, context_injected=injected)
        try:
            async with asyncio.timeout(self.presentation_timeout_seconds):
                completed = await asyncio.shield(completion)
            return PresentationOutcome(
                completed=completed,
                context_injected=injected,
            )
        except TimeoutError:
            if pending is not None:
                try:
                    state.pending_response_contexts.remove(pending)
                except ValueError:
                    pass
                for response_id, future in tuple(state.response_completions.items()):
                    if future is completion:
                        state.interrupted_responses.add(response_id)
                        self._forget_response(state, response_id)
            try:
                await state.provider.cancel_response()
            except Exception:
                pass
            return PresentationOutcome(completed=False, context_injected=injected)
        except Exception:
            return PresentationOutcome(completed=False, context_injected=injected)

    def _task_event(
        self,
        state: _RealtimeConnectionState,
        event: Mapping[str, Any],
    ) -> None:
        if state.closed:
            return
        task = event.get("task")
        if not isinstance(task, Mapping):
            return
        if not _task_event_belongs_to_owner(event, task, state.owner_id):
            return
        task_session = str(task.get("sessionId", task.get("session_id")) or "main")
        event_type = str(event.get("type") or "")
        authorization = task.get("authorization")
        if (
            event_type == "task.permission.requested"
            and task_session == state.session_id
            and isinstance(authorization, Mapping)
            and authorization.get("status") == "pending"
        ):
            self._queue_permission(state, task, authorization)
        elif event_type == "task.permission.resolved":
            resolved = event.get("permission")
            authorization_id = str(
                (
                    resolved.get("id")
                    if isinstance(resolved, Mapping)
                    else authorization.get("id")
                    if isinstance(authorization, Mapping)
                    else ""
                )
                or ""
            )
            if authorization_id:
                was_announced = (
                    authorization_id in state.announced_permissions
                    or any(
                        pending.authorization_id == authorization_id
                        for pending in state.pending_response_contexts
                    )
                    or any(
                        value == authorization_id
                        for value in state.response_authorizations.values()
                    )
                )
                state.pending_permissions.pop(authorization_id, None)
                state.announced_permissions.discard(authorization_id)
                self._spawn(
                    state,
                    self._resolve_permission(
                        state,
                        authorization_id,
                        inject_context=was_announced,
                    ),
                    name=f"permission-resolved:{authorization_id}",
                )
        presentation = (
            task.get("delegation", {}).get("presentation")
            if event_type == "task.delegated" and isinstance(task.get("delegation"), Mapping)
            else task.get("presentation")
        )
        inline = presentation.get("inline") if isinstance(presentation, Mapping) else None
        if (
            event_type in {"task.delegated", "task.completed", "task.failed"}
            and task_session == state.session_id
            and isinstance(inline, Mapping)
            and inline.get("content")
        ):
            suffix = "_delegated" if event_type == "task.delegated" else ""
            state.emit(
                {
                    "type": GatewayServerEvent.TIMELINE_INLINE.value,
                    "item": {
                        "id": f"inline_{task.get('id')}{suffix}",
                        "taskId": task.get("id"),
                        **dict(inline),
                    },
                }
            )
        if (
            state.delivery is not None
            and task.get("notificationStatus") == "pending"
            and event_type in {"task.completed", "task.failed"}
        ):
            self._queue_notification(state, task)
        if state.delivery is not None and event_type == "task.notification.delivered":
            task_id = str(task.get("id") or "")
            if task_id:
                self._spawn(
                    state,
                    state.delivery.discard_delivered([task_id]),
                    name=f"notification-delivered:{task_id}",
                )

    def _queue_notification(
        self,
        state: _RealtimeConnectionState,
        task: Mapping[str, Any],
    ) -> None:
        if state.delivery is None:
            return
        status = str(task.get("status") or "")
        if status == "completed":
            state.delivery.completed(task)
        elif status == "failed":
            state.delivery.failed(task)
        else:
            return
        result = str(task.get("result") or task.get("error") or "").strip()
        if result:
            self._record(
                state,
                {
                    "id": f"task:assistant:{task.get('id')}",
                    "role": "assistant",
                    "content": result,
                    "source": "task-result",
                    "taskId": task.get("id"),
                    "turnId": task.get("turnId"),
                },
            )

    def _queue_pending_notifications(self, state: _RealtimeConnectionState) -> None:
        method = getattr(self.task_source, "list", None)
        if not callable(method) or state.delivery is None:
            return
        try:
            result = method(owner_id=state.owner_id)
        except TypeError:
            result = method()
        if inspect.isawaitable(result):

            async def consume() -> None:
                values = await result
                for task in values or []:
                    if isinstance(task, Mapping):
                        self._queue_restored_task(state, task)

            task = asyncio.create_task(
                consume(),
                name=f"notification-backfill:{state.connection_id}",
            )
            state.tool_tasks.add(task)
            task.add_done_callback(state.tool_tasks.discard)
            return
        for task in result or []:
            if isinstance(task, Mapping):
                self._queue_restored_task(state, task)

    def _queue_restored_task(
        self,
        state: _RealtimeConnectionState,
        task: Mapping[str, Any],
    ) -> None:
        if _owner_id(task) != state.owner_id:
            return
        if task.get("notificationStatus") in {"pending", "delivering"}:
            self._queue_notification(state, task)
        authorization = task.get("authorization")
        if (
            str(task.get("sessionId", task.get("session_id")) or "main") == state.session_id
            and isinstance(authorization, Mapping)
            and authorization.get("status") == "pending"
        ):
            self._queue_permission(state, task, authorization)

    def _queue_permission(
        self,
        state: _RealtimeConnectionState,
        task: Mapping[str, Any],
        authorization: Mapping[str, Any],
    ) -> None:
        authorization_id = str(authorization.get("id") or "").strip()
        if not authorization_id or authorization_id in state.announced_permissions:
            return
        state.pending_permissions[authorization_id] = {
            "taskId": str(task.get("id") or ""),
            "authorization": dict(authorization),
        }
        if state.sleeping:
            state.sleeping = False
            state.output_enabled = True
            state.emit(
                {
                    "type": GatewayServerEvent.VOICE_SLEEP.value,
                    "state": "detected",
                }
            )
            self._ensure_connect_task(state)
        self._flush_permissions(state)

    def _flush_permissions(self, state: _RealtimeConnectionState) -> None:
        if (
            state.closed
            or state.announcing_permission
            or not state.pending_permissions
            or self._delivery_blocked(state)
        ):
            return
        authorization_id = next(iter(state.pending_permissions))
        state.announcing_permission = authorization_id
        state.permission_task = self._spawn(
            state,
            self._announce_permission(state, authorization_id),
            name=f"permission-announcement:{authorization_id}",
        )

    async def _announce_permission(
        self,
        state: _RealtimeConnectionState,
        authorization_id: str,
    ) -> None:
        completion: asyncio.Future[bool] | None = None
        advance_queue = False
        try:
            async with state.presentation_lock:
                queued = state.pending_permissions.get(authorization_id)
                if (
                    queued is None
                    or any(not pending.cancelled for pending in state.pending_response_contexts)
                    or self._delivery_blocked(state)
                ):
                    return
                authorization = queued["authorization"]
                if not isinstance(authorization, Mapping):
                    state.pending_permissions.pop(authorization_id, None)
                    return
                summary = str(authorization.get("summary") or "").strip()[:600]
                category = str(authorization.get("category") or "").strip()[:80]
                operation = authorization.get("operation")
                operation_text = ""
                if isinstance(operation, Mapping):
                    operation_text = "\n".join(
                        f"{label}: {str(operation.get(key) or '').strip()[:600]}"
                        for label, key in (
                            ("操作", "title"),
                            ("说明", "description"),
                            ("命令", "command"),
                            ("路径", "path"),
                        )
                        if str(operation.get(key) or "").strip()
                    )
                permission_text = "\n".join(
                    value
                    for value in (
                        "<backend_permission_request>",
                        f"authorization_id: {authorization_id}",
                        f"类别: {category}" if category else "",
                        f"摘要: {summary}" if summary else "",
                        operation_text,
                        "</backend_permission_request>",
                    )
                    if value
                )[:2_400]
                completion = asyncio.get_running_loop().create_future()
                pending = _PendingResponse(
                    context=TurnContext(
                        turn_id=f"gateway_{uuid4().hex}",
                        turn_generation=max(
                            state.turns.current().turn_generation,
                            state.turns.committed().turn_generation,
                        ),
                    ),
                    origin="permission",
                    task_ids=(str(queued.get("taskId") or ""),),
                    authorization_id=authorization_id,
                    completion=completion,
                )
                await state.provider.send_input({"text": permission_text})
                await self._request_response(
                    state,
                    pending,
                    {
                        "modalities": ["text"] if state.text_only else ["text", "audio"],
                        "tool_choice": "none",
                        "instructions": (
                            "这是后台 Agent 的权限请求。自然、简短地说明操作，"
                            "并询问用户是否同意授权。不要规定回答方式，不要调用工具，"
                            "也不要读出内部字段或 ID。"
                        ),
                    },
                )
            async with asyncio.timeout(self.presentation_timeout_seconds):
                completed = await asyncio.shield(completion)
            if completed and authorization_id in state.pending_permissions:
                state.pending_permissions.pop(authorization_id, None)
                state.announced_permissions.add(authorization_id)
                advance_queue = True
        except (TimeoutError, asyncio.CancelledError):
            if state.closed:
                return
        except Exception as error:
            if not state.closed:
                state.emit(
                    {
                        "type": GatewayServerEvent.ERROR.value,
                        "message": f"暂时无法询问权限: {error}",
                    }
                )
        finally:
            if state.announcing_permission == authorization_id:
                state.announcing_permission = ""
            if state.permission_task is asyncio.current_task():
                state.permission_task = None
            if advance_queue and not state.closed:
                self._flush_permissions(state)

    async def _resolve_permission(
        self,
        state: _RealtimeConnectionState,
        authorization_id: str,
        *,
        inject_context: bool,
    ) -> None:
        async with state.presentation_lock:
            pending_matches = [
                pending
                for pending in tuple(state.pending_response_contexts)
                if pending.authorization_id == authorization_id and not pending.cancelled
            ]
            for pending in pending_matches:
                pending.cancelled = True
                if pending.completion is not None and not pending.completion.done():
                    pending.completion.set_result(False)
            matching = [
                response_id
                for response_id, current_id in state.response_authorizations.items()
                if current_id == authorization_id
            ]
            should_inject = inject_context or bool(pending_matches or matching)
            if pending_matches or any(
                response_id in state.active_responses for response_id in matching
            ):
                try:
                    await state.provider.cancel_response()
                except Exception:
                    pass
            if pending_matches or matching:
                state.emit(
                    {
                        "type": GatewayServerEvent.PLAYBACK_CLEAR.value,
                        "reason": "permission_resolved",
                    }
                )
            for response_id in matching:
                state.active_responses.discard(response_id)
                state.interrupted_responses.add(response_id)
                completion = state.response_completions.get(response_id)
                if completion is not None and not completion.done():
                    completion.set_result(False)
                state.announcement_window.finish_playback(response_id)
                self._forget_response(state, response_id)
            if should_inject and state.provider.ready:
                try:
                    await state.provider.send_input(
                        {
                            "text": (
                                "(系统提示: 刚才的后台权限请求已处理完毕，无需再询问或回应该请求。)"
                            )
                        }
                    )
                except Exception:
                    pass
        self._flush_permissions(state)
        if state.delivery is not None:
            await state.delivery.flush()

    @staticmethod
    def _spawn(
        state: _RealtimeConnectionState,
        coroutine: Awaitable[Any],
        *,
        name: str,
    ) -> asyncio.Task[Any]:
        task = asyncio.create_task(coroutine, name=name)
        state.tool_tasks.add(task)
        task.add_done_callback(state.tool_tasks.discard)
        return task

    def _spawn_tool_call(
        self,
        state: _RealtimeConnectionState,
        event: Mapping[str, Any],
        context: TurnContext,
        response_id: str,
    ) -> None:
        if response_id in state.interrupted_responses or state.turns.is_stale(context):
            return
        state.tool_calls_pending[response_id] = state.tool_calls_pending.get(response_id, 0) + 1
        state.tool_contexts[response_id] = context
        task = asyncio.create_task(
            self._run_tool_call(state, event, context, response_id),
            name=f"frontend-tool:{state.connection_id}",
        )
        state.tool_tasks.add(task)
        state.frontend_tool_tasks[task] = response_id

        def cleanup(completed: asyncio.Task[Any]) -> None:
            state.tool_tasks.discard(completed)
            state.frontend_tool_tasks.pop(completed, None)

        task.add_done_callback(cleanup)

    async def _reject_non_model_tool_call(
        self,
        state: _RealtimeConnectionState,
        event: Mapping[str, Any],
        context: TurnContext,
        response_id: str,
        origin: str,
    ) -> None:
        call_id, _name = _tool_identity(event)
        try:
            await state.provider.send_tool_result(
                {
                    "callId": call_id,
                    "output": {
                        "status": "failed",
                        "error": True,
                        "error_code": "tool_origin_not_allowed",
                        "user_message": "非用户模型回合不能调用前台工具。",
                    },
                    "context": {
                        "turnId": context.turn_id,
                        "turnGeneration": context.turn_generation,
                        "responseId": response_id,
                        "origin": origin,
                    },
                }
            )
        except Exception as error:
            if not state.closed:
                state.emit(
                    {
                        "type": GatewayServerEvent.ERROR.value,
                        "message": f"前台工具拒绝结果发送失败: {error}",
                    }
                )

    async def _run_tool_call(
        self,
        state: _RealtimeConnectionState,
        event: Mapping[str, Any],
        context: TurnContext,
        response_id: str,
    ) -> None:
        try:
            async with state.frontend_tool_lock:
                if (
                    state.closed
                    or response_id in state.interrupted_responses
                    or state.turns.is_stale(context)
                ):
                    return
                await self._run_tool_call_serialized(state, event, context, response_id)
        finally:
            remaining = max(0, state.tool_calls_pending.get(response_id, 1) - 1)
            if remaining:
                state.tool_calls_pending[response_id] = remaining
            else:
                state.tool_calls_pending.pop(response_id, None)
            await self._maybe_create_tool_followup(state, response_id)

    async def _run_tool_call_serialized(
        self,
        state: _RealtimeConnectionState,
        event: Mapping[str, Any],
        context: TurnContext,
        response_id: str,
    ) -> None:
        call_id, name = _tool_identity(event)
        try:
            arguments: Any = event.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            if not isinstance(arguments, Mapping):
                arguments = {}
            execution = FrontendExecutionContext(
                owner_id=state.owner_id,
                session_id=state.session_id,
                turn_id=(
                    context.turn_id or state.turns.current().turn_id or f"turn-{time.time_ns()}"
                ),
                turn_generation=max(0, context.turn_generation),
                frontend_capabilities=state.frontend_capabilities,
                client_states=state.client_states,
                input_parts=state.input_parts,
                client_context=state.client_context,
            )
            output = await self.frontend_graph.ainvoke(
                FrontendToolCall(call_id=call_id, name=name, arguments=arguments),
                execution,
            )
            result = output.get("result") or {
                "status": "failed",
                "error": True,
                "error_code": "tool_execution_failed",
                "user_message": "前台工具没有返回结果。",
            }
        except Exception:
            result = {
                "status": "failed",
                "error": True,
                "error_code": "tool_execution_failed",
                "user_message": "前台工具暂时无法完成这次操作。",
            }
        try:
            await state.provider.send_tool_result(
                {
                    "callId": call_id,
                    "output": result,
                    "context": {
                        "turnId": context.turn_id,
                        "turnGeneration": context.turn_generation,
                        "responseId": response_id,
                        "origin": "model",
                    },
                }
            )
            if name == "enter_sleep" and result.get("status") == "sleeping":
                state.tool_followup_suppressed.add(response_id)
                state.sleeping = True
                state.input_enabled = False
                state.output_enabled = False
                state.provider.clear_pending_audio()
                state.announcement_window.reset()
                if state.delivery is not None:
                    await state.delivery.pause()
                state.emit(
                    {
                        "type": GatewayServerEvent.CLIENT_STATE.value,
                        "state": "sleeping",
                    }
                )
                state.emit(
                    {
                        "type": GatewayServerEvent.VOICE_SLEEP.value,
                        "state": "sleeping",
                    }
                )
                return
        except Exception as error:
            state.tool_followup_suppressed.add(response_id)
            if not state.closed:
                state.emit(
                    {
                        "type": GatewayServerEvent.ERROR.value,
                        "message": f"前台工具结果发送失败: {error}",
                    }
                )

    async def _maybe_create_tool_followup(
        self,
        state: _RealtimeConnectionState,
        response_id: str,
    ) -> None:
        if (
            not response_id
            or state.closed
            or state.tool_calls_pending.get(response_id, 0) > 0
            or response_id not in state.tool_response_done
            or response_id in state.tool_followup_created
            or response_id in state.tool_followup_suppressed
            or response_id in state.interrupted_responses
        ):
            return
        context = state.tool_contexts.get(response_id)
        if context is None or state.turns.is_stale(context):
            state.tool_followup_suppressed.add(response_id)
            return
        state.tool_followup_created.add(response_id)
        try:
            await self._request_response(
                state,
                _PendingResponse(context=context),
                {"modalities": ["text"] if state.text_only else ["text", "audio"]},
            )
        except Exception:
            state.tool_followup_created.discard(response_id)
            raise
        finally:
            state.tool_response_done.discard(response_id)
            state.tool_contexts.pop(response_id, None)
            while len(state.tool_followup_created) > 256:
                state.tool_followup_created.pop()
            while len(state.tool_followup_suppressed) > 256:
                state.tool_followup_suppressed.pop()

    @staticmethod
    def _provider_input(parts: tuple[dict[str, Any], ...]) -> str:
        text = _display_input(parts)
        files = [part for part in parts if part.get("type") == "file"]
        if not files:
            return text
        metadata = [
            {
                "type": "file",
                "filename": part.get("filename"),
                "mime": part.get("mime"),
                "reference": (
                    part.get("source", {}).get("text", {}).get("value")
                    if isinstance(part.get("source"), Mapping)
                    and isinstance(part.get("source", {}).get("text"), Mapping)
                    else None
                ),
            }
            for part in files
        ]
        return "\n\n".join(
            (
                text or "用户提交了附件, 但没有附带文字说明。请询问希望如何处理。",
                "<input_parts>",
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                "</input_parts>",
            )
        )

    def _record(
        self,
        state: _RealtimeConnectionState,
        message: Mapping[str, Any],
    ) -> None:
        if self.conversation_store is None or not str(message.get("content") or "").strip():
            return
        try:
            self.conversation_store.record(
                owner_id=state.owner_id,
                session_id=state.session_id,
                message=message,
            )
        except (OSError, ValueError):
            return
