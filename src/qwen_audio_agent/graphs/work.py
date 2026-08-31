"""Application facade around the checkpointed Work StateGraph."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import suppress
from copy import deepcopy
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, StateSnapshot

from qwen_audio_agent.backend.port import (
    BackendEventType,
    BackendPort,
    assert_backend_port,
)
from qwen_audio_agent.domain.task import (
    Task,
    TaskScope,
    TaskStatus,
    is_task_cancellable,
    is_task_terminal,
    now_ms,
    public_task,
    transition_task,
)
from qwen_audio_agent.graphs.work_graph import (
    _BACKEND_EVENT_OVERFLOW_ERROR,
    _BACKEND_EVENT_TYPES,
    _BACKEND_OVERFLOW_CANCEL_TIMEOUT_SECONDS,
    _COALESCED_EVENT_MARKER,
    _MAX_BACKEND_EVENT_QUEUE,
    _MAX_BUFFERED_BACKEND_EVENT_BYTES,
    _QUEUED_SIGNAL_TYPES,
    TaskListener,
    WorkState,
    _authorization_resolution_status,
    _backend_event_coalesce_key,
    _backend_event_size,
    _backend_recovery_task,
    _bounded_text,
    _cancel_confirmation_error,
    _maybe_await,
    _normalize_backend_outcome,
    _normalize_backend_signal,
    _WorkHandle,
    build_work_graph,
    reduce_activity,
    reduce_artifacts,
    reduce_processed_event_ids,
    reduce_work_events,
)
from qwen_audio_agent.runtime.delivery import DeliveryRecord
from qwen_audio_agent.runtime.repository import TaskRepository
from qwen_audio_agent.runtime.scheduler import TaskScheduler


class WorkGraph:
    """Application facade around the checkpointed Work StateGraph."""

    def __init__(
        self,
        *,
        backend: BackendPort,
        repository: TaskRepository[Task] | None = None,
        scheduler: TaskScheduler | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
    ) -> None:
        self.backend = assert_backend_port(backend, name="WorkGraph backend")
        self.repository = repository if repository is not None else TaskRepository()
        self.scheduler = scheduler if scheduler is not None else TaskScheduler()
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.graph = build_work_graph(checkpointer=self.checkpointer)
        self._handles: dict[str, _WorkHandle] = {}
        self._scheduled_timers: dict[str, asyncio.Task[None]] = {}
        self._listeners: set[TaskListener] = set()
        self._start_lock = asyncio.Lock()
        self._started = False
        self._closed = False

    @staticmethod
    def config(task_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": f"work:{task_id}"}}

    def subscribe(self, listener: TaskListener) -> Callable[[], None]:
        if not callable(listener):
            raise TypeError("WorkGraph subscriber must be callable")
        self._listeners.add(listener)
        return lambda: self._listeners.discard(listener)

    def _notify(self, event: Mapping[str, Any]) -> None:
        for listener in tuple(self._listeners):
            with suppress(Exception):
                listener(event)

    def notification_delivered(self, record: DeliveryRecord) -> None:
        """Publish a durable playback acknowledgement from the lease adapter."""

        task = self.repository.get(record.task_id)
        if task is None:
            return
        self._notify(
            {
                "type": "task.notification.delivered",
                "ownerId": task.owner_id,
                "task": public_task(task),
            }
        )

    def get(self, task_id: str, *, owner_id: str | None = None) -> dict[str, Any] | None:
        task = self.repository.get(str(task_id))
        if task is None or (owner_id is not None and task.owner_id != str(owner_id)):
            return None
        return public_task(task)

    def list(self, *, owner_id: str | None = None) -> list[dict[str, Any]]:
        tasks = [
            task
            for task in self.repository.values()
            if task.scope == TaskScope.USER and (owner_id is None or task.owner_id == str(owner_id))
        ]
        tasks.sort(key=lambda task: task.created_at)
        return [public_task(task) for task in tasks]

    def _initial_state(
        self,
        task: Task,
        *,
        initial_event_type: str = "task.accepted",
    ) -> WorkState:
        return {
            "task_id": task.id,
            "owner_id": task.owner_id,
            "session_id": task.session_id,
            "turn_id": task.turn_id,
            "scope": task.scope.value,
            "kind": task.kind,
            "objective": task.objective,
            "submission_key": task.submission_key,
            "lane_key": task.lane_key,
            "lane_limit": task.lane_limit,
            "input_parts": deepcopy(task.input_parts),
            "input_refs": list(task.input_refs),
            "status": task.status.value,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "elapsed_ms": task.elapsed_ms,
            "result": task.result,
            "error": task.error,
            "message": task.message,
            "presentation": task.presentation,
            "activity": [],
            "artifacts": [],
            "delegation": task.delegation,
            "cancellation": task.cancellation,
            "authorization": task.authorization,
            "notification_status": task.notification_status,
            "initial_event_type": initial_event_type,
            "backend_started": False,
            "pending_signal": None,
            "processed_event_ids": [],
            "events": [],
        }

    async def start(self) -> None:
        """Start the adapter and reconcile durable non-terminal Tasks once."""

        async with self._start_lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("WorkGraph is closed")
            await _maybe_await(self.backend.start({"runtime": "qwen-audio-agent"}))
            if self.repository.store is not None:
                self.repository.load()

            changed = False
            for task in self.repository.values():
                if is_task_terminal(task.status):
                    continue
                if task.status == TaskStatus.SCHEDULED:
                    handle = _WorkHandle(task_id=task.id)
                    self._handles[task.id] = handle
                    self._arm_scheduled(task)
                    continue
                if task.status == TaskStatus.CANCELLING:
                    try:
                        confirmation = await _maybe_await(
                            self.backend.cancel(task.id, {"owner_id": task.owner_id})
                        )
                    except Exception as error:
                        transition_task(task, TaskStatus.FAILED)
                        message = (
                            _bounded_text(error, 3_900) or "backend did not confirm cancellation"
                        )
                        task.error = f"取消失败: {message}"[:4_000]
                        task.notification_status = (
                            "pending" if task.scope == TaskScope.USER else "none"
                        )
                        event_type = "task.failed"
                    else:
                        confirmation_error = _cancel_confirmation_error(confirmation)
                        if confirmation_error is not None:
                            transition_task(task, TaskStatus.FAILED)
                            task.error = f"取消失败: {confirmation_error}"[:4_000]
                            task.notification_status = (
                                "pending" if task.scope == TaskScope.USER else "none"
                            )
                            event_type = "task.failed"
                        else:
                            transition_task(task, TaskStatus.CANCELLED)
                            task.result = None
                            task.error = None
                            task.cancellation = deepcopy(dict(confirmation))
                            task.notification_status = "none"
                            event_type = "task.cancelled"
                    timestamp = now_ms()
                    task.completed_at = timestamp
                    task.elapsed_ms = max(
                        0,
                        timestamp - int(task.started_at or timestamp),
                    )
                    task.authorization = None
                    self.repository.set(task.id, task)
                    self._notify(
                        {
                            "type": event_type,
                            "ownerId": task.owner_id,
                            "task": public_task(task),
                        }
                    )
                    changed = True
                    continue
                if task.status in {TaskStatus.DELEGATED, TaskStatus.FINALIZING} and (
                    await self._reattach_recoverable_task(task)
                ):
                    continue
                timestamp = now_ms()
                transition_task(task, TaskStatus.FAILED)
                task.completed_at = timestamp
                task.elapsed_ms = max(0, timestamp - int(task.started_at or timestamp))
                task.authorization = None
                task.error = "Gateway restarted before this work could be safely resumed"
                task.notification_status = "pending" if task.scope == TaskScope.USER else "none"
                self.repository.set(task.id, task)
                self._notify(
                    {
                        "type": "task.failed",
                        "ownerId": task.owner_id,
                        "task": public_task(task),
                    }
                )
                changed = True
            if changed:
                self.repository.save()
            self._started = True

    async def _reattach_recoverable_task(self, task: Task) -> bool:
        can_recover = getattr(self.backend, "can_recover_delegated_work", None)
        recover = getattr(self.backend, "recover_delegated_work", None)
        if not callable(can_recover) or not callable(recover):
            return False
        if not await _maybe_await(can_recover(_backend_recovery_task(task))):
            return False
        snapshot = await self.graph.aget_state(self.config(task.id), subgraphs=True)
        if not snapshot.values and not snapshot.tasks:
            return False
        handle = _WorkHandle(task_id=task.id)
        handle.ready.set()
        self._handles[task.id] = handle
        handle.event_pump_task = asyncio.create_task(
            self._event_pump(task.id, handle),
            name=f"work-events:{task.id}",
        )
        handle.backend_task = asyncio.create_task(
            self._run_backend_recovery(task.id, handle, recover),
            name=f"backend-recover:{task.id}",
        )
        return True

    async def submit(
        self,
        *,
        objective: str,
        owner_id: str,
        session_id: str = "main",
        turn_id: str | None = None,
        submission_key: str | None = None,
        lane_key: str | None = None,
        lane_limit: int = 1,
        input_parts: list[dict[str, Any]] | None = None,
        input_refs: list[str] | None = None,
        kind: str = "work",
    ) -> dict[str, Any]:
        await self.start()
        if self._closed:
            raise RuntimeError("WorkGraph is closed")
        normalized_objective = str(objective or "").strip()
        if not normalized_objective:
            raise ValueError("Work objective must not be empty")
        task = Task(
            id=self.repository.allocate_task_id(),
            status=TaskStatus.QUEUED,
            objective=normalized_objective,
            owner_id=str(owner_id or ""),
            session_id=str(session_id or "main"),
            turn_id=turn_id,
            submission_key=str(submission_key).strip() if submission_key else None,
            lane_key=str(lane_key) if lane_key else f"backend:{owner_id}",
            lane_limit=max(1, int(lane_limit or 1)),
            input_parts=deepcopy(input_parts or []),
            input_refs=[str(value) for value in (input_refs or []) if str(value).strip()][:32],
            kind=str(kind or "work"),
        )
        added = self.repository.add(task)
        if added.reused:
            return {**public_task(added.record), "reused": True}
        self.repository.save()
        handle = _WorkHandle(task_id=task.id)
        self._handles[task.id] = handle
        handle.bootstrap_task = asyncio.create_task(
            self._bootstrap(task, handle),
            name=f"work-bootstrap:{task.id}",
        )
        return {**public_task(task), "reused": False}

    async def submit_scheduled(
        self,
        *,
        objective: str,
        owner_id: str,
        at: int | None = None,
        schedule: Mapping[str, Any] | None = None,
        type: str = "reminder",
        session_id: str = "main",
        turn_id: str | None = None,
        submission_key: str | None = None,
        input_parts: list[dict[str, Any]] | None = None,
        input_refs: list[str] | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        """Persist a scheduled reminder or backend Task and arm its wake-up.

        Recurrence is retained for wire compatibility with the JavaScript
        runtime.  Like that runtime's current scheduler, firing remains
        one-shot until recurrence expansion is implemented separately.
        """

        await self.start()
        if self._closed:
            raise RuntimeError("WorkGraph is closed")
        normalized_objective = str(objective or "").strip()
        if not normalized_objective:
            raise ValueError("Scheduled Work objective must not be empty")
        schedule_value = dict(schedule or {})
        scheduled_at = at if at is not None else schedule_value.get("at")
        try:
            scheduled_at = int(scheduled_at)
        except (TypeError, ValueError) as error:
            raise ValueError("Scheduled Work requires an at timestamp in milliseconds") from error
        recurrence = str(schedule_value.get("recurrence") or "once")
        if recurrence not in {"once", "daily", "weekly", "weekdays"}:
            raise ValueError("Unsupported scheduled Work recurrence")
        if type not in {"reminder", "task"}:
            raise ValueError("Scheduled Work type must be reminder or task")
        owner = str(owner_id or "")
        task = Task(
            id=self.repository.allocate_task_id(),
            status=TaskStatus.SCHEDULED,
            objective=normalized_objective,
            owner_id=owner,
            session_id=str(session_id or "main"),
            turn_id=turn_id,
            submission_key=str(submission_key).strip() if submission_key else None,
            lane_key=f"backend:{owner}" if type == "task" else None,
            input_parts=deepcopy(input_parts or []),
            input_refs=[str(value) for value in (input_refs or []) if str(value).strip()][:32],
            kind="scheduled_task" if type == "task" else "reminder",
            schedule={"type": "at", "at": scheduled_at, "recurrence": recurrence},
            timeout_ms=int(timeout_ms) if type == "task" and timeout_ms else None,
        )
        added = self.repository.add(task)
        if added.reused:
            return {**public_task(added.record), "reused": True}
        self.repository.save()
        handle = _WorkHandle(task_id=task.id)
        self._handles[task.id] = handle
        self._notify(
            {
                "type": "task.scheduled",
                "ownerId": task.owner_id,
                "task": public_task(task),
            }
        )
        self._arm_scheduled(task)
        return {**public_task(task), "reused": False}

    def _arm_scheduled(self, task: Task) -> None:
        previous = self._scheduled_timers.pop(task.id, None)
        if previous is not None:
            previous.cancel()
        scheduled_at = int((task.schedule or {}).get("at") or 0)
        delay = max(0.0, (scheduled_at - now_ms()) / 1_000)
        timer = asyncio.create_task(
            self._scheduled_timer(task.id, delay),
            name=f"work-scheduled:{task.id}",
        )
        self._scheduled_timers[task.id] = timer

    async def _scheduled_timer(self, task_id: str, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            if not self._closed:
                await self._fire_scheduled(task_id)
        except asyncio.CancelledError:
            return
        finally:
            current = asyncio.current_task()
            if self._scheduled_timers.get(task_id) is current:
                self._scheduled_timers.pop(task_id, None)

    async def _fire_scheduled(self, task_id: str) -> None:
        task = self.repository.get(task_id)
        if task is None or task.status != TaskStatus.SCHEDULED:
            return
        transition_task(task, TaskStatus.QUEUED)
        self.repository.set(task.id, task)
        self.repository.save()
        if task.kind == "reminder":
            self._notify(
                {
                    "type": "task.scheduled.fired",
                    "ownerId": task.owner_id,
                    "task": public_task(task),
                }
            )
            timestamp = now_ms()
            transition_task(task, TaskStatus.RUNNING)
            task.started_at = timestamp
            self.repository.set(task.id, task)
            self.repository.save()
            self._notify(
                {
                    "type": "task.running",
                    "ownerId": task.owner_id,
                    "task": public_task(task),
                }
            )
            transition_task(task, TaskStatus.COMPLETED)
            task.completed_at = now_ms()
            task.elapsed_ms = max(0, task.completed_at - timestamp)
            task.result = task.objective
            task.presentation = {
                "speech": task.objective,
                "inline": None,
            }
            task.notification_status = "pending"
            self.repository.set(task.id, task)
            self.repository.save()
            self._notify(
                {
                    "type": "task.completed",
                    "ownerId": task.owner_id,
                    "task": public_task(task),
                }
            )
            handle = self._handles.get(task.id)
            if handle is not None:
                handle.ready.set()
                handle.done.set()
            return

        handle = self._handles.get(task.id)
        if handle is None:
            handle = _WorkHandle(task_id=task.id)
            self._handles[task.id] = handle
        handle.bootstrap_task = asyncio.create_task(
            self._bootstrap(
                task,
                handle,
                initial_event_type="task.scheduled.fired",
            ),
            name=f"work-bootstrap:{task.id}",
        )

    async def _bootstrap(
        self,
        task: Task,
        handle: _WorkHandle,
        *,
        initial_event_type: str = "task.accepted",
    ) -> None:
        try:
            await self._drive(
                handle,
                self._initial_state(task, initial_event_type=initial_event_type),
            )
            handle.ready.set()
            acquired = await self.scheduler.acquire(task)
            if not acquired:
                await self._drive(handle, Command(resume={"type": "control.cancel"}))
                return
            handle.scheduler_held = True
            state = await self._drive(
                handle,
                Command(resume={"type": "scheduler.acquired"}),
            )
            if state.get("status") == TaskStatus.RUNNING:
                self._start_backend_supervisor(task.id, handle)
        except Exception as error:
            handle.ready.set()
            await self._fail_outside_graph(task.id, handle, error)

    def _start_backend_supervisor(self, task_id: str, handle: _WorkHandle) -> None:
        if handle.backend_task is not None:
            return
        handle.event_pump_task = asyncio.create_task(
            self._event_pump(task_id, handle),
            name=f"work-events:{task_id}",
        )
        handle.backend_task = asyncio.create_task(
            self._run_backend(task_id, handle),
            name=f"backend-submit:{task_id}",
        )

    @staticmethod
    def _discard_backend_event_buffers(handle: _WorkHandle) -> None:
        """Release every queued payload after a Work reaches a terminal state."""

        with handle.foreign_event_lock:
            handle.foreign_events.clear()
            handle.foreign_coalesced_events.clear()
            handle.foreign_queued_coalesced_keys.clear()
            handle.foreign_drain_scheduled = False
            handle.foreign_queue_overflowed = False
            handle.buffered_event_sizes.clear()
            handle.buffered_event_bytes = 0
        handle.coalesced_events.clear()
        handle.queued_coalesced_keys.clear()
        while True:
            try:
                handle.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _close_backend_event_ingress(self, handle: _WorkHandle) -> None:
        """Stop adapter callbacks without marking the Work itself done."""

        with handle.foreign_event_lock:
            handle.backend_ingress_closed = True
        self._discard_backend_event_buffers(handle)

    def _fail_backend_event_queue(self, handle: _WorkHandle) -> None:
        if handle.queue_overflowed:
            return
        handle.queue_overflowed = True
        with handle.foreign_event_lock:
            handle.foreign_queue_overflowed = True
            handle.foreign_events.clear()
            handle.foreign_coalesced_events.clear()
            handle.foreign_queued_coalesced_keys.clear()
            handle.buffered_event_sizes.clear()
            handle.buffered_event_bytes = 0
        handle.coalesced_events.clear()
        handle.queued_coalesced_keys.clear()
        while True:
            try:
                handle.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        handle.event_queue.put_nowait(
            {
                "type": "runtime.failed",
                "error": _BACKEND_EVENT_OVERFLOW_ERROR,
                "_event_id": f"{handle.task_id}:backend:queue-overflow",
            }
        )
        backend_task = handle.backend_task
        if backend_task is not None and not backend_task.done():
            backend_task.cancel()

        handle.overflow_cancel_task = asyncio.create_task(
            self._cancel_overflowed_backend(handle.task_id),
            name=f"backend-overflow-cancel:{handle.task_id}",
        )

    async def _cancel_overflowed_backend(self, task_id: str) -> None:
        task = self.repository.get(task_id)
        if task is None:
            return
        with suppress(Exception):
            async with asyncio.timeout(_BACKEND_OVERFLOW_CANCEL_TIMEOUT_SECONDS):
                await _maybe_await(self.backend.cancel(task_id, {"owner_id": task.owner_id}))

    @staticmethod
    def _next_backend_event_id(handle: _WorkHandle, prefix: str) -> str:
        with handle.foreign_event_lock:
            handle.backend_event_number += 1
            return f"{prefix}:{handle.backend_event_number}"

    @staticmethod
    def _reserve_buffered_event_locked(
        handle: _WorkHandle,
        event: Mapping[str, Any],
        *,
        replacing: Mapping[str, Any] | None = None,
        already_reserved: bool = False,
    ) -> bool:
        event_id = _bounded_text(event.get("_event_id"), 240)
        if not event_id:
            return False
        replaced_id = (
            _bounded_text(replacing.get("_event_id"), 240) if isinstance(replacing, Mapping) else ""
        )
        replaced_size = (
            handle.buffered_event_sizes.get(replaced_id, 0)
            if replaced_id and replaced_id != event_id
            else 0
        )
        if already_reserved:
            if event_id not in handle.buffered_event_sizes:
                return False
            if replaced_size:
                handle.buffered_event_sizes.pop(replaced_id, None)
                handle.buffered_event_bytes -= replaced_size
            return True
        size = _backend_event_size(event)
        existing_size = handle.buffered_event_sizes.get(event_id, 0)
        prospective = handle.buffered_event_bytes - replaced_size - existing_size + size
        if prospective > _MAX_BUFFERED_BACKEND_EVENT_BYTES:
            return False
        if replaced_size:
            handle.buffered_event_sizes.pop(replaced_id, None)
        handle.buffered_event_sizes[event_id] = size
        handle.buffered_event_bytes = prospective
        return True

    @classmethod
    def _reserve_buffered_event(
        cls,
        handle: _WorkHandle,
        event: Mapping[str, Any],
        *,
        replacing: Mapping[str, Any] | None = None,
        already_reserved: bool = False,
    ) -> bool:
        with handle.foreign_event_lock:
            return cls._reserve_buffered_event_locked(
                handle,
                event,
                replacing=replacing,
                already_reserved=already_reserved,
            )

    @staticmethod
    def _release_buffered_event(handle: _WorkHandle, event: Mapping[str, Any]) -> None:
        event_id = _bounded_text(event.get("_event_id"), 240)
        if not event_id:
            return
        with handle.foreign_event_lock:
            size = handle.buffered_event_sizes.pop(event_id, 0)
            handle.buffered_event_bytes = max(0, handle.buffered_event_bytes - size)

    def _enqueue_backend_event(
        self,
        handle: _WorkHandle,
        event: dict[str, Any],
        *,
        already_reserved: bool = False,
    ) -> None:
        if handle.queue_overflowed or handle.done.is_set():
            if already_reserved:
                self._release_buffered_event(handle, event)
            return
        event_type = str(event.get("type") or "")
        if (
            event_type in _BACKEND_EVENT_TYPES
            and handle.backend_ingress_closed
            and not already_reserved
        ):
            return
        if event_type not in _QUEUED_SIGNAL_TYPES:
            if already_reserved:
                self._release_buffered_event(handle, event)
            return
        key = _backend_event_coalesce_key(event, handle.coalesced_events)
        replacing = handle.coalesced_events.get(key) if key else None
        if not self._reserve_buffered_event(
            handle,
            event,
            replacing=replacing,
            already_reserved=already_reserved,
        ):
            self._fail_backend_event_queue(handle)
            return
        if not key:
            try:
                handle.event_queue.put_nowait(event)
            except asyncio.QueueFull:
                self._fail_backend_event_queue(handle)
            return
        handle.coalesced_events[key] = event
        if key in handle.queued_coalesced_keys:
            return
        handle.queued_coalesced_keys.add(key)
        try:
            handle.event_queue.put_nowait(
                {
                    "type": _COALESCED_EVENT_MARKER,
                    "_coalesce_key": key,
                }
            )
        except asyncio.QueueFull:
            self._fail_backend_event_queue(handle)

    def _enqueue_foreign_backend_event(
        self,
        loop: asyncio.AbstractEventLoop,
        handle: _WorkHandle,
        event: dict[str, Any],
    ) -> None:
        event_type = str(event.get("type") or "")
        if event_type not in _BACKEND_EVENT_TYPES:
            return
        schedule_drain = False
        with handle.foreign_event_lock:
            if (
                handle.queue_overflowed
                or handle.foreign_queue_overflowed
                or handle.backend_ingress_closed
                or handle.done.is_set()
            ):
                return
            key = _backend_event_coalesce_key(event, handle.foreign_coalesced_events)
            replacing = handle.foreign_coalesced_events.get(key) if key else None
            reserved = self._reserve_buffered_event_locked(
                handle,
                event,
                replacing=replacing,
            )
            if not reserved:
                handle.foreign_queue_overflowed = True
            elif key:
                handle.foreign_coalesced_events[key] = event
                if key not in handle.foreign_queued_coalesced_keys:
                    if len(handle.foreign_events) >= _MAX_BACKEND_EVENT_QUEUE:
                        handle.foreign_queue_overflowed = True
                    else:
                        handle.foreign_queued_coalesced_keys.add(key)
                        handle.foreign_events.append(
                            {
                                "type": _COALESCED_EVENT_MARKER,
                                "_coalesce_key": key,
                            }
                        )
            elif len(handle.foreign_events) >= _MAX_BACKEND_EVENT_QUEUE:
                handle.foreign_queue_overflowed = True
            elif reserved:
                handle.foreign_events.append(event)
            if not handle.foreign_drain_scheduled:
                handle.foreign_drain_scheduled = True
                schedule_drain = True
        if schedule_drain:
            try:
                loop.call_soon_threadsafe(self._drain_foreign_backend_events, handle)
            except RuntimeError:
                with handle.foreign_event_lock:
                    handle.foreign_drain_scheduled = False
                    handle.foreign_queue_overflowed = True
                    handle.foreign_events.clear()
                    handle.foreign_coalesced_events.clear()
                    handle.foreign_queued_coalesced_keys.clear()

    def _seal_backend_event_ingress(self, handle: _WorkHandle) -> bool:
        """Close callbacks atomically and order all earlier foreign events first."""

        events: list[dict[str, Any]] = []
        with handle.foreign_event_lock:
            if handle.backend_ingress_closed:
                return False
            handle.backend_ingress_closed = True
            queue_overflowed = handle.queue_overflowed
            foreign_overflowed = handle.foreign_queue_overflowed
            if not queue_overflowed and not foreign_overflowed:
                while handle.foreign_events:
                    event = handle.foreign_events.popleft()
                    if event.get("type") == _COALESCED_EVENT_MARKER:
                        key = str(event.get("_coalesce_key") or "")
                        event = handle.foreign_coalesced_events.pop(key, None)
                        handle.foreign_queued_coalesced_keys.discard(key)
                        if event is None:
                            continue
                    events.append(event)
            handle.foreign_events.clear()
            handle.foreign_coalesced_events.clear()
            handle.foreign_queued_coalesced_keys.clear()
            handle.foreign_drain_scheduled = False
            handle.foreign_queue_overflowed = False
        if queue_overflowed:
            return False
        if foreign_overflowed:
            self._fail_backend_event_queue(handle)
            return False
        for event in events:
            self._enqueue_backend_event(handle, event, already_reserved=True)
            if handle.queue_overflowed:
                return False
        return True

    def _drain_foreign_backend_events(self, handle: _WorkHandle) -> None:
        events: list[dict[str, Any]] = []
        with handle.foreign_event_lock:
            overflowed = handle.foreign_queue_overflowed
            if not overflowed:
                while handle.foreign_events:
                    event = handle.foreign_events.popleft()
                    if event.get("type") == _COALESCED_EVENT_MARKER:
                        key = str(event.get("_coalesce_key") or "")
                        event = handle.foreign_coalesced_events.pop(key, None)
                        handle.foreign_queued_coalesced_keys.discard(key)
                        if event is None:
                            continue
                    events.append(event)
            handle.foreign_events.clear()
            handle.foreign_coalesced_events.clear()
            handle.foreign_queued_coalesced_keys.clear()
            handle.foreign_drain_scheduled = False
        if overflowed:
            self._fail_backend_event_queue(handle)
            return
        for event in events:
            self._enqueue_backend_event(handle, event, already_reserved=True)

    async def _run_backend(self, task_id: str, handle: _WorkHandle) -> None:
        task = self.repository.get(task_id)
        if task is None:
            return
        loop = asyncio.get_running_loop()

        def on_event(event: Mapping[str, Any]) -> None:
            if (
                handle.queue_overflowed
                or handle.foreign_queue_overflowed
                or handle.backend_ingress_closed
                or handle.done.is_set()
            ):
                return
            normalized = _normalize_backend_signal(
                event,
                task_id=task_id,
                event_id=self._next_backend_event_id(handle, f"{task_id}:backend"),
            )
            if normalized is None:
                return
            try:
                same_loop = asyncio.get_running_loop() is loop
            except RuntimeError:
                same_loop = False
            if same_loop:
                self._enqueue_backend_event(handle, normalized)
            else:
                self._enqueue_foreign_backend_event(loop, handle, normalized)

        work = {
            "id": task.id,
            "ownerId": task.owner_id,
            "objective": task.objective,
            "instruction": task.objective,
            "inputParts": deepcopy(task.input_parts),
        }
        if task.input_refs:
            work["inputRefs"] = list(task.input_refs)
        try:
            outcome = await _maybe_await(self.backend.submit(work, {"on_event": on_event}))
            # Thread-safe callbacks queued before the terminal result must enter
            # the coalescer first so runtime.completed cannot overtake them.
            await asyncio.sleep(0)
            if not self._seal_backend_event_ingress(handle):
                return
            self._enqueue_backend_event(
                handle,
                {
                    "type": "runtime.completed",
                    "outcome": _normalize_backend_outcome(outcome),
                    "_event_id": self._next_backend_event_id(
                        handle,
                        f"{task_id}:backend",
                    ),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self._seal_backend_event_ingress(handle):
                return
            self._enqueue_backend_event(
                handle,
                {
                    "type": "runtime.failed",
                    "error": _bounded_text(error, 3_900),
                    "_event_id": self._next_backend_event_id(
                        handle,
                        f"{task_id}:backend",
                    ),
                },
            )

    async def _run_backend_recovery(
        self,
        task_id: str,
        handle: _WorkHandle,
        recover: Callable[..., Any],
    ) -> None:
        task = self.repository.get(task_id)
        if task is None:
            return
        loop = asyncio.get_running_loop()

        def on_event(event: Mapping[str, Any]) -> None:
            if (
                handle.queue_overflowed
                or handle.foreign_queue_overflowed
                or handle.backend_ingress_closed
                or handle.done.is_set()
            ):
                return
            normalized = _normalize_backend_signal(
                event,
                task_id=task_id,
                event_id=self._next_backend_event_id(handle, f"{task_id}:recovered"),
            )
            if normalized is None:
                return
            try:
                same_loop = asyncio.get_running_loop() is loop
            except RuntimeError:
                same_loop = False
            if same_loop:
                self._enqueue_backend_event(handle, normalized)
            else:
                self._enqueue_foreign_backend_event(loop, handle, normalized)

        try:
            outcome = await _maybe_await(
                recover(_backend_recovery_task(task), {"on_event": on_event})
            )
            await asyncio.sleep(0)
            if not self._seal_backend_event_ingress(handle):
                return
            self._enqueue_backend_event(
                handle,
                {
                    "type": "runtime.completed",
                    "outcome": _normalize_backend_outcome(outcome),
                    "_event_id": self._next_backend_event_id(
                        handle,
                        f"{task_id}:recovered",
                    ),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self._seal_backend_event_ingress(handle):
                return
            self._enqueue_backend_event(
                handle,
                {
                    "type": "runtime.failed",
                    "error": _bounded_text(error, 3_900),
                    "_event_id": self._next_backend_event_id(
                        handle,
                        f"{task_id}:recovered",
                    ),
                },
            )

    async def _event_pump(self, task_id: str, handle: _WorkHandle) -> None:
        while not handle.done.is_set():
            event = await handle.event_queue.get()
            if event.get("type") == _COALESCED_EVENT_MARKER:
                key = str(event.get("_coalesce_key") or "")
                event = handle.coalesced_events.pop(key, None)
                handle.queued_coalesced_keys.discard(key)
                if event is None:
                    continue
            self._release_buffered_event(handle, event)
            task = self.repository.get(task_id)
            if task is None or is_task_terminal(task.status):
                return
            if task.status == TaskStatus.CANCELLING and event.get("type") not in {
                "control.cancel.confirmed",
                "control.cancel.failed",
            }:
                continue
            state = await self._drive(
                handle,
                Command(resume=event),
                backend_event=event,
            )
            if event.get("type") == BackendEventType.DELEGATED:
                await self._release_scheduler(handle)
            if is_task_terminal(state.get("status", "")):
                return

    async def _drive(
        self,
        handle: _WorkHandle,
        value: Any,
        *,
        backend_event: Mapping[str, Any] | None = None,
    ) -> WorkState:
        async with handle.invoke_lock:
            if backend_event is not None:
                event_type = str(backend_event.get("type") or "")
                task = self.repository.get(handle.task_id)
                if task is None:
                    return {"status": TaskStatus.FAILED}
                if is_task_terminal(task.status):
                    return {"status": task.status}
                if (
                    handle.cancellation_requested or task.status == TaskStatus.CANCELLING
                ) and event_type not in {
                    "control.cancel.confirmed",
                    "control.cancel.failed",
                }:
                    snapshot = await self.graph.aget_state(
                        self.config(handle.task_id),
                        subgraphs=True,
                    )
                    return self._deepest_state(snapshot)
            await self.graph.ainvoke(value, self.config(handle.task_id))
            snapshot = await self.graph.aget_state(self.config(handle.task_id), subgraphs=True)
            state = self._deepest_state(snapshot)
            self._sync_projection(handle, state)
            if is_task_terminal(state.get("status", "")):
                await self._release_scheduler(handle)
                handle.done.set()
                self._discard_backend_event_buffers(handle)
                event_pump = handle.event_pump_task
                if (
                    event_pump is not None
                    and event_pump is not asyncio.current_task()
                    and not event_pump.done()
                ):
                    event_pump.cancel()
            return state

    @classmethod
    def _deepest_state(cls, snapshot: StateSnapshot) -> WorkState:
        for task in snapshot.tasks:
            if isinstance(task.state, StateSnapshot):
                return cls._deepest_state(task.state)
        return dict(snapshot.values)

    def _sync_projection(self, handle: _WorkHandle, state: WorkState) -> None:
        task = self.repository.get(handle.task_id)
        if task is None:
            return
        next_status = TaskStatus(state.get("status", task.status))
        if task.status != next_status:
            transition_task(task, next_status)
        for state_key, task_key in (
            ("started_at", "started_at"),
            ("completed_at", "completed_at"),
            ("elapsed_ms", "elapsed_ms"),
            ("result", "result"),
            ("error", "error"),
            ("message", "message"),
            ("presentation", "presentation"),
            ("activity", "activity"),
            ("artifacts", "artifacts"),
            ("delegation", "delegation"),
            ("cancellation", "cancellation"),
            ("authorization", "authorization"),
            ("notification_status", "notification_status"),
        ):
            if state_key in state:
                setattr(task, task_key, deepcopy(state[state_key]))
        self.repository.set(task.id, task)
        self.repository.save()
        for event in state.get("events", []):
            event_id = event.get("eventId")
            if event_id and event_id in handle.published_event_ids:
                continue
            if event_id:
                handle.published_event_ids.add(event_id)
            details = {
                key: deepcopy(value)
                for key, value in event.items()
                if key not in {"eventId", "type"}
            }
            self._notify(
                {
                    "type": event.get("type"),
                    "ownerId": task.owner_id,
                    "task": public_task(task),
                    **details,
                }
            )

    async def _release_scheduler(self, handle: _WorkHandle) -> None:
        if not handle.scheduler_held:
            return
        await self.scheduler.release(handle.task_id)
        handle.scheduler_held = False

    async def _fail_outside_graph(
        self, task_id: str, handle: _WorkHandle, error: Exception
    ) -> None:
        task = self.repository.get(task_id)
        if task is None or is_task_terminal(task.status):
            return
        with suppress(Exception):
            await self._drive(
                handle,
                Command(
                    resume={
                        "type": "runtime.failed",
                        "error": str(error),
                        "_event_id": f"{task_id}:runtime.failure",
                    }
                ),
            )

    async def cancel(self, task_id: str, *, owner_id: str | None = None) -> dict[str, Any] | None:
        task = self.repository.get(str(task_id))
        if task is None or (owner_id is not None and task.owner_id != str(owner_id)):
            return None
        if is_task_terminal(task.status):
            return None
        handle = self._handles.get(task.id)
        if handle is None:
            return None
        async with handle.cancel_lock:
            task = self.repository.get(task.id)
            if task is None:
                return None
            if task.status == TaskStatus.SCHEDULED:
                timer = self._scheduled_timers.pop(task.id, None)
                if timer is not None:
                    timer.cancel()
                transition_task(task, TaskStatus.CANCELLED)
                task.completed_at = now_ms()
                task.elapsed_ms = 0
                task.authorization = None
                task.notification_status = "none"
                self.repository.set(task.id, task)
                self.repository.save()
                handle.ready.set()
                handle.done.set()
                self._notify(
                    {
                        "type": "task.cancelled",
                        "ownerId": task.owner_id,
                        "task": public_task(task),
                    }
                )
                return public_task(task)
            if task.status == TaskStatus.CANCELLING:
                await handle.done.wait()
                return self.get(task.id, owner_id=owner_id)
            if not is_task_cancellable(task.status):
                return None
            await handle.ready.wait()
            if task.status == TaskStatus.QUEUED:
                await self.scheduler.cancel(task.id)
                await self._drive(handle, Command(resume={"type": "control.cancel"}))
                return self.get(task.id, owner_id=owner_id)
            handle.cancellation_requested = True
            self._close_backend_event_ingress(handle)
            state = await self._drive(
                handle,
                Command(
                    resume={
                        "type": "control.cancel",
                        "_event_id": f"{task.id}:cancel.requested",
                    }
                ),
            )
            if state.get("status") != TaskStatus.CANCELLING:
                return self.get(task.id, owner_id=owner_id)
            try:
                confirmation = await _maybe_await(
                    self.backend.cancel(task.id, {"owner_id": task.owner_id})
                )
                confirmation_error = _cancel_confirmation_error(confirmation)
                if confirmation_error is None:
                    signal = {
                        "type": "control.cancel.confirmed",
                        "confirmation": confirmation,
                    }
                else:
                    signal = {
                        "type": "control.cancel.failed",
                        "error": confirmation_error,
                    }
                await self._drive(handle, Command(resume=signal))
            except Exception as error:
                await self._drive(
                    handle,
                    Command(resume={"type": "control.cancel.failed", "error": str(error)}),
                )
            return self.get(task.id, owner_id=owner_id)

    async def respond_authorization(
        self,
        task_id: str,
        authorization_id: str,
        decision: Any,
        *,
        owner_id: str | None = None,
    ) -> dict[str, Any] | None:
        task = self.repository.get(str(task_id))
        if task is None or (owner_id is not None and task.owner_id != str(owner_id)):
            return None
        authorization = task.authorization
        if (
            not isinstance(authorization, Mapping)
            or authorization.get("status") != "pending"
            or authorization.get("id") != str(authorization_id)
        ):
            return None
        backend_outcome = await _maybe_await(
            self.backend.respond_authorization(
                task.id,
                str(authorization_id),
                decision,
                {"owner_id": task.owner_id},
            )
        )
        status = _authorization_resolution_status(backend_outcome, str(authorization_id))
        handle = self._handles.get(task.id)
        if handle is not None:
            self._enqueue_backend_event(
                handle,
                {
                    "type": BackendEventType.AUTHORIZATION_RESOLVED,
                    "permission": {"id": authorization_id, "status": status},
                    "_event_id": self._next_backend_event_id(
                        handle,
                        f"{task.id}:authorization",
                    ),
                },
            )
        return self.get(task.id, owner_id=owner_id)

    async def wait(self, task_id: str) -> dict[str, Any] | None:
        task = self.repository.get(str(task_id))
        if task is None:
            return None
        if is_task_terminal(task.status):
            return public_task(task)
        handle = self._handles.get(task.id)
        if handle is None:
            return public_task(task)
        await handle.done.wait()
        return self.get(task.id)

    async def checkpoint(self, task_id: str) -> Any:
        return await self.graph.aget_state(self.config(task_id))

    async def astream_checkpoint_updates(self, task_id: str, value: Any) -> AsyncIterator[Any]:
        """Expose native internal streams without projecting graph details publicly."""

        async for chunk in self.graph.astream(
            value,
            self.config(task_id),
            stream_mode=["updates", "custom"],
            subgraphs=True,
        ):
            yield chunk

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        scheduled_timers = tuple(self._scheduled_timers.values())
        self._scheduled_timers.clear()
        for timer in scheduled_timers:
            timer.cancel()
        for handle in self._handles.values():
            for task in (
                handle.bootstrap_task,
                handle.backend_task,
                handle.event_pump_task,
                handle.overflow_cancel_task,
            ):
                if task is not None and not task.done():
                    task.cancel()
        await _maybe_await(self.backend.close())


__all__ = [
    "WorkGraph",
    "WorkState",
    "build_work_graph",
    "reduce_activity",
    "reduce_artifacts",
    "reduce_processed_event_ids",
    "reduce_work_events",
]
