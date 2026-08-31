"""Coordinate durable result leases with realtime presentation.

This service intentionally lives outside LangGraph.  Checkpoints describe one
graph execution; they cannot prove that a client started playing a completed
result.  The coordinator therefore combines the durable lease queue with the
transport-owned duplex window and playback receipts.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from qwen_audio_agent.runtime.delivery import DeliveryLeaseQueue


@dataclass(frozen=True, slots=True)
class PresentationOutcome:
    """Result of asking the realtime frontend to generate an announcement."""

    completed: bool
    context_injected: bool = False


@dataclass(frozen=True, slots=True)
class DeliveryAnnouncement:
    """Bounded final Work material that may be presented to the user."""

    task_id: str
    status: str
    objective: str = ""
    result: str = ""
    error: str = ""
    sequence: int = 0


@dataclass(slots=True)
class _DeliveryBatch:
    announcements: list[DeliveryAnnouncement]
    task_ids: list[str]
    delivery_turn_id: str
    context_injected: bool = False
    response_completed: bool = False
    retry_requested: bool = False
    acknowledged: set[str] = field(default_factory=set)

    @property
    def outstanding_task_ids(self) -> list[str]:
        return [task_id for task_id in self.task_ids if task_id not in self.acknowledged]


type PresentResult = Callable[
    [str, Mapping[str, Any], bool],
    Awaitable[PresentationOutcome | Mapping[str, Any] | None],
]


def _value(record: Any, snake: str, camel: str | None = None, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        if snake in record:
            return record[snake]
        if camel and camel in record:
            return record[camel]
        return default
    if hasattr(record, snake):
        return getattr(record, snake)
    if camel and hasattr(record, camel):
        return getattr(record, camel)
    return default


def format_work_results(announcements: list[DeliveryAnnouncement]) -> str:
    """Build the bounded, model-facing final-result handoff."""

    blocks: list[str] = []
    for item in announcements:
        lines = [f"task_id: {item.task_id}", f"状态: {item.status}"]
        if item.objective:
            lines.append(f"工作: {item.objective}")
        if item.status == "completed":
            lines.append(f"结果: {item.result.strip()}")
        else:
            lines.append(f"错误: {item.error.strip()}")
        blocks.append("\n".join(lines))
    return "\n".join(
        [
            "以下是你先前异步执行工作的最终更新，不是用户的新请求。"
            "请作为自己的工作结果自然告知用户，不要提后台 Agent、协议或 Task，"
            "也不要读出 task_id。",
            *(f"\n{block}" for block in blocks),
        ]
    )


def truncate_result(value: str, maximum_characters: int) -> str:
    """Truncate by Unicode code point while keeping the durable result intact."""

    text = str(value or "").strip()
    limit = max(1, int(maximum_characters or 1))
    if len(text) <= limit:
        return text
    suffix = "\n…（事件内容过长，已截断；完整结果仍保留在任务记录中）"  # noqa: RUF001
    if limit <= len(suffix):
        return text[:limit]
    return f"{text[: limit - len(suffix)]}{suffix}"


def _artifact_summary(value: Any) -> str:
    """Describe artifact metadata without copying artifact payloads."""

    artifacts = value if isinstance(value, list | tuple) else ()
    descriptors: list[str] = []
    for index, artifact in enumerate(artifacts[:8]):
        if not isinstance(artifact, Mapping):
            continue
        name = str(artifact.get("name") or artifact.get("artifactId") or "").strip()
        name = " ".join(name.split())[:120] or f"产物 {index + 1}"
        raw_parts = artifact.get("parts")
        parts = raw_parts if isinstance(raw_parts, list | tuple) else ()
        mime_types: list[str] = []
        for part in parts[:8]:
            if not isinstance(part, Mapping):
                continue
            mime = " ".join(
                str(
                    part.get("mediaType")
                    or part.get("media_type")
                    or part.get("mimeType")
                    or part.get("mime")
                    or ""
                )
                .strip()
                .split()
            )[:120]
            if mime and mime not in mime_types:
                mime_types.append(mime)
            if len(mime_types) == 3:
                break
        kind = "、".join(mime_types) if mime_types else "未标注类型"
        descriptors.append(f"{name} ({kind})")
    if not artifacts:
        return ""
    count = len(artifacts)
    shown = "、".join(descriptors)
    remainder = max(0, count - len(descriptors))
    suffix = f"，另有 {remainder} 个产物" if remainder else ""
    if shown:
        return f"已生成 {count} 个产物: {shown}{suffix}。"[:1_200]
    return f"已生成 {count} 个产物。"


def _announcement_result(task: Any, *, status: str) -> str:
    result = str(_value(task, "result", default="") or "").strip()
    if status != "completed" or result:
        return result
    presentation = _value(task, "presentation", default={})
    if isinstance(presentation, Mapping):
        speech = str(presentation.get("speech") or "").strip()
        if speech:
            return speech[:6_000]
    return _artifact_summary(_value(task, "artifacts", default=[])) or "工作已完成。"


class ResultDeliveryCoordinator:
    """Drive one claimant's result presentation without owning audio playback."""

    def __init__(
        self,
        *,
        leases: DeliveryLeaseQueue,
        owner_id: str,
        session_id: str,
        claimant_id: str,
        present_result: PresentResult,
        is_delivery_blocked: Callable[[], bool],
        include_other_sessions: bool = False,
        max_batch_items: int = 8,
        result_context_max_chars: int = 6_000,
        batch_window_seconds: float = 0.12,
        retry_base_seconds: float = 1.0,
        retry_max_seconds: float = 10.0,
        acknowledgement_timeout_seconds: float = 120.0,
        max_retry_attempts: int = 8,
        lease_renew_interval_seconds: float = 20.0,
        create_turn_id: Callable[[], str] = lambda: f"gateway_{uuid4().hex}",
        on_error: Callable[[Exception], None] = lambda _error: None,
    ) -> None:
        if not owner_id or not claimant_id:
            raise ValueError("owner_id and claimant_id are required")
        if max_batch_items <= 0 or result_context_max_chars <= 0:
            raise ValueError("delivery batch limits must be positive")
        if retry_base_seconds < 0 or retry_max_seconds < 0:
            raise ValueError("retry delays cannot be negative")
        if acknowledgement_timeout_seconds <= 0 or lease_renew_interval_seconds <= 0:
            raise ValueError("delivery timeouts must be positive")
        if max_retry_attempts < 0:
            raise ValueError("max_retry_attempts cannot be negative")

        self.leases = leases
        self.owner_id = owner_id
        self.session_id = session_id
        self.claimant_id = claimant_id
        self.present_result = present_result
        self.is_delivery_blocked = is_delivery_blocked
        self.include_other_sessions = include_other_sessions
        self.max_batch_items = max_batch_items
        self.result_context_max_chars = result_context_max_chars
        self.batch_window_seconds = max(0.0, batch_window_seconds)
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = max(retry_base_seconds, retry_max_seconds)
        self.acknowledgement_timeout_seconds = acknowledgement_timeout_seconds
        self.max_retry_attempts = max_retry_attempts
        self.lease_renew_interval_seconds = lease_renew_interval_seconds
        self.create_turn_id = create_turn_id
        self.on_error = on_error

        self._pending: dict[str, DeliveryAnnouncement] = {}
        self._active: _DeliveryBatch | None = None
        self._sequence = 0
        self._retry_count = 0
        self._claim_retry_not_before: dict[str, float] = {}
        self._claim_retry_counts: dict[str, int] = {}
        self._claim_retry_base_seconds = max(0.05, retry_base_seconds)
        self._claim_retry_max_seconds = max(
            self._claim_retry_base_seconds,
            retry_max_seconds,
        )
        self._generation = 0
        self._delivering = False
        self._closed = False
        self._lock = asyncio.Lock()
        self._delivery_task: asyncio.Task[None] | None = None
        self._delivery_due_at: float | None = None
        self._retry_task: asyncio.Task[None] | None = None
        self._acknowledgement_task: asyncio.Task[None] | None = None
        self._lease_renew_task: asyncio.Task[None] | None = None
        self._attempt_task: asyncio.Task[Any] | None = None

    @property
    def active_task_ids(self) -> tuple[str, ...]:
        return tuple(self._active.task_ids) if self._active is not None else ()

    @property
    def pending_task_ids(self) -> tuple[str, ...]:
        return tuple(
            item.task_id for item in sorted(self._pending.values(), key=lambda item: item.sequence)
        )

    def completed(self, task: Any) -> None:
        self._enqueue(task, status="completed")

    def failed(self, task: Any) -> None:
        self._enqueue(task, status="failed")

    def _enqueue(self, task: Any, *, status: str) -> None:
        if self._closed:
            return
        task_id = str(_value(task, "id", "taskId", "") or "").strip()
        if not task_id:
            raise ValueError("delivery announcement requires a task id")
        self._sequence += 1
        self._pending[task_id] = DeliveryAnnouncement(
            task_id=task_id,
            status=status,
            objective=str(_value(task, "objective", default="") or ""),
            result=_announcement_result(task, status=status),
            error=str(_value(task, "error", default="") or ""),
            sequence=self._sequence,
        )
        if self._active is None:
            self._schedule_delivery(self.batch_window_seconds)

    async def flush(self) -> None:
        """Recheck the safe window and attempt the oldest claim immediately."""

        self._cancel_task(self._delivery_task)
        self._delivery_task = None
        self._delivery_due_at = None
        await self._deliver()

    async def playback_started(self, task_ids: list[str] | tuple[str, ...]) -> int:
        """Durably acknowledge only results whose client playback has started."""

        delivered = 0
        async with self._lock:
            batch = self._active
            if batch is None:
                return 0
            for task_id in dict.fromkeys(str(value) for value in task_ids if value):
                if task_id not in batch.task_ids or task_id in batch.acknowledged:
                    continue
                if self.leases.mark_delivered([task_id], claimant_id=self.claimant_id) != 1:
                    continue
                batch.acknowledged.add(task_id)
                self._pending.pop(task_id, None)
                self._clear_claim_retry_locked(task_id)
                delivered += 1
            self._finish_acknowledged_locked()
        return delivered

    async def playback_failed(self, task_ids: list[str] | tuple[str, ...]) -> None:
        """Retry an active result whose generated audio never started playback."""

        async with self._lock:
            batch = self._active
            failed_ids = {str(task_id) for task_id in task_ids if task_id}
            if batch is None or not failed_ids.intersection(batch.task_ids):
                return
            batch.response_completed = False
            batch.retry_requested = True
            self._cancel_task(self._acknowledgement_task)
            self._acknowledgement_task = None
            self._cancel_task(self._retry_task)
            self._retry_task = None
            if not self._delivering:
                batch.retry_requested = False
                self._schedule_retry_locked()

    async def discard_delivered(self, task_ids: list[str] | tuple[str, ...]) -> None:
        """Drop results another claimant has already delivered.

        A losing frontend can keep already-delivered rows at the head of its
        in-memory queue.  Since those rows can no longer be claimed, a full
        stale batch would otherwise prevent newer results from ever reaching
        the claimant.  If one of the rows belongs to the active batch, rebuild
        that batch from the still-pending rows instead of presenting a mixed
        stale result.
        """

        delivered_ids = {str(task_id) for task_id in task_ids if task_id}
        if not delivered_ids:
            return
        async with self._lock:
            for task_id in delivered_ids:
                self._pending.pop(task_id, None)
                self._clear_claim_retry_locked(task_id)

            batch = self._active
            if batch is None or delivered_ids.isdisjoint(batch.task_ids):
                if self._active is None and self._pending and not self._delivering:
                    self._cancel_task(self._delivery_task)
                    self._delivery_task = None
                    self._delivery_due_at = None
                    self._schedule_next_pending_locked()
                return

            remaining_claims = [
                task_id for task_id in batch.outstanding_task_ids if task_id not in delivered_ids
            ]
            self._generation += 1
            self._cancel_scheduled_locked()
            self._active = None
            self._retry_count = 0
            self._delivering = False
            if remaining_claims:
                self.leases.release(remaining_claims, claimant_id=self.claimant_id)
            if self._pending:
                self._schedule_next_pending_locked()

    async def provider_disconnected(self) -> None:
        """Fence an old provider attempt and reinject context after reconnect."""

        async with self._lock:
            batch = self._active
            if batch is None:
                return
            self._generation += 1
            batch.context_injected = False
            batch.response_completed = False
            batch.retry_requested = False
            self._cancel_scheduled_locked()
            self._delivering = False
            self._ensure_lease_renewal_locked(batch, self._generation)
            self._schedule_delivery(0)

    async def pause(self) -> None:
        """Release claims when the owning voice client sleeps or is replaced."""

        async with self._lock:
            self._generation += 1
            active = self._active.outstanding_task_ids if self._active is not None else []
            self._cancel_scheduled_locked()
            self._pending.clear()
            self._claim_retry_not_before.clear()
            self._claim_retry_counts.clear()
            self._active = None
            self._retry_count = 0
            self._delivering = False
            if active:
                self.leases.release(active, claimant_id=self.claimant_id)

    async def close(self) -> None:
        self._closed = True
        await self.pause()

    async def _deliver(self) -> None:
        batch: _DeliveryBatch | None = None
        generation = 0
        async with self._lock:
            if self._closed or self._delivering or not self._pending:
                return
            if self.is_delivery_blocked():
                self._schedule_delivery(self.retry_base_seconds)
                return
            if self._active is None:
                self._active = self._claim_batch_locked()
            batch = self._active
            if batch is None:
                return
            generation = self._generation
            self._delivering = True
            self._ensure_lease_renewal_locked(batch, generation)
            text = truncate_result(
                format_work_results(batch.announcements),
                self.result_context_max_chars,
            )
            context = {
                "turnId": batch.delivery_turn_id,
                "taskId": batch.task_ids[0] if len(batch.task_ids) == 1 else None,
                "taskIds": list(batch.task_ids),
            }
            attempt = asyncio.create_task(
                self._present(text, context, not batch.context_injected),
                name="result-delivery-presentation",
            )
            self._attempt_task = attempt

        try:
            raw_outcome = await attempt
            outcome = self._normalize_outcome(raw_outcome)
        except asyncio.CancelledError:
            # Pause and lease loss fence stale work by cancelling this dedicated
            # presentation task. Do not cancel a caller merely awaiting flush().
            current = asyncio.current_task()
            caller_cancelled = current is not None and current.cancelling()
            async with self._lock:
                if self._attempt_task is attempt:
                    self._attempt_task = None
                    self._delivering = False
                    if generation == self._generation and self._active is batch:
                        self._schedule_retry_locked()
            if caller_cancelled:
                raise
            return
        except Exception as error:  # presentation diagnostics must not break recovery
            try:
                self.on_error(error)
            except Exception:
                pass
            outcome = PresentationOutcome(completed=False)

        async with self._lock:
            if self._attempt_task is attempt:
                self._attempt_task = None
                self._delivering = False
            if generation != self._generation or self._active is not batch:
                if self._pending and not self._delivering:
                    self._schedule_next_pending_locked()
                return
            if outcome.context_injected:
                batch.context_injected = True
            if batch.retry_requested:
                batch.retry_requested = False
                batch.response_completed = False
                self._schedule_retry_locked()
            elif outcome.completed:
                batch.response_completed = True
                self._schedule_acknowledgement_timeout_locked(batch, generation)
            else:
                self._schedule_retry_locked()

    def _claim_batch_locked(self) -> _DeliveryBatch | None:
        queued: list[DeliveryAnnouncement] = []
        now = asyncio.get_running_loop().time()
        for item in sorted(self._pending.values(), key=lambda value: value.sequence):
            if self._claim_retry_not_before.get(item.task_id, 0.0) > now:
                continue
            candidate = [*queued, item]
            if queued and (
                len(queued) >= self.max_batch_items
                or len(format_work_results(candidate)) > self.result_context_max_chars
            ):
                break
            queued.append(item)
        if not queued:
            self._schedule_next_pending_locked()
            return None
        claimed = self.leases.claim(
            owner_id=self.owner_id,
            session_id=self.session_id,
            include_other_sessions=self.include_other_sessions,
            claimant_id=self.claimant_id,
            task_ids=[item.task_id for item in queued],
        )
        claimed_ids = {record.task_id for record in claimed}
        deferred_retry = False
        for item in queued:
            if item.task_id in claimed_ids:
                self._clear_claim_retry_locked(item.task_id)
            elif item.task_id in self._claim_retry_not_before:
                self._defer_claim_retry_locked(item.task_id)
                deferred_retry = True
        announcements = [item for item in queued if item.task_id in claimed_ids]
        if not announcements:
            if deferred_retry:
                self._schedule_next_pending_locked()
            else:
                self._schedule_delivery(self.retry_base_seconds)
            return None
        return _DeliveryBatch(
            announcements=announcements,
            task_ids=[item.task_id for item in announcements],
            delivery_turn_id=self.create_turn_id(),
        )

    @staticmethod
    def _normalize_outcome(
        value: PresentationOutcome | Mapping[str, Any] | None,
    ) -> PresentationOutcome:
        if isinstance(value, PresentationOutcome):
            return value
        if isinstance(value, Mapping):
            return PresentationOutcome(
                completed=value.get("completed") is True,
                context_injected=value.get("contextInjected") is True
                or value.get("context_injected") is True,
            )
        return PresentationOutcome(completed=False)

    async def _present(
        self,
        text: str,
        context: Mapping[str, Any],
        inject_context: bool,
    ) -> PresentationOutcome | Mapping[str, Any] | None:
        return await self.present_result(text, context, inject_context)

    def _finish_acknowledged_locked(self) -> None:
        batch = self._active
        if batch is None or batch.outstanding_task_ids:
            return
        self._active = None
        self._retry_count = 0
        self._cancel_task(self._retry_task)
        self._retry_task = None
        self._cancel_task(self._acknowledgement_task)
        self._acknowledgement_task = None
        self._cancel_task(self._lease_renew_task)
        self._lease_renew_task = None
        if self._pending and not self._delivering:
            self._schedule_next_pending_locked()

    def _abandon_active_locked(self, *, release: bool) -> None:
        batch = self._active
        if batch is None:
            return
        outstanding = batch.outstanding_task_ids
        if release:
            for task_id in batch.task_ids:
                self._pending.pop(task_id, None)
                self._clear_claim_retry_locked(task_id)
        else:
            # A lost lease is not a delivered result. Keep the announcement in
            # memory, but defer its next claim so a later queued result can pass
            # it and the new claimant cannot cause duplicate presentation.
            for task_id in batch.acknowledged:
                self._pending.pop(task_id, None)
                self._clear_claim_retry_locked(task_id)
            for task_id in outstanding:
                self._defer_claim_retry_locked(task_id)
        self._generation += 1
        self._active = None
        self._retry_count = 0
        self._delivering = False
        self._cancel_task(self._retry_task)
        self._retry_task = None
        self._cancel_task(self._acknowledgement_task)
        self._acknowledgement_task = None
        self._cancel_task(self._lease_renew_task)
        self._lease_renew_task = None
        self._cancel_task(self._attempt_task)
        self._attempt_task = None
        if release and outstanding:
            self.leases.release(outstanding, claimant_id=self.claimant_id)
        if self._pending:
            self._schedule_next_pending_locked()

    def _defer_claim_retry_locked(self, task_id: str) -> None:
        if task_id not in self._pending:
            return
        count = self._claim_retry_counts.get(task_id, 0) + 1
        self._claim_retry_counts[task_id] = count
        delay = min(
            self._claim_retry_max_seconds,
            self._claim_retry_base_seconds * (2 ** min(count - 1, 4)),
        )
        self._claim_retry_not_before[task_id] = asyncio.get_running_loop().time() + delay

    def _clear_claim_retry_locked(self, task_id: str) -> None:
        self._claim_retry_not_before.pop(task_id, None)
        self._claim_retry_counts.pop(task_id, None)

    def _schedule_next_pending_locked(self) -> None:
        if self._closed or self._active is not None or not self._pending:
            return
        now = asyncio.get_running_loop().time()
        deadlines = [self._claim_retry_not_before.get(task_id, now) for task_id in self._pending]
        self._schedule_delivery(max(0.0, min(deadlines) - now))

    def _schedule_delivery(self, delay: float) -> None:
        if self._closed:
            return
        delay = max(0.0, delay)
        due_at = asyncio.get_running_loop().time() + delay
        if self._delivery_task is not None:
            if self._delivery_due_at is not None and self._delivery_due_at <= due_at:
                return
            self._cancel_task(self._delivery_task)
        self._delivery_due_at = due_at
        self._delivery_task = asyncio.create_task(
            self._run_delivery_after(delay),
            name="result-delivery",
        )

    async def _run_delivery_after(self, delay: float) -> None:
        try:
            if delay:
                await asyncio.sleep(delay)
            if self._delivery_task is asyncio.current_task():
                self._delivery_task = None
                self._delivery_due_at = None
            await self._deliver()
        except asyncio.CancelledError:
            return

    def _schedule_retry_locked(self) -> None:
        if self._closed or self._active is None or self._retry_task is not None:
            return
        if self._retry_count >= self.max_retry_attempts:
            self._abandon_active_locked(release=True)
            return
        self._retry_count += 1
        delay = min(
            self.retry_max_seconds,
            self.retry_base_seconds * (2 ** min(self._retry_count - 1, 4)),
        )
        self._retry_task = asyncio.create_task(
            self._run_retry_after(delay),
            name="result-delivery-retry",
        )

    async def _run_retry_after(self, delay: float) -> None:
        try:
            if delay:
                await asyncio.sleep(delay)
            async with self._lock:
                if self._retry_task is asyncio.current_task():
                    self._retry_task = None
            await self._deliver()
        except asyncio.CancelledError:
            return

    def _schedule_acknowledgement_timeout_locked(
        self,
        batch: _DeliveryBatch,
        generation: int,
    ) -> None:
        self._cancel_task(self._acknowledgement_task)
        self._acknowledgement_task = asyncio.create_task(
            self._run_acknowledgement_timeout(batch, generation),
            name="result-delivery-acknowledgement",
        )

    async def _run_acknowledgement_timeout(
        self,
        batch: _DeliveryBatch,
        generation: int,
    ) -> None:
        try:
            await asyncio.sleep(self.acknowledgement_timeout_seconds)
            async with self._lock:
                if self._acknowledgement_task is asyncio.current_task():
                    self._acknowledgement_task = None
                if (
                    generation != self._generation
                    or self._active is not batch
                    or not batch.response_completed
                ):
                    return
                if self.is_delivery_blocked():
                    self._schedule_acknowledgement_timeout_locked(batch, generation)
                    return
                batch.response_completed = False
                self._schedule_retry_locked()
        except asyncio.CancelledError:
            return

    def _ensure_lease_renewal_locked(
        self,
        batch: _DeliveryBatch,
        generation: int,
    ) -> None:
        outstanding = batch.outstanding_task_ids
        if outstanding:
            self.leases.renew(outstanding, claimant_id=self.claimant_id)
        if self._lease_renew_task is None:
            self._lease_renew_task = asyncio.create_task(
                self._run_lease_renewal(batch, generation),
                name="result-delivery-lease-renewal",
            )

    async def _run_lease_renewal(
        self,
        batch: _DeliveryBatch,
        generation: int,
    ) -> None:
        try:
            while True:
                await asyncio.sleep(self.lease_renew_interval_seconds)
                async with self._lock:
                    if generation != self._generation or self._active is not batch:
                        return
                    outstanding = batch.outstanding_task_ids
                    if not outstanding:
                        return
                    renewed = self.leases.renew(outstanding, claimant_id=self.claimant_id)
                    if renewed != len(outstanding):
                        self._abandon_active_locked(release=False)
                        return
        except asyncio.CancelledError:
            return
        finally:
            if self._lease_renew_task is asyncio.current_task():
                self._lease_renew_task = None

    def _cancel_scheduled_locked(self) -> None:
        for task in (
            self._delivery_task,
            self._retry_task,
            self._acknowledgement_task,
            self._lease_renew_task,
            self._attempt_task,
        ):
            self._cancel_task(task)
        self._delivery_task = None
        self._delivery_due_at = None
        self._retry_task = None
        self._acknowledgement_task = None
        self._lease_renew_task = None
        self._attempt_task = None

    @staticmethod
    def _cancel_task(task: asyncio.Task[Any] | None) -> None:
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()


__all__ = [
    "DeliveryAnnouncement",
    "PresentationOutcome",
    "ResultDeliveryCoordinator",
    "format_work_results",
    "truncate_result",
]
