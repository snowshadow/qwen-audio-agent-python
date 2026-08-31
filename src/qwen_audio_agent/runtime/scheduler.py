"""FIFO scheduler for WorkGraph backend submissions."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _positive_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _value(task: Any, snake: str, camel: str | None = None, default: Any = None) -> Any:
    if isinstance(task, Mapping):
        if snake in task:
            return task[snake]
        if camel and camel in task:
            return task[camel]
        return default
    if hasattr(task, snake):
        return getattr(task, snake)
    if camel and hasattr(task, camel):
        return getattr(task, camel)
    return default


@dataclass(frozen=True, slots=True)
class SchedulerTicket:
    task_id: str
    owner_id: str
    lane_key: str | None = None
    lane_limit: int = 1

    @classmethod
    def from_task(cls, task: Any) -> SchedulerTicket:
        task_id = str(_value(task, "id", default="") or "")
        if not task_id:
            raise ValueError("Scheduled Task requires an id")
        lane = _value(task, "lane_key", "laneKey")
        return cls(
            task_id=task_id,
            owner_id=str(_value(task, "owner_id", "ownerId", "") or ""),
            lane_key=str(lane) if lane else None,
            lane_limit=_positive_int(_value(task, "lane_limit", "laneLimit", 1), 1),
        )


@dataclass(slots=True)
class _Pending:
    ticket: SchedulerTicket
    future: asyncio.Future[bool]


class TaskScheduler:
    """Bounded scheduler with FIFO ordering per owner and serialization lane.

    A blocked owner or lane never blocks unrelated owners.  A later Task may
    not overtake an earlier pending Task owned by the same owner or using the
    same lane.
    """

    def __init__(
        self,
        *,
        max_concurrent: int = 4,
        max_concurrent_per_owner: int = 1,
    ) -> None:
        self.max_concurrent = _positive_int(max_concurrent, 4)
        self.max_concurrent_per_owner = _positive_int(max_concurrent_per_owner, 1)
        self._active: dict[str, SchedulerTicket] = {}
        self._pending: list[_Pending] = []
        self._lock = asyncio.Lock()

    @property
    def active(self) -> Mapping[str, SchedulerTicket]:
        return dict(self._active)

    @property
    def pending(self) -> tuple[SchedulerTicket, ...]:
        return tuple(item.ticket for item in self._pending)

    def count(self, predicate: Any = None) -> int:
        if predicate is None:
            return len(self._active)
        return sum(1 for ticket in self._active.values() if predicate(ticket))

    def can_start(self, task: Any) -> bool:
        ticket = task if isinstance(task, SchedulerTicket) else SchedulerTicket.from_task(task)
        if ticket.task_id in self._active:
            return True
        if len(self._active) >= self.max_concurrent:
            return False
        owner_count = sum(active.owner_id == ticket.owner_id for active in self._active.values())
        if owner_count >= self.max_concurrent_per_owner:
            return False
        if ticket.lane_key:
            lane_count = sum(active.lane_key == ticket.lane_key for active in self._active.values())
            if lane_count >= ticket.lane_limit:
                return False
        return True

    def _has_predecessor(self, index: int, ticket: SchedulerTicket) -> bool:
        for pending in self._pending[:index]:
            earlier = pending.ticket
            if earlier.owner_id == ticket.owner_id:
                return True
            if ticket.lane_key and earlier.lane_key == ticket.lane_key:
                return True
        return False

    def _drain_locked(self) -> None:
        made_progress = True
        while made_progress and len(self._active) < self.max_concurrent:
            made_progress = False
            for index, pending in enumerate(self._pending):
                if self._has_predecessor(index, pending.ticket):
                    continue
                if not self.can_start(pending.ticket):
                    continue
                self._pending.pop(index)
                self._active[pending.ticket.task_id] = pending.ticket
                if not pending.future.done():
                    pending.future.set_result(True)
                made_progress = True
                break

    async def acquire(self, task: Any) -> bool:
        ticket = task if isinstance(task, SchedulerTicket) else SchedulerTicket.from_task(task)
        loop = asyncio.get_running_loop()
        async with self._lock:
            if ticket.task_id in self._active:
                return True
            existing = next(
                (item for item in self._pending if item.ticket.task_id == ticket.task_id),
                None,
            )
            if existing is None:
                existing = _Pending(ticket=ticket, future=loop.create_future())
                self._pending.append(existing)
                self._drain_locked()
            future = existing.future
        return await asyncio.shield(future)

    async def cancel(self, task_id: str) -> bool:
        normalized = str(task_id)
        async with self._lock:
            for index, pending in enumerate(self._pending):
                if pending.ticket.task_id != normalized:
                    continue
                self._pending.pop(index)
                if not pending.future.done():
                    pending.future.set_result(False)
                self._drain_locked()
                return True
            return False

    async def release(self, task: Any) -> bool:
        task_id = str(task if isinstance(task, str) else _value(task, "id", default=""))
        async with self._lock:
            removed = self._active.pop(task_id, None) is not None
            self._drain_locked()
            return removed

    def active_counts(self) -> dict[str, Mapping[str, int]]:
        owners = Counter(ticket.owner_id for ticket in self._active.values())
        lanes = Counter(ticket.lane_key for ticket in self._active.values() if ticket.lane_key)
        return {"owners": dict(owners), "lanes": dict(lanes)}


__all__ = ["SchedulerTicket", "TaskScheduler"]
