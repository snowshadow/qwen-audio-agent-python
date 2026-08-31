"""Adapt the durable Task ledger to the result-delivery lease interface.

The Task repository remains the only source of truth. ``DeliveryRecord``
instances returned here are transient projections for the presentation
coordinator; no parallel notification ledger is created or persisted.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from qwen_audio_agent.domain.task import Task, now_ms
from qwen_audio_agent.runtime.delivery import DeliveryRecord, DeliveryStatus
from qwen_audio_agent.runtime.repository import TaskRepository


class TaskNotificationLeaseAdapter:
    """Expose ``Task.notification_*`` fields as durable delivery leases."""

    def __init__(
        self,
        repository: TaskRepository[Task],
        *,
        claim_ttl_seconds: float = 60.0,
        clock_ms: Callable[[], int] = now_ms,
        on_delivered: Callable[[DeliveryRecord], None] = lambda _record: None,
    ) -> None:
        if claim_ttl_seconds <= 0:
            raise ValueError("claim_ttl_seconds must be positive")
        self.repository = repository
        self.claim_ttl_ms = max(1, int(claim_ttl_seconds * 1_000))
        self._clock_ms = clock_ms
        self._on_delivered = on_delivered

        # Notification updates are read-modify-write operations across several
        # repository calls. Reusing its reentrant lock keeps two adapters over
        # the same Task ledger from claiming the same notification.
        self._lock = repository._lock

    def claim(
        self,
        *,
        owner_id: str,
        claimant_id: str,
        session_id: str | None = None,
        include_other_sessions: bool = False,
        task_ids: Iterable[str] | None = None,
    ) -> list[DeliveryRecord]:
        """Claim matching pending Task notifications in creation order."""

        requested = {str(task_id) for task_id in task_ids} if task_ids is not None else None
        claimed: list[DeliveryRecord] = []
        timestamp = self._now()
        changed = False
        with self._lock:
            changed = self._reclaim_expired_locked(timestamp) > 0
            for task in self.repository.values():
                if (
                    task.owner_id != str(owner_id)
                    or task.notification_status != DeliveryStatus.PENDING.value
                    or (
                        session_id is not None
                        and not include_other_sessions
                        and task.session_id != str(session_id)
                    )
                    or (requested is not None and task.id not in requested)
                ):
                    continue
                task.notification_status = DeliveryStatus.DELIVERING.value
                task.notification_claimant_id = str(claimant_id)
                task.notification_claimed_at = timestamp
                self.repository.set(task.id, task)
                claimed.append(self._project(task))
                changed = True
            if changed:
                self.repository.save()
        return sorted(claimed, key=lambda record: record.created_at)

    def renew(self, task_ids: Iterable[str], *, claimant_id: str) -> int:
        """Renew leases still owned by ``claimant_id`` and persist the timestamp."""

        renewed = 0
        timestamp = self._now()
        with self._lock:
            for task_id in task_ids:
                task = self._owned_claim(task_id, claimant_id)
                if task is None:
                    continue
                task.notification_claimed_at = timestamp
                self.repository.set(task.id, task)
                renewed += 1
            if renewed:
                self.repository.save()
        return renewed

    def release(self, task_ids: Iterable[str], *, claimant_id: str) -> int:
        """Return leases owned by ``claimant_id`` to the pending state."""

        released = 0
        with self._lock:
            for task_id in task_ids:
                task = self._owned_claim(task_id, claimant_id)
                if task is None:
                    continue
                task.notification_status = DeliveryStatus.PENDING.value
                task.notification_claimant_id = None
                task.notification_claimed_at = None
                self.repository.set(task.id, task)
                released += 1
            if released:
                self.repository.save()
        return released

    def mark_delivered(self, task_ids: Iterable[str], *, claimant_id: str) -> int:
        """Persist playback acknowledgement before publishing delivery events."""

        delivered: list[DeliveryRecord] = []
        timestamp = self._now()
        with self._lock:
            for task_id in task_ids:
                task = self._owned_claim(task_id, claimant_id)
                if task is None:
                    continue
                task.notification_status = DeliveryStatus.DELIVERED.value
                task.notification_claimant_id = None
                task.notification_claimed_at = None
                task.notification_delivered_at = timestamp
                self.repository.set(task.id, task)
                delivered.append(self._project(task))
            if delivered:
                self.repository.save()

        # Observers cannot see an acknowledgement that is absent from disk.
        for record in delivered:
            self._on_delivered(record)
        return len(delivered)

    def reclaim_expired(self) -> int:
        """Release expired claims so another live frontend can reclaim them."""

        with self._lock:
            reclaimed = self._reclaim_expired_locked(self._now())
            if reclaimed:
                self.repository.save()
            return reclaimed

    def _reclaim_expired_locked(self, timestamp: int) -> int:
        reclaimed = 0
        for task in self.repository.values():
            claimed_at = task.notification_claimed_at
            if (
                task.notification_status != DeliveryStatus.DELIVERING.value
                or claimed_at is None
                or timestamp - int(claimed_at) < self.claim_ttl_ms
            ):
                continue
            task.notification_status = DeliveryStatus.PENDING.value
            task.notification_claimant_id = None
            task.notification_claimed_at = None
            self.repository.set(task.id, task)
            reclaimed += 1
        return reclaimed

    def _owned_claim(self, task_id: str, claimant_id: str) -> Task | None:
        task = self.repository.get(str(task_id))
        if (
            task is None
            or task.notification_status != DeliveryStatus.DELIVERING.value
            or task.notification_claimant_id != str(claimant_id)
        ):
            return None
        return task

    def _now(self) -> int:
        return int(self._clock_ms())

    @staticmethod
    def _project(task: Task) -> DeliveryRecord:
        try:
            status = DeliveryStatus(task.notification_status)
        except ValueError:
            status = DeliveryStatus.PENDING
        return DeliveryRecord(
            task_id=task.id,
            owner_id=task.owner_id,
            session_id=task.session_id,
            created_at=float(task.created_at),
            status=status,
            claimant_id=task.notification_claimant_id,
            claimed_at=(
                float(task.notification_claimed_at)
                if task.notification_claimed_at is not None
                else None
            ),
            delivered_at=(
                float(task.notification_delivered_at)
                if task.notification_delivered_at is not None
                else None
            ),
        )


__all__ = ["TaskNotificationLeaseAdapter"]
