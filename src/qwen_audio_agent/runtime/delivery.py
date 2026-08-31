"""Exactly-once presentation leases for completed user work.

Execution completion and voice playback are separate facts. A result is only
delivered after the active output client acknowledges playback; an abandoned
claim becomes pending again after its lease expires.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass, replace
from enum import StrEnum
from time import time


class DeliveryStatus(StrEnum):
    PENDING = "pending"
    DELIVERING = "delivering"
    DELIVERED = "delivered"


@dataclass(slots=True)
class DeliveryRecord:
    task_id: str
    owner_id: str
    session_id: str
    created_at: float
    status: DeliveryStatus = DeliveryStatus.PENDING
    claimant_id: str | None = None
    claimed_at: float | None = None
    delivered_at: float | None = None


class DeliveryLeaseQueue:
    """Own result-delivery leases without owning playback or transport."""

    def __init__(
        self,
        records: MutableMapping[str, DeliveryRecord],
        *,
        claim_ttl_seconds: float = 30.0,
        now: Callable[[], float] = time,
        persist: Callable[[], None] = lambda: None,
        on_delivered: Callable[[DeliveryRecord], None] = lambda _record: None,
    ) -> None:
        if claim_ttl_seconds <= 0:
            raise ValueError("claim_ttl_seconds must be positive")
        self._records = records
        self._claim_ttl_seconds = claim_ttl_seconds
        self._now = now
        self._persist = persist
        self._on_delivered = on_delivered

    def claim(
        self,
        *,
        owner_id: str,
        claimant_id: str,
        session_id: str | None = None,
        include_other_sessions: bool = False,
        task_ids: Iterable[str] | None = None,
    ) -> list[DeliveryRecord]:
        self.reclaim_expired()
        requested = {str(task_id) for task_id in task_ids} if task_ids is not None else None
        claimed: list[DeliveryRecord] = []
        claimed_at = self._now()
        for record in self._records.values():
            if record.owner_id != owner_id or record.status is not DeliveryStatus.PENDING:
                continue
            if (
                session_id is not None
                and not include_other_sessions
                and record.session_id != session_id
            ):
                continue
            if requested is not None and record.task_id not in requested:
                continue
            record.status = DeliveryStatus.DELIVERING
            record.claimant_id = claimant_id
            record.claimed_at = claimed_at
            claimed.append(replace(record))
        if claimed:
            self._persist()
        return sorted(claimed, key=lambda record: record.created_at)

    def renew(self, task_ids: Iterable[str], *, claimant_id: str) -> int:
        renewed = 0
        claimed_at = self._now()
        for task_id in task_ids:
            record = self._owned_claim(task_id, claimant_id)
            if record is None:
                continue
            record.claimed_at = claimed_at
            renewed += 1
        if renewed:
            self._persist()
        return renewed

    def release(self, task_ids: Iterable[str], *, claimant_id: str) -> int:
        released = 0
        for task_id in task_ids:
            record = self._owned_claim(task_id, claimant_id)
            if record is None:
                continue
            record.status = DeliveryStatus.PENDING
            record.claimant_id = None
            record.claimed_at = None
            released += 1
        if released:
            self._persist()
        return released

    def mark_delivered(self, task_ids: Iterable[str], *, claimant_id: str) -> int:
        delivered: list[DeliveryRecord] = []
        delivered_at = self._now()
        for task_id in task_ids:
            record = self._owned_claim(task_id, claimant_id)
            if record is None:
                continue
            record.status = DeliveryStatus.DELIVERED
            record.claimant_id = None
            record.claimed_at = None
            record.delivered_at = delivered_at
            delivered.append(record)
        if not delivered:
            return 0

        # The acknowledgement is durable before observers can emit the public
        # delivered event. A crash after the event therefore cannot replay it.
        self._persist()
        for record in delivered:
            self._on_delivered(replace(record))
        return len(delivered)

    def reclaim_expired(self) -> int:
        reclaimed = 0
        now = self._now()
        for record in self._records.values():
            if (
                record.status is not DeliveryStatus.DELIVERING
                or record.claimed_at is None
                or now - record.claimed_at < self._claim_ttl_seconds
            ):
                continue
            record.status = DeliveryStatus.PENDING
            record.claimant_id = None
            record.claimed_at = None
            reclaimed += 1
        if reclaimed:
            self._persist()
        return reclaimed

    def _owned_claim(self, task_id: str, claimant_id: str) -> DeliveryRecord | None:
        record = self._records.get(str(task_id))
        if (
            record is None
            or record.status is not DeliveryStatus.DELIVERING
            or record.claimant_id != claimant_id
        ):
            return None
        return record
