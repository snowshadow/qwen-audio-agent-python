"""Durable, idempotent repository for Gateway Task records."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any

from qwen_audio_agent.domain.task import TaskScope, persisted_task

MAX_TASK_NUMBER = 99_999


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


def _normalized_task_number(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 1
    return number if 1 <= number <= MAX_TASK_NUMBER else 1


@dataclass(frozen=True, slots=True)
class RepositoryAddResult[RecordT]:
    record: RecordT
    reused: bool


class TaskRepository[RecordT]:
    """Own records, the short-ID cursor, persistence, and submission idempotency.

    ``submission_key`` is scoped by ``(scope, owner_id)``.  It is indexed at
    insertion time under one lock, so concurrent duplicate submissions cannot
    create two backend runs.
    """

    def __init__(
        self,
        *,
        store: Any = None,
        serialize: Callable[[RecordT], Mapping[str, Any]] | None = None,
    ) -> None:
        self.store = store
        self.serialize = serialize or persisted_task
        self._records: dict[str, RecordT] = {}
        self._submission_index: dict[tuple[str, str, str], str] = {}
        self._lock = RLock()
        self.next_task_number = 1

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(tuple(self._records))

    def _submission_identity(self, record: RecordT) -> tuple[str, str, str] | None:
        key = str(_value(record, "submission_key", "submissionKey", "") or "").strip()
        if not key:
            return None
        scope = str(_value(record, "scope", default=TaskScope.USER) or TaskScope.USER)
        owner_id = str(_value(record, "owner_id", "ownerId", "") or "")
        return (scope, owner_id, key)

    def _remove_index(self, record: RecordT) -> None:
        identity = self._submission_identity(record)
        if identity is not None:
            self._submission_index.pop(identity, None)

    def _index(self, record: RecordT) -> None:
        identity = self._submission_identity(record)
        if identity is not None:
            self._submission_index[identity] = str(_value(record, "id", default=""))

    def load(self) -> list[RecordT]:
        saved = self.store.load() if callable(getattr(self.store, "load", None)) else []
        records = list(saved or [])
        cursor = None
        if isinstance(self.store, Mapping):
            cursor = self.store.get("next_task_number", self.store.get("nextTaskNumber"))
        elif self.store is not None:
            cursor = getattr(
                self.store,
                "next_task_number",
                getattr(self.store, "nextTaskNumber", getattr(self.store, "nextJobNumber", None)),
            )
        with self._lock:
            self.next_task_number = _normalized_task_number(cursor)
            self._records.clear()
            self._submission_index.clear()
            for record in records:
                record_id = str(_value(record, "id", default=""))
                if not record_id:
                    continue
                self._records[record_id] = record
                self._index(record)
            return list(self._records.values())

    def get(self, task_id: str) -> RecordT | None:
        with self._lock:
            return self._records.get(str(task_id))

    def values(self) -> tuple[RecordT, ...]:
        with self._lock:
            return tuple(self._records.values())

    def items(self) -> tuple[tuple[str, RecordT], ...]:
        with self._lock:
            return tuple(self._records.items())

    def set(self, task_id: str, record: RecordT) -> TaskRepository[RecordT]:
        normalized_id = str(task_id)
        with self._lock:
            previous = self._records.get(normalized_id)
            if previous is not None:
                self._remove_index(previous)
            self._records[normalized_id] = record
            self._index(record)
        return self

    def add(self, record: RecordT) -> RepositoryAddResult[RecordT]:
        record_id = str(_value(record, "id", default=""))
        if not record_id:
            raise ValueError("Task record requires an id")
        with self._lock:
            identity = self._submission_identity(record)
            if identity is not None:
                existing_id = self._submission_index.get(identity)
                if existing_id is not None and existing_id in self._records:
                    return RepositoryAddResult(self._records[existing_id], reused=True)
            existing = self._records.get(record_id)
            if existing is not None:
                return RepositoryAddResult(existing, reused=True)
            self._records[record_id] = record
            self._index(record)
            return RepositoryAddResult(record, reused=False)

    def get_by_submission(
        self,
        *,
        owner_id: str,
        submission_key: str,
        scope: TaskScope | str = TaskScope.USER,
    ) -> RecordT | None:
        identity = (str(scope), str(owner_id), str(submission_key).strip())
        with self._lock:
            task_id = self._submission_index.get(identity)
            return self._records.get(task_id) if task_id is not None else None

    def delete(self, task_id: str) -> bool:
        with self._lock:
            record = self._records.pop(str(task_id), None)
            if record is None:
                return False
            self._remove_index(record)
            return True

    def allocate_task_id(self) -> str:
        with self._lock:
            for _ in range(MAX_TASK_NUMBER):
                current = _normalized_task_number(self.next_task_number)
                self.next_task_number = 1 if current >= MAX_TASK_NUMBER else current + 1
                task_id = f"task_{current}"
                if task_id not in self._records:
                    return task_id
        raise RuntimeError("No available task IDs")

    def _serialized_records(self) -> list[Mapping[str, Any]]:
        with self._lock:
            return [self.serialize(record) for record in self._records.values()]

    def save(self) -> None:
        save = getattr(self.store, "save", None)
        if not callable(save):
            return
        with self._lock:
            records = self._serialized_records()
            state = {"nextTaskNumber": self.next_task_number}
        save(records, state)

    def save_deferred(self) -> None:
        with self._lock:
            records = self._serialized_records()
            state = {"nextTaskNumber": self.next_task_number}
        save_deferred = getattr(self.store, "save_deferred", None) or getattr(
            self.store, "saveDeferred", None
        )
        if callable(save_deferred):
            save_deferred(records, state)
            return
        save = getattr(self.store, "save", None)
        if callable(save):
            save(records, state)

    def replace_all(self, records: Iterable[RecordT]) -> None:
        with self._lock:
            self._records.clear()
            self._submission_index.clear()
            for record in records:
                record_id = str(_value(record, "id", default=""))
                if not record_id:
                    continue
                self._records[record_id] = record
                self._index(record)


__all__ = [
    "MAX_TASK_NUMBER",
    "RepositoryAddResult",
    "TaskRepository",
]
