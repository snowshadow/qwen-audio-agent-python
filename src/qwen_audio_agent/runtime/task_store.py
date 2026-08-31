"""Atomic JSON Task ledger; LangGraph checkpoints are not the product ledger."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_audio_agent.domain.task import Task, task_from_persisted

TASK_STORE_SCHEMA = "qwaudio.tasks/v1"


class JsonTaskStore:
    """Persist the bounded Task projection and short-ID cursor atomically."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.next_task_number = 1

    def load(self) -> list[Task]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"unable to read Task store: {self.path}") from error
        if not isinstance(value, Mapping) or value.get("schema") != TASK_STORE_SCHEMA:
            raise RuntimeError(f"unsupported Task store: {self.path}")
        self.next_task_number = int(value.get("nextTaskNumber") or 1)
        tasks: list[Task] = []
        for raw in value.get("tasks") or []:
            if not isinstance(raw, Mapping):
                continue
            try:
                tasks.append(task_from_persisted(raw))
            except (TypeError, ValueError):
                continue
        return tasks

    def save(
        self,
        records: Sequence[Mapping[str, Any]],
        state: Mapping[str, Any],
    ) -> None:
        self.next_task_number = int(state.get("nextTaskNumber") or 1)
        payload = {
            "schema": TASK_STORE_SCHEMA,
            "nextTaskNumber": self.next_task_number,
            "tasks": list(records),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def save_deferred(
        self,
        records: Sequence[Mapping[str, Any]],
        state: Mapping[str, Any],
    ) -> None:
        self.save(records, state)
