"""Bounded persistent frontend conversation projection."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)


class ConversationStore:
    def __init__(
        self,
        path: str | Path,
        *,
        max_sessions: int = 100,
        max_messages_per_session: int = 200,
    ) -> None:
        self.path = Path(path)
        self.max_sessions = max_sessions
        self.max_messages_per_session = max_messages_per_session
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {"version": 1, "sessions": {}}
        if not isinstance(value, dict) or value.get("version") != 1:
            return {"version": 1, "sessions": {}}
        if not isinstance(value.get("sessions"), dict):
            value["sessions"] = {}
        return value

    @staticmethod
    def _key(owner_id: str, session_id: str) -> str:
        return f"{owner_id}\x00{session_id}"

    def messages(self, *, owner_id: str, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            entry = self._data["sessions"].get(self._key(owner_id, session_id), {})
            messages = entry.get("messages") if isinstance(entry, Mapping) else []
            if not isinstance(messages, list):
                return []
            return [
                dict(value)
                for value in messages[-max(0, min(limit, 200)) :]
                if isinstance(value, Mapping)
            ]

    def record(
        self,
        *,
        owner_id: str,
        session_id: str,
        message: Mapping[str, Any],
    ) -> dict[str, Any]:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "").replace("\x00", "").strip()[:100_000]
        if role not in {"user", "assistant"} or not content:
            raise ValueError("conversation message requires user/assistant role and content")
        now = int(time.time() * 1_000)
        public = {
            "id": str(message.get("id") or f"message_{now}"),
            "role": role,
            "content": content,
            "source": str(message.get("source") or "conversation-history"),
            "createdAt": int(message.get("createdAt") or now),
        }
        for key in ("turnId", "responseId", "taskId", "taskIds", "citations", "inputs"):
            if key in message:
                public[key] = message[key]
        with self._lock:
            sessions = self._data["sessions"]
            key = self._key(owner_id, session_id)
            entry = sessions.setdefault(key, {"updatedAt": now, "messages": []})
            values = entry.setdefault("messages", [])
            existing = next(
                (index for index, value in enumerate(values) if value.get("id") == public["id"]),
                None,
            )
            if existing is None:
                values.append(public)
            else:
                values[existing] = public
            entry["updatedAt"] = now
            entry["messages"] = values[-self.max_messages_per_session :]
            if len(sessions) > self.max_sessions:
                oldest = sorted(
                    sessions,
                    key=lambda item: int(sessions[item].get("updatedAt") or 0),
                )[: len(sessions) - self.max_sessions]
                for stale in oldest:
                    del sessions[stale]
            _atomic_json(self.path, self._data)
        return dict(public)


__all__ = ["ConversationStore"]
