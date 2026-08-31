"""Small owner-scoped stores for the realtime frontend's local tools."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SENSITIVE = re.compile(
    r"(?:pass(?:word)?|secret|api[_ -]?key|access[_ -]?token|credential|"
    r"验证码|密码|密钥|令牌|\bsk-[a-z0-9_-]+)",
    re.IGNORECASE,
)
_MAX_LISTS = 20
_MAX_ITEMS = 100


def _digest(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def _clean(value: Any, maximum: int) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())[:maximum]


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)


class MarkdownMemoryService:
    """Atomic USER.md/MEMORY.md operations with exact replace semantics."""

    def __init__(
        self,
        *,
        user_path: str | Path,
        memory_path: str | Path,
        personal_owner_id: str = "user_personal",
        user_max_chars: int = 6_000,
        memory_max_chars: int = 8_000,
    ) -> None:
        self.paths = {"user": Path(user_path), "memory": Path(memory_path)}
        self.templates = {"user": "# USER", "memory": "# MEMORY"}
        self.limits = {"user": user_max_chars, "memory": memory_max_chars}
        self.personal_owner_id = personal_owner_id
        self._lock = threading.RLock()
        for document in ("user", "memory"):
            path = self.paths[document]
            if not path.exists():
                _atomic_text(path, f"{self.templates[document]}\n")

    def _path(self, owner_id: str, document: str) -> Path:
        base = self.paths[document]
        owner = str(owner_id or self.personal_owner_id)
        if owner in {"personal", self.personal_owner_id}:
            return base
        return base.parent / "users" / _digest(owner) / base.name

    def read_document(self, owner_id: str, document: str) -> str:
        if document not in self.paths:
            raise ValueError("unsupported memory document")
        path = self._path(owner_id, document)
        try:
            return path.read_text(encoding="utf-8").replace("\x00", "").strip()
        except FileNotFoundError:
            return self.templates[document]

    def read(self, owner_id: str, document: str = "all") -> dict[str, Any]:
        if document == "all":
            names = ("user", "memory")
        elif document in self.paths:
            names = (document,)
        else:
            raise ValueError("read document must be all, user, or memory")
        return {
            "status": "ok",
            "documents": [
                {
                    "id": f"{name}_document",
                    "document": name,
                    "format": "markdown",
                    "content": self.read_document(owner_id, name),
                }
                for name in names
            ],
        }

    def apply(self, owner_id: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        action = str(arguments.get("action") or "").strip()
        document = str(arguments.get("document") or ("all" if action == "read" else ""))
        if action == "read":
            return self.read(owner_id, document)
        if document not in self.paths:
            return self._failure("invalid_memory_document", "写入必须指定 user 或 memory。")
        proposed = str(arguments.get("content") or arguments.get("new_text") or "")
        if _SENSITIVE.search(proposed):
            return self._failure("sensitive_memory", "不能把密码、密钥或令牌保存到记忆中。")
        with self._lock:
            current = self.read_document(owner_id, document)
            if action == "append":
                addition = str(arguments.get("content") or "").replace("\x00", "").strip()
                if not addition:
                    return self._failure("invalid_memory_edit", "追加内容不能为空。")
                next_value = f"{current.rstrip()}\n\n{addition}".strip()
            elif action == "replace":
                old = str(arguments.get("old_text") or "").replace("\x00", "")
                new = str(arguments.get("new_text") or "").replace("\x00", "")
                if not old:
                    return self._failure("invalid_memory_edit", "精确替换必须提供 old_text。")
                matches = current.count(old)
                if matches != 1:
                    return self._failure(
                        "ambiguous_edit" if matches else "edit_not_found",
                        "old_text 必须在当前文档中恰好出现一次。",
                    )
                next_value = current.replace(old, new)
            else:
                return self._failure("invalid_memory_action", "不支持这个记忆操作。")
            next_value = self._normalize_markdown(next_value)
            if len(next_value) > self.limits[document]:
                return self._failure("document_too_large", "这份记忆文档已经达到容量上限。")
            _atomic_text(self._path(owner_id, document), f"{next_value}\n")
        return {
            "status": "ok",
            "changed": 1,
            "document": document,
            "content": next_value,
            "revision": _digest(next_value),
        }

    @staticmethod
    def _normalize_markdown(content: str) -> str:
        seen: set[str] = set()
        lines: list[str] = []
        for line in content.replace("\r\n", "\n").replace("\r", "\n").splitlines():
            match = re.match(r"^\s*[-*+]\s*(.*?)\s*$", line)
            if match:
                item = " ".join(match.group(1).split())
                if not item or item in seen:
                    continue
                seen.add(item)
            lines.append(line.rstrip())
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    @staticmethod
    def _failure(code: str, message: str) -> dict[str, Any]:
        return {"status": "failed", "error": True, "error_code": code, "user_message": message}


class JsonNotesStore:
    """Bounded, JSON-compatible owner-scoped named lists."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {"version": 1, "owners": {}}
        if not isinstance(value, dict) or value.get("version") != 1:
            return {"version": 1, "owners": {}}
        if not isinstance(value.get("owners"), dict):
            value["owners"] = {}
        return value

    def _save(self) -> None:
        _atomic_text(
            self.path,
            json.dumps(self._data, ensure_ascii=False, separators=(",", ":")),
        )

    def _owner(self, owner_id: str) -> dict[str, Any]:
        owners = self._data.setdefault("owners", {})
        return owners.setdefault(str(owner_id), {})

    @staticmethod
    def _resolve(lists: Mapping[str, Any], name: str) -> tuple[str | None, list[str]]:
        needle = _clean(name, 30).casefold()
        if not needle:
            return None, []
        if needle in lists:
            return needle, []
        matches = [key for key in lists if needle in key.casefold() or key.casefold() in needle]
        if len(matches) == 1:
            return matches[0], []
        candidates = [str(lists[key].get("name") or key) for key in (matches or list(lists))[:10]]
        return None, candidates

    def execute(self, owner_id: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        action = str(arguments.get("action") or "")
        name = _clean(arguments.get("list"), 30)
        raw_items = arguments.get("items")
        items = [
            _clean(value, 100)
            for value in (raw_items if isinstance(raw_items, list) else [])[:20]
            if _clean(value, 100)
        ]
        if any(_SENSITIVE.search(value) for value in [name, *items]):
            return self._failure("sensitive_note", "不能把密码、密钥或令牌保存到清单中。")
        with self._lock:
            lists = self._owner(owner_id)
            if action == "lists":
                return {
                    "status": "ok",
                    "lists": [
                        {
                            "list": value.get("name") or key,
                            "count": len(value.get("items") or []),
                            "updated_at": value.get("updatedAt"),
                        }
                        for key, value in lists.items()
                    ],
                }
            if not name:
                return self._failure("missing_list", "这个操作需要清单名称。")
            key, candidates = self._resolve(lists, name)
            if action == "add":
                if not items:
                    return self._failure("missing_items", "没有提供要添加的条目。")
                key = key or name.casefold()
                if key not in lists and len(lists) >= _MAX_LISTS:
                    return self._failure("notes_limit", "清单数量已经达到上限。")
                now = int(time.time() * 1_000)
                entry = lists.setdefault(
                    key,
                    {"name": name, "items": [], "createdAt": now, "updatedAt": now},
                )
                existing = {str(item.get("text") or "").casefold() for item in entry["items"]}
                added = []
                for text in items:
                    if text.casefold() in existing or len(entry["items"]) >= _MAX_ITEMS:
                        continue
                    item = {"id": f"item_{_digest(text, 12)}", "text": text, "addedAt": now}
                    entry["items"].append(item)
                    existing.add(text.casefold())
                    added.append(item)
                entry["updatedAt"] = now
                self._save()
                return {"status": "ok", "list": entry["name"], "added": added}
            if key is None:
                return {
                    "status": "ambiguous" if candidates else "not_found",
                    "error": True,
                    "error_code": "list_ambiguous" if candidates else "list_not_found",
                    "candidates": candidates,
                }
            entry = lists[key]
            if action == "show":
                return {"status": "ok", "list": entry["name"], "items": list(entry["items"])}
            if action == "remove":
                if not items:
                    return self._failure("missing_items", "没有提供要移除的条目。")
                removed: list[dict[str, Any]] = []
                unresolved: list[dict[str, Any]] = []
                for requested in items:
                    lowered = requested.casefold()
                    exact = [item for item in entry["items"] if item["text"].casefold() == lowered]
                    matches = exact or [
                        item
                        for item in entry["items"]
                        if lowered in item["text"].casefold() or item["text"].casefold() in lowered
                    ]
                    if len(matches) == 1:
                        entry["items"].remove(matches[0])
                        removed.append(matches[0])
                    else:
                        unresolved.append(
                            {
                                "item": requested,
                                "candidates": [item["text"] for item in matches[:10]],
                            }
                        )
                entry["updatedAt"] = int(time.time() * 1_000)
                self._save()
                return {
                    "status": "ok" if not unresolved else "ambiguous",
                    "list": entry["name"],
                    "removed": removed,
                    "unresolved": unresolved,
                }
            if action == "clear":
                entry["items"] = []
                entry["updatedAt"] = int(time.time() * 1_000)
                self._save()
                return {"status": "ok", "list": entry["name"], "cleared": True}
            if action == "drop":
                dropped = entry["name"]
                del lists[key]
                self._save()
                return {"status": "ok", "list": dropped, "dropped": True}
            return self._failure("invalid_notes_action", "不支持这个清单操作。")

    @staticmethod
    def _failure(code: str, message: str) -> dict[str, Any]:
        return {"status": "failed", "error": True, "error_code": code, "user_message": message}


__all__ = ["JsonNotesStore", "MarkdownMemoryService"]
