"""Fail-closed normalization shared by domain and public protocol projections."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from typing import Any

_REDACTED = "[REDACTED]"
_MAX_SAFE_INTEGER = 9_007_199_254_740_991
_AUTH_SECRET = re.compile(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_ASSIGNED_SECRET = re.compile(
    r"(?i)\b((?:[a-z0-9]+[_-])*(?:api[_-]?key|access[_-]?key|secret(?:[_-]?key)?|"
    r"token|password|passwd|credentials?)(?:[_-][a-z0-9]+)*)"
    r"(\s*(?:=|:)\s*[\"']?)([^\s\"',;&]+)"
)
_CLI_SECRET = re.compile(
    r"(?i)(--(?:[a-z0-9]+[_-])*(?:api[_-]?key|access[_-]?key|secret|token|password|"
    r"passwd|credentials?)(?:[_-][a-z0-9]+)*(?:=|\s+))"
    r"([^\s]+)"
)
_URL_SECRET = re.compile(r"(?i)([?&](?:api[_-]?key|access_token|secret|token|password)=)[^&\s]+")
_COOKIE_SECRET = re.compile(r"(?im)\b((?:set-)?cookie\s*:\s*)[^\r\n]+")
_URL_USERINFO = re.compile(r"(?i)\b((?:https?|wss?)://)[^/@\s]+@")
_KNOWN_TOKEN = re.compile(
    r"\b(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})\b"
)
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)


def _clean(value: Any, maximum: int) -> str:
    return str(value or "").replace("\x00", "").strip()[:maximum]


def redact_public_text(value: Any, maximum: int) -> str:
    """Redact common credentials, then apply the public text budget."""

    text = str(value or "").replace("\x00", "")
    text = _PRIVATE_KEY.sub(_REDACTED, text)
    text = _AUTH_SECRET.sub(_REDACTED, text)
    text = _ASSIGNED_SECRET.sub(rf"\1\2{_REDACTED}", text)
    text = _CLI_SECRET.sub(rf"\1{_REDACTED}", text)
    text = _URL_SECRET.sub(rf"\1{_REDACTED}", text)
    text = _COOKIE_SECRET.sub(rf"\1{_REDACTED}", text)
    text = _URL_USERINFO.sub(rf"\1{_REDACTED}@", text)
    return _KNOWN_TOKEN.sub(_REDACTED, text).strip()[:maximum]


def normalize_public_authorization_operation(value: Any) -> dict[str, Any] | None:
    """Project one permission operation through a closed, redacted schema."""

    if not isinstance(value, Mapping):
        return None
    title = redact_public_text(value.get("title"), 160)
    kind = redact_public_text(value.get("kind"), 80)
    if not title or not kind:
        return None
    operation: dict[str, Any] = {"title": title, "kind": kind}
    for key, maximum in (("description", 600), ("command", 1_200), ("path", 600)):
        normalized = redact_public_text(value.get(key), maximum)
        if normalized:
            operation[key] = normalized
    locations: list[dict[str, Any]] = []
    raw_locations = value.get("locations")
    for item in (raw_locations if isinstance(raw_locations, list | tuple) else [])[:16]:
        if not isinstance(item, Mapping):
            continue
        path = redact_public_text(item.get("path"), 600)
        if not path:
            continue
        location: dict[str, Any] = {"path": path}
        line = item.get("line")
        if isinstance(line, int) and not isinstance(line, bool) and 0 < line <= _MAX_SAFE_INTEGER:
            location["line"] = line
        locations.append(location)
    if locations:
        operation["locations"] = locations
    return operation


def _timestamp(value: Any, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value if 0 <= value <= _MAX_SAFE_INTEGER else fallback
    if isinstance(value, float):
        return int(value) if 0 <= value <= _MAX_SAFE_INTEGER else fallback
    if isinstance(value, str):
        normalized = value.strip()
        if re.fullmatch(r"\d{1,16}", normalized):
            parsed = int(normalized)
            return parsed if parsed <= _MAX_SAFE_INTEGER else fallback
    return fallback


def normalize_public_authorization(
    value: Any,
    *,
    task_id: str = "",
    timestamp: int | None = None,
    force_pending: bool = False,
) -> dict[str, Any] | None:
    """Bind, bound and redact one public permission record."""

    if not isinstance(value, Mapping):
        return None
    authorization_id = _clean(value.get("id"), 160)
    summary = " ".join(redact_public_text(value.get("summary"), 600).split())
    if not authorization_id or not summary:
        return None
    clock = time.time_ns() // 1_000_000
    now = _timestamp(timestamp, clock) if timestamp is not None else clock
    status = "pending" if force_pending else _clean(value.get("status"), 40).lower()
    if status not in {"pending", "approved", "denied", "cancelled"}:
        status = "pending"
    created_at = _timestamp(value.get("createdAt", value.get("created_at")), now)
    resolved_at = None
    if status != "pending":
        resolved_at = _timestamp(value.get("resolvedAt", value.get("resolved_at")), now)
    raw_patterns = value.get("patterns")
    patterns = [
        normalized
        for item in (raw_patterns if isinstance(raw_patterns, list | tuple) else [])[:32]
        if (normalized := redact_public_text(item, 300))
    ]
    bound_task_id = _clean(task_id or value.get("taskId") or value.get("task_id"), 160)
    return {
        "id": authorization_id,
        "taskId": bound_task_id or None,
        "status": status,
        "category": redact_public_text(value.get("category"), 80) or "unknown",
        "summary": summary,
        "patterns": patterns,
        "approvalScope": "once",
        "operation": normalize_public_authorization_operation(value.get("operation")),
        "createdAt": created_at,
        "resolvedAt": resolved_at,
    }


__all__ = [
    "normalize_public_authorization",
    "normalize_public_authorization_operation",
    "redact_public_text",
]
