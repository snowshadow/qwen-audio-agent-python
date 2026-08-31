"""Reference-counted, expiring microphone suspension for external hosts."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import Timer
from time import time
from typing import Any, Protocol

DEFAULT_INPUT_SUSPEND_TTL_MS = 15_000
MAX_INPUT_SUSPEND_TTL_MS = 300_000


class TimerHandle(Protocol):
    def cancel(self) -> Any: ...


class InputOwnerRequiredError(ValueError):
    code = "QWAUDIO_INPUT_OWNER_REQUIRED"


@dataclass(slots=True)
class _Holder:
    owner: str
    reason: str
    ttl_ms: int
    since: int
    expires_at: int
    timer: TimerHandle


def _now_ms() -> int:
    return int(time() * 1000)


def _set_timer(callback: Callable[[], None], delay_ms: int) -> Timer:
    timer = Timer(delay_ms / 1000, callback)
    timer.daemon = True
    timer.start()
    return timer


class InputArbitration:
    """Compose idempotent per-owner input holds with automatic expiry."""

    def __init__(
        self,
        *,
        default_ttl_ms: int = DEFAULT_INPUT_SUSPEND_TTL_MS,
        max_ttl_ms: int = MAX_INPUT_SUSPEND_TTL_MS,
        now: Callable[[], int] = _now_ms,
        set_timer: Callable[[Callable[[], None], int], TimerHandle] = _set_timer,
        clear_timer: Callable[[TimerHandle], Any] = lambda timer: timer.cancel(),
        logger: logging.Logger | None = None,
    ) -> None:
        self.default_ttl_ms = default_ttl_ms
        self.max_ttl_ms = max_ttl_ms
        self._now = now
        self._set_timer = set_timer
        self._clear_timer = clear_timer
        self._logger = logger or logging.getLogger(__name__)
        self._holders: dict[str, _Holder] = {}
        self._listeners: set[Callable[[dict[str, Any], dict[str, Any]], None]] = set()

    @property
    def suspended(self) -> bool:
        return bool(self._holders)

    def subscribe(
        self,
        listener: Callable[[dict[str, Any], dict[str, Any]], None],
    ) -> Callable[[], None]:
        self._listeners.add(listener)

        def unsubscribe() -> None:
            self._listeners.discard(listener)

        return unsubscribe

    def _notify(self, change: dict[str, Any]) -> None:
        status = self.status()
        for listener in tuple(self._listeners):
            try:
                listener(status, change)
            except Exception:
                self._logger.exception("input.arbitration_listener_failed")

    def _ttl_for(self, ttl_ms: Any) -> int:
        try:
            requested = float(ttl_ms)
        except (TypeError, ValueError):
            return self.default_ttl_ms
        if requested <= 0:
            return self.default_ttl_ms
        return min(self.max_ttl_ms, int(requested))

    def suspend(
        self,
        *,
        owner: str = "",
        reason: str = "",
        ttl_ms: Any = None,
    ) -> dict[str, Any]:
        holder_id = str(owner or "").strip()[:80]
        if not holder_id:
            raise InputOwnerRequiredError("input.suspend requires owner")
        was_suspended = self.suspended
        ttl = self._ttl_for(ttl_ms)
        existing = self._holders.get(holder_id)
        if existing is not None:
            self._clear_timer(existing.timer)
        now = self._now()
        expires_at = now + ttl
        self._holders[holder_id] = _Holder(
            owner=holder_id,
            reason=str(reason or "")[:200],
            ttl_ms=ttl,
            since=existing.since if existing is not None else now,
            expires_at=expires_at,
            timer=self._set_timer(lambda: self.expire(holder_id), ttl),
        )
        if not was_suspended:
            self._notify({"state": "suspended", "owner": holder_id, "reason": reason})
        return self.status()

    def resume(self, *, owner: str = "") -> dict[str, Any]:
        holder_id = str(owner or "").strip()[:80]
        existing = self._holders.pop(holder_id, None)
        if existing is None:
            return self.status()
        self._clear_timer(existing.timer)
        if not self.suspended:
            self._notify({"state": "resumed", "owner": holder_id})
        return self.status()

    def expire(self, owner: str) -> None:
        existing = self._holders.pop(owner, None)
        if existing is None:
            return
        if not self.suspended:
            self._notify({"state": "resumed", "owner": owner, "expired": True})

    def status(self) -> dict[str, Any]:
        holders = [
            {
                "owner": holder.owner,
                "reason": holder.reason,
                "ttlMs": holder.ttl_ms,
                "since": holder.since,
                "expiresAt": holder.expires_at,
            }
            for holder in self._holders.values()
        ]
        return {
            "suspended": bool(holders),
            "holders": holders,
            "owner": holders[0]["owner"] if holders else None,
            "reason": holders[0]["reason"] if holders else "",
            "expiresAt": max((holder["expiresAt"] for holder in holders), default=None),
        }

    def release_all(self) -> None:
        was_suspended = self.suspended
        for holder in self._holders.values():
            self._clear_timer(holder.timer)
        self._holders.clear()
        if was_suspended:
            self._notify({"state": "resumed", "owner": None})

    def close(self) -> None:
        for holder in self._holders.values():
            self._clear_timer(holder.timer)
        self._holders.clear()
        self._listeners.clear()
