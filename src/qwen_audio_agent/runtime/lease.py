"""Cross-process Gateway instance lease compatible with the JS host contract."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

GATEWAY_LOCK_SCHEMA = "qwaudio.gateway-lock/v1"


class GatewayAlreadyRunningError(RuntimeError):
    code = "QWAUDIO_GATEWAY_ALREADY_RUNNING"

    def __init__(self, lease: dict[str, Any]) -> None:
        origin = f": {lease['origin']}" if lease.get("origin") else ""
        super().__init__(f"Gateway is already running{origin}")
        self.lease = lease


def gateway_lock_path(config_directory: Path) -> Path:
    return config_directory / "gateway.lock"


def read_gateway_lease(config_directory: Path) -> dict[str, Any] | None:
    try:
        lease = json.loads(gateway_lock_path(config_directory).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if lease.get("schema") != GATEWAY_LOCK_SCHEMA or not isinstance(lease.get("instanceId"), str):
        return None
    return lease


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump(value, file, separators=(",", ":"))
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def _replace(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{value['instanceId']}.tmp")
    try:
        _write_exclusive(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _same_lease(left: dict[str, Any] | None, right: dict[str, Any]) -> bool:
    return bool(
        left
        and left.get("instanceId") == right.get("instanceId")
        and left.get("pid") == right.get("pid")
    )


@dataclass(slots=True)
class GatewayLease:
    config_directory: Path
    value: dict[str, Any]
    now: Callable[[], datetime]
    released: bool = False

    @property
    def path(self) -> Path:
        return gateway_lock_path(self.config_directory)

    @property
    def instance_id(self) -> str:
        return str(self.value["instanceId"])

    def update(self, **fields: Any) -> bool:
        if self.released:
            return False
        current = read_gateway_lease(self.config_directory)
        if not _same_lease(current, self.value):
            return False
        self.value.update(
            fields,
            schema=GATEWAY_LOCK_SCHEMA,
            instanceId=self.value["instanceId"],
            pid=self.value["pid"],
            heartbeatAt=self.now().isoformat(),
        )
        _replace(self.path, self.value)
        return True

    def release(self) -> bool:
        if self.released:
            return False
        self.released = True
        current = read_gateway_lease(self.config_directory)
        if not _same_lease(current, self.value):
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            return False
        return True


def acquire_gateway_lease(
    config_directory: Path,
    *,
    pid: int = os.getpid(),
    owner: str = "gateway",
    instance_id: str | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    process_is_alive: Callable[[int], bool] = _process_is_alive,
) -> GatewayLease:
    config_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = gateway_lock_path(config_directory)
    instance_id = instance_id or str(uuid4())
    for _attempt in range(4):
        timestamp = now().isoformat()
        value = {
            "schema": GATEWAY_LOCK_SCHEMA,
            "instanceId": instance_id,
            "pid": pid,
            "owner": owner,
            "state": "starting",
            "origin": "",
            "startedAt": timestamp,
            "heartbeatAt": timestamp,
        }
        try:
            _write_exclusive(path, value)
            return GatewayLease(config_directory=config_directory, value=value, now=now)
        except FileExistsError:
            existing = read_gateway_lease(config_directory)
            if existing is not None and process_is_alive(int(existing.get("pid", 0))):
                raise GatewayAlreadyRunningError(existing) from None
            stale = path.with_name(f"{path.name}.stale.{instance_id}")
            try:
                os.replace(path, stale)
            except FileNotFoundError:
                continue
            stale.unlink(missing_ok=True)
    raise RuntimeError("unable to acquire Gateway instance lease")
