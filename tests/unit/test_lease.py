import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from qwen_audio_agent.runtime.lease import (
    GATEWAY_LOCK_SCHEMA,
    GatewayAlreadyRunningError,
    acquire_gateway_lease,
    gateway_lock_path,
    read_gateway_lease,
)


def test_lease_updates_atomically_and_releases_only_its_own_file(tmp_path: Path) -> None:
    clock = [datetime(2026, 8, 28, tzinfo=UTC)]
    lease = acquire_gateway_lease(
        tmp_path,
        pid=os.getpid(),
        owner="desktop",
        instance_id="instance-a",
        now=lambda: clock[0],
    )
    assert oct(gateway_lock_path(tmp_path).stat().st_mode & 0o777) == "0o600"

    clock[0] += timedelta(seconds=15)
    assert lease.update(state="ready", origin="http://127.0.0.1:3210")
    stored = read_gateway_lease(tmp_path)
    assert stored == {
        "schema": GATEWAY_LOCK_SCHEMA,
        "instanceId": "instance-a",
        "pid": os.getpid(),
        "owner": "desktop",
        "state": "ready",
        "origin": "http://127.0.0.1:3210",
        "startedAt": "2026-08-28T00:00:00+00:00",
        "heartbeatAt": "2026-08-28T00:00:15+00:00",
    }
    assert lease.release()
    assert read_gateway_lease(tmp_path) is None


def test_live_lease_conflicts_and_stale_lease_is_replaced(tmp_path: Path) -> None:
    live = acquire_gateway_lease(tmp_path, pid=111, instance_id="live")
    with pytest.raises(GatewayAlreadyRunningError) as raised:
        acquire_gateway_lease(
            tmp_path,
            pid=222,
            instance_id="other",
            process_is_alive=lambda pid: pid == 111,
        )
    assert raised.value.lease["instanceId"] == "live"

    replacement = acquire_gateway_lease(
        tmp_path,
        pid=222,
        instance_id="replacement",
        process_is_alive=lambda _pid: False,
    )
    assert replacement.instance_id == "replacement"
    assert not live.release()
    assert replacement.release()


def test_invalid_lock_is_treated_as_stale(tmp_path: Path) -> None:
    gateway_lock_path(tmp_path).write_text("not-json", encoding="utf-8")

    lease = acquire_gateway_lease(tmp_path, instance_id="fresh")

    assert read_gateway_lease(tmp_path)["instanceId"] == "fresh"
    assert lease.release()
