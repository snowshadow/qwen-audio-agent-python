from collections.abc import Callable
from dataclasses import dataclass

import pytest

from qwen_audio_agent.runtime.input_arbitration import (
    DEFAULT_INPUT_SUSPEND_TTL_MS,
    MAX_INPUT_SUSPEND_TTL_MS,
    InputArbitration,
    InputOwnerRequiredError,
)


@dataclass
class FakeTimer:
    callback: Callable[[], None]
    fires_at: int

    def cancel(self) -> None:
        return None


class ControlledClock:
    def __init__(self) -> None:
        self.now = 1_000
        self.timers: list[FakeTimer] = []
        self.arbitration = InputArbitration(
            now=lambda: self.now,
            set_timer=self.set_timer,
            clear_timer=self.clear_timer,
        )

    def set_timer(self, callback: Callable[[], None], delay: int) -> FakeTimer:
        timer = FakeTimer(callback, self.now + delay)
        self.timers.append(timer)
        return timer

    def clear_timer(self, timer: FakeTimer) -> None:
        if timer in self.timers:
            self.timers.remove(timer)

    def advance(self, milliseconds: int) -> None:
        self.now += milliseconds
        due = [timer for timer in self.timers if timer.fires_at <= self.now]
        for timer in due:
            self.clear_timer(timer)
            timer.callback()


def test_suspend_is_idempotent_per_owner_and_last_release_wins() -> None:
    controls = ControlledClock()
    changes: list[tuple[bool, str]] = []
    controls.arbitration.subscribe(
        lambda status, change: changes.append((status["suspended"], change["state"]))
    )
    first = controls.arbitration.suspend(owner="dictation", reason="voice", ttl_ms=1_000)
    controls.advance(500)
    renewed = controls.arbitration.suspend(owner="dictation", ttl_ms=1_000)
    controls.arbitration.suspend(owner="overlay")

    assert renewed["holders"][0]["since"] == first["holders"][0]["since"]
    assert renewed["holders"][0]["expiresAt"] > first["holders"][0]["expiresAt"]
    assert len(controls.timers) == 2
    assert controls.arbitration.resume(owner="dictation")["suspended"]
    assert not controls.arbitration.resume(owner="overlay")["suspended"]
    assert changes == [(True, "suspended"), (False, "resumed")]


def test_hold_expires_and_ttl_is_bounded() -> None:
    controls = ControlledClock()
    expired: list[bool] = []
    controls.arbitration.subscribe(
        lambda _status, change: expired.append(bool(change.get("expired")))
    )
    controls.arbitration.suspend(owner="a", ttl_ms=0)
    controls.arbitration.suspend(owner="b", ttl_ms=10**20)
    status = controls.arbitration.status()

    assert status["holders"][0]["ttlMs"] == DEFAULT_INPUT_SUSPEND_TTL_MS
    assert status["holders"][1]["ttlMs"] == MAX_INPUT_SUSPEND_TTL_MS
    controls.arbitration.resume(owner="b")
    controls.advance(DEFAULT_INPUT_SUSPEND_TTL_MS)
    assert not controls.arbitration.suspended
    assert expired[-1]


def test_owner_is_required_and_release_all_keeps_subscribers() -> None:
    controls = ControlledClock()
    changes: list[str] = []
    controls.arbitration.subscribe(lambda _status, change: changes.append(change["state"]))

    with pytest.raises(InputOwnerRequiredError) as raised:
        controls.arbitration.suspend(reason="dictation")
    assert raised.value.code == "QWAUDIO_INPUT_OWNER_REQUIRED"

    controls.arbitration.suspend(owner="a")
    controls.arbitration.release_all()
    controls.arbitration.suspend(owner="a")
    assert changes == ["suspended", "resumed", "suspended"]


def test_listener_failure_does_not_break_other_listeners() -> None:
    controls = ControlledClock()
    seen: list[bool] = []

    def fail(_status: object, _change: object) -> None:
        raise RuntimeError("dispatch failed")

    controls.arbitration.subscribe(fail)
    controls.arbitration.subscribe(lambda status, _change: seen.append(status["suspended"]))

    controls.arbitration.suspend(owner="a")
    assert seen == [True]
