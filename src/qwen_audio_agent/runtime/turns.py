"""Realtime turn correlation and stale-event fencing.

This state intentionally stays outside LangGraph: provider audio events and
barge-in live on a latency-sensitive connection timeline, while graph
interrupts represent durable workflow pauses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import time


@dataclass(frozen=True, slots=True)
class TurnContext:
    turn_id: str
    turn_generation: int


@dataclass(frozen=True, slots=True)
class CompletedInput:
    context: TurnContext
    invalid: bool
    duplicate: bool = False


class TurnCorrelation:
    """Bound provider item IDs to their original Gateway turn."""

    def __init__(self, *, max_items: int = 100) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        self.max_items = max_items
        self._turns: dict[str, TurnContext] = {}
        self._invalid_items: set[str] = set()
        self._completed_items: set[str] = set()

    def remember(
        self,
        item_id: str,
        context: TurnContext,
        *,
        replace: bool = False,
    ) -> TurnContext:
        if not item_id:
            return context
        existing = self._turns.get(item_id)
        if existing is not None and not replace:
            return existing
        if replace:
            self._invalid_items.discard(item_id)
            self._completed_items.discard(item_id)
        self._turns[item_id] = context
        while len(self._turns) > self.max_items:
            oldest = next(iter(self._turns))
            del self._turns[oldest]
            self._invalid_items.discard(oldest)
            self._completed_items.discard(oldest)
        return context

    def resolve(self, item_id: str, fallback: TurnContext | None) -> TurnContext | None:
        return self._turns.get(item_id, fallback)

    def invalidate(self, item_id: str) -> None:
        if item_id:
            self._invalid_items.add(item_id)

    def invalidate_before_generation(self, generation: int) -> None:
        for item_id, context in self._turns.items():
            if context.turn_generation < generation:
                self._invalid_items.add(item_id)

    def is_invalid(self, item_id: str) -> bool:
        return item_id in self._invalid_items

    def is_complete(self, item_id: str) -> bool:
        return item_id in self._completed_items

    def complete(self, item_id: str, fallback: TurnContext) -> CompletedInput:
        context = self.resolve(item_id, fallback)
        if context is None:
            context = fallback
        duplicate = item_id in self._completed_items
        if item_id:
            self._completed_items.add(item_id)
        return CompletedInput(
            context=context,
            invalid=item_id in self._invalid_items,
            duplicate=duplicate,
        )

    def clear(self) -> None:
        self._turns.clear()
        self._invalid_items.clear()
        self._completed_items.clear()


@dataclass(frozen=True, slots=True)
class BeginVoiceResult:
    accepted: bool
    context: TurnContext | None


@dataclass(frozen=True, slots=True)
class BeginManualResult:
    context: TurnContext
    superseded_voice_turn: TurnContext | None


def _default_voice_turn_id(generation: int) -> str:
    return f"voice-{int(time() * 1000)}-{generation}"


class RealtimeTurnState:
    """Own mutable turn state for one realtime client connection."""

    def __init__(
        self,
        *,
        correlation: TurnCorrelation | None = None,
        create_voice_turn_id: Callable[[int], str] = _default_voice_turn_id,
    ) -> None:
        self.correlation = correlation or TurnCorrelation()
        self.create_voice_turn_id = create_voice_turn_id
        self.turn_id = ""
        self.turn_generation = 0
        self.turn_sequence = 0
        self.committed_turn_id = ""
        self.committed_turn_generation = 0
        self.user_speaking = False
        self.voice_start_item_id = ""
        self.manual_input_generation: int | None = None

    def current(self) -> TurnContext:
        return TurnContext(self.turn_id, self.turn_generation)

    def committed(self) -> TurnContext:
        return TurnContext(self.committed_turn_id, self.committed_turn_generation)

    def begin_voice(self, item_id: str) -> BeginVoiceResult:
        if self.manual_input_generation is not None:
            self.correlation.invalidate(item_id)
            return BeginVoiceResult(accepted=False, context=None)

        self.user_speaking = True
        known = (
            self.correlation.resolve(item_id, None)
            if item_id and not self.correlation.is_complete(item_id)
            else None
        )
        if known is not None:
            self.turn_id = known.turn_id
            self.turn_generation = known.turn_generation
        else:
            self.turn_sequence += 1
            self.turn_generation = self.turn_sequence
            self.turn_id = self.create_voice_turn_id(self.turn_generation)
            self.correlation.remember(item_id, self.current(), replace=True)
        self.voice_start_item_id = item_id
        return BeginVoiceResult(accepted=True, context=self.current())

    def end_speech(self) -> None:
        self.user_speaking = False

    def begin_manual(self, turn_id: str) -> BeginManualResult:
        superseded = self.current() if self.user_speaking else None
        self.user_speaking = False
        self.voice_start_item_id = ""
        self.turn_sequence += 1
        self.turn_generation = self.turn_sequence
        self.turn_id = turn_id
        self.manual_input_generation = self.turn_generation
        self.correlation.invalidate_before_generation(self.turn_generation)
        context = self.current()
        self.commit(context)
        return BeginManualResult(context=context, superseded_voice_turn=superseded)

    def resolve_input(self, item_id: str) -> TurnContext:
        return self.correlation.resolve(item_id, self.current()) or self.current()

    def complete_input(self, item_id: str) -> CompletedInput:
        completed = self.correlation.complete(item_id, self.current())
        if (
            self.voice_start_item_id
            and self.voice_start_item_id != item_id
            and completed.context.turn_generation == self.turn_generation
        ):
            self.correlation.complete(self.voice_start_item_id, completed.context)
        if completed.context.turn_generation == self.turn_generation:
            self.voice_start_item_id = ""
        return completed

    def is_stale(self, context: TurnContext) -> bool:
        return context.turn_generation < self.committed_turn_generation

    def commit(self, context: TurnContext) -> bool:
        if not context.turn_id:
            return False
        if self.committed() == context or self.is_stale(context):
            return False
        self.committed_turn_id = context.turn_id
        self.committed_turn_generation = context.turn_generation
        return True

    def finish_manual_response(self, context: TurnContext, *, automatic: bool = False) -> bool:
        if (
            self.manual_input_generation is None
            or automatic
            or context.turn_generation != self.manual_input_generation
        ):
            return False
        self.manual_input_generation = None
        return True

    def fail_manual_input(self, context: TurnContext) -> bool:
        if context.turn_generation != self.manual_input_generation:
            return False
        self.manual_input_generation = None
        return True

    def advance_boundary(self) -> TurnContext:
        self.turn_sequence += 1
        self.turn_generation = self.turn_sequence
        self.committed_turn_generation = self.turn_generation
        return self.current()

    def close(self) -> None:
        self.advance_boundary()
        self.voice_start_item_id = ""
        self.correlation.clear()
