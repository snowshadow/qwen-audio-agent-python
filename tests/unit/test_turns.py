from qwen_audio_agent.runtime.turns import RealtimeTurnState, TurnContext, TurnCorrelation


def make_state() -> RealtimeTurnState:
    return RealtimeTurnState(create_voice_turn_id=lambda generation: f"voice-{generation}")


def test_correlates_repeated_provider_events_with_one_voice_turn() -> None:
    turns = make_state()
    first = turns.begin_voice("item-1")
    turns.end_speech()
    second = turns.begin_voice("item-1")

    assert first.accepted and first.context == TurnContext("voice-1", 1)
    assert second.context == first.context
    assert turns.resolve_input("item-1") == first.context


def test_completed_provider_item_id_starts_a_new_turn_when_reused() -> None:
    turns = make_state()
    first = turns.begin_voice("item-reused").context
    turns.end_speech()
    turns.complete_input("item-reused")

    second = turns.begin_voice("item-reused").context

    assert first == TurnContext("voice-1", 1)
    assert second == TurnContext("voice-2", 2)


def test_manual_input_supersedes_voice_and_rejects_late_provider_speech() -> None:
    turns = make_state()
    active = turns.begin_voice("voice-item").context
    manual = turns.begin_manual("text-1")

    assert manual.context == TurnContext("text-1", 2)
    assert manual.superseded_voice_turn == active
    assert turns.correlation.is_invalid("voice-item")
    assert not turns.begin_voice("late-item").accepted
    assert turns.correlation.is_invalid("late-item")


def test_committed_generation_rejects_stale_events() -> None:
    turns = make_state()
    first = turns.begin_voice("item-1").context
    assert first is not None and turns.commit(first)
    turns.end_speech()
    second = turns.begin_voice("item-2").context
    assert second is not None and turns.commit(second)

    assert turns.is_stale(first)
    assert not turns.commit(first)
    assert turns.committed() == second


def test_interrupt_boundary_drops_late_events_without_touching_work() -> None:
    turns = make_state()
    active = turns.begin_voice("item-1").context
    assert active is not None

    turns.advance_boundary()

    assert turns.is_stale(active)


def test_correlation_is_bounded_and_keeps_duplicate_completion_invalid() -> None:
    correlation = TurnCorrelation(max_items=2)
    one = TurnContext("one", 1)
    two = TurnContext("two", 2)
    three = TurnContext("three", 3)
    correlation.remember("one", one)
    correlation.remember("two", two)
    correlation.invalidate("two")
    correlation.complete("two", three)
    correlation.remember("three", three)

    assert correlation.resolve("one", None) is None
    duplicate = correlation.complete("two", three)
    assert duplicate.context == two
    assert duplicate.invalid
    assert duplicate.duplicate
