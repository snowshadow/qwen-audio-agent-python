from qwen_audio_agent.runtime.announcement_window import AnnouncementWindow


def test_blocks_through_foreground_reasoning_and_playback() -> None:
    window = AnnouncementWindow()

    window.begin_turn("turn-1")
    assert window.blocked
    window.end_speech()
    assert window.blocked

    window.queue_audio("response-1", turn_id="turn-1")
    window.response_done(turn_id="turn-1", has_audio=True)
    window.start_playback("response-1")
    assert window.blocked
    assert window.playing

    window.finish_playback("response-1")
    assert not window.blocked
    assert not window.playing


def test_tool_call_keeps_window_closed_until_follow_up() -> None:
    window = AnnouncementWindow()
    window.begin_turn("turn-tool")
    window.end_speech()

    window.response_done(turn_id="turn-tool", has_function_call=True)
    assert window.blocked

    window.response_done(turn_id="turn-tool", origin="agent")
    assert not window.blocked


def test_spoken_tool_ack_does_not_complete_turn() -> None:
    window = AnnouncementWindow()
    window.begin_turn("turn-tool")
    window.end_speech()
    window.queue_audio("ack", turn_id="turn-tool")
    window.response_done(
        turn_id="turn-tool",
        has_audio=True,
        has_function_call=True,
    )

    window.finish_playback("ack", has_function_call=True)
    assert window.blocked

    window.queue_audio("follow-up", turn_id="turn-tool", origin="agent")
    window.finish_playback("follow-up")
    assert not window.blocked


def test_announcement_never_completes_active_user_turn() -> None:
    window = AnnouncementWindow()
    window.begin_turn("turn-user")
    window.end_speech()

    window.response_done(
        turn_id="turn-background",
        origin="announcement",
    )

    assert window.blocked


def test_old_playback_cannot_release_a_new_turn() -> None:
    window = AnnouncementWindow()
    window.begin_turn("turn-old")
    window.end_speech()
    window.queue_audio("response-old", turn_id="turn-old")

    window.begin_turn("turn-new")
    window.finish_playback("response-old")
    assert window.blocked

    window.end_speech()
    window.response_done(turn_id="turn-new")
    assert not window.blocked


def test_interrupt_does_not_ignore_still_queued_audio() -> None:
    window = AnnouncementWindow()
    window.begin_turn("turn-1")
    window.end_speech()
    window.queue_audio("response-1", turn_id="turn-1")

    window.interrupt()
    assert window.blocked

    window.finish_playback("response-1")
    assert not window.blocked


def test_late_and_duplicate_playback_receipts_are_harmless() -> None:
    window = AnnouncementWindow()
    window.begin_turn("turn-1")
    window.end_speech()

    window.finish_playback("missing")
    assert window.blocked

    window.queue_audio("response-1", turn_id="turn-1")
    window.finish_playback("response-1")
    window.finish_playback("response-1")
    assert not window.blocked
