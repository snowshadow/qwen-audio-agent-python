"""Foreground voice safety window for background result presentation.

This state intentionally lives outside LangGraph.  Audio queue and playback
receipts are transport facts; a background result may only enter the realtime
session after the active foreground turn has fully released the duplex window.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioResponseContext:
    turn_id: str
    origin: str = "model"


class AnnouncementWindow:
    """Track whether foreground speech, reasoning, or audio blocks insertion."""

    def __init__(self) -> None:
        self.user_speaking = False
        self.active_turn_id = ""
        self.turn_pending = False
        self._audio_responses: dict[str, AudioResponseContext] = {}
        self._playing_responses: set[str] = set()

    def begin_turn(self, turn_id: str) -> None:
        self.user_speaking = True
        self.active_turn_id = str(turn_id or "")
        self.turn_pending = True

    def end_speech(self) -> None:
        self.user_speaking = False

    def response_done(
        self,
        *,
        turn_id: str,
        origin: str = "model",
        has_audio: bool = False,
        has_function_call: bool = False,
        suppressed: bool = False,
        failed: bool = False,
    ) -> None:
        """Record a provider response terminal event.

        A function call keeps the turn open for its follow-up response unless
        it was suppressed or failed.  Announcement responses never complete a
        user turn.
        """

        if (
            origin == "announcement"
            or str(turn_id or "") != self.active_turn_id
            or has_audio
            or (has_function_call and not suppressed and not failed)
        ):
            return
        self.turn_pending = False

    def queue_audio(
        self,
        response_id: str,
        *,
        turn_id: str = "",
        origin: str = "model",
    ) -> None:
        response_id = str(response_id or "")
        if not response_id:
            return
        self._audio_responses[response_id] = AudioResponseContext(
            turn_id=str(turn_id or ""),
            origin=origin,
        )

    def start_playback(self, response_id: str) -> None:
        response_id = str(response_id or "")
        if response_id:
            self._playing_responses.add(response_id)

    def finish_playback(
        self,
        response_id: str,
        *,
        has_function_call: bool = False,
    ) -> None:
        response_id = str(response_id or "")
        context = self._audio_responses.pop(response_id, None)
        self._playing_responses.discard(response_id)
        if (
            context is not None
            and context.origin != "announcement"
            and context.turn_id
            and context.turn_id == self.active_turn_id
            and not has_function_call
        ):
            self.turn_pending = False

    def interrupt(self) -> None:
        """Release foreground reasoning; queued audio still blocks until cleared."""

        self.turn_pending = False

    def reset(self) -> None:
        self.user_speaking = False
        self.active_turn_id = ""
        self.turn_pending = False
        self._audio_responses.clear()
        self._playing_responses.clear()

    @property
    def blocked(self) -> bool:
        return self.user_speaking or self.turn_pending or bool(self._audio_responses)

    @property
    def playing(self) -> bool:
        return bool(self._playing_responses)
