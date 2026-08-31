"""Realtime model provider contracts and graph-outside session lifecycle."""

from .openai_compatible import (
    DEFAULT_SPEECH_TO_SPEECH_REALTIME_URL,
    GARealtimeProtocol,
    SpeechToSpeechRealtimeProvider,
    build_speech_to_speech_session,
    classify_speech_to_speech_error,
)
from .realtime import (
    REALTIME_PROVIDER_METHODS,
    RealtimeProviderContractError,
    RealtimeProviderPort,
    RealtimeProviderSession,
    assert_realtime_provider,
)

__all__ = [
    "DEFAULT_SPEECH_TO_SPEECH_REALTIME_URL",
    "REALTIME_PROVIDER_METHODS",
    "GARealtimeProtocol",
    "RealtimeProviderContractError",
    "RealtimeProviderPort",
    "RealtimeProviderSession",
    "SpeechToSpeechRealtimeProvider",
    "assert_realtime_provider",
    "build_speech_to_speech_session",
    "classify_speech_to_speech_error",
]
