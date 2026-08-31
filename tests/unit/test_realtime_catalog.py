from qwen_audio_agent.providers.realtime_catalog import (
    list_dashscope_realtime_model_profiles,
    realtime_health_metadata,
    resolve_dashscope_realtime_model_profile,
)


def test_dashscope_catalog_matches_the_shared_node_contract() -> None:
    profiles = list_dashscope_realtime_model_profiles()

    assert [profile["id"] for profile in profiles] == [
        "qwen3.5-omni-flash-realtime",
        "qwen3.5-omni-plus-realtime",
        "qwen-audio-3.0-realtime-plus",
        "qwen-audio-3.0-realtime-flash",
    ]
    assert profiles[0] == {
        "id": "qwen3.5-omni-flash-realtime",
        "label": "Qwen3.5 Omni Flash Realtime",
        "family": "omni",
        "sessionDefaults": {"voice": "Ethan", "turnDetection": {"type": "semantic_vad"}},
        "modelCapabilities": {
            "textInput": True,
            "audioInput": True,
            "imageInput": True,
            "videoInput": False,
            "textOutput": True,
            "audioOutput": True,
            "functionCalling": True,
        },
        "transportCapabilities": {
            "textInput": True,
            "audioInput": True,
            "imageInput": False,
            "observationInput": False,
            "nativeVideoInput": False,
        },
    }
    assert profiles[2]["sessionDefaults"] == {
        "voice": "longanqian",
        "turnDetection": {"type": "smart_turn"},
    }
    assert profiles[2]["modelCapabilities"]["imageInput"] is False


def test_unknown_dashscope_model_fails_closed() -> None:
    profile = resolve_dashscope_realtime_model_profile(
        "qwen3.5-omni-plus-realtime-future"
    )

    assert profile["family"] == "unknown"
    assert profile["sessionDefaults"] == {"voice": None, "turnDetection": None}
    assert not any(profile["modelCapabilities"].values())
    assert not any(profile["transportCapabilities"].values())


def test_health_metadata_matches_node_provider_descriptors() -> None:
    metadata = realtime_health_metadata(
        "dashscope",
        {
            "DASHSCOPE_API_KEY": "secret",
            "QWEN_AUDIO_REALTIME_BASE_URL": " wss://gateway.example/realtime?? ",
            "QWEN_AUDIO_REALTIME_MODEL": "qwen3.5-omni-plus-realtime",
            "QWEN_OMNI_REALTIME_VOICE": "EthanCustom",
            "SPEECH_TO_SPEECH_REALTIME_URL": "ws://127.0.0.1:8765/v1/realtime",
        },
    )

    assert metadata["realtimeLabel"] == "Qwen-Audio-Realtime"
    assert metadata["realtimeModel"] == "qwen3.5-omni-plus-realtime"
    assert metadata["realtimeModelProfile"]["family"] == "omni"
    assert metadata["realtimeInputSampleRate"] == 16_000
    assert metadata["realtimeConfigurationSignature"] == (
        "553907ed714d69cbbd12963a6374cf780f27e16179904c3917436911129c687f"
    )
    assert metadata["realtimeProviders"] == [
        {
            "key": "dashscope",
            "label": "Qwen-Audio-Realtime",
            "model": "qwen3.5-omni-plus-realtime",
            "realtimeModelIds": [
                "qwen3.5-omni-flash-realtime",
                "qwen3.5-omni-plus-realtime",
                "qwen-audio-3.0-realtime-plus",
                "qwen-audio-3.0-realtime-flash",
            ],
            "configured": True,
        },
        {
            "key": "speech-to-speech",
            "label": "Hugging Face Speech-to-Speech",
            "model": None,
            "realtimeModelIds": None,
            "configured": True,
        },
    ]


def test_speech_to_speech_health_metadata_and_signature_match_node() -> None:
    metadata = realtime_health_metadata(
        "speech-to-speech",
        {
            "SPEECH_TO_SPEECH_REALTIME_URL": " ws://localhost:8765/v1/realtime/// ",
            "SPEECH_TO_SPEECH_AUTH_TOKEN": "token",
        },
    )

    assert metadata == {
        "realtimeLabel": "Hugging Face Speech-to-Speech",
        "realtimeModel": None,
        "realtimeModelProfile": None,
        "realtimeModelCatalog": [],
        "realtimeInputSampleRate": 16_000,
        "realtimeConfigurationSignature": (
            "98a823dd640bfd641160c8a1a356dedf748987f339f9fcfdae2b0fcf32f078b4"
        ),
        "realtimeProviders": [
            {
                "key": "speech-to-speech",
                "label": "Hugging Face Speech-to-Speech",
                "model": None,
                "realtimeModelIds": None,
                "configured": True,
            }
        ],
    }
