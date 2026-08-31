"""Public Realtime model metadata kept compatible with the Node Gateway."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

DEFAULT_DASHSCOPE_REALTIME_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
DEFAULT_DASHSCOPE_REALTIME_MODEL = "qwen-audio-3.0-realtime-plus"
DEFAULT_DASHSCOPE_REALTIME_VOICE = "longanqian"
DEFAULT_SPEECH_TO_SPEECH_REALTIME_URL = "ws://127.0.0.1:8765/v1/realtime"
DASHSCOPE_AUDIO_FLASH_REALTIME_MODEL = "qwen-audio-3.0-realtime-flash"
DASHSCOPE_OMNI_FLASH_REALTIME_MODEL = "qwen3.5-omni-flash-realtime"
DASHSCOPE_OMNI_PLUS_REALTIME_MODEL = "qwen3.5-omni-plus-realtime"

DASHSCOPE_AUDIO_REALTIME_MODELS = frozenset(
    {DEFAULT_DASHSCOPE_REALTIME_MODEL, DASHSCOPE_AUDIO_FLASH_REALTIME_MODEL}
)
DASHSCOPE_OMNI_REALTIME_MODELS = frozenset(
    {DASHSCOPE_OMNI_FLASH_REALTIME_MODEL, DASHSCOPE_OMNI_PLUS_REALTIME_MODEL}
)

_OMNI_MODEL_CAPABILITIES = {
    "textInput": True,
    "audioInput": True,
    "imageInput": True,
    "videoInput": False,
    "textOutput": True,
    "audioOutput": True,
    "functionCalling": True,
}
_OMNI_TRANSPORT_CAPABILITIES = {
    "textInput": True,
    "audioInput": True,
    "imageInput": False,
    "observationInput": False,
    "nativeVideoInput": False,
}
_AUDIO_MODEL_CAPABILITIES = {
    "textInput": True,
    "audioInput": True,
    "imageInput": False,
    "videoInput": False,
    "textOutput": True,
    "audioOutput": True,
    "functionCalling": True,
}
_AUDIO_TRANSPORT_CAPABILITIES = {
    "textInput": True,
    "audioInput": True,
    "imageInput": False,
    "observationInput": False,
    "nativeVideoInput": False,
}
_UNKNOWN_MODEL_CAPABILITIES = {
    "textInput": False,
    "audioInput": False,
    "imageInput": False,
    "videoInput": False,
    "textOutput": False,
    "audioOutput": False,
    "functionCalling": False,
}
_UNKNOWN_TRANSPORT_CAPABILITIES = {
    "textInput": False,
    "audioInput": False,
    "imageInput": False,
    "observationInput": False,
    "nativeVideoInput": False,
}

_DASHSCOPE_REALTIME_MODEL_PROFILES = (
    {
        "id": DASHSCOPE_OMNI_FLASH_REALTIME_MODEL,
        "label": "Qwen3.5 Omni Flash Realtime",
        "family": "omni",
        "sessionDefaults": {"voice": "Ethan", "turnDetection": {"type": "semantic_vad"}},
        "modelCapabilities": _OMNI_MODEL_CAPABILITIES,
        "transportCapabilities": _OMNI_TRANSPORT_CAPABILITIES,
    },
    {
        "id": DASHSCOPE_OMNI_PLUS_REALTIME_MODEL,
        "label": "Qwen3.5 Omni Plus Realtime",
        "family": "omni",
        "sessionDefaults": {"voice": "Ethan", "turnDetection": {"type": "semantic_vad"}},
        "modelCapabilities": _OMNI_MODEL_CAPABILITIES,
        "transportCapabilities": _OMNI_TRANSPORT_CAPABILITIES,
    },
    {
        "id": DEFAULT_DASHSCOPE_REALTIME_MODEL,
        "label": "Qwen Audio 3.0 Realtime Plus",
        "family": "audio",
        "sessionDefaults": {
            "voice": DEFAULT_DASHSCOPE_REALTIME_VOICE,
            "turnDetection": {"type": "smart_turn"},
        },
        "modelCapabilities": _AUDIO_MODEL_CAPABILITIES,
        "transportCapabilities": _AUDIO_TRANSPORT_CAPABILITIES,
    },
    {
        "id": DASHSCOPE_AUDIO_FLASH_REALTIME_MODEL,
        "label": "Qwen Audio 3.0 Realtime Flash",
        "family": "audio",
        "sessionDefaults": {
            "voice": DEFAULT_DASHSCOPE_REALTIME_VOICE,
            "turnDetection": {"type": "smart_turn"},
        },
        "modelCapabilities": _AUDIO_MODEL_CAPABILITIES,
        "transportCapabilities": _AUDIO_TRANSPORT_CAPABILITIES,
    },
)


def list_dashscope_realtime_model_profiles() -> list[dict[str, Any]]:
    """Return the public catalog in the product order used by existing clients."""

    return deepcopy(list(_DASHSCOPE_REALTIME_MODEL_PROFILES))


def resolve_dashscope_realtime_model_profile(model: str = "") -> dict[str, Any]:
    """Resolve an exact model ID, failing closed for unknown IDs."""

    model_id = str(model or "").strip() or DEFAULT_DASHSCOPE_REALTIME_MODEL
    for profile in _DASHSCOPE_REALTIME_MODEL_PROFILES:
        if profile["id"] == model_id:
            return deepcopy(profile)
    return {
        "id": model_id,
        "label": model_id,
        "family": "unknown",
        "sessionDefaults": {"voice": None, "turnDetection": None},
        "modelCapabilities": deepcopy(_UNKNOWN_MODEL_CAPABILITIES),
        "transportCapabilities": deepcopy(_UNKNOWN_TRANSPORT_CAPABILITIES),
    }


def realtime_health_metadata(
    provider: str,
    env: Mapping[str, str],
) -> dict[str, Any]:
    """Build the model/provider fields published by the legacy Node health route."""

    model = str(env.get("QWEN_AUDIO_REALTIME_MODEL") or "").strip()
    model = model or DEFAULT_DASHSCOPE_REALTIME_MODEL
    model_catalog = list_dashscope_realtime_model_profiles()
    dashscope_api_key = str(
        env.get("QWEN_AUDIO_REALTIME_API_KEY") or env.get("DASHSCOPE_API_KEY") or ""
    ).strip()
    dashscope_configured = bool(dashscope_api_key)
    workspace_id = str(env.get("DASHSCOPE_WORKSPACE_ID") or "")
    workspace_url = (
        f"wss://{workspace_id}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime"
        if workspace_id
        else ""
    )
    dashscope_url = str(
        env.get("QWEN_AUDIO_REALTIME_BASE_URL")
        or env.get("QWEN_AUDIO_REALTIME_URL")
        or workspace_url
        or DEFAULT_DASHSCOPE_REALTIME_URL
    ).strip().rstrip("?")
    profile = resolve_dashscope_realtime_model_profile(model)
    voice_environment = {
        "audio": "QWEN_AUDIO_REALTIME_VOICE",
        "omni": "QWEN_OMNI_REALTIME_VOICE",
    }.get(profile["family"])
    voice = str(env.get(voice_environment) or "").strip() if voice_environment else ""
    speech_to_speech_url = str(
        env.get("SPEECH_TO_SPEECH_REALTIME_URL")
        or env.get("S2S_REALTIME_URL")
        or DEFAULT_SPEECH_TO_SPEECH_REALTIME_URL
    ).strip().rstrip("/")
    speech_to_speech_token = str(
        env.get("SPEECH_TO_SPEECH_AUTH_TOKEN") or env.get("S2S_API_KEY") or ""
    ).strip()
    speech_to_speech_configured = bool(
        str(env.get("SPEECH_TO_SPEECH_REALTIME_URL") or "").strip()
        or str(env.get("S2S_REALTIME_URL") or "").strip()
        or provider == "speech-to-speech"
    )
    providers: list[dict[str, Any]] = []
    if dashscope_configured:
        providers.append(
            {
                "key": "dashscope",
                "label": "Qwen-Audio-Realtime",
                "model": model,
                "realtimeModelIds": [profile["id"] for profile in model_catalog],
                "configured": True,
            }
        )
    if speech_to_speech_configured:
        providers.append(
            {
                "key": "speech-to-speech",
                "label": "Hugging Face Speech-to-Speech",
                "model": None,
                "realtimeModelIds": None,
                "configured": True,
            }
        )

    if provider == "dashscope":
        identity = {
            "provider": provider,
            "endpoint": dashscope_url,
            "model": model,
            "voice": voice,
            "credential": dashscope_api_key,
        }
        return {
            "realtimeLabel": "Qwen-Audio-Realtime",
            "realtimeModel": model,
            "realtimeModelProfile": profile,
            "realtimeModelCatalog": model_catalog,
            "realtimeInputSampleRate": 16_000,
            "realtimeConfigurationSignature": _configuration_signature(identity),
            "realtimeProviders": providers,
        }
    if provider == "speech-to-speech":
        identity = {
            "provider": provider,
            "endpoint": speech_to_speech_url,
            "credential": speech_to_speech_token,
        }
        return {
            "realtimeLabel": "Hugging Face Speech-to-Speech",
            "realtimeModel": None,
            "realtimeModelProfile": None,
            "realtimeModelCatalog": [],
            "realtimeInputSampleRate": 16_000,
            "realtimeConfigurationSignature": _configuration_signature(identity),
            "realtimeProviders": providers,
        }
    raise ValueError(f"unsupported realtime provider: {provider}")


def _configuration_signature(identity: Mapping[str, Any]) -> str:
    serialized = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()
