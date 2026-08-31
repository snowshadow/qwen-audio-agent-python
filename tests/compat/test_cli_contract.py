from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest

from qwen_audio_agent.cli import (
    GatewaySetupRequiredError,
    assert_gateway_setup,
    build_parser,
    gateway_setup_status,
)


def test_setup_gate_matches_dashscope_and_speech_to_speech_rules() -> None:
    missing = gateway_setup_status({})
    assert missing == {
        "ready": False,
        "provider": "dashscope",
        "missing": [
            {
                "field": "dashscopeApiKey",
                "key": "DASHSCOPE_API_KEY",
                "message": "DashScope realtime API key is missing",
            }
        ],
    }
    assert gateway_setup_status({"DASHSCOPE_API_KEY": "sk-test"})["ready"] is True
    assert (
        gateway_setup_status({"QWEN_AUDIO_REALTIME_PROVIDER": "speech-to-speech"})["ready"] is True
    )


def test_setup_gate_fails_before_start_unless_explicitly_bypassed() -> None:
    with pytest.raises(GatewaySetupRequiredError) as raised:
        assert_gateway_setup({})
    assert raised.value.code == "QWAUDIO_GATEWAY_SETUP_REQUIRED"
    assert raised.value.missing[0]["key"] == "DASHSCOPE_API_KEY"
    assert_gateway_setup({"QWEN_AUDIO_ALLOW_UNCONFIGURED": "1"})
    assert_gateway_setup({"DASHSCOPE_API_KEY": "sk-test"})


def test_setup_status_does_not_mutate_the_environment() -> None:
    before = dict(os.environ)
    gateway_setup_status(os.environ)
    assert dict(os.environ) == before


def test_cli_separates_runtime_state_from_shared_data(tmp_path: Path) -> None:
    config_directory = tmp_path / "desktop-runtime"
    data_directory = tmp_path / "shared-assets"
    args = build_parser(
        {
            "QWAUDIO_CONFIG_DIR": str(config_directory),
            "QWAUDIO_DATA_DIR": str(data_directory),
        }
    ).parse_args([])

    assert args.config_dir == config_directory.resolve()
    assert args.data_dir == data_directory.resolve()

    fallback = build_parser({"QWAUDIO_CONFIG_DIR": str(config_directory)}).parse_args([])
    assert fallback.data_dir == config_directory.resolve()


def test_cli_readiness_health_and_lease_lifecycle(tmp_path: Path) -> None:
    config_directory = tmp_path / "desktop-runtime"
    data_directory = tmp_path / "shared-assets"
    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "QWEN_AUDIO_ALLOW_UNCONFIGURED": "1",
        "QWEN_AUDIO_AGENT_AUTH_SECRET": "",
        "QWEN_AUDIO_REALTIME_PROVIDER": "dashscope",
        "QWEN_AUDIO_REALTIME_MODEL": "qwen3.5-omni-flash-realtime",
        "DASHSCOPE_API_KEY": "health-secret-api-key",
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "qwen_audio_agent.cli",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--config-dir",
            str(config_directory),
            "--data-dir",
            str(data_directory),
            "--allow-unconfigured",
            "--log-level",
            "warning",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        assert process.stdout is not None
        ready_line = ""
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            readable, _, _ = select.select([process.stdout], [], [], 0.1)
            if readable:
                ready_line = process.stdout.readline().strip()
                if ready_line:
                    break
            if process.poll() is not None:
                error_output = process.stderr.read() if process.stderr else ""
                if "PermissionError" in error_output and "Operation not permitted" in error_output:
                    pytest.skip("sandbox does not allow binding a loopback socket")
                pytest.fail(error_output)
        assert ready_line, "Gateway never emitted its readiness record"
        ready = json.loads(ready_line)
        assert ready["type"] == "qwen-audio-agent:gateway-ready"
        assert ready["protocolVersion"] == "4.0.0"

        with urlopen(f"{ready['origin']}/api/health", timeout=3) as response:
            health = json.load(response)
        lease_path = config_directory / "gateway.lock"
        lease = json.loads(lease_path.read_text(encoding="utf-8"))
        assert lease["state"] == "ready"
        assert lease["origin"] == ready["origin"]
        assert lease["instanceId"] == ready["instanceId"]
        assert health["gatewayInstanceId"] == ready["instanceId"]
        assert health["protocolVersion"] == "4.0.0"
        assert health["realtimeLabel"] == "Qwen-Audio-Realtime"
        assert health["realtimeModel"] == "qwen3.5-omni-flash-realtime"
        assert health["realtimeModelProfile"]["family"] == "omni"
        assert health["realtimeModelProfile"]["modelCapabilities"]["imageInput"] is True
        assert health["realtimeModelProfile"]["transportCapabilities"]["imageInput"] is False
        assert [profile["id"] for profile in health["realtimeModelCatalog"]] == [
            "qwen3.5-omni-flash-realtime",
            "qwen3.5-omni-plus-realtime",
            "qwen-audio-3.0-realtime-plus",
            "qwen-audio-3.0-realtime-flash",
        ]
        assert health["realtimeInputSampleRate"] == 16_000
        assert len(health["realtimeConfigurationSignature"]) == 64
        assert health["realtimeProviders"][0]["key"] == "dashscope"
        serialized_health = json.dumps(health)
        assert "health-secret-api-key" not in serialized_health
        assert (data_directory / "state.env").exists()
        assert not (config_directory / "state.env").exists()

        process.terminate()
        assert process.wait(timeout=10) == 0
        assert not lease_path.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
