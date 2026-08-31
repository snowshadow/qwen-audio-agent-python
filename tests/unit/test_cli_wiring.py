from __future__ import annotations

import stat
from pathlib import Path

import pytest

from qwen_audio_agent.cli import (
    _allowed_origins,
    _build_identity_manager,
    _create_realtime_provider,
    _provider_tool_definitions,
    load_runtime_environment,
    parse_arguments,
)
from qwen_audio_agent.providers.openai_compatible import (
    DashScopeRealtimeProvider,
    SpeechToSpeechRealtimeProvider,
)


def _tool_names(tools: object) -> list[str]:
    return [
        str(tool["function"]["name"])
        for tool in tools  # type: ignore[union-attr]
    ]


def test_provider_tools_hide_backend_work_when_backend_is_disabled() -> None:
    disabled = _provider_tool_definitions(backend_enabled=False)
    enabled = _provider_tool_definitions(backend_enabled=True)
    desktop = _provider_tool_definitions(
        backend_enabled=True,
        client_states=("sleeping",),
    )

    assert "spawn_thinking" not in _tool_names(disabled)
    assert "spawn_thinking" in _tool_names(enabled)
    assert "schedule_reminder" in _tool_names(disabled)
    assert "enter_sleep" not in _tool_names(enabled)
    assert "enter_sleep" in _tool_names(desktop)

    spawn = next(
        tool["function"] for tool in enabled if tool["function"]["name"] == "spawn_thinking"
    )
    notes = next(tool["function"] for tool in enabled if tool["function"]["name"] == "notes")
    reminder = next(
        tool["function"] for tool in enabled if tool["function"]["name"] == "schedule_reminder"
    )
    assert set(spawn["parameters"]["properties"]) == {"objective"}
    assert "持久化命名清单" in notes["description"]
    assert "易变条目" not in notes["description"]
    assert reminder["parameters"]["properties"]["recurrence"]["enum"] == ["once"]


def test_frontend_prompt_does_not_advertise_cross_turn_attachment_refs() -> None:
    prompt = (Path(__file__).parents[2] / "config" / "frontend-agent" / "PROMPT.md").read_text(
        encoding="utf-8"
    )

    assert "input_refs" not in prompt
    assert "不在当前 `<input_parts>` 中的更早附件" in prompt


def test_cli_builds_the_speech_to_speech_profile_from_environment() -> None:
    tools = _provider_tool_definitions(backend_enabled=False)
    provider = _create_realtime_provider(
        requested="s2s",
        instructions="be helpful",
        tools=tools,
        env={
            "SPEECH_TO_SPEECH_REALTIME_URL": "ws://localhost:9876/v1/realtime",
            "SPEECH_TO_SPEECH_AUTH_TOKEN": "proxy-token",
        },
    )

    assert isinstance(provider, SpeechToSpeechRealtimeProvider)
    assert provider.url == "ws://localhost:9876/v1/realtime"
    assert provider.headers == {"Authorization": "Bearer proxy-token"}
    assert provider.initial_session["instructions"] == "be helpful"
    assert "spawn_thinking" not in [tool["name"] for tool in provider.initial_session["tools"]]


def test_cli_preserves_realtime_environment_aliases() -> None:
    tools = _provider_tool_definitions(backend_enabled=False)
    dashscope = _create_realtime_provider(
        requested="qwen",
        instructions="be helpful",
        tools=tools,
        env={
            "DASHSCOPE_API_KEY": "dashscope-token",
            "QWEN_AUDIO_REALTIME_URL": "wss://workspace.example/realtime?",
        },
    )
    speech = _create_realtime_provider(
        requested="speech-to-speech",
        instructions="be helpful",
        tools=tools,
        env={
            "S2S_REALTIME_URL": "ws://localhost:9000/v1/realtime/",
            "S2S_API_KEY": "s2s-token",
        },
    )

    assert isinstance(dashscope, DashScopeRealtimeProvider)
    assert dashscope.url.partition("?")[0] == "wss://workspace.example/realtime"
    assert speech.url == "ws://localhost:9000/v1/realtime"
    assert speech.headers == {"Authorization": "Bearer s2s-token"}


def test_direct_python_cli_loads_source_and_user_config_without_overwriting_env(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    data = tmp_path / "data"
    root.mkdir()
    data.mkdir()
    (root / ".env.local").write_text(
        "DASHSCOPE_API_KEY=source-token\nAGENT_PROTOCOL=opencode\n",
        encoding="utf-8",
    )
    (root / ".env").write_text("HOST=127.0.0.2\n", encoding="utf-8")
    (data / "config.env").write_text(
        'DASHSCOPE_API_KEY="user-token"\nS2S_API_KEY=user-s2s # comment\n',
        encoding="utf-8",
    )
    env = {"AGENT_PROTOCOL": "none"}

    loaded = load_runtime_environment(
        ["--data-dir", str(data)],
        env=env,
        runtime_root=root,
    )

    assert loaded == (root / ".env.local", root / ".env", data / "config.env")
    assert env["DASHSCOPE_API_KEY"] == "source-token"
    assert env["AGENT_PROTOCOL"] == "none"
    assert env["HOST"] == "127.0.0.2"
    assert env["S2S_API_KEY"] == "user-s2s"


def test_direct_cli_config_dir_is_the_legacy_data_dir_fallback(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    config = tmp_path / "config"
    root.mkdir()
    config.mkdir()
    (config / "config.env").write_text("HOST=127.0.0.9\n", encoding="utf-8")
    env: dict[str, str] = {}

    loaded = load_runtime_environment(
        ["--config-dir", str(config)],
        env=env,
        runtime_root=root,
    )
    args = parse_arguments(["--config-dir", str(config)], env=env)

    assert loaded == (config / "config.env",)
    assert args.config_dir == config
    assert args.data_dir == config
    assert env["HOST"] == "127.0.0.9"


@pytest.mark.parametrize(
    ("legacy", "normalized"),
    [("warn", "warning"), ("fatal", "critical"), ("silent", "critical")],
)
def test_cli_normalizes_legacy_log_levels(legacy: str, normalized: str) -> None:
    assert parse_arguments(["--log-level", legacy], env={}).log_level == normalized
    assert parse_arguments([], env={"QWEN_AUDIO_LOG_LEVEL": legacy}).log_level == normalized


def test_cli_persists_identity_secret_and_parses_security_configuration(
    tmp_path: Path,
) -> None:
    first = _build_identity_manager(tmp_path, {})
    second = _build_identity_manager(tmp_path, {})
    state_path = tmp_path / "state.env"

    assert first.secret == second.secret
    assert len(first.secret) == 64
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600

    browser = _build_identity_manager(
        tmp_path,
        {
            "QWEN_AUDIO_AGENT_AUTH_SECRET": "x" * 32,
            "QWEN_AUDIO_AGENT_IDENTITY_MODE": "browser",
            "QWEN_AUDIO_AGENT_PERSONAL_OWNER_ID": "user_custom",
        },
    )
    assert browser.mode == "browser"
    assert _allowed_origins(
        {"QWEN_AUDIO_AGENT_ALLOWED_ORIGINS": " https://voice.example.com, http://localhost:3101 "}
    ) == ("https://voice.example.com", "http://localhost:3101")
