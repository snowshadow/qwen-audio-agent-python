from __future__ import annotations

from pathlib import Path

from qwen_audio_agent.backend.node_bridge import NodeBackendBridge


def test_packaged_bridge_uses_app_asar_with_the_electron_node_runtime(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "Resources" / "runtime"
    source_root = tmp_path / "Resources" / "app.asar"
    electron = tmp_path / "MacOS" / "Qwen Audio Agent"

    bridge = NodeBackendBridge(
        root=runtime_root,
        env={
            "QWEN_AUDIO_AGENT_SOURCE_ROOT": str(source_root),
            "QWEN_AUDIO_AGENT_NODE": str(electron),
        },
    )

    assert bridge.command == (
        str(electron),
        str(source_root / "server" / "src" / "python-backend-bridge.mjs"),
    )
    assert bridge.env["QWEN_AUDIO_AGENT_RUNTIME_ROOT"] == str(runtime_root.resolve())
    assert bridge.env["QWEN_AUDIO_AGENT_SOURCE_ROOT"] == str(source_root.resolve())
    assert bridge.env["ELECTRON_RUN_AS_NODE"] == "1"


def test_regular_cli_bridge_keeps_the_runtime_tree_and_system_node(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("ELECTRON_RUN_AS_NODE", raising=False)
    monkeypatch.delenv("QWEN_AUDIO_AGENT_SOURCE_ROOT", raising=False)
    monkeypatch.delenv("QWEN_AUDIO_AGENT_NODE", raising=False)
    bridge = NodeBackendBridge(
        root=tmp_path,
        env={
            "QWEN_AUDIO_AGENT_SOURCE_ROOT": "",
            "QWEN_AUDIO_AGENT_NODE": "",
        },
    )

    assert bridge.command == (
        "node",
        str(tmp_path / "server" / "src" / "python-backend-bridge.mjs"),
    )
    assert "ELECTRON_RUN_AS_NODE" not in bridge.env
