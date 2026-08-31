"""Command-line entry point for the Python Gateway."""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import secrets
import signal
import socket
import sys
from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn

from qwen_audio_agent.application import (
    FrontendLocalServices,
    FrontendRealtimeApplication,
    WorkApplication,
    build_frontend_instructions,
    frontend_graph_tools,
    provider_frontend_tool_definitions,
)
from qwen_audio_agent.backend.disabled import DisabledBackend
from qwen_audio_agent.backend.node_bridge import NodeBackendBridge
from qwen_audio_agent.graphs.frontend import FrontendTurnGraph
from qwen_audio_agent.graphs.work import WorkGraph
from qwen_audio_agent.protocol import GATEWAY_PROTOCOL_VERSION
from qwen_audio_agent.providers.openai_compatible import (
    DEFAULT_DASHSCOPE_REALTIME_MODEL,
    DEFAULT_DASHSCOPE_REALTIME_URL,
    DEFAULT_SPEECH_TO_SPEECH_REALTIME_URL,
    DashScopeRealtimeProvider,
    SpeechToSpeechRealtimeProvider,
)
from qwen_audio_agent.providers.realtime_catalog import realtime_health_metadata
from qwen_audio_agent.runtime.conversation import ConversationStore
from qwen_audio_agent.runtime.frontend_data import JsonNotesStore, MarkdownMemoryService
from qwen_audio_agent.runtime.input_arbitration import InputArbitration
from qwen_audio_agent.runtime.lease import GatewayLease, acquire_gateway_lease
from qwen_audio_agent.runtime.persistence import open_runtime_persistence
from qwen_audio_agent.runtime.repository import TaskRepository
from qwen_audio_agent.runtime.task_notification_lease import TaskNotificationLeaseAdapter
from qwen_audio_agent.runtime.task_store import JsonTaskStore
from qwen_audio_agent.transport import IdentityManager, create_gateway_app

_AUTH_SECRET_KEY = "QWEN_AUDIO_AGENT_AUTH_SECRET"
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UVICORN_LOG_LEVELS = {
    "trace": "trace",
    "debug": "debug",
    "info": "info",
    "warn": "warning",
    "warning": "warning",
    "error": "error",
    "fatal": "critical",
    "critical": "critical",
    # Uvicorn has no silent level. Critical preserves the old quiet startup
    # intent without passing an invalid value that prevents the server boot.
    "silent": "critical",
}


class GatewaySetupRequiredError(RuntimeError):
    """Raised before lease acquisition when realtime is not configured."""

    code = "QWAUDIO_GATEWAY_SETUP_REQUIRED"

    def __init__(self, missing: list[dict[str, str]]) -> None:
        self.missing = missing
        details = "; ".join(f"{item['key']} ({item['message']})" for item in missing)
        super().__init__(f"Gateway startup refused; missing required configuration: {details}")


def gateway_setup_status(env: Mapping[str, str] = os.environ) -> dict[str, Any]:
    """Return the setup-gate shape used by the JavaScript host contract."""

    requested = str(env.get("QWEN_AUDIO_REALTIME_PROVIDER", "dashscope")).strip().lower()
    aliases = {"qwen": "dashscope", "s2s": "speech-to-speech"}
    provider = aliases.get(requested, requested)
    if provider not in {"dashscope", "speech-to-speech"}:
        raise ValueError(f"unsupported realtime provider: {requested}")
    missing: list[dict[str, str]] = []
    if provider == "dashscope":
        configured = bool(
            str(
                env.get("QWEN_AUDIO_REALTIME_API_KEY") or env.get("DASHSCOPE_API_KEY") or ""
            ).strip()
        )
        if not configured:
            missing.append(
                {
                    "field": "dashscopeApiKey",
                    "key": "DASHSCOPE_API_KEY",
                    "message": "DashScope realtime API key is missing",
                }
            )
    else:
        # Selecting speech-to-speech opts into its documented local default;
        # an explicit URL is therefore optional, matching the JS catalog.
        configured = True
    return {"ready": not missing, "provider": provider, "missing": missing}


def assert_gateway_setup(env: Mapping[str, str] = os.environ) -> None:
    if str(env.get("QWEN_AUDIO_ALLOW_UNCONFIGURED", "")) == "1":
        return
    status = gateway_setup_status(env)
    if not status["ready"]:
        raise GatewaySetupRequiredError(status["missing"])


def _default_config_directory(env: Mapping[str, str] = os.environ) -> Path:
    explicit = str(env.get("QWAUDIO_CONFIG_DIR", "")).strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    xdg = str(env.get("XDG_CONFIG_HOME", "")).strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return (base / "qwaudio").resolve()


def _default_data_directory(env: Mapping[str, str] = os.environ) -> Path:
    explicit = str(env.get("QWAUDIO_DATA_DIR", "")).strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return _default_config_directory(env)


def _decode_env_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        try:
            decoded = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]
        return str(decoded)
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def _load_env_file(path: Path, env: MutableMapping[str, str]) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return False
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or not _ENV_ASSIGNMENT.fullmatch(key) or key in env:
            continue
        env[key] = _decode_env_value(raw_value)
    return True


def load_runtime_environment(
    argv: Sequence[str] | None = None,
    *,
    env: MutableMapping[str, str] = os.environ,
    runtime_root: Path | None = None,
) -> tuple[Path, ...]:
    """Load the same source and user config files as the retained Node host."""

    root = runtime_root.resolve() if runtime_root is not None else _runtime_root(env)
    loaded: list[Path] = []
    for path in (root / ".env.local", root / ".env"):
        if _load_env_file(path, env):
            loaded.append(path)

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-dir", type=Path)
    pre_parser.add_argument("--data-dir", type=Path)
    parsed, _unknown = pre_parser.parse_known_args(argv)
    data_directory = (
        parsed.data_dir.expanduser().resolve()
        if parsed.data_dir is not None
        else (
            parsed.config_dir.expanduser().resolve()
            if parsed.config_dir is not None and not str(env.get("QWAUDIO_DATA_DIR", "")).strip()
            else _default_data_directory(env)
        )
    )
    config_path = data_directory / "config.env"
    if _load_env_file(config_path, env):
        loaded.append(config_path)
    return tuple(loaded)


def _uvicorn_log_level(value: str) -> str:
    return _UVICORN_LOG_LEVELS.get(str(value).strip().lower(), "info")


def build_parser(env: Mapping[str, str] = os.environ) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Python qwen-audio-agent Gateway")
    parser.add_argument("--host", default=env.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(env.get("PORT", "3101")))
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=_default_config_directory(env),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_directory(env),
    )
    parser.add_argument(
        "--owner",
        default=env.get(
            "QWEN_AUDIO_GATEWAY_OWNER",
            "desktop" if env.get("QWEN_AUDIO_AGENT_DESKTOP") == "1" else "cli",
        ),
    )
    parser.add_argument(
        "--log-level",
        type=_uvicorn_log_level,
        default=_uvicorn_log_level(env.get("QWEN_AUDIO_LOG_LEVEL", "info")),
    )
    parser.add_argument(
        "--allow-unconfigured",
        action="store_true",
        help="start transport-only harnesses without realtime credentials",
    )
    return parser


def parse_arguments(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] = os.environ,
) -> argparse.Namespace:
    """Parse CLI arguments while preserving the legacy single-root layout."""

    args = build_parser(env).parse_args(argv)
    explicit_data_dir = any(
        value == "--data-dir" or value.startswith("--data-dir=") for value in argv
    )
    explicit_config_dir = any(
        value == "--config-dir" or value.startswith("--config-dir=") for value in argv
    )
    if (
        explicit_config_dir
        and not explicit_data_dir
        and not str(env.get("QWAUDIO_DATA_DIR", "")).strip()
    ):
        args.data_dir = args.config_dir
    return args


def _listen_socket(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    listener = socket.socket(family, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(2048)
    listener.setblocking(False)
    return listener


def _origin(host: str, port: int) -> str:
    display_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{display_host}:{port}"


def _runtime_root(env: Mapping[str, str] = os.environ) -> Path:
    configured = str(env.get("QWEN_AUDIO_AGENT_RUNTIME_ROOT", "")).strip()
    return Path(configured).expanduser().resolve() if configured else Path(__file__).parents[2]


def _frontend_tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    return str(function.get("name") or "") if isinstance(function, Mapping) else ""


def _provider_tool_definitions(
    *,
    backend_enabled: bool,
    client_states: Sequence[str] = (),
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        tool
        for tool in provider_frontend_tool_definitions(client_states=client_states)
        if backend_enabled or _frontend_tool_name(tool) != "spawn_thinking"
    )


def _create_realtime_provider(
    *,
    requested: str,
    instructions: str,
    tools: Sequence[Mapping[str, Any]],
    env: Mapping[str, str] = os.environ,
) -> DashScopeRealtimeProvider | SpeechToSpeechRealtimeProvider:
    aliases = {"qwen": "dashscope", "s2s": "speech-to-speech"}
    selected = aliases.get(str(requested).strip().lower(), str(requested).strip().lower())
    if selected == "dashscope":
        model = str(
            env.get("QWEN_AUDIO_REALTIME_MODEL") or DEFAULT_DASHSCOPE_REALTIME_MODEL
        ).strip()
        voice_key = "QWEN_OMNI_REALTIME_VOICE" if "omni" in model else "QWEN_AUDIO_REALTIME_VOICE"
        workspace_id = str(env.get("DASHSCOPE_WORKSPACE_ID") or "").strip()
        workspace_url = (
            f"wss://{workspace_id}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime"
            if workspace_id
            else ""
        )
        base_url = str(
            env.get("QWEN_AUDIO_REALTIME_BASE_URL")
            or env.get("QWEN_AUDIO_REALTIME_URL")
            or workspace_url
            or DEFAULT_DASHSCOPE_REALTIME_URL
        ).rstrip("?")
        return DashScopeRealtimeProvider(
            api_key=str(
                env.get("QWEN_AUDIO_REALTIME_API_KEY") or env.get("DASHSCOPE_API_KEY") or ""
            ),
            model=model,
            base_url=base_url,
            voice=str(env.get(voice_key) or "") or None,
            instructions=instructions,
            tools=tools,
        )
    if selected == "speech-to-speech":
        base_url = str(
            env.get("SPEECH_TO_SPEECH_REALTIME_URL")
            or env.get("S2S_REALTIME_URL")
            or DEFAULT_SPEECH_TO_SPEECH_REALTIME_URL
        ).rstrip("/")
        return SpeechToSpeechRealtimeProvider(
            base_url=base_url,
            auth_token=str(env.get("SPEECH_TO_SPEECH_AUTH_TOKEN") or env.get("S2S_API_KEY") or "")
            or None,
            instructions=instructions,
            tools=tools,
        )
    raise ValueError(f"unsupported realtime provider: {requested}")


def _read_persisted_auth_secret(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return ""
    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key.strip() == _AUTH_SECRET_KEY:
            return value.strip().strip("\"'")
    return ""


def _identity_secret(
    config_directory: Path,
    env: Mapping[str, str] = os.environ,
) -> str:
    configured = str(env.get(_AUTH_SECRET_KEY) or "").strip()
    if configured:
        return configured
    state_path = config_directory / "state.env"
    persisted = _read_persisted_auth_secret(state_path)
    if persisted:
        state_path.chmod(0o600)
        return persisted
    if state_path.exists():
        raise ValueError(f"generated local identity configuration is invalid: {state_path}")

    config_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    generated = secrets.token_hex(32)
    try:
        descriptor = os.open(state_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        persisted = _read_persisted_auth_secret(state_path)
        if not persisted:
            raise ValueError(
                f"generated local identity configuration is invalid: {state_path}"
            ) from error
        state_path.chmod(0o600)
        return persisted
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(f"{_AUTH_SECRET_KEY}={generated}\n")
    return generated


def _build_identity_manager(
    config_directory: Path,
    env: Mapping[str, str] = os.environ,
) -> IdentityManager:
    mode = (
        "browser"
        if str(env.get("QWEN_AUDIO_AGENT_IDENTITY_MODE") or "").strip().lower() == "browser"
        else "personal"
    )
    return IdentityManager(
        secret=_identity_secret(config_directory, env),
        mode=mode,
        personal_owner_id=str(env.get("QWEN_AUDIO_AGENT_PERSONAL_OWNER_ID") or "user_personal"),
    )


def _allowed_origins(env: Mapping[str, str] = os.environ) -> tuple[str, ...]:
    return tuple(
        value
        for raw in str(env.get("QWEN_AUDIO_AGENT_ALLOWED_ORIGINS") or "").split(",")
        if (value := raw.strip())
    )


async def _heartbeat(lease: GatewayLease) -> None:
    while True:
        await asyncio.sleep(15)
        lease.update()


async def serve(args: argparse.Namespace) -> None:
    """Acquire the lease, run Uvicorn, then release the lease cleanly."""

    if args.allow_unconfigured:
        os.environ["QWEN_AUDIO_ALLOW_UNCONFIGURED"] = "1"
    config_directory = args.config_dir.expanduser().resolve()
    data_directory = args.data_dir.expanduser().resolve()
    os.environ["QWAUDIO_CONFIG_DIR"] = str(config_directory)
    os.environ["QWAUDIO_DATA_DIR"] = str(data_directory)
    # The setup gate intentionally precedes both lease acquisition and bind.
    assert_gateway_setup()
    lease = acquire_gateway_lease(config_directory, owner=args.owner)
    started_at = datetime.now(UTC).isoformat()
    os.environ["QWEN_AUDIO_GATEWAY_INSTANCE_ID"] = lease.instance_id
    os.environ["QWEN_AUDIO_GATEWAY_STARTED_AT"] = started_at
    arbitration = InputArbitration()
    listener: socket.socket | None = None
    heartbeat: asyncio.Task[None] | None = None
    previous_signal_handlers: dict[signal.Signals, Any] = {}
    try:
        identity_manager = _build_identity_manager(data_directory)
        allowed_origins = _allowed_origins()
        listener = _listen_socket(args.host, args.port)
        actual_port = int(listener.getsockname()[1])
        setup = gateway_setup_status()
        runtime_root = _runtime_root()
        async with AsyncExitStack() as runtime_stack:
            persistence = await runtime_stack.enter_async_context(
                open_runtime_persistence(config_directory / "state" / "runtime")
            )
            backend_protocol = str(os.getenv("AGENT_PROTOCOL", "")).strip().lower()
            backend_enabled = backend_protocol not in {"", "none"}
            backend: NodeBackendBridge | None = None
            backend_port: Any
            if backend_enabled:
                backend = NodeBackendBridge(root=runtime_root)
                backend_port = backend
            else:
                backend_port = DisabledBackend()
            task_repository = TaskRepository(
                store=JsonTaskStore(config_directory / "state" / "tasks.json")
            )
            work_runtime = WorkGraph(
                backend=backend_port,
                repository=task_repository,
                checkpointer=persistence.checkpointer,
            )
            notification_leases = TaskNotificationLeaseAdapter(
                task_repository,
                on_delivered=work_runtime.notification_delivered,
            )
            work_application = WorkApplication(work_runtime)
            memory = MarkdownMemoryService(
                user_path=data_directory / "USER.md",
                memory_path=data_directory / "MEMORY.md",
                personal_owner_id=os.getenv(
                    "QWEN_AUDIO_AGENT_PERSONAL_OWNER_ID",
                    "user_personal",
                ),
            )
            notes = JsonNotesStore(data_directory / "frontend-notes.json")
            conversations = ConversationStore(
                config_directory / "state" / "frontend-conversations.json"
            )
            local_services = FrontendLocalServices(
                memory=memory,
                notes=notes,
                work=work_application,
                backend_enabled=backend_enabled,
            )
            frontend_graph = FrontendTurnGraph(
                frontend_graph_tools(),
                local_services.ports(),
                checkpointer=persistence.checkpointer,
                store=persistence.store,
            )

            def provider_factory(
                connection_context: Mapping[str, Any],
                connect_event: Mapping[str, Any],
            ) -> DashScopeRealtimeProvider | SpeechToSpeechRealtimeProvider:
                requested = (
                    str(
                        connect_event.get("provider")
                        or os.getenv("QWEN_AUDIO_REALTIME_PROVIDER", "dashscope")
                    )
                    .strip()
                    .lower()
                )
                owner_id = str(connection_context.get("ownerId") or "personal")
                session_id = str(connection_context.get("sessionId") or "main")
                profile_path = data_directory / "ASSISTANT.md"
                instructions = build_frontend_instructions(
                    runtime_root=runtime_root,
                    assistant_profile_path=profile_path if profile_path.exists() else None,
                    user_preferences=memory.read_document(owner_id, "user"),
                    user_memory=memory.read_document(owner_id, "memory"),
                    recent_messages=conversations.messages(
                        owner_id=owner_id,
                        session_id=session_id,
                        limit=10,
                    ),
                    client_context={
                        "client_time_zone": connect_event.get("timeZone"),
                        "client_locale": connect_event.get("locale"),
                        "client_working_directory": connect_event.get("workingDirectory"),
                    },
                )
                client_states = (
                    tuple(str(value) for value in connect_event.get("clientStates", ()))
                    if isinstance(connect_event.get("clientStates"), list)
                    else ()
                )
                return _create_realtime_provider(
                    requested=requested,
                    instructions=instructions,
                    tools=_provider_tool_definitions(
                        backend_enabled=backend_enabled,
                        client_states=client_states,
                    ),
                )

            frontend_runtime = FrontendRealtimeApplication(
                provider_factory=provider_factory,
                frontend_graph=frontend_graph,
                conversation_store=conversations,
                task_source=work_runtime,
                delivery_leases=notification_leases,
            )
            runtime_stack.push_async_callback(frontend_runtime.close)

            async def health_extensions() -> dict[str, Any]:
                if backend is None:
                    backend_status: dict[str, Any] = {
                        "enabled": False,
                        "configured": False,
                        "ok": True,
                        "status": "not_configured",
                        "capabilities": {},
                    }
                else:
                    description = dict(backend.describe())
                    try:
                        current = await backend.health()
                        health = dict(current) if isinstance(current, Mapping) else {}
                    except Exception as error:
                        health = {"ok": False, "status": "unavailable", "error": str(error)}
                    backend_status = {**description, **health, "enabled": True, "configured": True}
                return {
                    "voiceConfigured": setup["ready"],
                    "realtimeProvider": setup["provider"],
                    **realtime_health_metadata(setup["provider"], os.environ),
                    "backend": backend_status,
                }

            app = create_gateway_app(
                input_arbitration=arbitration,
                task_source=work_runtime,
                conversation_source=conversations,
                on_client_event=frontend_runtime,
                on_client_disconnect=frontend_runtime.disconnect,
                identity_manager=identity_manager,
                allowed_origins=allowed_origins,
                gateway_instance_id=lease.instance_id,
                gateway_started_at=started_at,
                web_root=runtime_root / "web" / "dist",
                skins_root=config_directory / "skins",
                health_provider=health_extensions,
                close_task_source_on_shutdown=True,
            )
            config = uvicorn.Config(
                app,
                host=args.host,
                port=actual_port,
                log_level=args.log_level,
            )
            server = uvicorn.Server(config)

            # Uvicorn restores the previous handlers and re-raises captured
            # signals after graceful shutdown.  If the previous handler is the OS
            # default, SIGTERM can terminate this process before our outer
            # ``finally`` releases the Gateway lease.  A small outer handler keeps
            # that re-raise inside Python so cleanup always completes.
            def request_exit(_signal: int, _frame: Any) -> None:
                server.should_exit = True

            for handled_signal in (signal.SIGINT, signal.SIGTERM):
                previous_signal_handlers[handled_signal] = signal.signal(
                    handled_signal,
                    request_exit,
                )
            server_task = asyncio.create_task(server.serve(sockets=[listener]))
            while not server.started:
                if server_task.done():
                    await server_task
                await asyncio.sleep(0.01)
            origin = _origin(args.host, actual_port)
            lease.update(state="ready", origin=origin)
            print(
                json.dumps(
                    {
                        "type": "qwen-audio-agent:gateway-ready",
                        "origin": origin,
                        "instanceId": lease.instance_id,
                        "protocolVersion": GATEWAY_PROTOCOL_VERSION,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            heartbeat = asyncio.create_task(_heartbeat(lease))
            await server_task
    finally:
        for handled_signal, previous in previous_signal_handlers.items():
            signal.signal(handled_signal, previous)
        if heartbeat is not None:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        if listener is not None:
            listener.close()
        arbitration.close()
        lease.release()


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    load_runtime_environment(arguments)
    args = parse_arguments(arguments)
    try:
        asyncio.run(serve(args))
    except GatewaySetupRequiredError as error:
        print(f"qwenaudio-python-gateway: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
