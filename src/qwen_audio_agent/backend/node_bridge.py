"""BackendPort adapter for the retained ACP/A2A JavaScript adapter boundary."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

# A legal backend result may contain 1M result characters plus 16M artifact
# characters. JSON escaping can expand those characters beyond their UTF-8
# representation, so the framing budget must cover the complete envelope.
MAX_BRIDGE_LINE_BYTES = 128 * 1024 * 1024
DEFAULT_CONTROL_RPC_TIMEOUT_SECONDS = 30.0

_CONTEXT_KEY_ALIASES = {
    "owner_id": "ownerId",
    "session_id": "sessionId",
    "turn_id": "turnId",
    "task_id": "taskId",
    "coordination_run_id": "coordinationRunId",
    "coordination_request_id": "coordinationRequestId",
    "permission_scope_id": "permissionScopeId",
}


class BackendBridgeError(RuntimeError):
    def __init__(self, message: str, *, code: str = "BACKEND_BRIDGE_ERROR") -> None:
        super().__init__(message)
        self.code = code


def _safe_context(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    context = {
        str(key): item
        for key, item in value.items()
        if not callable(item) and key not in {"on_event", "onEvent", "signal"}
    }
    for source, target in _CONTEXT_KEY_ALIASES.items():
        if source not in context:
            continue
        context.setdefault(target, context[source])
        del context[source]
    return context


class NodeBackendBridge:
    """Async NDJSON RPC client exposing the complete Python BackendPort."""

    def __init__(
        self,
        *,
        root: str | Path,
        command: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        control_timeout_seconds: float | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        inherited_env = {**os.environ, **dict(env or {})}
        source_root = Path(inherited_env.get("QWEN_AUDIO_AGENT_SOURCE_ROOT") or self.root).resolve()
        node_command = str(inherited_env.get("QWEN_AUDIO_AGENT_NODE") or "node").strip()
        self.command = tuple(
            command
            or (
                node_command,
                str(source_root / "server" / "src" / "python-backend-bridge.mjs"),
            )
        )
        self.env = {
            **inherited_env,
            "QWEN_AUDIO_AGENT_RUNTIME_ROOT": str(self.root),
            "QWEN_AUDIO_AGENT_SOURCE_ROOT": str(source_root),
            "QWEN_AUDIO_LOG_CONSOLE": "0",
        }
        configured_timeout = (
            control_timeout_seconds
            if control_timeout_seconds is not None
            else float(
                inherited_env.get("QWEN_AUDIO_BACKEND_CONTROL_TIMEOUT_SECONDS")
                or DEFAULT_CONTROL_RPC_TIMEOUT_SECONDS
            )
        )
        if configured_timeout <= 0:
            raise ValueError("control_timeout_seconds must be positive")
        self.control_timeout_seconds = configured_timeout
        if inherited_env.get("QWEN_AUDIO_AGENT_NODE"):
            # The Desktop host supplies its Electron executable. Node mode can
            # read the bridge and dependencies directly from app.asar.
            self.env["ELECTRON_RUN_AS_NODE"] = "1"
        self.process: asyncio.subprocess.Process | None = None
        self._process_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._listeners: set[Callable[[Mapping[str, Any]], Any]] = set()
        self._task_listeners: dict[str, Callable[[Mapping[str, Any]], Any]] = {}
        self._callback_tasks: set[asyncio.Task[Any]] = set()
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._description: dict[str, Any] = {
            "configured": True,
            "enabled": True,
            "protocol": self.env.get("AGENT_PROTOCOL") or "node-bridge",
            "label": "Backend adapter bridge",
            "capabilities": {},
        }
        self._closed = False

    def describe(self) -> Mapping[str, Any]:
        return dict(self._description)

    async def _ensure_process(self) -> None:
        async with self._process_lock:
            if self.process is not None and self.process.returncode is None:
                return
            if self._closed:
                raise BackendBridgeError("Backend bridge is closed", code="BACKEND_CLOSED")
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                cwd=self.root,
                env=self.env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=MAX_BRIDGE_LINE_BYTES,
            )
            self._reader_task = asyncio.create_task(
                self._read_stdout(),
                name="backend-bridge-stdout",
            )
            self._stderr_task = asyncio.create_task(
                self._drain_stderr(),
                name="backend-bridge-stderr",
            )

    async def _read_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        try:
            while line := await process.stdout.readline():
                try:
                    message = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(message, Mapping) and isinstance(message.get("event"), Mapping):
                    self._dispatch_event(message["event"])
                    continue
                request_id = message.get("id") if isinstance(message, Mapping) else None
                future = self._pending.pop(request_id, None)
                if future is None or future.done():
                    continue
                error = message.get("error")
                if isinstance(error, Mapping):
                    future.set_exception(
                        BackendBridgeError(
                            str(error.get("message") or "Backend bridge failed"),
                            code=str(error.get("code") or "BACKEND_BRIDGE_ERROR"),
                        )
                    )
                else:
                    future.set_result(message.get("result"))
        finally:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.terminate()
            if self.process is process:
                self.process = None
            error = BackendBridgeError(
                "Backend bridge exited",
                code="BACKEND_BRIDGE_EXITED",
            )
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(error)
            self._pending.clear()

    async def _drain_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        while await process.stderr.readline():
            pass

    def _dispatch_event(self, event: Mapping[str, Any]) -> None:
        normalized = dict(event)
        task_id = str(normalized.get("taskId") or normalized.get("task_id") or "")
        task_listener = self._task_listeners.get(task_id)
        callbacks = [*self._listeners, *([task_listener] if task_listener else [])]
        for callback in callbacks:
            try:
                outcome = callback(normalized)
                if inspect.isawaitable(outcome):
                    task = asyncio.create_task(outcome)
                    self._callback_tasks.add(task)
                    task.add_done_callback(self._callback_tasks.discard)
            except Exception:
                continue

    async def _abort_process(self, process: asyncio.subprocess.Process) -> None:
        async with self._process_lock:
            reader_task = self._reader_task
            stderr_task = self._stderr_task
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except TimeoutError:
                    with suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
            if self.process is process:
                self.process = None
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(
                        BackendBridgeError(
                            "Backend bridge was restarted",
                            code="BACKEND_BRIDGE_RESTARTED",
                        )
                    )
            self._pending.clear()
            await asyncio.gather(
                *(task for task in (reader_task, stderr_task) if task is not None),
                return_exceptions=True,
            )
            if self._reader_task is reader_task:
                self._reader_task = None
            if self._stderr_task is stderr_task:
                self._stderr_task = None

    async def _rpc(
        self,
        method: str,
        *,
        _timeout_seconds: float | None = None,
        **params: Any,
    ) -> Any:
        await self._ensure_process()
        process = self.process
        if process is None or process.stdin is None:
            raise BackendBridgeError("Backend bridge stdin is unavailable")
        loop = asyncio.get_running_loop()
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        payload = (
            json.dumps(
                {"id": request_id, "method": method, "params": params},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        try:
            async with self._write_lock:
                process.stdin.write(payload)
                await process.stdin.drain()
            if _timeout_seconds is None:
                return await future
            try:
                return await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=_timeout_seconds,
                )
            except TimeoutError as error:
                self._pending.pop(request_id, None)
                future.cancel()
                await self._abort_process(process)
                raise BackendBridgeError(
                    f"Backend bridge {method} timed out",
                    code="BACKEND_RPC_TIMEOUT",
                ) from error
        except asyncio.CancelledError:
            self._pending.pop(request_id, None)
            future.cancel()
            raise
        except Exception:
            self._pending.pop(request_id, None)
            raise

    async def start(self, context: Mapping[str, Any] | None = None) -> Any:
        result = await self._rpc(
            "start",
            _timeout_seconds=self.control_timeout_seconds,
            context=_safe_context(context),
        )
        try:
            described = await self._rpc(
                "describe",
                _timeout_seconds=self.control_timeout_seconds,
            )
            if isinstance(described, Mapping):
                self._description = dict(described)
        except BackendBridgeError:
            pass
        return result

    async def health(self) -> Any:
        return await self._rpc(
            "health",
            _timeout_seconds=self.control_timeout_seconds,
        )

    async def submit(
        self,
        task: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        task_id = str(task.get("id") or "")
        callback = (context or {}).get("on_event") if isinstance(context, Mapping) else None
        if task_id and callable(callback):
            self._task_listeners[task_id] = callback
        try:
            return await self._rpc(
                "submit",
                task=dict(task),
                context=_safe_context(context),
            )
        finally:
            if task_id:
                self._task_listeners.pop(task_id, None)

    async def status(
        self,
        task_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        return await self._rpc(
            "status",
            _timeout_seconds=self.control_timeout_seconds,
            taskId=task_id,
            context=_safe_context(context),
        )

    async def cancel(
        self,
        task_id: str,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        return await self._rpc(
            "cancel",
            _timeout_seconds=self.control_timeout_seconds,
            taskId=task_id,
            context=_safe_context(context),
        )

    async def respond_authorization(
        self,
        task_id: str,
        authorization_id: str,
        decision: Any,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        return await self._rpc(
            "respondAuthorization",
            _timeout_seconds=self.control_timeout_seconds,
            taskId=task_id,
            authorizationId=authorization_id,
            decision=decision,
            context=_safe_context(context),
        )

    def subscribe(
        self,
        listener: Callable[[Mapping[str, Any]], Any],
    ) -> Callable[[], None]:
        self._listeners.add(listener)
        return lambda: self._listeners.discard(listener)

    async def can_recover_delegated_work(self, task: Mapping[str, Any]) -> bool:
        return bool(
            await self._rpc(
                "canRecoverDelegatedWork",
                _timeout_seconds=self.control_timeout_seconds,
                task=dict(task),
            )
        )

    async def recover_delegated_work(
        self,
        task: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        task_id = str(task.get("id") or "")
        callback = (context or {}).get("on_event") if isinstance(context, Mapping) else None
        if task_id and callable(callback):
            self._task_listeners[task_id] = callback
        try:
            return await self._rpc(
                "recoverDelegatedWork",
                task=dict(task),
                context=_safe_context(context),
            )
        finally:
            if task_id:
                self._task_listeners.pop(task_id, None)

    async def close(self) -> None:
        if self._closed:
            return
        process = self.process
        if process is not None and process.returncode is None:
            with suppress(Exception):
                await asyncio.wait_for(self._rpc("close"), timeout=5)
            if process.returncode is None:
                process.terminate()
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=5)
            if process.returncode is None:
                process.kill()
                await process.wait()
        self._closed = True
        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
        for task in tuple(self._callback_tasks):
            if not task.done():
                task.cancel()
        await asyncio.gather(
            *(
                task
                for task in (
                    self._reader_task,
                    self._stderr_task,
                    *self._callback_tasks,
                )
                if task is not None
            ),
            return_exceptions=True,
        )
        self._callback_tasks.clear()
