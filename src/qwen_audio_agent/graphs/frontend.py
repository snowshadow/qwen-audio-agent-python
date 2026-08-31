"""LangGraph implementation of the bounded Frontend Tool Runtime."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, RunnableConfig, interrupt

from qwen_audio_agent.domain.frontend import (
    FrontendExecutionContext,
    FrontendTool,
    FrontendToolCall,
    FrontendToolResult,
    FrontendTurnState,
    JsonValue,
    SubmissionReceipt,
    ToolHandler,
    ToolMode,
    WorkSubmissionPort,
    WorkSubmissionRequest,
)

DEFAULT_MAX_CALLS_PER_TURN = 12
DEFAULT_MAX_DURATION_MS = 30_000
AUTHORIZATION_NAMESPACE = ("qwen_audio_agent", "frontend_authorizations")


@dataclass(frozen=True, slots=True)
class FrontendPorts:
    """Injected effects grouped by their architecture boundary."""

    inline: Mapping[str, ToolHandler] = field(default_factory=dict)
    control: Mapping[str, ToolHandler] = field(default_factory=dict)
    submission: WorkSubmissionPort | None = None


def _failure(
    error_code: str,
    message: str,
    *,
    retryable: bool = False,
    status: str = "failed",
) -> dict[str, JsonValue]:
    return {
        "status": status,
        "error": True,
        "error_code": error_code,
        "user_message": message,
        "retryable": retryable,
    }


def _stable_signature(name: str, arguments: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        arguments,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return f"{name}:{encoded}"


def _authorization_id(state: FrontendTurnState) -> str:
    source = "\0".join(
        (
            state["turn_key"],
            state["call_id"],
            state["call_signature"],
        )
    )
    digest = hashlib.sha256(source.encode()).hexdigest()[:24]
    return f"frontend_auth_{digest}"


def _submission_key(context: FrontendExecutionContext, call: FrontendToolCall) -> str:
    objective = " ".join(str(call.arguments.get("objective", "")).split())
    fingerprint = hashlib.sha256(objective.encode()).hexdigest()[:24]
    return ":".join(
        ("delegation", context.session_id, context.turn_id or call.call_id, fingerprint)
    )


def _normalize_handler_result(value: Any) -> dict[str, Any]:
    if isinstance(value, FrontendToolResult):
        return value.as_dict()
    if value is None:
        return {"status": "ok"}
    if isinstance(value, Mapping):
        return dict(value)
    return {"status": "ok", "value": value}


class FrontendTurnGraph:
    """One checkpointed StateGraph per realtime turn.

    The graph owns admission, routing and an optional approval pause.  Audio,
    model responses, Work execution and backend selection stay outside it.
    Callers should serialize calls for the same ``thread_id``, matching the
    realtime runtime's existing per-turn mailbox.
    """

    def __init__(
        self,
        tools: Iterable[FrontendTool],
        ports: FrontendPorts,
        *,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        store: BaseStore | None = None,
        max_calls_per_turn: int = DEFAULT_MAX_CALLS_PER_TURN,
        max_duration_ms: int = DEFAULT_MAX_DURATION_MS,
        clock_ms: Any | None = None,
    ) -> None:
        if max_calls_per_turn <= 0:
            raise ValueError("max_calls_per_turn must be a positive integer")
        if max_duration_ms <= 0:
            raise ValueError("max_duration_ms must be a positive integer")
        entries: dict[str, FrontendTool] = {}
        for tool in tools:
            if tool.name in entries:
                raise ValueError(f"duplicate frontend tool: {tool.name}")
            if tool.policy.mode is ToolMode.BACKGROUND and tool.name != "spawn_thinking":
                raise ValueError("spawn_thinking is the only supported background tool")
            entries[tool.name] = tool
        if not entries:
            raise ValueError("FrontendTurnGraph requires at least one tool")

        self.tools = entries
        self.ports = ports
        self.max_calls_per_turn = max_calls_per_turn
        self.max_duration_ms = max_duration_ms
        self.clock_ms = clock_ms or (lambda: int(time.time() * 1_000))
        self.checkpointer = checkpointer or InMemorySaver()
        self.store = store or InMemoryStore()
        self.compiled = self._build().compile(
            checkpointer=self.checkpointer,
            store=self.store,
            name="frontend_turn",
        )

    def _build(self) -> StateGraph[FrontendTurnState]:
        builder = StateGraph(FrontendTurnState)
        builder.add_node("validate", self._validate)
        builder.add_node("policy", self._policy)
        builder.add_node("prepare_authorization", self._prepare_authorization)
        builder.add_node("await_authorization", self._await_authorization)
        builder.add_node("inline", self._inline)
        builder.add_node("background", self._background)
        builder.add_node("control", self._control)
        builder.add_node("bound_result", self._bound_result)
        builder.add_node("finish", self._finish)
        builder.add_edge(START, "validate")
        return builder

    @staticmethod
    def config_for(context: FrontendExecutionContext) -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": (
                    f"frontend:{context.owner_id}:{context.session_id}:"
                    f"{context.turn_id}:{context.turn_generation}"
                )
            }
        }

    async def ainvoke(
        self,
        call: FrontendToolCall | Mapping[str, Any],
        context: FrontendExecutionContext,
        *,
        config: RunnableConfig | None = None,
    ) -> FrontendTurnState:
        """Run one normalized frontend call in its turn-scoped graph thread."""

        raw_call = call.as_dict() if isinstance(call, FrontendToolCall) else dict(call)
        graph_input: FrontendTurnState = {
            "request_context": {
                "owner_id": context.owner_id,
                "session_id": context.session_id,
                "turn_id": context.turn_id,
                "turn_generation": context.turn_generation,
                "frontend_capabilities": list(context.frontend_capabilities),
                "client_states": list(context.client_states),
                "input_parts": [dict(value) for value in context.input_parts],
                "client_context": dict(context.client_context),
                "turn_key": context.turn_key,
            },
            "call": raw_call,
        }
        output = await self.compiled.ainvoke(
            graph_input,
            config=config or self.config_for(context),
        )
        return cast(FrontendTurnState, output)

    async def aresume_authorization(
        self,
        authorization_id: str,
        decision: Literal["allow", "reject"],
        *,
        owner_id: str,
    ) -> FrontendTurnState:
        """Resume exactly one interrupted side effect via LangGraph Command."""

        normalized_id = authorization_id.strip()
        item = await self.store.aget(AUTHORIZATION_NAMESPACE, normalized_id)
        if item is None or decision not in {"allow", "reject"}:
            return {
                "result": _failure(
                    "external_authorization_not_pending",
                    "没有找到仍在等待决定的前台工具请求。",
                )
            }
        metadata = item.value
        if metadata.get("owner_id") != owner_id:
            return {
                "result": _failure(
                    "external_authorization_not_pending",
                    "没有找到仍在等待决定的前台工具请求。",
                )
            }
        config: RunnableConfig = {"configurable": {"thread_id": str(metadata["thread_id"])}}
        output = await self.compiled.ainvoke(
            Command(
                resume={
                    "authorization_id": normalized_id,
                    "decision": decision,
                }
            ),
            config=config,
        )
        return cast(FrontendTurnState, output)

    def _validate(
        self,
        state: FrontendTurnState,
    ) -> Command[Literal["policy", "finish"]]:
        context_data = state.get("request_context") or {}
        requested_turn_key = str(context_data.get("turn_key", ""))
        existing_turn_key = state.get("turn_key")
        base_update: FrontendTurnState = {
            "result": None,
            "pending_authorization": None,
            "authorization_decision": None,
            "admitted": False,
        }
        if not requested_turn_key or (
            existing_turn_key and existing_turn_key != requested_turn_key
        ):
            return Command(
                update={
                    **base_update,
                    "result": _failure("stale_turn", "这次工具调用已被新的对话轮次取代。"),
                },
                goto="finish",
            )

        raw_call = state.get("call") or {}
        call_id = str(raw_call.get("call_id") or "").strip()
        tool_name = str(raw_call.get("name") or "").strip()
        raw_arguments = raw_call.get("arguments", {})
        if isinstance(raw_arguments, str):
            try:
                raw_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                raw_arguments = None
        if not call_id or not tool_name or not isinstance(raw_arguments, Mapping):
            return Command(
                update={
                    **base_update,
                    "turn_key": existing_turn_key or requested_turn_key,
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "result": _failure("invalid_tool_call", "工具调用缺少必要的识别信息或参数。"),
                },
                goto="finish",
            )
        arguments = dict(raw_arguments)
        try:
            signature = _stable_signature(tool_name, arguments)
        except (TypeError, ValueError):
            return Command(
                update={
                    **base_update,
                    "turn_key": existing_turn_key or requested_turn_key,
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "result": _failure("invalid_tool_arguments", "工具参数无法安全解析。"),
                },
                goto="finish",
            )

        update: FrontendTurnState = {
            **base_update,
            "turn_key": existing_turn_key or requested_turn_key,
            "call_id": call_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "call_signature": signature,
        }
        if call_id in state.get("processed_call_ids", ()):
            update["result"] = {
                "status": "duplicate",
                "message": "这次工具调用已经处理。不再重复执行。",
            }
            return Command(update=update, goto="finish")
        if tool_name not in self.tools:
            update["result"] = _failure("unsupported_tool", "当前无法执行这个操作。")
            return Command(update=update, goto="finish")
        return Command(update=update, goto="policy")

    def _policy(
        self,
        state: FrontendTurnState,
    ) -> Command[Literal["prepare_authorization", "inline", "background", "control", "finish"]]:
        tool = self.tools[state["tool_name"]]
        policy = tool.policy
        context_data = state["request_context"]
        capabilities = {str(value) for value in context_data.get("frontend_capabilities", ())}
        client_states = {str(value) for value in context_data.get("client_states", ())}
        if not set(policy.required_capabilities).issubset(capabilities) or not set(
            policy.required_client_states
        ).issubset(client_states):
            return Command(
                update={
                    "result": _failure("tool_unavailable", "当前前台没有启用这个能力。"),
                },
                goto="finish",
            )

        now_ms = int(self.clock_ms())
        started_at = state.get("started_at_ms", now_ms)
        total_calls = state.get("total_calls", 0)
        calls_by_tool = dict(state.get("calls_by_tool", {}))
        tool_calls = calls_by_tool.get(tool.name, 0)
        if now_ms - started_at > self.max_duration_ms:
            return Command(
                update={
                    "started_at_ms": started_at,
                    "result": _failure(
                        "tool_loop_limit",
                        "本轮工具调用已超过时间边界。",
                        retryable=True,
                    ),
                },
                goto="finish",
            )
        if total_calls >= self.max_calls_per_turn or (
            policy.max_calls_per_turn is not None and tool_calls >= policy.max_calls_per_turn
        ):
            return Command(
                update={
                    "started_at_ms": started_at,
                    "result": _failure(
                        "tool_loop_limit",
                        "本轮工具调用已达到次数边界。",
                        retryable=True,
                    ),
                },
                goto="finish",
            )
        if policy.repeat_handling != "handler" and state["call_signature"] in state.get(
            "processed_signatures", ()
        ):
            return Command(
                update={
                    "started_at_ms": started_at,
                    "result": {
                        "status": "duplicate",
                        "message": "本轮相同操作已经处理。不再重复执行。",
                    },
                },
                goto="finish",
            )

        calls_by_tool[tool.name] = tool_calls + 1
        route = policy.mode.value
        update: FrontendTurnState = {
            "started_at_ms": started_at,
            "total_calls": total_calls + 1,
            "calls_by_tool": calls_by_tool,
            "mode": route,
            "route": route,
            "admitted": True,
        }
        if policy.requires_approval:
            return Command(update=update, goto="prepare_authorization")
        return Command(update=update, goto=route)

    async def _prepare_authorization(
        self,
        state: FrontendTurnState,
        config: RunnableConfig,
    ) -> Command[Literal["await_authorization"]]:
        authorization_id = _authorization_id(state)
        tool = self.tools[state["tool_name"]]
        operation = " ".join((tool.description or tool.name).split())[:400]
        pending = {
            "authorization_id": authorization_id,
            "tool_name": tool.name,
            "operation": operation,
        }
        thread_id = str(config.get("configurable", {}).get("thread_id", ""))
        await self.store.aput(
            AUTHORIZATION_NAMESPACE,
            authorization_id,
            {
                "thread_id": thread_id,
                "owner_id": state["request_context"]["owner_id"],
            },
        )
        return Command(
            update={
                "pending_authorization": pending,
                "result": {
                    "status": "confirmation_required",
                    **pending,
                    "message": "此操作会修改外部系统。必须先获得用户明确同意。",
                },
            },
            goto="await_authorization",
        )

    def _await_authorization(
        self,
        state: FrontendTurnState,
    ) -> Command[Literal["inline", "control", "finish"]]:
        pending = state["pending_authorization"] or {}
        resumed = interrupt(
            {
                "kind": "frontend_tool_authorization",
                **pending,
            }
        )
        if not isinstance(resumed, Mapping):
            return Command(
                update={
                    "result": _failure(
                        "invalid_authorization_decision",
                        "没有获得有效的前台工具授权决定。",
                    )
                },
                goto="finish",
            )
        authorization_id = str(resumed.get("authorization_id", ""))
        decision = str(resumed.get("decision", ""))
        if authorization_id != pending.get("authorization_id") or decision not in {
            "allow",
            "reject",
        }:
            return Command(
                update={
                    "result": _failure(
                        "invalid_authorization_decision",
                        "授权决定与当前等待的操作不匹配。",
                    )
                },
                goto="finish",
            )
        if decision == "reject":
            return Command(
                update={
                    "authorization_decision": decision,
                    "result": {"status": "rejected", "message": "用户拒绝了这次操作。"},
                },
                goto="finish",
            )
        route = cast(Literal["inline", "control"], state["route"])
        return Command(
            update={"authorization_decision": decision, "result": None},
            goto=route,
        )

    def _execution_context(self, state: FrontendTurnState) -> FrontendExecutionContext:
        data = state["request_context"]
        return FrontendExecutionContext(
            owner_id=str(data["owner_id"]),
            session_id=str(data["session_id"]),
            turn_id=str(data["turn_id"]),
            turn_generation=int(data["turn_generation"]),
            frontend_capabilities=tuple(data.get("frontend_capabilities", ())),
            client_states=tuple(data.get("client_states", ())),
            input_parts=tuple(data.get("input_parts", ())),
            client_context=dict(data.get("client_context", {})),
        )

    def _call(self, state: FrontendTurnState) -> FrontendToolCall:
        return FrontendToolCall(
            call_id=state["call_id"],
            name=state["tool_name"],
            arguments=state["arguments"],
        )

    async def _execute_handler(
        self,
        handler: ToolHandler | None,
        state: FrontendTurnState,
    ) -> Command[Literal["bound_result", "finish"]]:
        if handler is None:
            return Command(
                update={"result": _failure("tool_unavailable", "当前前台没有启用这个能力。")},
                goto="finish",
            )
        policy = self.tools[state["tool_name"]].policy
        elapsed_ms = int(self.clock_ms()) - state["started_at_ms"]
        remaining_ms = max(1, self.max_duration_ms - elapsed_ms)
        timeout_seconds = min(policy.timeout_ms, remaining_ms) / 1_000
        try:
            async with asyncio.timeout(timeout_seconds):
                value = handler(self._call(state), self._execution_context(state))
                if inspect.isawaitable(value):
                    value = await value
        except TimeoutError:
            return Command(
                update={
                    "result": _failure(
                        "tool_timeout",
                        "前台工具没有在当前轮次的时间边界内完成。",
                        retryable=True,
                    )
                },
                goto="finish",
            )
        except Exception:
            return Command(
                update={
                    "result": _failure(
                        "tool_execution_failed",
                        "前台工具暂时无法完成这次操作。",
                        retryable=True,
                    )
                },
                goto="finish",
            )
        return Command(update={"result": _normalize_handler_result(value)}, goto="bound_result")

    async def _inline(
        self,
        state: FrontendTurnState,
    ) -> Command[Literal["bound_result", "finish"]]:
        return await self._execute_handler(self.ports.inline.get(state["tool_name"]), state)

    async def _control(
        self,
        state: FrontendTurnState,
    ) -> Command[Literal["bound_result", "finish"]]:
        return await self._execute_handler(self.ports.control.get(state["tool_name"]), state)

    async def _background(
        self,
        state: FrontendTurnState,
    ) -> Command[Literal["bound_result", "finish"]]:
        if self.ports.submission is None:
            return Command(
                update={
                    "result": _failure(
                        "backend_unavailable",
                        "当前未配置后台 Agent。无法执行需要后台处理的任务。",
                    )
                },
                goto="finish",
            )
        call = self._call(state)
        objective = " ".join(str(call.arguments.get("objective", "")).split())
        if not objective:
            return Command(
                update={
                    "result": _failure(
                        "missing_objective",
                        "没有获得完整、可执行的目标。",
                        retryable=True,
                    )
                },
                goto="finish",
            )
        raw_refs = call.arguments.get("input_refs", ())
        if raw_refs is None:
            raw_refs = ()
        if (
            not isinstance(raw_refs, (list, tuple))
            or len(raw_refs) > 8
            or any(not isinstance(value, str) or not value.strip() for value in raw_refs)
        ):
            return Command(
                update={
                    "result": _failure(
                        "invalid_input_ref",
                        "引用的图片或文件标识无效。",
                        retryable=True,
                    )
                },
                goto="finish",
            )
        context = self._execution_context(state)
        request = WorkSubmissionRequest(
            owner_id=context.owner_id,
            session_id=context.session_id,
            turn_id=context.turn_id,
            call_id=call.call_id,
            objective=objective,
            input_parts=context.input_parts,
            input_refs=tuple(value.strip() for value in raw_refs),
            submission_key=_submission_key(context, call),
        )
        policy = self.tools[state["tool_name"]].policy
        elapsed_ms = int(self.clock_ms()) - state["started_at_ms"]
        remaining_ms = max(1, self.max_duration_ms - elapsed_ms)
        try:
            async with asyncio.timeout(min(policy.timeout_ms, remaining_ms) / 1_000):
                receipt: Any = self.ports.submission.submit(request)
                if inspect.isawaitable(receipt):
                    receipt = await receipt
            normalized = self._submission_receipt(receipt)
        except TimeoutError:
            return Command(
                update={
                    "result": _failure(
                        "work_submission_timeout",
                        "暂时没有在受理时间内创建工作。",
                        retryable=True,
                    )
                },
                goto="finish",
            )
        except Exception:
            return Command(
                update={
                    "result": _failure(
                        "work_submission_failed",
                        "暂时没有成功提交这次请求。",
                        retryable=True,
                    )
                },
                goto="finish",
            )
        return Command(update={"result": normalized.as_tool_result()}, goto="bound_result")

    @staticmethod
    def _submission_receipt(value: Any) -> SubmissionReceipt:
        if isinstance(value, SubmissionReceipt):
            return value
        if isinstance(value, Mapping):
            task_id = value.get("task_id", value.get("id", ""))
            return SubmissionReceipt(
                task_id=str(task_id),
                reused=bool(value.get("reused", value.get("status") == "duplicate")),
            )
        task_id = getattr(value, "task_id", getattr(value, "id", ""))
        return SubmissionReceipt(task_id=str(task_id), reused=bool(getattr(value, "reused", False)))

    def _bound_result(
        self,
        state: FrontendTurnState,
    ) -> Command[Literal["finish"]]:
        result = state.get("result") or {"status": "ok"}
        try:
            encoded = json.dumps(
                result,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        except (TypeError, ValueError, RecursionError):
            encoded = b""
        limit = self.tools[state["tool_name"]].policy.max_result_bytes
        if not encoded or len(encoded) > limit:
            result = _failure(
                "tool_result_too_large",
                "工具结果过大。无法在当前语音轮次中安全返回。",
                retryable=True,
            )
        return Command(update={"result": result}, goto="finish")

    async def _finish(
        self,
        state: FrontendTurnState,
    ) -> Command[Literal[END]]:
        pending = state.get("pending_authorization") or {}
        authorization_id = str(pending.get("authorization_id", ""))
        if authorization_id:
            await self.store.adelete(AUTHORIZATION_NAMESPACE, authorization_id)
        update: FrontendTurnState = {}
        call_id = state.get("call_id", "")
        if call_id:
            update["processed_call_ids"] = [call_id]
        if state.get("admitted") and state.get("call_signature"):
            update["processed_signatures"] = [state["call_signature"]]
        return Command(update=update, goto=END)
