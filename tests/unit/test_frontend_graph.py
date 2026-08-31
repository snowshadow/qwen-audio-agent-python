from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from qwen_audio_agent.domain.frontend import (
    MAX_PROCESSED_CALL_IDS,
    FrontendExecutionContext,
    FrontendTool,
    FrontendToolCall,
    SubmissionReceipt,
    ToolMode,
    ToolPolicy,
    WorkSubmissionRequest,
    bounded_unique_ids,
)
from qwen_audio_agent.graphs.frontend import FrontendPorts, FrontendTurnGraph


def context(*, turn: str = "turn-one", capabilities: tuple[str, ...] = ()):
    return FrontendExecutionContext(
        owner_id="owner",
        session_id="voice",
        turn_id=turn,
        turn_generation=1,
        frontend_capabilities=capabilities,
    )


def call(call_id: str, name: str, **arguments: Any) -> FrontendToolCall:
    return FrontendToolCall(call_id=call_id, name=name, arguments=arguments)


@dataclass
class SubmissionPort:
    requests: list[WorkSubmissionRequest] = field(default_factory=list)

    def submit(self, request: WorkSubmissionRequest) -> SubmissionReceipt:
        self.requests.append(request)
        return SubmissionReceipt(task_id=f"task_{len(self.requests)}")


def test_processed_id_reducer_is_unique_and_bounded() -> None:
    current = [f"call_{index}" for index in range(MAX_PROCESSED_CALL_IDS)]
    reduced = bounded_unique_ids(current, ["call_1", "call_new"])

    assert len(reduced) == MAX_PROCESSED_CALL_IDS
    assert reduced[0] == "call_1"
    assert reduced[-1] == "call_new"
    assert reduced.count("call_1") == 1


@pytest.mark.asyncio
async def test_routes_inline_control_and_background_through_distinct_ports() -> None:
    calls: list[tuple[str, str]] = []
    submissions = SubmissionPort()

    async def inline(tool_call: FrontendToolCall, _: FrontendExecutionContext):
        calls.append(("inline", tool_call.call_id))
        return {"status": "ok", "time": "12:00"}

    def control(tool_call: FrontendToolCall, _: FrontendExecutionContext):
        calls.append(("control", tool_call.call_id))
        return {"status": "working"}

    graph = FrontendTurnGraph(
        (
            FrontendTool("get_current_time", ToolPolicy(ToolMode.INLINE)),
            FrontendTool("get_agent_task_status", ToolPolicy(ToolMode.CONTROL)),
            FrontendTool("spawn_thinking", ToolPolicy(ToolMode.BACKGROUND)),
        ),
        FrontendPorts(
            inline={"get_current_time": inline},
            control={"get_agent_task_status": control},
            submission=submissions,
        ),
    )
    turn = context()

    inline_state = await graph.ainvoke(call("one", "get_current_time"), turn)
    control_state = await graph.ainvoke(call("two", "get_agent_task_status"), turn)
    background_state = await graph.ainvoke(
        call("three", "spawn_thinking", objective="  整理   会议纪要  "),
        turn,
    )

    assert inline_state["result"] == {"status": "ok", "time": "12:00"}
    assert control_state["result"] == {"status": "working"}
    assert background_state["result"]["status"] == "accepted"
    assert calls == [("inline", "one"), ("control", "two")]
    assert len(submissions.requests) == 1
    request = submissions.requests[0]
    assert request.objective == "整理 会议纪要"
    assert request.submission_key.startswith("delegation:voice:turn-one:")
    assert not hasattr(request, "history")
    assert not hasattr(request, "memory")


@pytest.mark.asyncio
async def test_dedupe_budget_and_visibility_fail_before_handlers() -> None:
    executed: list[str] = []

    def handler(tool_call: FrontendToolCall, _: FrontendExecutionContext):
        executed.append(tool_call.call_id)
        return {"status": "ok"}

    graph = FrontendTurnGraph(
        (
            FrontendTool("limited", ToolPolicy(ToolMode.INLINE, max_calls_per_turn=2)),
            FrontendTool(
                "gated",
                ToolPolicy(ToolMode.INLINE, required_capabilities=("knowledge",)),
            ),
        ),
        FrontendPorts(inline={"limited": handler, "gated": handler}),
        max_calls_per_turn=2,
    )
    turn = context()

    first = await graph.ainvoke(call("one", "limited", value=1), turn)
    replay = await graph.ainvoke(call("one", "limited", value=2), turn)
    repeated = await graph.ainvoke(call("two", "limited", value=1), turn)
    second = await graph.ainvoke(call("three", "limited", value=2), turn)
    over_tool_budget = await graph.ainvoke(call("four", "limited", value=3), turn)
    unavailable = await graph.ainvoke(call("five", "gated"), turn)
    unsupported = await graph.ainvoke(call("six", "missing"), turn)

    assert first["result"]["status"] == "ok"
    assert replay["result"]["status"] == "duplicate"
    assert repeated["result"]["status"] == "duplicate"
    assert second["result"]["status"] == "ok"
    assert over_tool_budget["result"]["error_code"] == "tool_loop_limit"
    assert unavailable["result"]["error_code"] == "tool_unavailable"
    assert unsupported["result"]["error_code"] == "unsupported_tool"
    assert executed == ["one", "three"]


@pytest.mark.asyncio
async def test_new_turn_gets_a_fresh_budget() -> None:
    executed: list[str] = []

    def handler(tool_call: FrontendToolCall, _: FrontendExecutionContext):
        executed.append(tool_call.call_id)
        return {"status": "ok"}

    graph = FrontendTurnGraph(
        (FrontendTool("only", ToolPolicy(ToolMode.INLINE)),),
        FrontendPorts(inline={"only": handler}),
        max_calls_per_turn=1,
    )

    await graph.ainvoke(call("one", "only", value=1), context(turn="first"))
    blocked = await graph.ainvoke(call("two", "only", value=2), context(turn="first"))
    fresh = await graph.ainvoke(call("three", "only", value=3), context(turn="second"))

    assert blocked["result"]["error_code"] == "tool_loop_limit"
    assert fresh["result"]["status"] == "ok"
    assert executed == ["one", "three"]


@pytest.mark.asyncio
async def test_result_size_and_handler_time_are_bounded() -> None:
    async def slow(_: FrontendToolCall, __: FrontendExecutionContext):
        await asyncio.sleep(0.05)
        return {"status": "ok"}

    def large(_: FrontendToolCall, __: FrontendExecutionContext):
        return {"status": "ok", "content": "x" * 200}

    graph = FrontendTurnGraph(
        (
            FrontendTool("slow", ToolPolicy(ToolMode.INLINE, timeout_ms=1)),
            FrontendTool("large", ToolPolicy(ToolMode.INLINE, max_result_bytes=50)),
        ),
        FrontendPorts(inline={"slow": slow, "large": large}),
    )

    timeout = await graph.ainvoke(call("slow", "slow"), context())
    oversized = await graph.ainvoke(call("large", "large"), context())

    assert timeout["result"]["error_code"] == "tool_timeout"
    assert oversized["result"]["error_code"] == "tool_result_too_large"


@pytest.mark.asyncio
async def test_side_effect_interrupts_before_execution_and_resumes_with_command() -> None:
    effects: list[dict[str, Any]] = []
    checkpointer = InMemorySaver()
    store = InMemoryStore()

    def create_issue(tool_call: FrontendToolCall, _: FrontendExecutionContext):
        effects.append(dict(tool_call.arguments))
        return {"status": "ok", "issue": 42}

    graph = FrontendTurnGraph(
        (
            FrontendTool(
                "mcp__issues__create",
                ToolPolicy(
                    ToolMode.INLINE,
                    read_only=False,
                    requires_approval=True,
                ),
                description="Create an issue",
            ),
        ),
        FrontendPorts(inline={"mcp__issues__create": create_issue}),
        checkpointer=checkpointer,
        store=store,
    )

    paused = await graph.ainvoke(
        call("write", "mcp__issues__create", title="Fix it"),
        context(),
    )
    authorization_id = paused["result"]["authorization_id"]

    assert paused["result"]["status"] == "confirmation_required"
    assert paused["__interrupt__"][0].value["authorization_id"] == authorization_id
    assert effects == []

    wrong_owner = await graph.aresume_authorization(
        authorization_id,
        "allow",
        owner_id="other-owner",
    )
    assert wrong_owner["result"]["error_code"] == "external_authorization_not_pending"
    assert effects == []

    completed = await graph.aresume_authorization(
        authorization_id,
        "allow",
        owner_id="owner",
    )
    assert completed["result"] == {"status": "ok", "issue": 42}
    assert effects == [{"title": "Fix it"}]

    replay = await graph.aresume_authorization(
        authorization_id,
        "allow",
        owner_id="owner",
    )
    assert replay["result"]["error_code"] == "external_authorization_not_pending"
    assert effects == [{"title": "Fix it"}]


@pytest.mark.asyncio
async def test_rejected_authorization_never_executes_side_effect() -> None:
    executed = False

    def destructive(_: FrontendToolCall, __: FrontendExecutionContext):
        nonlocal executed
        executed = True

    graph = FrontendTurnGraph(
        (
            FrontendTool(
                "delete",
                ToolPolicy(ToolMode.INLINE, read_only=False, requires_approval=True),
            ),
        ),
        FrontendPorts(inline={"delete": destructive}),
    )
    paused = await graph.ainvoke(call("delete-one", "delete", item=42), context())

    result = await graph.aresume_authorization(
        paused["result"]["authorization_id"],
        "reject",
        owner_id="owner",
    )

    assert result["result"]["status"] == "rejected"
    assert executed is False


@pytest.mark.asyncio
async def test_background_port_returns_only_an_intake_receipt() -> None:
    completion = asyncio.Event()
    submissions: list[WorkSubmissionRequest] = []

    class IntakeOnlyPort:
        def submit(self, request: WorkSubmissionRequest) -> SubmissionReceipt:
            submissions.append(request)
            # A Work completion primitive may exist elsewhere, but intake does
            # not and cannot wait for it.
            assert not completion.is_set()
            return SubmissionReceipt("task_1")

    graph = FrontendTurnGraph(
        (FrontendTool("spawn_thinking", ToolPolicy(ToolMode.BACKGROUND)),),
        FrontendPorts(submission=IntakeOnlyPort()),
    )

    result = await asyncio.wait_for(
        graph.ainvoke(call("spawn", "spawn_thinking", objective="Do it"), context()),
        timeout=0.1,
    )

    assert result["result"]["status"] == "accepted"
    assert result["result"]["task_id"] == "task_1"
    assert len(submissions) == 1
