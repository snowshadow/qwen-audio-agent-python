"""Protocol-neutral contracts for one bounded frontend tool turn.

The approved migration keeps the realtime conversation and transport outside
LangGraph.  These types therefore describe only one already assembled function
call and the narrow ports it may use.  In particular, a background call can
submit a :class:`WorkSubmissionRequest`; it cannot see a backend adapter or run
the background work itself.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, Protocol, TypedDict

MAX_PROCESSED_CALL_IDS = 500
DEFAULT_MAX_RESULT_BYTES = 64 * 1024
DEFAULT_TOOL_TIMEOUT_MS = 10_000

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def bounded_unique_ids(
    current: Sequence[str] | None,
    updates: Sequence[str] | str | None,
) -> list[str]:
    """Append unique identifiers while retaining only the newest bounded set.

    LangGraph calls reducers for both graph input and node updates, so this
    function deliberately accepts either one identifier or a sequence.  It is
    also used for call signatures; both collections have the same replay-risk
    versus memory-bound trade-off as the JavaScript runtime's bounded Sets.
    """

    combined = [str(value).strip() for value in (current or ()) if str(value).strip()]
    if isinstance(updates, str):
        incoming = (updates,)
    else:
        incoming = updates or ()
    seen = set(combined)
    for value in incoming:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            combined.append(normalized)
            seen.add(normalized)
    return combined[-MAX_PROCESSED_CALL_IDS:]


class ToolMode(StrEnum):
    """Execution boundary declared by a frontend tool."""

    INLINE = "inline"
    BACKGROUND = "background"
    CONTROL = "control"


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """Safety and visibility limits enforced before a handler can run."""

    mode: ToolMode | str
    read_only: bool = True
    requires_approval: bool = False
    timeout_ms: int = DEFAULT_TOOL_TIMEOUT_MS
    max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES
    max_calls_per_turn: int | None = None
    repeat_handling: str = "block"
    required_capabilities: tuple[str, ...] = ()
    required_client_states: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            mode = ToolMode(self.mode)
        except ValueError as exc:
            choices = ", ".join(item.value for item in ToolMode)
            raise ValueError(f"frontend tool mode must be one of: {choices}") from exc
        object.__setattr__(self, "mode", mode)
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")
        if self.max_result_bytes <= 0:
            raise ValueError("max_result_bytes must be a positive integer")
        if self.max_calls_per_turn is not None and self.max_calls_per_turn <= 0:
            raise ValueError("max_calls_per_turn must be a positive integer")
        if self.repeat_handling not in {"block", "handler"}:
            raise ValueError("repeat_handling must be 'block' or 'handler'")
        object.__setattr__(
            self,
            "required_capabilities",
            tuple(str(value) for value in self.required_capabilities),
        )
        object.__setattr__(
            self,
            "required_client_states",
            tuple(str(value) for value in self.required_client_states),
        )


@dataclass(frozen=True, slots=True)
class FrontendTool:
    """Registry metadata; executable functions remain injected ports."""

    name: str
    policy: ToolPolicy
    description: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("frontend tool requires a name")
        object.__setattr__(self, "name", name)


@dataclass(frozen=True, slots=True)
class FrontendToolCall:
    """One provider-normalized function call."""

    call_id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "call_id", self.call_id.strip())
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "arguments", dict(self.arguments))

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True, slots=True)
class FrontendExecutionContext:
    """Trusted correlation data supplied by the realtime runtime."""

    owner_id: str
    session_id: str
    turn_id: str
    turn_generation: int
    frontend_capabilities: tuple[str, ...] = ()
    client_states: tuple[str, ...] = ()
    input_parts: tuple[Mapping[str, Any], ...] = ()
    client_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("owner_id", "session_id", "turn_id"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} is required")
            object.__setattr__(self, field_name, value)
        if self.turn_generation < 0:
            raise ValueError("turn_generation must be non-negative")
        object.__setattr__(
            self,
            "frontend_capabilities",
            tuple(str(value) for value in self.frontend_capabilities),
        )
        object.__setattr__(
            self,
            "client_states",
            tuple(str(value) for value in self.client_states),
        )
        object.__setattr__(
            self,
            "input_parts",
            tuple(dict(value) for value in self.input_parts if isinstance(value, Mapping))[:8],
        )
        object.__setattr__(self, "client_context", dict(self.client_context))

    @property
    def turn_key(self) -> str:
        return ":".join((self.owner_id, self.session_id, self.turn_id, str(self.turn_generation)))


@dataclass(frozen=True, slots=True)
class WorkSubmissionRequest:
    """The complete and intentionally narrow frontend-to-work handoff."""

    owner_id: str
    session_id: str
    turn_id: str
    call_id: str
    objective: str
    input_parts: tuple[Mapping[str, Any], ...]
    input_refs: tuple[str, ...]
    submission_key: str


@dataclass(frozen=True, slots=True)
class SubmissionReceipt:
    """Receipt from the work intake boundary; never a work result."""

    task_id: str
    reused: bool = False

    def __post_init__(self) -> None:
        task_id = self.task_id.strip()
        if not task_id:
            raise ValueError("submission receipt requires task_id")
        object.__setattr__(self, "task_id", task_id)

    def as_tool_result(self) -> dict[str, JsonValue]:
        if self.reused:
            return {
                "status": "duplicate",
                "task_id": self.task_id,
                "message": "同一工作此前已受理。不再重复提交。",
            }
        return {
            "status": "accepted",
            "task_id": self.task_id,
            "message": "工作已受理。",
        }


@dataclass(frozen=True, slots=True)
class FrontendToolResult:
    """Convenience result accepted from injected tool handlers."""

    status: str
    value: JsonValue | None = None
    error_code: str | None = None
    retryable: bool = False

    def as_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {"status": self.status}
        if self.value is not None:
            result["value"] = self.value
        if self.error_code:
            result.update(
                {
                    "error": True,
                    "error_code": self.error_code,
                    "retryable": self.retryable,
                }
            )
        return result


class WorkSubmissionPort(Protocol):
    """Atomic work intake. Implementations must not wait for work completion."""

    def submit(
        self,
        request: WorkSubmissionRequest,
    ) -> SubmissionReceipt | Mapping[str, Any] | Awaitable[SubmissionReceipt | Mapping[str, Any]]:
        """Persist or reuse one Work and return its intake receipt."""


type ToolHandler = Callable[
    [FrontendToolCall, FrontendExecutionContext],
    FrontendToolResult | JsonValue | Mapping[str, Any] | Awaitable[Any],
]


class FrontendTurnState(TypedDict, total=False):
    """Checkpointed state for one realtime turn's bounded tool loop."""

    request_context: dict[str, Any]
    turn_key: str
    call: dict[str, Any]
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    mode: str
    route: str
    call_signature: str
    started_at_ms: int
    total_calls: int
    calls_by_tool: dict[str, int]
    admitted: bool
    result: dict[str, Any] | None
    pending_authorization: dict[str, Any] | None
    authorization_decision: str | None
    processed_call_ids: Annotated[list[str], bounded_unique_ids]
    processed_signatures: Annotated[list[str], bounded_unique_ids]


BASE_FRONTEND_TOOLS: tuple[FrontendTool, ...] = (
    FrontendTool(
        "spawn_thinking",
        ToolPolicy(mode=ToolMode.BACKGROUND, repeat_handling="handler"),
    ),
    FrontendTool("schedule_reminder", ToolPolicy(mode=ToolMode.INLINE, read_only=False)),
    FrontendTool("cancel_agent_task", ToolPolicy(mode=ToolMode.CONTROL, read_only=False)),
    FrontendTool("get_agent_task_status", ToolPolicy(mode=ToolMode.CONTROL)),
    FrontendTool("get_current_time", ToolPolicy(mode=ToolMode.INLINE)),
    FrontendTool("memory", ToolPolicy(mode=ToolMode.INLINE, read_only=False)),
    FrontendTool("notes", ToolPolicy(mode=ToolMode.INLINE, read_only=False)),
    FrontendTool(
        "respond_agent_permission",
        ToolPolicy(mode=ToolMode.CONTROL, read_only=False),
    ),
)
