"""Stable frontend tool catalog shared by provider setup and LangGraph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_audio_agent.domain.frontend import FrontendTool, ToolMode, ToolPolicy


def _function(
    name: str,
    description: str,
    properties: Mapping[str, Any],
    *,
    required: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": dict(properties),
                **({"required": list(required)} if required else {}),
                "additionalProperties": False,
            },
        },
    }


FRONTEND_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    _function(
        "spawn_thinking",
        "把需要工具、文件、应用、代码、设备控制或多步执行的请求交给后台。只提交忠实、自包含的目标，不写执行计划；调用立即返回受理回执，不等待完成。",
        {
            "objective": {"type": "string", "description": "忠实且自包含的用户目标。"},
        },
        required=("objective",),
    ),
    _function(
        "schedule_reminder",
        "创建定时提醒或定时任务。先用 get_current_time 确定准确时区和时间。",
        {
            "execute_at": {"type": "string", "description": "ISO 8601 触发时间。"},
            "reminder": {"type": "string", "description": "提醒内容或任务目标。"},
            "type": {"type": "string", "enum": ["reminder", "task"]},
            "recurrence": {
                "type": "string",
                "enum": ["once"],
                "description": "当前仅支持单次触发。",
            },
        },
        required=("execute_at", "reminder"),
    ),
    _function(
        "cancel_agent_task",
        (
            "取消此前开始且仍可取消的工作、定时任务或提醒。"
            "只能使用系统已返回的 task_id；目标不清楚时先查询状态。"
        ),
        {
            "task_id": {"type": "string"},
            "all": {"type": "boolean"},
        },
    ),
    _function(
        "get_agent_task_status",
        (
            "仅在用户主动询问此前工作的状态、进度或列表时读取 Gateway 自己的 Task 记录；"
            "不会创建新工作。"
        ),
        {
            "task_id": {"type": "string"},
            "question": {"type": "string"},
            "list_all": {"type": "boolean"},
        },
    ),
    _function(
        "get_current_time",
        "获取用户本地时区中准确的日期、时间和星期。",
        {},
    ),
    _function(
        "memory",
        (
            "原子读取、追加或精确替换 USER.md 与 MEMORY.md。"
            "user 保存长期个性化；memory 保存只用于理解和回答的事实。"
            "不得保存密码、密钥、验证码或令牌。"
        ),
        {
            "action": {"type": "string", "enum": ["read", "append", "replace"]},
            "document": {"type": "string", "enum": ["all", "user", "memory"]},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
            "content": {"type": "string"},
        },
        required=("action",),
    ),
    _function(
        "notes",
        "管理当前用户的持久化命名清单。清单条目不是长期记忆；clear/drop 仅在用户明确要求时使用。",
        {
            "action": {
                "type": "string",
                "enum": ["lists", "show", "add", "remove", "clear", "drop"],
            },
            "list": {"type": "string"},
            "items": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        },
        required=("action",),
    ),
    _function(
        "respond_agent_permission",
        (
            "只回复当前仍在等待的后台权限请求。明确同意当前这一项用 always，"
            "明确拒绝用 reject；always 是兼容字段值，只授权当前请求；"
            "不得猜造 authorization_id。"
        ),
        {
            "authorization_id": {"type": "string"},
            "decision": {"type": "string", "enum": ["always", "reject"]},
        },
        required=("authorization_id", "decision"),
    ),
)

_ENTER_SLEEP_TOOL = _function(
    "enter_sleep",
    (
        "让当前语音入口进入它支持的休眠状态。仅在用户明确要求当前入口"
        "退下、隐藏、收起或暂时休息时调用；不用于取消后台工作、静音或退出应用。"
    ),
    {},
)


def provider_frontend_tool_definitions(
    *,
    client_states: Sequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    """Return only tools the current realtime client can actually use."""

    states = {str(value) for value in client_states}
    return (
        (*FRONTEND_TOOL_DEFINITIONS, _ENTER_SLEEP_TOOL)
        if "sleeping" in states
        else FRONTEND_TOOL_DEFINITIONS
    )


def frontend_graph_tools() -> tuple[FrontendTool, ...]:
    modes = {
        "spawn_thinking": ToolMode.BACKGROUND,
        "cancel_agent_task": ToolMode.CONTROL,
        "get_agent_task_status": ToolMode.CONTROL,
        "respond_agent_permission": ToolMode.CONTROL,
        "enter_sleep": ToolMode.CONTROL,
    }
    tools: list[FrontendTool] = []
    for definition in (*FRONTEND_TOOL_DEFINITIONS, _ENTER_SLEEP_TOOL):
        function = definition["function"]
        name = str(function["name"])
        tools.append(
            FrontendTool(
                name=name,
                description=str(function["description"]),
                input_schema=dict(function["parameters"]),
                policy=ToolPolicy(
                    mode=modes.get(name, ToolMode.INLINE),
                    repeat_handling="handler" if name == "spawn_thinking" else "block",
                    required_client_states=("sleeping",) if name == "enter_sleep" else (),
                ),
            )
        )
    return tuple(tools)


def build_frontend_instructions(
    *,
    runtime_root: str | Path,
    assistant_profile_path: str | Path | None = None,
    user_preferences: str = "",
    user_memory: str = "",
    recent_messages: Sequence[Mapping[str, Any]] = (),
    client_context: Mapping[str, Any] | None = None,
) -> str:
    root = Path(runtime_root)
    prompt_path = root / "config" / "frontend-agent" / "PROMPT.md"
    packaged_profile = root / "config" / "frontend-agent" / "ASSISTANT.md"
    profile_path = Path(assistant_profile_path) if assistant_profile_path else packaged_profile
    prompt = prompt_path.read_text(encoding="utf-8")
    profile = profile_path.read_text(encoding="utf-8") if profile_path.exists() else ""
    recent = [
        {
            "role": str(message.get("role") or ""),
            "content": str(message.get("content") or "")[:4_000],
        }
        for message in recent_messages[-12:]
        if isinstance(message, Mapping)
    ]
    runtime = dict(client_context or {})
    return "\n\n".join(
        (
            prompt,
            '# Assistant Profile\n<assistant_profile authority="persona_only">\n'
            f"{profile}\n</assistant_profile>",
            f"# User Preferences\n<user_preferences>\n{user_preferences}\n</user_preferences>",
            f"# User Memory\n<user_memory>\n{user_memory}\n</user_memory>",
            f"# Recent Conversation\n<recent_conversation>\n{recent!r}\n</recent_conversation>",
            f"# Runtime Context\n<runtime_context>\n{runtime!r}\n</runtime_context>",
        )
    )


__all__ = [
    "FRONTEND_TOOL_DEFINITIONS",
    "build_frontend_instructions",
    "frontend_graph_tools",
    "provider_frontend_tool_definitions",
]
