"""Built-in single-step handlers for the FrontendTurnGraph."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from qwen_audio_agent.domain.frontend import FrontendExecutionContext, FrontendToolCall
from qwen_audio_agent.graphs.frontend import FrontendPorts
from qwen_audio_agent.runtime.frontend_data import JsonNotesStore, MarkdownMemoryService

from .work import WorkApplication


def _unavailable(code: str, message: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "error": True,
        "error_code": code,
        "user_message": message,
    }


def _time_zone(context: FrontendExecutionContext) -> tuple[str, Any]:
    name = str(context.client_context.get("timeZone") or "").strip()
    if name:
        try:
            return name, ZoneInfo(name)
        except ZoneInfoNotFoundError:
            pass
    local = datetime.now().astimezone().tzinfo or UTC
    return str(local), local


@dataclass(slots=True)
class FrontendLocalServices:
    memory: MarkdownMemoryService
    notes: JsonNotesStore
    work: WorkApplication | None = None
    backend_enabled: bool = False

    async def current_time(
        self,
        call: FrontendToolCall,
        context: FrontendExecutionContext,
    ) -> dict[str, Any]:
        del call
        time_zone, zone = _time_zone(context)
        current = datetime.now(zone)
        return {
            "status": "ok",
            "iso": current.isoformat(),
            "date": current.date().isoformat(),
            "time": current.strftime("%H:%M:%S"),
            "weekday": current.strftime("%A"),
            "time_zone": time_zone,
        }

    async def memory_tool(
        self,
        call: FrontendToolCall,
        context: FrontendExecutionContext,
    ) -> dict[str, Any]:
        return self.memory.apply(context.owner_id, call.arguments)

    async def notes_tool(
        self,
        call: FrontendToolCall,
        context: FrontendExecutionContext,
    ) -> dict[str, Any]:
        return self.notes.execute(context.owner_id, call.arguments)

    async def schedule_reminder(
        self,
        call: FrontendToolCall,
        context: FrontendExecutionContext,
    ) -> dict[str, Any]:
        if self.work is None or not hasattr(self.work.runtime, "submit_scheduled"):
            return _unavailable("reminder_unavailable", "当前未启用提醒服务。")
        execute_at = str(call.arguments.get("execute_at") or "").strip()
        reminder = " ".join(str(call.arguments.get("reminder") or "").split())
        recurrence = str(call.arguments.get("recurrence") or "once")
        reminder_type = str(call.arguments.get("type") or "reminder")
        if not execute_at or not reminder:
            return _unavailable("invalid_reminder", "提醒需要明确的时间和内容。")
        if recurrence != "once":
            return _unavailable("invalid_recurrence", "当前只支持单次提醒。")
        if reminder_type not in {"reminder", "task"}:
            return _unavailable("invalid_reminder_type", "提醒类型必须是 reminder 或 task。")
        if reminder_type == "task" and not self.backend_enabled:
            return _unavailable("backend_unavailable", "当前未配置后台 Agent，不能定时执行工作。")
        try:
            target = datetime.fromisoformat(execute_at.replace("Z", "+00:00"))
        except ValueError:
            return _unavailable("invalid_reminder_time", "提醒时间不是有效的 ISO 8601 时间。")
        if target.tzinfo is None:
            _, zone = _time_zone(context)
            target = target.replace(tzinfo=zone)
        execute_at_ms = int(target.timestamp() * 1_000)
        if execute_at_ms <= int(datetime.now(UTC).timestamp() * 1_000):
            return _unavailable("reminder_in_past", "提醒时间必须晚于当前时间。")
        task = await self.work.runtime.submit_scheduled(
            objective=reminder,
            owner_id=context.owner_id,
            session_id=context.session_id,
            turn_id=context.turn_id,
            at=execute_at_ms,
            schedule={"at": execute_at_ms, "recurrence": recurrence},
            type=reminder_type,
            submission_key=f"reminder:{context.session_id}:{call.call_id}",
        )
        return {
            "status": "scheduled",
            "task_id": task["id"],
            "execute_at": target.isoformat(),
            "recurrence": recurrence,
            "type": reminder_type,
        }

    async def backend_unavailable(
        self,
        call: FrontendToolCall,
        context: FrontendExecutionContext,
    ) -> dict[str, Any]:
        del call, context
        return _unavailable("backend_unavailable", "当前未配置后台 Agent。")

    async def enter_sleep(
        self,
        call: FrontendToolCall,
        context: FrontendExecutionContext,
    ) -> dict[str, Any]:
        del call, context
        return {"status": "sleeping"}

    def ports(self) -> FrontendPorts:
        inline = {
            "get_current_time": self.current_time,
            "memory": self.memory_tool,
            "notes": self.notes_tool,
            "schedule_reminder": self.schedule_reminder,
        }
        if self.work is None:
            return FrontendPorts(
                inline=inline,
                control={
                    "cancel_agent_task": self.backend_unavailable,
                    "get_agent_task_status": self.backend_unavailable,
                    "respond_agent_permission": self.backend_unavailable,
                    "enter_sleep": self.enter_sleep,
                },
                submission=None,
            )
        work_ports = self.work.frontend_ports()
        return FrontendPorts(
            inline=inline,
            control={**work_ports.control, "enter_sleep": self.enter_sleep},
            submission=work_ports.submission if self.backend_enabled else None,
        )


__all__ = ["FrontendLocalServices"]
