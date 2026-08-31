"""Stable Gateway 4.0.0 protocol constants.

These values mirror the JavaScript contract at upstream commit ``1d35173b``.
Clients branch on capabilities rather than the product package version.
"""

from __future__ import annotations

from enum import StrEnum

GATEWAY_PROTOCOL_VERSION = "4.0.0"

# A tuple gives callers an immutable, deterministic capability list while
# remaining JSON serializable through FastAPI/Pydantic.
GATEWAY_CAPABILITIES: tuple[str, ...] = (
    "web.same-origin-ui",
    "web.skin-assets",
    "gateway.instance-lease",
    "gateway.setup-gate",
    "gateway.settings-store",
    "host.electron-entry",
    "host.gateway-process",
    "input.suspend-protocol",
    "input.suspend-clears-playback",
    "input.suspend-ttl",
    "input.suspend-ack",
    "tasks.ag-ui-event-stream",
    "tasks.work-artifacts-authorization",
    "tasks.unified-id-updates",
    "messages.citations",
    "desktop.orb-shell",
    "desktop.orb-window-factory",
    "desktop.orb-placement",
    "desktop.orb-position-store",
    "desktop.skin-store",
)


class GatewayClientEvent(StrEnum):
    """Events accepted from the existing Web, TUI and Desktop clients."""

    CONNECT = "connect"
    UNMUTE = "unmute"
    MUTE = "mute"
    INPUT_UNMUTE = "input.unmute"
    INPUT_MUTE = "input.mute"
    AUDIO_APPEND = "audio.append"
    TEXT_MESSAGE = "text.message"
    INPUT_MESSAGE = "input.message"
    INTERRUPT = "interrupt"
    SLEEP = "sleep"
    WAKE = "wake"
    PLAYBACK_STARTED = "playback.started"
    PLAYBACK_ENDED = "playback.ended"
    PLAYBACK_CANCELLED = "playback.cancelled"
    INPUT_SUSPEND_ACK = "input.suspend.ack"


class GatewayServerEvent(StrEnum):
    """Voice/control events emitted by the Gateway."""

    GATEWAY_CONNECTED = "gateway.connected"
    GATEWAY_DISCONNECTED = "gateway.disconnected"
    VOICE_CONNECTION = "voice.connection"
    VOICE_READY = "voice.ready"
    VOICE_STATE = "voice.state"
    VOICE_OWNERSHIP = "voice.ownership"
    VOICE_DEACTIVATED = "voice.deactivated"
    VOICE_SLEEP = "voice.sleep"
    TURN_STARTED = "turn.started"
    PLAYBACK_CLEAR = "playback.clear"
    INPUT_SUSPEND = "input.suspend"
    INPUT_RESUME = "input.resume"
    AUDIO_DELTA = "audio.delta"
    AUDIO_DONE = "audio.done"
    RESPONSE_STARTED = "response.started"
    RESPONSE_INTERRUPTED = "response.interrupted"
    TRANSCRIPT_DELTA = "transcript.delta"
    TRANSCRIPT_FINAL = "transcript.final"
    TRANSCRIPT_DISCARD = "transcript.discard"
    TIMELINE_INLINE = "timeline.inline"
    AGENT_ACTIVITY = "agent.activity"
    CLIENT_STATE = "client.state"
    ERROR = "error"


class GatewayTaskEvent(StrEnum):
    """Public Task lifecycle events."""

    SNAPSHOT = "task.snapshot"
    ACCEPTED = "task.accepted"
    SCHEDULED = "task.scheduled"
    SCHEDULED_FIRED = "task.scheduled.fired"
    RUNNING = "task.running"
    DELEGATED = "task.delegated"
    FINALIZING = "task.finalizing"
    CANCELLING = "task.cancelling"
    UPDATED = "task.updated"
    PROGRESS = "task.progress"
    COMPLETED = "task.completed"
    FAILED = "task.failed"
    CANCELLED = "task.cancelled"
    PERMISSION_REQUESTED = "task.permission.requested"
    PERMISSION_RESOLVED = "task.permission.resolved"
    NOTIFICATION_PENDING = "task.notification.pending"
    NOTIFICATION_DELIVERED = "task.notification.delivered"
    NOTIFICATION_OFFLINE = "task.notification.offline"


GATEWAY_CLIENT_EVENT_TYPES = frozenset(GatewayClientEvent)
GATEWAY_SERVER_EVENT_TYPES = frozenset(GatewayServerEvent)
GATEWAY_TASK_EVENT_TYPES = frozenset(GatewayTaskEvent)


class AguiEventType(StrEnum):
    """The deliberately small AG-UI compatibility surface."""

    ACTIVITY_SNAPSHOT = "ACTIVITY_SNAPSHOT"


AGUI_TASK_ACTIVITY_TYPE = "qwen.audio.task"
