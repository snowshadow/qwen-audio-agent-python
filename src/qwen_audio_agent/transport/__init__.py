"""HTTP and WebSocket composition roots."""

from .app import create_gateway_app
from .ports import EmptyTaskSource, InputArbitrationPort, TaskEventBroker, TaskSource
from .realtime import ClientEventHandler, RealtimeGateway
from .security import IdentityManager, ResolvedIdentity, is_allowed_origin

__all__ = [
    "ClientEventHandler",
    "EmptyTaskSource",
    "IdentityManager",
    "InputArbitrationPort",
    "RealtimeGateway",
    "ResolvedIdentity",
    "TaskEventBroker",
    "TaskSource",
    "create_gateway_app",
    "is_allowed_origin",
]
