"""Pydantic models for the public Gateway protocol.

The parsing boundary deliberately allows unknown fields. Existing clients and
providers can add compatible extensions without the Python rewrite silently
dropping them. Public projectors use an explicit allow-list separately, so
adapter-private state still cannot leak into Task events.
"""

from __future__ import annotations

import json
import re
from typing import Any, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import (
    AguiEventType,
    GatewayClientEvent,
    GatewayServerEvent,
    GatewayTaskEvent,
)


def _to_camel(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(part.capitalize() for part in tail)


class ExtensibleModel(BaseModel):
    """Wire model that keeps additive protocol fields intact."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        extra="allow",
        populate_by_name=True,
        serialize_by_alias=True,
        use_enum_values=True,
    )


class ClosedModel(ExtensibleModel):
    """Nested public value whose field set is closed by the JS v4 schema."""

    model_config = ConfigDict(extra="ignore")


class GatewayEventEnvelope(ExtensibleModel):
    type: str = Field(min_length=1)


_MIME_TYPE = re.compile(r"^[a-z\d!#$&^_.+-]+/[a-z\d!#$&^_.+-]+$")
_BASE64 = re.compile(r"^[a-z\d+/]*={0,2}$", re.IGNORECASE)
_ARTIFACT_PAYLOAD_FIELDS = ("text", "raw", "url", "data")
_PUBLIC_ARTIFACT_URL_SCHEMES = {"data", "http", "https"}


class GatewayArtifactPart(ClosedModel):
    """One MIME-labelled artifact part.

    The Zod union serializes exactly one of text/raw/url/data. Python rejects
    ambiguous parts instead of retaining a second payload as an extension.
    """

    media_type: str = Field(min_length=3, max_length=160, pattern=_MIME_TYPE.pattern)
    filename: str | None = Field(default=None, max_length=240)
    text: str | None = Field(default=None, max_length=1_000_000)
    raw: str | None = Field(default=None, max_length=16_000_000)
    url: str | None = Field(default=None, max_length=20_000)
    data: Any = None

    @field_validator("media_type", mode="before")
    @classmethod
    def normalize_media_type(cls, value: Any) -> str:
        return str(value or "").replace("\x00", "").strip().lower()

    @model_validator(mode="after")
    def validate_one_safe_payload(self) -> Self:
        selected = [name for name in _ARTIFACT_PAYLOAD_FIELDS if name in self.model_fields_set]
        if len(selected) != 1:
            raise ValueError("artifact part requires exactly one of text, raw, url, or data")
        payload_type = selected[0]
        if payload_type == "text":
            text = str(self.text or "").replace("\x00", "").strip()
            if not text:
                raise ValueError("artifact text must not be empty")
            self.text = text
        elif payload_type == "raw":
            raw = re.sub(r"\s", "", str(self.raw or "").replace("\x00", "").strip())
            if not raw or not _BASE64.fullmatch(raw):
                raise ValueError("raw artifact data must be base64")
            self.raw = raw
        elif payload_type == "url":
            url = str(self.url or "").replace("\x00", "").strip()
            if not url or any(character.isspace() for character in url):
                raise ValueError("artifact URL must not be empty or contain whitespace")
            try:
                parsed = urlsplit(url)
                if parsed.scheme.lower() not in _PUBLIC_ARTIFACT_URL_SCHEMES:
                    raise ValueError("artifact URL must use data, HTTP, or HTTPS")
                if parsed.scheme.lower() in {"http", "https"} and not parsed.hostname:
                    raise ValueError("artifact HTTP(S) URL requires a host")
                if parsed.username is not None or parsed.password is not None:
                    raise ValueError("artifact URL must not include credentials")
            except ValueError as error:
                raise ValueError(str(error)) from error
            self.url = url
        else:
            try:
                serialized = json.dumps(
                    self.data,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("artifact data must be JSON serializable") from error
            if len(serialized) > 1_000_000:
                raise ValueError("artifact data exceeds the public size limit")
        return self


class GatewayArtifact(ClosedModel):
    artifact_id: str = Field(min_length=1, max_length=160)
    name: str | None = Field(default=None, max_length=240)
    description: str | None = Field(default=None, max_length=1_000)
    parts: list[GatewayArtifactPart] = Field(min_length=1, max_length=64)


class GatewayInlinePresentation(ExtensibleModel):
    title: str
    format: str = Field(pattern=r"^(markdown|code|link)$")
    content: str


class GatewayPresentation(ExtensibleModel):
    speech: str
    inline: GatewayInlinePresentation | None


class GatewayCitation(ExtensibleModel):
    id: str = Field(min_length=1, max_length=40)
    title: str = Field(min_length=1, max_length=300)
    url: str = Field(max_length=2048)
    snippet: str | None = Field(default=None, max_length=1200)
    source: str | None = Field(default=None, max_length=120)
    published_at: str | None = Field(default=None, max_length=80)

    @field_validator("url")
    @classmethod
    def public_http_url_without_credentials(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("citation URL must be HTTP(S)")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("citation URL must not include credentials")
        return value


class GatewayAuthorizationLocation(ClosedModel):
    path: str = Field(min_length=1, max_length=600)
    line: int | None = Field(default=None, gt=0)


class GatewayAuthorizationOperation(ClosedModel):
    title: str = Field(min_length=1, max_length=160)
    kind: str = Field(min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=600)
    command: str | None = Field(default=None, max_length=1_200)
    path: str | None = Field(default=None, max_length=600)
    locations: list[GatewayAuthorizationLocation] | None = Field(
        default=None,
        max_length=16,
    )


class GatewayAuthorization(ClosedModel):
    id: str = Field(min_length=1, max_length=160)
    task_id: str | None = Field(max_length=160)
    status: str = Field(pattern=r"^(pending|approved|denied|cancelled)$")
    category: str = Field(min_length=1, max_length=80)
    summary: str = Field(min_length=1, max_length=600)
    patterns: list[str] = Field(max_length=32)
    approval_scope: str | None = Field(
        default=None,
        pattern=r"^(once|session|persistent)$",
    )
    operation: GatewayAuthorizationOperation | None = None
    created_at: int | float
    resolved_at: int | float | None

    @field_validator("patterns")
    @classmethod
    def bounded_nonempty_patterns(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value or len(value) > 300 or "\x00" in value:
                raise ValueError("authorization patterns must be 1..300 safe characters")
        return values

    @field_validator("created_at", "resolved_at", mode="before")
    @classmethod
    def reject_boolean_timestamps(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("authorization timestamps must be numbers")
        return value

    @model_validator(mode="after")
    def validate_resolution_time(self) -> Self:
        if self.status == "pending" and self.resolved_at is not None:
            raise ValueError("pending authorization must not have resolvedAt")
        if self.status != "pending" and self.resolved_at is None:
            raise ValueError("resolved authorization requires resolvedAt")
        return self


class GatewayActivity(ExtensibleModel):
    id: str | None = None
    kind: str = Field(min_length=1)
    status: str | None = None
    message: str | None = None
    label: str | None = None
    detail: str | None = None
    category: str | None = None
    tool: str | None = None
    title: str | None = None
    updated_at: str | None = None
    mode: str | None = None
    completed: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=0)


class GatewayTask(ExtensibleModel):
    id: str = Field(min_length=1)
    work_state: str = Field(
        pattern=r"^(submitted|working|auth_required|completed|failed|cancelled)$"
    )
    status: str = Field(min_length=1)
    scope: str | None = None
    kind: str = Field(min_length=1)
    parent_task_id: str | None = None
    objective: str
    owner_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    created_at: int | float
    started_at: int | float | None = None
    completed_at: int | float | None = None
    elapsed_ms: int | float
    result: str | None = None
    error: str | None = None
    message: str | None = None
    artifacts: list[GatewayArtifact] | None = Field(default=None, max_length=32)
    presentation: GatewayPresentation | None = None
    activity: list[GatewayActivity] | None = None
    delegation: Any = None
    authorization: GatewayAuthorization | None = None
    notification_status: str | None = None
    notification_delivered_at: int | float | None = None
    schedule: Any = None
    timeout_ms: int | float | None = None

    @model_validator(mode="after")
    def validate_artifact_budget(self) -> Self:
        if not self.artifacts:
            return self
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("artifactId values must be unique")
        total = 0
        for artifact in self.artifacts:
            for part in artifact.parts:
                payload_type = next(
                    name for name in _ARTIFACT_PAYLOAD_FIELDS if name in part.model_fields_set
                )
                value = getattr(part, payload_type)
                if isinstance(value, str):
                    total += len(value)
                else:
                    total += len(
                        json.dumps(
                            value,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                    )
                if total > 16_000_000:
                    raise ValueError("artifacts exceed the public total size limit")
        return self


class GatewayClientMessage(GatewayEventEnvelope):
    type: GatewayClientEvent


class GatewayVoiceMessage(GatewayEventEnvelope):
    type: GatewayServerEvent
    citations: list[GatewayCitation] | None = Field(default=None, max_length=16)

    @model_validator(mode="after")
    def citations_only_on_final_assistant_transcript(self) -> Self:
        if self.citations is not None and (
            self.type != GatewayServerEvent.TRANSCRIPT_FINAL
            or getattr(self, "role", None) != "assistant"
        ):
            raise ValueError("citations are allowed only on final assistant transcripts")
        return self


class GatewayTaskEventMessage(GatewayEventEnvelope):
    type: GatewayTaskEvent
    task: GatewayTask
    permission: GatewayAuthorization | None = None
    message: str | None = None


class AguiActivitySnapshotEvent(ExtensibleModel):
    type: AguiEventType = AguiEventType.ACTIVITY_SNAPSHOT
    message_id: str = Field(min_length=1)
    activity_type: str = Field(min_length=1)
    content: dict[str, Any]
    replace: bool = True
    timestamp: int | float | None = None
    raw_event: Any = None
    metadata: dict[str, Any] | None = None


class GatewayInputHolder(ExtensibleModel):
    owner: str
    reason: str
    ttl_ms: int | float
    since: int | float
    expires_at: int | float


class GatewayInputStatus(ExtensibleModel):
    suspended: bool
    holders: list[GatewayInputHolder]
    owner: str | None
    reason: str
    expires_at: int | float | None


class InputSuspendRequest(ExtensibleModel):
    owner: str | None = None
    reason: str | None = None
    ttl_ms: int | float | None = None


class InputResumeRequest(ExtensibleModel):
    owner: str | None = None


class GatewayVoiceClients(ExtensibleModel):
    connected: int = 0
    active_owners: int = 0
    by_type: dict[str, int] = Field(default_factory=dict)


class GatewayHealth(ExtensibleModel):
    ok: bool = True
    status: str = "ready"
    protocol_version: str
    capabilities: list[str] | tuple[str, ...]
    gateway_instance_id: str | None = None
    gateway_started_at: str | int | float | None = None
    input_suspension: GatewayInputStatus
    voice_configured: bool = False
    voice_clients: GatewayVoiceClients
    backend: dict[str, Any]


def parse_gateway_client_message(value: Any) -> GatewayClientMessage:
    """Validate one client event while retaining extension fields."""

    return GatewayClientMessage.model_validate(value)


def parse_gateway_server_message(
    value: Any,
) -> GatewayVoiceMessage | GatewayTaskEventMessage:
    """Validate one server event against the voice or Task namespace."""

    event_type = value.get("type") if isinstance(value, dict) else None
    if event_type in {item.value for item in GatewayTaskEvent}:
        return GatewayTaskEventMessage.model_validate(value)
    return GatewayVoiceMessage.model_validate(value)


def parse_agui_gateway_event(value: Any) -> AguiActivitySnapshotEvent:
    """Validate the supported AG-UI compatibility event."""

    return AguiActivitySnapshotEvent.model_validate(value)
