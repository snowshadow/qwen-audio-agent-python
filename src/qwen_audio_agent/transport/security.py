"""Framework-neutral HTTP identity and DNS-rebinding protection.

The transport remains responsible for applying ``set_cookie`` to an HTTP
response. WebSocket upgrades only resolve an already signed cookie and never
mint an identity during the upgrade handshake.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import re
import socket
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote_to_bytes, urlsplit
from uuid import uuid4

IDENTITY_COOKIE_NAME = "qwen_audio_agent_identity"
IDENTITY_COOKIE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})

_HEX_ESCAPE = re.compile(r"%[0-9a-fA-F]{2}")
_IPV4_NUMBER = re.compile(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)(?:\.(?:0[xX][0-9a-fA-F]+|[0-9]+)){0,3}\.?")
_DEFAULT_PORTS = {
    "ftp": 21,
    "http": 80,
    "https": 443,
    "ws": 80,
    "wss": 443,
}


@dataclass(frozen=True, slots=True)
class ResolvedIdentity:
    """Authenticated owner and an optional HTTP cookie to apply."""

    owner_id: str
    set_cookie: str | None = None


def _header(headers: Mapping[str, Any], name: str) -> str:
    expected = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == expected:
            if isinstance(value, bytes):
                return value.decode("latin-1")
            return str(value or "")
    return ""


def _has_header(headers: Mapping[str, Any], name: str) -> bool:
    expected = name.casefold()
    return any(str(key).casefold() == expected for key in headers)


def _strict_unquote(value: str) -> str:
    index = 0
    while True:
        index = value.find("%", index)
        if index < 0:
            break
        if _HEX_ESCAPE.fullmatch(value[index : index + 3]) is None:
            raise ValueError("malformed percent escape")
        index += 3
    return unquote_to_bytes(value).decode("utf-8", errors="strict")


def _cookies(header: str = "") -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_part in str(header or "").split(";"):
        part = raw_part.strip()
        if not part:
            continue
        name, separator, raw_value = part.partition("=")
        if not separator:
            parsed[name] = ""
            continue
        try:
            parsed[name] = _strict_unquote(raw_value)
        except (UnicodeDecodeError, ValueError):
            parsed[name] = ""
    return parsed


class IdentityManager:
    """Issue and verify server-owned HMAC browser identities."""

    def __init__(
        self,
        *,
        secret: str,
        cookie_name: str = IDENTITY_COOKIE_NAME,
        mode: str = "browser",
        personal_owner_id: str = "user_personal",
    ) -> None:
        if not secret or len(str(secret)) < 32:
            raise ValueError("QWEN_AUDIO_AGENT_AUTH_SECRET must contain at least 32 characters")
        self.secret = str(secret)
        self.cookie_name = str(cookie_name)
        self.mode = "personal" if mode == "personal" else "browser"
        self.personal_identity = ResolvedIdentity(
            owner_id=str(personal_owner_id or "user_personal")
        )

    def sign(self, owner_id: str) -> str:
        digest = hmac.new(
            self.secret.encode("utf-8"),
            str(owner_id).encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    def resolve_token(self, cookie_header: str | None) -> ResolvedIdentity | None:
        token = _cookies(cookie_header or "").get(self.cookie_name)
        if not token:
            return None
        owner_id, separator, supplied = token.rpartition(".")
        if not separator or not owner_id.startswith("user_"):
            return None
        supplied_bytes = supplied.encode("utf-8")
        expected_bytes = self.sign(owner_id).encode("ascii")
        if len(supplied_bytes) != len(expected_bytes) or not hmac.compare_digest(
            supplied_bytes,
            expected_bytes,
        ):
            return None
        return ResolvedIdentity(owner_id=owner_id)

    def issue(self, *, secure: bool = False) -> ResolvedIdentity:
        owner_id = f"user_{uuid4()}"
        token = f"{owner_id}.{self.sign(owner_id)}"
        attributes = [
            f"{self.cookie_name}={quote(token, safe='')}",
            "Path=/",
            "HttpOnly",
            "SameSite=Strict",
            f"Max-Age={IDENTITY_COOKIE_MAX_AGE_SECONDS}",
        ]
        if secure:
            attributes.append("Secure")
        return ResolvedIdentity(owner_id=owner_id, set_cookie="; ".join(attributes))

    def resolve_http(
        self,
        headers: Mapping[str, Any],
        *,
        encrypted: bool = False,
    ) -> ResolvedIdentity:
        if self.mode == "personal":
            return self.personal_identity
        existing = self.resolve_token(_header(headers, "cookie"))
        if existing is not None:
            return existing
        return self.issue(secure=encrypted or _header(headers, "x-forwarded-proto") == "https")

    def resolve_upgrade(self, headers: Mapping[str, Any]) -> ResolvedIdentity | None:
        if self.mode == "personal":
            return self.personal_identity
        return self.resolve_token(_header(headers, "cookie"))


@dataclass(frozen=True, slots=True)
class _Authority:
    scheme: str
    hostname: str
    host: str
    origin: str


def _canonical_hostname(value: str) -> str:
    hostname = value.casefold()
    try:
        return str(ipaddress.ip_address(hostname))
    except ValueError:
        pass
    if _IPV4_NUMBER.fullmatch(hostname):
        try:
            return socket.inet_ntoa(socket.inet_aton(hostname.removesuffix(".")))
        except OSError as error:
            raise ValueError("invalid numeric hostname") from error
    return hostname.encode("idna").decode("ascii")


def _authority(value: str, *, host_header: bool = False) -> _Authority | None:
    raw = str(value or "")
    if any(character in raw for character in "\r\n\x00"):
        return None
    text = raw.strip()
    if not text:
        return None
    candidate = f"http://{text}" if host_header else text
    try:
        parsed = urlsplit(candidate)
        if not parsed.scheme or not parsed.netloc or parsed.hostname is None:
            return None
        if host_header and (
            parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            return None
        hostname = _canonical_hostname(parsed.hostname)
        port = parsed.port
    except (UnicodeError, ValueError):
        return None
    scheme = parsed.scheme.casefold()
    bracketed = f"[{hostname}]" if ":" in hostname else hostname
    host = (
        bracketed if port is None or _DEFAULT_PORTS.get(scheme) == port else f"{bracketed}:{port}"
    )
    return _Authority(
        scheme=scheme,
        hostname=hostname,
        host=host,
        origin=f"{scheme}://{host}",
    )


def _trusted_origins(allowed_origins: Iterable[str]) -> tuple[_Authority, ...]:
    trusted: list[_Authority] = []
    for value in allowed_origins:
        origin = _authority(str(value))
        if origin is None:
            continue
        if origin.scheme == "https" or (
            origin.scheme == "http" and origin.hostname in LOOPBACK_HOSTS
        ):
            trusted.append(origin)
    return tuple(trusted)


def is_allowed_origin(
    headers: Mapping[str, Any],
    *,
    allowed_origins: Iterable[str] = (),
) -> bool:
    """Reject DNS rebinding unless Host and Origin meet the trusted policy."""

    request_host = _authority(_header(headers, "host"), host_header=True)
    if request_host is None:
        return False
    has_origin = _has_header(headers, "origin")
    origin = _authority(_header(headers, "origin"))
    configured = _trusted_origins(allowed_origins)
    trusted_host = any(value.host == request_host.host for value in configured)

    # Only a genuinely absent Origin identifies a non-browser client. Opaque
    # browser origins (`null`) and malformed/empty Origin headers must never be
    # downgraded into this more permissive path.
    if origin is None:
        return not has_origin and (request_host.hostname in LOOPBACK_HOSTS or trusted_host)

    if any(value.origin == origin.origin for value in configured):
        return origin.host == request_host.host

    # Arbitrary same-origin hostnames are still vulnerable to DNS rebinding.
    return (
        request_host.hostname in LOOPBACK_HOSTS
        and origin.hostname in LOOPBACK_HOSTS
        and origin.host == request_host.host
    )


__all__ = [
    "IDENTITY_COOKIE_MAX_AGE_SECONDS",
    "IDENTITY_COOKIE_NAME",
    "LOOPBACK_HOSTS",
    "IdentityManager",
    "ResolvedIdentity",
    "is_allowed_origin",
]
