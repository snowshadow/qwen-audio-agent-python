from __future__ import annotations

from urllib.parse import quote

import pytest

from qwen_audio_agent.transport import IdentityManager, is_allowed_origin
from qwen_audio_agent.transport.security import (
    IDENTITY_COOKIE_MAX_AGE_SECONDS,
    IDENTITY_COOKIE_NAME,
)

SECRET = "test-secret-that-is-longer-than-32-characters"


def cookie_token(set_cookie: str) -> str:
    prefix = f"{IDENTITY_COOKIE_NAME}="
    assert set_cookie.startswith(prefix)
    return set_cookie[len(prefix) :].partition(";")[0]


def test_browser_http_issues_signed_cookie_and_websocket_only_resolves_it() -> None:
    manager = IdentityManager(secret=SECRET)

    issued = manager.resolve_http({})
    assert issued.owner_id.startswith("user_")
    assert issued.set_cookie is not None
    assert "; Path=/" in issued.set_cookie
    assert "; HttpOnly" in issued.set_cookie
    assert "; SameSite=Strict" in issued.set_cookie
    assert f"; Max-Age={IDENTITY_COOKIE_MAX_AGE_SECONDS}" in issued.set_cookie

    token = cookie_token(issued.set_cookie)
    recognized = manager.resolve_upgrade(
        {"cookie": f"unrelated=value; {IDENTITY_COOKIE_NAME}={token}"}
    )
    assert recognized is not None
    assert recognized.owner_id == issued.owner_id
    assert recognized.set_cookie is None

    assert manager.resolve_upgrade({}) is None
    assert (
        manager.resolve_upgrade({"cookie": f"{IDENTITY_COOKIE_NAME}=attacker-selected-client-id"})
        is None
    )


def test_valid_http_cookie_is_reused_but_forged_cookie_is_replaced() -> None:
    manager = IdentityManager(secret=SECRET)
    issued = manager.resolve_http({})
    assert issued.set_cookie is not None
    token = cookie_token(issued.set_cookie)

    reused = manager.resolve_http({"Cookie": f"{IDENTITY_COOKIE_NAME}={token}"})
    replaced = manager.resolve_http({"cookie": f"{IDENTITY_COOKIE_NAME}=user_attacker.invalid"})

    assert reused.owner_id == issued.owner_id
    assert reused.set_cookie is None
    assert replaced.owner_id != "user_attacker"
    assert replaced.set_cookie is not None


def test_signature_is_stable_across_restart_and_browsers_are_isolated() -> None:
    first_manager = IdentityManager(secret=SECRET)
    first = first_manager.resolve_http({})
    second = first_manager.resolve_http({})
    assert first.owner_id != second.owner_id
    assert first.set_cookie is not None

    restarted = IdentityManager(secret=SECRET)
    recognized = restarted.resolve_upgrade(
        {"cookie": f"{IDENTITY_COOKIE_NAME}={cookie_token(first.set_cookie)}"}
    )
    assert recognized is not None
    assert recognized.owner_id == first.owner_id


def test_token_validation_rejects_wrong_owner_signature_and_cookie_encoding() -> None:
    manager = IdentityManager(secret=SECRET)
    owner_id = "user_browser"
    assert manager.sign(owner_id) == "7CjuunTHkAZeHuBxwopkLG5UdoeFUQMTqVMtfpJcnAc"
    valid = f"{owner_id}.{manager.sign(owner_id)}"

    assert manager.resolve_token(f"{IDENTITY_COOKIE_NAME}={quote(valid, safe='')}") is not None
    assert (
        manager.resolve_token(f"{IDENTITY_COOKIE_NAME}=browser.{manager.sign('browser')}") is None
    )
    assert manager.resolve_token(f"{IDENTITY_COOKIE_NAME}={valid[:-1]}x") is None
    assert manager.resolve_token(f"{IDENTITY_COOKIE_NAME}={valid}.extra") is None
    assert manager.resolve_token(f"{IDENTITY_COOKIE_NAME}={owner_id}.{'é' * 43}") is None
    assert manager.resolve_token(f"{IDENTITY_COOKIE_NAME}=%E0%A4%A") is None
    assert (
        manager.resolve_token(f"{IDENTITY_COOKIE_NAME}={valid}; {IDENTITY_COOKIE_NAME}=forged")
        is None
    )


def test_personal_mode_has_one_owner_and_never_sets_a_cookie() -> None:
    manager = IdentityManager(
        secret=SECRET,
        mode="personal",
        personal_owner_id="user_my_assistant",
    )

    first = manager.resolve_http({})
    second = manager.resolve_http({"cookie": f"{IDENTITY_COOKIE_NAME}=ignored.invalid"})
    websocket = manager.resolve_upgrade({})

    assert first is second
    assert websocket is first
    assert first.owner_id == "user_my_assistant"
    assert first.set_cookie is None


def test_secure_cookie_follows_encrypted_socket_or_exact_forwarded_protocol() -> None:
    manager = IdentityManager(secret=SECRET)

    encrypted = manager.resolve_http({}, encrypted=True)
    forwarded = manager.resolve_http({"x-forwarded-proto": "https"})
    uppercase = manager.resolve_http({"x-forwarded-proto": "HTTPS"})

    assert encrypted.set_cookie is not None and encrypted.set_cookie.endswith("; Secure")
    assert forwarded.set_cookie is not None and forwarded.set_cookie.endswith("; Secure")
    assert uppercase.set_cookie is not None and not uppercase.set_cookie.endswith("; Secure")


@pytest.mark.parametrize("secret", ["", "short", "x" * 31])
def test_identity_secret_requires_at_least_32_characters(secret: str) -> None:
    with pytest.raises(ValueError, match="at least 32 characters"):
        IdentityManager(secret=secret)


def test_allows_loopback_same_origin_and_non_browser_requests() -> None:
    assert is_allowed_origin({"host": "localhost:3101"})
    assert is_allowed_origin(
        {
            "host": "localhost:3101",
            "origin": "http://localhost:3101",
        }
    )
    assert is_allowed_origin(
        {
            "Host": "[::1]:3101",
            "Origin": "http://[::1]:3101",
        }
    )
    assert not is_allowed_origin(
        {
            "host": "localhost:3101",
            "origin": "https://attacker.example",
        }
    )


def test_rejects_dns_rebinding_direct_network_access_and_port_mismatch() -> None:
    assert not is_allowed_origin(
        {
            "host": "attacker.example:3101",
            "origin": "http://attacker.example:3101",
        }
    )
    assert not is_allowed_origin({"host": "192.168.1.20:3101"})
    assert not is_allowed_origin(
        {
            "host": "localhost:3101",
            "origin": "http://localhost:3102",
        }
    )
    assert not is_allowed_origin({"host": "localhost.attacker.example:3101"})
    assert not is_allowed_origin({"host": "localhost:bad-port"})
    assert not is_allowed_origin({"host": "999999999999"})
    assert not is_allowed_origin({})


@pytest.mark.parametrize(
    "origin",
    ["null", "", "not-an-origin", "http://localhost:bad-port", "http://localhost:3101\n"],
)
def test_rejects_present_but_opaque_or_malformed_origins(origin: str) -> None:
    assert not is_allowed_origin({"host": "localhost:3101", "origin": origin})


def test_explicit_https_reverse_proxy_is_the_only_public_allowlist_path() -> None:
    allowed = ["https://voice.example.com/some/config/path"]

    assert is_allowed_origin(
        {
            "host": "voice.example.com",
            "origin": "https://voice.example.com",
        },
        allowed_origins=allowed,
    )
    assert is_allowed_origin(
        {"host": "voice.example.com"},
        allowed_origins=allowed,
    )
    assert not is_allowed_origin(
        {
            "host": "other.example.com",
            "origin": "https://other.example.com",
        },
        allowed_origins=allowed,
    )
    assert not is_allowed_origin(
        {
            "host": "other.example.com",
            "origin": "https://voice.example.com",
        },
        allowed_origins=allowed,
    )
    assert not is_allowed_origin(
        {
            "host": "voice.example.com",
            "origin": "http://voice.example.com",
        },
        allowed_origins=["http://voice.example.com"],
    )


def test_default_ports_are_normalized_like_url_origins() -> None:
    assert is_allowed_origin(
        {
            "host": "localhost:80",
            "origin": "http://localhost:80",
        }
    )
    assert is_allowed_origin(
        {
            "host": "voice.example.com",
            "origin": "https://voice.example.com:443",
        },
        allowed_origins=["https://voice.example.com:443"],
    )
    # Host parsing intentionally uses an HTTP authority, matching the JS code;
    # :443 is therefore not collapsed on the request Host side.
    assert not is_allowed_origin(
        {
            "host": "voice.example.com:443",
            "origin": "https://voice.example.com:443",
        },
        allowed_origins=["https://voice.example.com:443"],
    )


@pytest.mark.parametrize("host", ["127.1", "2130706433", "0x7f000001", "0177.0.0.1"])
def test_whatwg_numeric_loopback_spellings_remain_literal_loopback(host: str) -> None:
    assert is_allowed_origin({"host": host})
