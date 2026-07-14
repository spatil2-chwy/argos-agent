from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from argos_src.provider_api.errors import ProviderTimeout
from argos_src.provider_api.factory import create_provider_client
from argos_src.provider_api.manifest import parse_provider_manifest
from argos_src.provider_api.transports.http import HttpProviderClient
from argos_src.provider_api.wire import (
    OP_DISPLAY_AWAIT_RESPONSE,
    OP_DISPLAY_COMMAND,
    OP_DISPLAY_HEALTH,
    OP_DISPLAY_IMAGE,
    OP_DISPLAY_STATE,
)


KEY_PREFIX = "argos/providers/puffle-go2-display"
RESOURCE_ID = "screen_001"
RESOURCE_PREFIX = f"{KEY_PREFIX}/resources/{RESOURCE_ID}"


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_http_display_command_posts_to_display():
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return _Response({"ok": True})

    client = HttpProviderClient(
        base_url="http://localhost:4173",
        key_prefix=KEY_PREFIX,
        resource_id=RESOURCE_ID,
        urlopen_fn=fake_urlopen,
    )

    result = client.request(
        resource_id=RESOURCE_ID,
        operation=OP_DISPLAY_COMMAND,
        args={"type": "face", "face": "happy"},
    )

    assert result == {"ok": True}
    request, _timeout = calls[-1]
    assert request.full_url == f"http://localhost:4173/{RESOURCE_PREFIX}/display"
    assert request.get_method() == "POST"
    assert json.loads(request.data.decode("utf-8")) == {
        "type": "face",
        "face": "happy",
    }


def test_http_health_and_state_use_operational_get_endpoints():
    paths = []

    def fake_urlopen(request, timeout):
        paths.append(request.full_url)
        return _Response({"ok": True, "path": request.full_url})

    client = HttpProviderClient(
        base_url="http://localhost:4173",
        key_prefix=KEY_PREFIX,
        resource_id=RESOURCE_ID,
        urlopen_fn=fake_urlopen,
    )

    assert client.request(
        resource_id=RESOURCE_ID,
        operation=OP_DISPLAY_HEALTH,
    )["path"].endswith("/health")
    assert client.request(
        resource_id=RESOURCE_ID,
        operation=OP_DISPLAY_STATE,
    )["path"].endswith("/state")
    assert paths == [
        f"http://localhost:4173/{RESOURCE_PREFIX}/health",
        f"http://localhost:4173/{RESOURCE_PREFIX}/state",
    ]


def test_http_display_image_posts_to_image_endpoint():
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return _Response({"ok": True})

    client = HttpProviderClient(
        base_url="http://localhost:4173",
        key_prefix=KEY_PREFIX,
        resource_id=RESOURCE_ID,
        urlopen_fn=fake_urlopen,
    )

    result = client.request(
        resource_id=RESOURCE_ID,
        operation=OP_DISPLAY_IMAGE,
        args={
            "dataUrl": "data:image/png;base64,abc",
            "title": "Camera",
            "ttlMs": 1000,
        },
    )

    assert result == {"ok": True}
    request, _timeout = calls[-1]
    assert request.full_url == f"http://localhost:4173/{RESOURCE_PREFIX}/image"
    assert request.get_method() == "POST"
    assert json.loads(request.data.decode("utf-8")) == {
        "dataUrl": "data:image/png;base64,abc",
        "title": "Camera",
        "ttlMs": 1000,
    }


def test_http_display_await_response_matches_request_id():
    paths = []
    responses = [
        {"requestId": "other", "accepted": False},
        {"requestId": "capture-1", "accepted": True, "action": "accept"},
    ]

    def fake_urlopen(request, timeout):
        paths.append(request.full_url)
        return _Response(responses.pop(0))

    client = HttpProviderClient(
        base_url="http://localhost:4173",
        key_prefix=KEY_PREFIX,
        resource_id=RESOURCE_ID,
        urlopen_fn=fake_urlopen,
    )

    result = client.request(
        resource_id=RESOURCE_ID,
        operation=OP_DISPLAY_AWAIT_RESPONSE,
        args={"requestId": "capture-1", "poll_sec": 0.001},
        timeout_ms=200,
    )

    assert result["accepted"] is True
    assert result["action"] == "accept"
    assert paths == [
        f"http://localhost:4173/{RESOURCE_PREFIX}/response",
        f"http://localhost:4173/{RESOURCE_PREFIX}/response",
    ]


def test_http_display_await_response_times_out():
    def fake_urlopen(request, timeout):
        return _Response({"requestId": "other", "accepted": False})

    client = HttpProviderClient(
        base_url="http://localhost:4173",
        key_prefix=KEY_PREFIX,
        resource_id=RESOURCE_ID,
        urlopen_fn=fake_urlopen,
    )

    with pytest.raises(ProviderTimeout):
        client.request(
            resource_id=RESOURCE_ID,
            operation=OP_DISPLAY_AWAIT_RESPONSE,
            args={"requestId": "capture-1", "poll_sec": 0.001},
            timeout_ms=1,
        )


def test_http_provider_maps_408_response_to_timeout():
    def fake_urlopen(request, timeout):
        raise HTTPError(
            request.full_url,
            408,
            "Timed out waiting for response",
            hdrs=None,
            fp=None,
        )

    client = HttpProviderClient(
        base_url="http://localhost:4173",
        key_prefix=KEY_PREFIX,
        resource_id=RESOURCE_ID,
        urlopen_fn=fake_urlopen,
    )

    with pytest.raises(ProviderTimeout):
        client.request(
            resource_id=RESOURCE_ID,
            operation=OP_DISPLAY_AWAIT_RESPONSE,
            args={"requestId": "capture-1"},
            timeout_ms=200,
        )


def test_http_provider_returns_top_level_json_null():
    def fake_urlopen(request, timeout):
        del request, timeout
        return _Response(None)

    client = HttpProviderClient(
        base_url="http://localhost:8000",
        key_prefix="argos/providers/memory",
        resource_id="memory",
        urlopen_fn=fake_urlopen,
    )

    assert client.request(
        resource_id="memory",
        operation="memory.people_profile",
        args={"person_id": "missing"},
    ) is None


def test_http_provider_sends_bearer_token_from_manifest_auth(monkeypatch):
    monkeypatch.setenv("TAILWAG_API_BEARER_TOKEN", "secret-token")
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return _Response({"ok": True})

    manifest = parse_provider_manifest(
        {
            "id": "memory-test",
            "providers": [
                {
                    "id": "memory",
                    "transport": "http",
                    "key_prefix": "argos/providers/memory",
                    "connect_endpoints": ["http://localhost:8000"],
                    "auth": {
                        "type": "bearer",
                        "token_env": "TAILWAG_API_BEARER_TOKEN",
                    },
                }
            ],
            "resources": [
                {
                    "id": "memory",
                    "kind": "memory",
                    "provider": "memory",
                    "capabilities": ["memory.identity"],
                }
            ],
        }
    )
    provider = manifest.provider_by_id("memory")
    assert provider is not None
    client = HttpProviderClient(
        base_url="http://localhost:8000",
        key_prefix=provider.key_prefix,
        resource_id="memory",
        manifest=manifest,
        auth_token_env=provider.auth.token_env,
        urlopen_fn=fake_urlopen,
    )

    client.request(
        resource_id="memory",
        operation="memory.semantic_search",
        args={"text": "robot", "person_id": "person-1"},
    )

    request, _timeout = calls[-1]
    assert request.full_url == (
        "http://localhost:8000/argos/providers/memory/resources/memory/"
        "request/semantic_search"
    )
    assert request.get_header("Authorization") == "Bearer secret-token"


def test_factory_passes_http_provider_routing_options():
    client = create_provider_client(
        transport="http",
        key_prefix=KEY_PREFIX,
        connect_endpoints=("http://localhost:4173",),
        resource_id=RESOURCE_ID,
    )

    assert isinstance(client, HttpProviderClient)
    assert client.base_url == "http://localhost:4173"
    assert client.key_prefix == KEY_PREFIX
    assert client._resource_id == RESOURCE_ID
