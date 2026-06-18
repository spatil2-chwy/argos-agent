from __future__ import annotations

import json

import pytest

from argos_src.provider_api.errors import ProviderTimeout
from argos_src.provider_api.transports.http import HttpProviderClient
from argos_src.provider_api.wire import (
    OP_DISPLAY_AWAIT_RESPONSE,
    OP_DISPLAY_COMMAND,
    OP_DISPLAY_HEALTH,
    OP_DISPLAY_IMAGE,
    OP_DISPLAY_STATE,
)


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
        resource_id="interaction_display",
        urlopen_fn=fake_urlopen,
    )

    result = client.request(
        resource_id="interaction_display",
        operation=OP_DISPLAY_COMMAND,
        args={"type": "face", "face": "happy"},
    )

    assert result == {"ok": True}
    request, _timeout = calls[-1]
    assert request.full_url == "http://localhost:4173/display"
    assert request.get_method() == "POST"
    assert json.loads(request.data.decode("utf-8")) == {
        "type": "face",
        "face": "happy",
    }


def test_http_display_health_and_state_use_get_endpoints():
    paths = []

    def fake_urlopen(request, timeout):
        paths.append(request.full_url)
        return _Response({"ok": True, "path": request.full_url})

    client = HttpProviderClient(
        base_url="http://localhost:4173",
        resource_id="interaction_display",
        urlopen_fn=fake_urlopen,
    )

    assert client.request(
        resource_id="interaction_display",
        operation=OP_DISPLAY_HEALTH,
    )["path"].endswith("/health")
    assert client.request(
        resource_id="interaction_display",
        operation=OP_DISPLAY_STATE,
    )["path"].endswith("/state")
    assert paths == [
        "http://localhost:4173/health",
        "http://localhost:4173/state",
    ]


def test_http_display_image_posts_to_image_endpoint():
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return _Response({"ok": True})

    client = HttpProviderClient(
        base_url="http://localhost:4173",
        resource_id="interaction_display",
        urlopen_fn=fake_urlopen,
    )

    result = client.request(
        resource_id="interaction_display",
        operation=OP_DISPLAY_IMAGE,
        args={
            "dataUrl": "data:image/png;base64,abc",
            "title": "Camera",
            "ttlMs": 1000,
        },
    )

    assert result == {"ok": True}
    request, _timeout = calls[-1]
    assert request.full_url == "http://localhost:4173/image"
    assert request.get_method() == "POST"
    assert json.loads(request.data.decode("utf-8")) == {
        "dataUrl": "data:image/png;base64,abc",
        "title": "Camera",
        "ttlMs": 1000,
    }


def test_http_display_await_response_matches_request_id():
    responses = [
        {"requestId": "other", "accepted": False},
        {"requestId": "capture-1", "accepted": True, "action": "accept"},
    ]

    def fake_urlopen(request, timeout):
        return _Response(responses.pop(0))

    client = HttpProviderClient(
        base_url="http://localhost:4173",
        resource_id="interaction_display",
        urlopen_fn=fake_urlopen,
    )

    result = client.request(
        resource_id="interaction_display",
        operation=OP_DISPLAY_AWAIT_RESPONSE,
        args={"requestId": "capture-1", "poll_sec": 0.001},
        timeout_ms=200,
    )

    assert result["accepted"] is True
    assert result["action"] == "accept"


def test_http_display_await_response_times_out():
    def fake_urlopen(request, timeout):
        return _Response({"requestId": "other", "accepted": False})

    client = HttpProviderClient(
        base_url="http://localhost:4173",
        resource_id="interaction_display",
        urlopen_fn=fake_urlopen,
    )

    with pytest.raises(ProviderTimeout):
        client.request(
            resource_id="interaction_display",
            operation=OP_DISPLAY_AWAIT_RESPONSE,
            args={"requestId": "capture-1", "poll_sec": 0.001},
            timeout_ms=1,
        )
