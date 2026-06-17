from __future__ import annotations

from argos_src.display import DisplayRuntime
from argos_src.provider_api.wire import OP_DISPLAY_AWAIT_RESPONSE, OP_DISPLAY_COMMAND


class _Client:
    def __init__(self):
        self.requests = []
        self.response = {"requestId": "capture-1", "accepted": True, "action": "accept"}

    def request(self, *, resource_id, operation, args=None, timeout_ms=None):
        self.requests.append(
            {
                "resource_id": resource_id,
                "operation": operation,
                "args": dict(args or {}),
                "timeout_ms": timeout_ms,
            }
        )
        if operation == OP_DISPLAY_AWAIT_RESPONSE:
            return dict(self.response)
        return {"ok": True}


def test_display_runtime_review_posts_preview_and_waits_for_response():
    client = _Client()
    runtime = DisplayRuntime(client=client, resource_id="interaction_display")

    result = runtime.review_face_capture(
        request_id="capture-1",
        image_url="data:image/png;base64,abc",
    )

    assert result["accepted"] is True
    assert client.requests[0]["operation"] == OP_DISPLAY_COMMAND
    assert client.requests[0]["args"]["type"] == "face_capture_preview"
    assert client.requests[1]["operation"] == OP_DISPLAY_AWAIT_RESPONSE
    assert client.requests[1]["args"]["requestId"] == "capture-1"


def test_display_runtime_deduplicates_faces_and_subtitles():
    client = _Client()
    runtime = DisplayRuntime(client=client, resource_id="interaction_display")

    runtime.set_face("happy")
    runtime.set_face("happy")
    runtime.show_subtitle("hello")
    runtime.show_subtitle("hello")

    commands = [
        request for request in client.requests if request["operation"] == OP_DISPLAY_COMMAND
    ]
    assert len(commands) == 2


def test_display_runtime_state_modes_are_face_only_except_explicit_subtitles():
    client = _Client()
    runtime = DisplayRuntime(client=client, resource_id="interaction_display")

    runtime.show_idle()
    runtime.show_alert()
    runtime.show_recording()
    runtime.show_thinking()

    commands = [
        request["args"]
        for request in client.requests
        if request["operation"] == OP_DISPLAY_COMMAND
    ]
    assert commands == [
        {"type": "face", "face": "happy"},
        {"type": "face", "face": "think"},
    ]
