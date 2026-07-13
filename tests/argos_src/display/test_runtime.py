from __future__ import annotations

from argos_src.display import DisplayRuntime
from argos_src.display.runtime import _wrap_display_prompt_text
from argos_src.provider_api.wire import (
    OP_DISPLAY_AWAIT_RESPONSE,
    OP_DISPLAY_COMMAND,
    OP_DISPLAY_IMAGE,
)


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


class _MeasuredText:
    def textbbox(self, xy, text, font=None):
        return (0, 0, len(str(text)) * 10, 10)


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


def test_display_runtime_text_prompt_uses_preview_accept_reject():
    client = _Client()
    runtime = DisplayRuntime(client=client, resource_id="interaction_display")

    result = runtime.review_text_prompt(
        request_id="prompt-1",
        title="Face enrollment",
        message="I will snap 5 photos with a countdown before each.",
        accept_label="Start photos",
        reject_label="Cancel",
    )

    assert result["accepted"] is True
    assert client.requests[0]["operation"] == OP_DISPLAY_COMMAND
    assert client.requests[0]["args"]["type"] == "face_capture_preview"
    assert client.requests[0]["args"]["requestId"] == "prompt-1"
    assert client.requests[0]["args"]["imageUrl"].startswith("data:image/png;base64,")
    assert client.requests[0]["args"]["title"] == ""
    assert client.requests[0]["args"]["acceptLabel"] == "Start photos"
    assert client.requests[0]["args"]["rejectLabel"] == "Cancel"
    assert client.requests[1]["operation"] == OP_DISPLAY_AWAIT_RESPONSE


def test_display_runtime_text_prompt_wraps_all_instruction_paragraphs():
    paragraphs = _wrap_display_prompt_text(
        _MeasuredText(),
        text=(
            "I will snap 5 photos with a countdown before each.\n\n"
            "Change angle a little each time: smile, tilt, crouch, or move closer."
        ),
        font=object(),
        max_width=260,
    )

    rendered_lines = [line for paragraph in paragraphs for line in paragraph]
    assert any("I will snap" in line for line in rendered_lines)
    assert any("Change angle" in line for line in rendered_lines)


def test_display_runtime_image_message_preview_posts_and_clears():
    client = _Client()
    runtime = DisplayRuntime(client=client, resource_id="interaction_display")

    assert runtime.show_image_message_preview(
        image_url="data:image/png;base64,abc",
        title="Multiple Faces Detected",
        message="Please make sure you are the only person in view.",
        hold_sec=0,
    )

    commands = [
        request["args"]
        for request in client.requests
        if request["operation"] == OP_DISPLAY_COMMAND
    ]
    assert commands == [
        {
            "type": "image_message_preview",
            "imageUrl": "data:image/png;base64,abc",
            "title": "Multiple Faces Detected",
            "message": "Please make sure you are the only person in view.",
        },
        {"type": "clear"},
    ]


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


def test_display_runtime_state_modes_send_expected_commands():
    client = _Client()
    runtime = DisplayRuntime(client=client, resource_id="interaction_display")

    runtime.show_idle()
    runtime.show_alert()
    runtime.show_recording()
    runtime.show_thinking()
    runtime.show_speaking()

    commands = [
        request["args"]
        for request in client.requests
        if request["operation"] == OP_DISPLAY_COMMAND
    ]
    assert commands == [
        {"type": "face", "face": "happy"},
        {"type": "face", "face": "think"},
        {"type": "face", "face": "think"},
        {"type": "subtitle", "text": "Recording...", "durationMs": 5000},
        {"type": "message", "text": "Thinking..."},
        {"type": "face", "face": "excited"},
    ]


def test_display_runtime_clear_transient_view_resets_face_cache():
    client = _Client()
    runtime = DisplayRuntime(client=client, resource_id="interaction_display")

    runtime.show_idle()
    runtime.show_thinking()
    runtime.show_idle()

    commands = [
        request["args"]
        for request in client.requests
        if request["operation"] == OP_DISPLAY_COMMAND
    ]
    assert commands == [
        {"type": "face", "face": "happy"},
        {"type": "message", "text": "Thinking..."},
        {"type": "face", "face": "happy"},
    ]


def test_display_runtime_live_image_posts_image_payload_and_clear():
    client = _Client()
    runtime = DisplayRuntime(client=client, resource_id="interaction_display")

    assert runtime.show_live_image(
        data_url="data:image/png;base64,abc",
        title="Camera",
        ttl_ms=1000,
    )
    assert runtime.clear_live_image()

    assert client.requests[0]["operation"] == OP_DISPLAY_IMAGE
    assert client.requests[0]["args"] == {
        "title": "Camera",
        "ttlMs": 1000,
        "dataUrl": "data:image/png;base64,abc",
    }
    assert client.requests[1]["operation"] == OP_DISPLAY_IMAGE
    assert client.requests[1]["args"] == {"type": "clear"}
