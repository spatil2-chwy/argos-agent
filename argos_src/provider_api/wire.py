"""Argos provider wire protocol."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

OP_GO2_ACTION = "go2.action"
OP_STOP_MOTION = "motion.stop"
OP_MOVE_VELOCITY = "motion.velocity_for_duration"
OP_PUBLISH_VELOCITY = "motion.velocity_sample"
OP_CAMERA_LATEST_IMAGE = "camera.latest_image"
OP_CAMERA_LATEST_RGBD = "camera.latest_rgbd"
OP_CAMERA_INTRINSICS = "camera.intrinsics"
OP_BATTERY_SNAPSHOT = "battery.snapshot"
OP_BATTERY_EVENT = "battery.event"
OP_TRANSFORM = "tf.transform"
OP_VOICE_COMMAND = "voice.command"
OP_FACE_PRESENCE = "face.presence"
OP_NAVIGATE_TO_POSE = "navigation.go_to_pose"
OP_FOLLOW_WAYPOINTS = "navigation.follow_waypoints"
OP_CANCEL_NAVIGATION = "navigation.cancel"
OP_NAVIGATION_EVENT = "navigation.event"
OP_CHARGING_DOCK = "dock.charging_sequence"
OP_SPOT_COMMAND = "spot.command"
OP_DISPLAY_COMMAND = "display.command"
OP_DISPLAY_AWAIT_RESPONSE = "display.await_response"
OP_DISPLAY_HEALTH = "display.health"
OP_DISPLAY_STATE = "display.state"

REQUEST_TYPE = "request"
RESPONSE_TYPE = "response"
EVENT_TYPE = "event"


def new_request_id() -> str:
    """Create a request id for request/response correlation."""
    return uuid.uuid4().hex


def now_s() -> float:
    """Return wall-clock seconds for provider messages."""
    return time.time()


def build_request(
    *,
    op: str,
    args: dict[str, Any] | None = None,
    timeout_ms: int,
    request_id: str | None = None,
    ts: float | None = None,
) -> dict[str, Any]:
    """Build one provider request message."""
    return {
        "id": request_id or new_request_id(),
        "type": REQUEST_TYPE,
        "op": str(op or "").strip(),
        "args": dict(args or {}),
        "timeout_ms": int(timeout_ms),
        "ts": float(now_s() if ts is None else ts),
    }


def build_response(
    *,
    request_id: str,
    ok: bool,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    ts: float | None = None,
) -> dict[str, Any]:
    """Build one provider response message."""
    return {
        "id": str(request_id or "").strip(),
        "type": RESPONSE_TYPE,
        "ok": bool(ok),
        "result": dict(result or {}),
        "error": error,
        "ts": float(now_s() if ts is None else ts),
    }


def build_event(
    *,
    op: str,
    data: dict[str, Any] | None = None,
    ts: float | None = None,
) -> dict[str, Any]:
    """Build one provider event message."""
    return {
        "type": EVENT_TYPE,
        "op": str(op or "").strip(),
        "data": dict(data or {}),
        "ts": float(now_s() if ts is None else ts),
    }


def encode_message(message: dict[str, Any]) -> bytes:
    """Encode a provider message as compact UTF-8 JSON."""
    return json.dumps(message, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_message(payload: bytes | bytearray | memoryview | str) -> dict[str, Any]:
    """Decode one provider message from UTF-8 JSON."""
    if isinstance(payload, str):
        text = payload
    else:
        text = bytes(payload).decode("utf-8")
    decoded = json.loads(text)
    if not isinstance(decoded, dict):
        raise ValueError("Provider message must decode to a JSON object")
    return decoded


__all__ = [
    "EVENT_TYPE",
    "OP_BATTERY_SNAPSHOT",
    "OP_BATTERY_EVENT",
    "OP_CAMERA_INTRINSICS",
    "OP_CAMERA_LATEST_IMAGE",
    "OP_CAMERA_LATEST_RGBD",
    "OP_FACE_PRESENCE",
    "OP_CANCEL_NAVIGATION",
    "OP_CHARGING_DOCK",
    "OP_DISPLAY_AWAIT_RESPONSE",
    "OP_DISPLAY_COMMAND",
    "OP_DISPLAY_HEALTH",
    "OP_DISPLAY_STATE",
    "OP_FOLLOW_WAYPOINTS",
    "OP_GO2_ACTION",
    "OP_STOP_MOTION",
    "OP_MOVE_VELOCITY",
    "OP_NAVIGATE_TO_POSE",
    "OP_NAVIGATION_EVENT",
    "OP_PUBLISH_VELOCITY",
    "OP_SPOT_COMMAND",
    "OP_TRANSFORM",
    "OP_VOICE_COMMAND",
    "REQUEST_TYPE",
    "RESPONSE_TYPE",
    "build_event",
    "build_request",
    "build_response",
    "decode_message",
    "encode_message",
    "new_request_id",
    "now_s",
]
