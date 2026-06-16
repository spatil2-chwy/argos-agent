"""Zenoh-backed provider transport for Argos.

This module is the default real provider transport. It speaks Argos capability
messages over provider/resource Zenoh keys and does not import ROS or robot SDK
message packages.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from typing import Any, Callable

import numpy as np

from argos_src.provider_api.manifest import ProviderManifest
from argos_src.provider_api.namespaces import (
    normalize_provider_prefix,
    provider_event_key,
    provider_request_key,
    provider_response_key,
)
from argos_src.provider_api.errors import ProviderError, ProviderTimeout
from argos_src.provider_api.models import (
    BatterySnapshot,
    CameraIntrinsics,
    ImageFrame,
    RGBDFrame,
    RobotTransform,
)
from argos_src.provider_api.wire import (
    OP_BATTERY_EVENT,
    OP_BATTERY_SNAPSHOT,
    OP_CANCEL_NAVIGATION,
    OP_CAMERA_INTRINSICS,
    OP_CAMERA_LATEST_IMAGE,
    OP_CAMERA_LATEST_RGBD,
    OP_CHARGING_DOCK,
    OP_FACE_PRESENCE,
    OP_FOLLOW_WAYPOINTS,
    OP_GO2_ACTION,
    OP_MOVE_VELOCITY,
    OP_NAVIGATE_TO_POSE,
    OP_NAVIGATION_EVENT,
    OP_PUBLISH_VELOCITY,
    OP_SPOT_COMMAND,
    OP_STOP_MOTION,
    OP_TRANSFORM,
    OP_VOICE_COMMAND,
    build_event,
    build_request,
    decode_message,
    encode_message,
)


DEFAULT_TIMEOUT_MS = 3000
DEFAULT_IMAGE_TIMEOUT_MS = 5000
DEFAULT_GO2_ACTION_TOPIC = "rt/api/sport/request"


class ZenohProviderClient:
    """Provider client that sends Argos capability messages over Zenoh."""

    def __init__(
        self,
        *,
        key_prefix: str | None = None,
        connect_endpoints: list[str] | tuple[str, ...] | None = None,
        timeout_ms: int | None = None,
        session: Any | None = None,
        zenoh_module: Any | None = None,
        resource_id: str | None = None,
        manifest: ProviderManifest | None = None,
    ) -> None:
        self.key_prefix = normalize_provider_prefix(
            key_prefix or os.getenv("ARGOS_ZENOH_KEY_PREFIX", "")
        )
        self.timeout_ms = int(
            timeout_ms
            if timeout_ms is not None
            else os.getenv("ARGOS_ZENOH_TIMEOUT_MS", DEFAULT_TIMEOUT_MS)
        )
        if self.timeout_ms <= 0:
            raise ValueError("ARGOS_ZENOH_TIMEOUT_MS must be > 0")
        self._connect_endpoints = tuple(
            connect_endpoints
            if connect_endpoints is not None
            else _parse_endpoints(os.getenv("ARGOS_ZENOH_CONNECT", ""))
        )
        self._zenoh = zenoh_module
        self._session = session
        self._owns_session = session is None
        self._resource_id = str(resource_id or os.getenv("ARGOS_PROVIDER_RESOURCE_ID", "")).strip()
        if not self._resource_id:
            raise ValueError("Zenoh provider transport requires resource_id.")
        self._manifest = manifest
        self._lock = threading.Lock()
        self._pending: dict[str, dict[str, Any]] = {}
        self._event_subscriber = None
        self._battery_callbacks: list[Callable[[BatterySnapshot], None]] = []
        self._navigation_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._battery_snapshot: BatterySnapshot | None = None

    def start(self) -> None:
        self._ensure_session()

    def shutdown(self) -> None:
        subscriber = self._event_subscriber
        self._event_subscriber = None
        if subscriber is not None:
            _undeclare(subscriber)
        if self._owns_session and self._session is not None:
            closer = getattr(self._session, "close", None)
            if callable(closer):
                closer()
        self._session = None

    def get_manifest(self) -> ProviderManifest | None:
        return self._manifest

    def request(
        self,
        *,
        resource_id: str,
        operation: str,
        args: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        return self._request(
            operation,
            dict(args or {}),
            timeout_ms=timeout_ms,
            resource_id=resource_id,
        )

    def publish_event(
        self,
        *,
        resource_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self._publish_event(event_type, dict(data or {}), resource_id=resource_id)

    def subscribe_event(
        self,
        *,
        resource_id: str,
        event_type: str,
        callback: Callable[[dict[str, Any]], None],
    ) -> Callable[[], None]:
        session = self._ensure_session()

        def _handle(sample: Any) -> None:
            try:
                event = decode_message(_sample_payload_bytes(sample))
            except Exception:
                return
            if event.get("type") != "event":
                return
            callback(dict(event))

        subscriber = session.declare_subscriber(
            self._event_key(event_type, resource_id=resource_id),
            _handle,
        )

        def unsubscribe() -> None:
            _undeclare(subscriber)

        return unsubscribe

    def perform_go2_action(
        self,
        *,
        api_id: int,
        parameter: dict[str, Any] | None = None,
        priority: int = 0,
        topic: str = DEFAULT_GO2_ACTION_TOPIC,
    ) -> None:
        self._request(
            OP_GO2_ACTION,
            {
                "api_id": int(api_id),
                "parameter": dict(parameter or {}),
                "topic": str(topic or DEFAULT_GO2_ACTION_TOPIC),
                "priority": int(priority),
            },
        )

    def move_velocity(
        self,
        *,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        angular_z: float = 0.0,
        duration: float = 0.5,
        hz: float = 10.0,
        max_duration: float = 10.0,
    ) -> float | None:
        result = self._request(
            OP_MOVE_VELOCITY,
            {
                "linear_x": float(linear_x),
                "linear_y": float(linear_y),
                "angular_z": float(angular_z),
                "duration": float(duration),
                "hz": float(hz),
                "max_duration": float(max_duration),
            },
        )
        duration_result = result.get("duration")
        if duration_result is None:
            return None
        return float(duration_result)

    def publish_velocity(
        self,
        *,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        angular_z: float = 0.0,
    ) -> None:
        if (
            float(linear_x) == 0.0
            and float(linear_y) == 0.0
            and float(angular_z) == 0.0
        ):
            self._request(OP_STOP_MOTION, {})
            return
        self._request(
            OP_PUBLISH_VELOCITY,
            {
                "linear_x": float(linear_x),
                "linear_y": float(linear_y),
                "angular_z": float(angular_z),
            },
        )

    def stop(self) -> None:
        self._request(OP_STOP_MOTION, {})

    def get_latest_image(
        self,
        resource_id: str | None = None,
        timeout: float = 2.0,
    ) -> ImageFrame | None:
        rendered_resource = self._effective_resource_id(resource_id)
        result = self._request(
            OP_CAMERA_LATEST_IMAGE,
            {
                "timeout": float(timeout),
            },
            timeout_ms=max(_seconds_to_ms(timeout), DEFAULT_IMAGE_TIMEOUT_MS),
            resource_id=rendered_resource,
        )
        if not result:
            return None
        image = _decode_image_payload(result.get("image", result))
        if image is None:
            return None
        return ImageFrame(
            image=image,
            resource_id=str(result.get("resource_id", rendered_resource) or ""),
            captured_at=float(result.get("captured_at", 0.0) or 0.0),
            stamp_s=float(result.get("stamp_s", 0.0) or 0.0),
        )

    def get_latest_rgbd(
        self,
        *,
        resource_id: str | None = None,
        timeout: float = 2.0,
        sync_slop_sec: float = 0.12,
        queue_size: int = 10,
    ) -> RGBDFrame | None:
        rendered_resource = self._effective_resource_id(resource_id)
        result = self._request(
            OP_CAMERA_LATEST_RGBD,
            {
                "timeout": float(timeout),
                "sync_slop_sec": float(sync_slop_sec),
                "queue_size": int(queue_size),
            },
            timeout_ms=max(_seconds_to_ms(timeout), DEFAULT_IMAGE_TIMEOUT_MS),
            resource_id=rendered_resource,
        )
        if not result:
            return None
        color = _decode_image_payload(result.get("color_image"))
        depth = _decode_array_payload(result.get("depth_m"))
        if color is None or depth is None:
            return None
        return RGBDFrame(
            color_image=color,
            depth_m=depth,
            color_stamp_s=float(result.get("color_stamp_s", 0.0) or 0.0),
            depth_stamp_s=float(result.get("depth_stamp_s", 0.0) or 0.0),
        )

    def get_latest_intrinsics(
        self,
        resource_id: str | None = None,
        timeout: float = 0.05,
    ) -> CameraIntrinsics | None:
        rendered_resource = self._effective_resource_id(resource_id)
        result = self._request(
            OP_CAMERA_INTRINSICS,
            {
                "timeout": float(timeout),
            },
            timeout_ms=max(_seconds_to_ms(timeout), self.timeout_ms),
            resource_id=rendered_resource,
        )
        if not result:
            return None
        return CameraIntrinsics(
            fx=float(result["fx"]),
            fy=float(result["fy"]),
            cx=float(result["cx"]),
            cy=float(result["cy"]),
            width=int(result["width"]),
            height=int(result["height"]),
            stamp_s=float(result.get("stamp_s", 0.0) or 0.0),
        )

    def get_transform(
        self,
        parent_frame: str,
        child_frame: str,
        timeout: float = 0.05,
    ) -> RobotTransform:
        result = self._request(
            OP_TRANSFORM,
            {
                "parent_frame": str(parent_frame or ""),
                "child_frame": str(child_frame or ""),
                "timeout": float(timeout),
            },
        )
        return RobotTransform(
            translation=_tuple3(result.get("translation"), default=(0.0, 0.0, 0.0)),
            rotation=_tuple4(result.get("rotation"), default=(0.0, 0.0, 0.0, 1.0)),
            stamp_s=float(result.get("stamp_s", 0.0) or 0.0),
            raw=result,
        )

    def get_battery_snapshot(self) -> BatterySnapshot | None:
        result = self._request(OP_BATTERY_SNAPSHOT, {})
        if not result:
            return None
        snapshot = _battery_from_payload(result)
        with self._lock:
            self._battery_snapshot = snapshot
        return snapshot

    def subscribe_battery(
        self,
        callback: Callable[[BatterySnapshot], None],
    ) -> Callable[[], None]:
        with self._lock:
            self._battery_callbacks.append(callback)
        self._ensure_event_subscriber()

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._battery_callbacks:
                    self._battery_callbacks.remove(callback)

        return unsubscribe

    def navigate_to_pose(
        self,
        *,
        goal_id: str,
        x: float,
        y: float,
        theta: float,
        target_label: str = "",
        tool_name: str = "navigation",
        blocking: bool = False,
        timeout_sec: float | None = None,
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            OP_NAVIGATE_TO_POSE,
            {
                "goal_id": str(goal_id or ""),
                "pose": {
                    "x": float(x),
                    "y": float(y),
                    "theta": float(theta),
                    "frame_id": "map",
                },
                "target_label": str(target_label or ""),
                "tool_name": str(tool_name or "navigation"),
                "blocking": bool(blocking),
                "timeout_sec": None if timeout_sec is None else float(timeout_sec),
                "policy": dict(policy or {}),
            },
            timeout_ms=(
                max(_seconds_to_ms(float(timeout_sec)), self.timeout_ms)
                if blocking and timeout_sec is not None
                else self.timeout_ms
            ),
        )

    def follow_waypoints(
        self,
        *,
        goal_id: str,
        waypoints: list[dict[str, Any]],
        target_label: str = "",
        tool_name: str = "follow_waypoints",
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            OP_FOLLOW_WAYPOINTS,
            {
                "goal_id": str(goal_id or ""),
                "waypoints": [dict(item) for item in waypoints],
                "target_label": str(target_label or ""),
                "tool_name": str(tool_name or "follow_waypoints"),
                "policy": dict(policy or {}),
            },
        )

    def cancel_navigation(self, *, goal_id: str | None = None) -> dict[str, Any]:
        return self._request(OP_CANCEL_NAVIGATION, {"goal_id": str(goal_id or "")})

    def subscribe_navigation(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> Callable[[], None]:
        with self._lock:
            self._navigation_callbacks.append(callback)
        self._ensure_event_subscriber()

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._navigation_callbacks:
                    self._navigation_callbacks.remove(callback)

        return unsubscribe

    def dock_for_charging(
        self,
        *,
        approach_pose: dict[str, float],
        dock_timeout_sec: float = 60.0,
    ) -> dict[str, Any]:
        return self._request(
            OP_CHARGING_DOCK,
            {
                "approach_pose": dict(approach_pose or {}),
                "dock_timeout_sec": float(dock_timeout_sec),
            },
            timeout_ms=max(_seconds_to_ms(float(dock_timeout_sec) + 60.0), self.timeout_ms),
        )

    def perform_spot_command(
        self,
        command: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            OP_SPOT_COMMAND,
            {
                "command": str(command or "").strip(),
                "params": dict(params or {}),
            },
        )

    def publish_voice_command(self, command: str) -> None:
        rendered = str(command or "").strip()
        if rendered:
            self._publish_event(OP_VOICE_COMMAND, {"command": rendered})

    def publish_face_presence(self, snapshot: dict[str, Any]) -> None:
        self._publish_event(OP_FACE_PRESENCE, dict(snapshot or {}))

    def _request(
        self,
        op: str,
        args: dict[str, Any],
        *,
        timeout_ms: int | None = None,
        resource_id: str | None = None,
    ) -> dict[str, Any]:
        session = self._ensure_session()
        rendered_timeout_ms = int(timeout_ms or self.timeout_ms)
        request = build_request(
            op=op,
            args=args,
            timeout_ms=rendered_timeout_ms,
        )
        request_id = str(request["id"])
        request_key = self._request_key(request_id, resource_id=resource_id)
        response_key = self._response_key(request_id, resource_id=resource_id)
        rendered_resource_id = self._effective_resource_id(resource_id)
        done = threading.Event()
        slot = {"event": done, "response": None}
        with self._lock:
            self._pending[request_id] = slot

        subscriber = session.declare_subscriber(
            response_key,
            lambda sample: self._handle_response_sample(request_id, sample),
        )
        try:
            session.put(
                request_key,
                encode_message(request),
            )
            if not done.wait(rendered_timeout_ms / 1000.0):
                raise ProviderTimeout(
                    f"Timed out waiting for provider response op={op} id={request_id}"
                )
            response = slot.get("response")
            if not isinstance(response, dict):
                raise ProviderError(
                    f"Invalid provider response op={op} id={request_id}"
                )
            if not bool(response.get("ok", False)):
                raise ProviderError(
                    "Provider request failed "
                    f"op={op} id={request_id} resource_id={rendered_resource_id} "
                    f"request_key={request_key} response_key={response_key} "
                    f"args={args} error={response.get('error')}"
                )
            result = response.get("result", {})
            if not isinstance(result, dict):
                raise ProviderError(
                    f"Provider response result must be an object op={op} id={request_id}"
                )
            return result
        finally:
            _undeclare(subscriber)
            with self._lock:
                self._pending.pop(request_id, None)

    def _publish_event(
        self,
        op: str,
        data: dict[str, Any],
        *,
        resource_id: str | None = None,
    ) -> None:
        session = self._ensure_session()
        session.put(
            self._event_key(op, resource_id=resource_id),
            encode_message(build_event(op=op, data=data)),
        )

    def _handle_response_sample(self, request_id: str, sample: Any) -> None:
        try:
            response = decode_message(_sample_payload_bytes(sample))
        except Exception as exc:
            response = {
                "ok": False,
                "error": f"Failed to decode provider response: {exc}",
            }
        with self._lock:
            slot = self._pending.get(request_id)
            if slot is None:
                return
            slot["response"] = response
            event = slot.get("event")
        if isinstance(event, threading.Event):
            event.set()

    def _ensure_event_subscriber(self) -> None:
        if self._event_subscriber is not None:
            return
        session = self._ensure_session()
        self._event_subscriber = session.declare_subscriber(
            self._event_key("*"),
            self._handle_event_sample,
        )

    def _handle_event_sample(self, sample: Any) -> None:
        try:
            event = decode_message(_sample_payload_bytes(sample))
        except Exception:
            return
        if event.get("type") != "event":
            return
        op = event.get("op")
        if op == OP_NAVIGATION_EVENT:
            data = event.get("data", {})
            if not isinstance(data, dict):
                return
            with self._lock:
                callbacks = list(self._navigation_callbacks)
            for callback in callbacks:
                callback(dict(data))
            return
        if op != OP_BATTERY_EVENT:
            return
        data = event.get("data", {})
        if not isinstance(data, dict):
            return
        snapshot = _battery_from_payload(data)
        with self._lock:
            self._battery_snapshot = snapshot
            callbacks = list(self._battery_callbacks)
        for callback in callbacks:
            callback(snapshot)

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        if self._zenoh is None:
            try:
                import zenoh  # type: ignore
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "ARGOS_PROVIDER_TRANSPORT=zenoh requires the Python 'zenoh' "
                    "module. Install it with 'python3 -m pip install "
                    "eclipse-zenoh' in the Argos environment or use a "
                    "test-injected session."
                ) from exc
            self._zenoh = zenoh
        config = self._zenoh.Config()
        if self._connect_endpoints:
            config.insert_json5(
                "connect/endpoints",
                json.dumps(list(self._connect_endpoints)),
            )
        self._session = self._zenoh.open(config)
        return self._session

    def _effective_resource_id(self, resource_id: str | None = None) -> str:
        rendered = str(resource_id or self._resource_id or "").strip()
        if not rendered:
            raise ValueError("resource_id must not be empty")
        return rendered

    def _request_key(self, request_id: str, *, resource_id: str | None = None) -> str:
        rendered_resource = self._effective_resource_id(resource_id)
        return provider_request_key(self.key_prefix, rendered_resource, request_id)

    def _response_key(self, request_id: str, *, resource_id: str | None = None) -> str:
        rendered_resource = self._effective_resource_id(resource_id)
        return provider_response_key(self.key_prefix, rendered_resource, request_id)

    def _event_key(self, op: str = "*", *, resource_id: str | None = None) -> str:
        rendered_resource = self._effective_resource_id(resource_id)
        return provider_event_key(self.key_prefix, rendered_resource, op)


def _parse_endpoints(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw or "").split(",") if part.strip())


def _seconds_to_ms(seconds: float) -> int:
    return max(1, int(float(seconds) * 1000.0))


def _undeclare(handle: Any) -> None:
    undeclare = getattr(handle, "undeclare", None)
    if callable(undeclare):
        undeclare()


def _sample_payload_bytes(sample: Any) -> bytes:
    payload = getattr(sample, "payload", sample)
    to_bytes = getattr(payload, "to_bytes", None)
    if callable(to_bytes):
        return bytes(to_bytes())
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, memoryview):
        return payload.tobytes()
    return bytes(payload)


def _decode_image_payload(payload: Any) -> np.ndarray | None:
    if payload is None:
        return None
    if isinstance(payload, list):
        return np.asarray(payload)
    if not isinstance(payload, dict):
        return None
    encoding = str(payload.get("encoding", "raw") or "raw").lower()
    if "data_b64" not in payload:
        return None
    data = base64.b64decode(str(payload["data_b64"]))
    if encoding in {"jpeg", "jpg", "png"}:
        import cv2  # type: ignore

        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return image
    dtype = np.dtype(str(payload.get("dtype", "uint8")))
    shape = tuple(int(value) for value in payload.get("shape", ()))
    array = np.frombuffer(data, dtype=dtype)
    if shape:
        array = array.reshape(shape)
    return array


def _decode_array_payload(payload: Any) -> np.ndarray | None:
    if payload is None:
        return None
    if isinstance(payload, list):
        return np.asarray(payload, dtype=np.float32)
    if not isinstance(payload, dict):
        return None
    if "data_b64" not in payload:
        return None
    data = base64.b64decode(str(payload["data_b64"]))
    dtype = np.dtype(str(payload.get("dtype", "float32")))
    shape = tuple(int(value) for value in payload.get("shape", ()))
    array = np.frombuffer(data, dtype=dtype)
    if shape:
        array = array.reshape(shape)
    return array


def _tuple3(value: Any, *, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, dict):
        return (
            float(value.get("x", default[0]) or 0.0),
            float(value.get("y", default[1]) or 0.0),
            float(value.get("z", default[2]) or 0.0),
        )
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return default


def _tuple4(
    value: Any,
    *,
    default: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if isinstance(value, dict):
        return (
            float(value.get("x", default[0]) or 0.0),
            float(value.get("y", default[1]) or 0.0),
            float(value.get("z", default[2]) or 0.0),
            float(value.get("w", default[3]) or 1.0),
        )
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    return default


def _battery_from_payload(payload: dict[str, Any]) -> BatterySnapshot:
    return BatterySnapshot(
        percentage=float(payload.get("percentage", 0.0) or 0.0),
        current=float(payload.get("current", 0.0) or 0.0),
        power_supply_status=int(payload.get("power_supply_status", 0) or 0),
        raw=payload,
    )


__all__ = [
    "ProviderError",
    "ProviderTimeout",
    "ZenohProviderClient",
]
