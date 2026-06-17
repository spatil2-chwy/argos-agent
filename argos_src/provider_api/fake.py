"""Fake provider client for local development and tests."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from argos_src.provider_api.models import (
    BatterySnapshot,
    CameraIntrinsics,
    ImageFrame,
    RGBDFrame,
    RobotTransform,
)


class FakeProviderClient:
    """In-memory provider client with no external transport."""

    def __init__(self) -> None:
        self.actions: list[dict[str, Any]] = []
        self.velocity_commands: list[dict[str, float]] = []
        self.navigation_goals: list[dict[str, Any]] = []
        self.spot_commands: list[dict[str, Any]] = []
        self.dock_requests: list[dict[str, Any]] = []
        self.voice_commands: list[str] = []
        self.face_presence_snapshots: list[dict[str, Any]] = []
        self._battery: BatterySnapshot | None = None
        self._battery_callbacks: list[Callable[[BatterySnapshot], None]] = []
        self._navigation_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def get_manifest(self):
        return None

    def request(
        self,
        *,
        resource_id: str,
        operation: str,
        args: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        del resource_id, timeout_ms
        raise NotImplementedError(f"FakeProviderClient request unsupported: {operation}")

    def publish_event(
        self,
        *,
        resource_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        del resource_id
        if event_type == "voice.command":
            self.publish_voice_command(str((data or {}).get("command", "")))
            return
        if event_type == "face.presence":
            self.publish_face_presence(dict(data or {}))

    def subscribe_event(
        self,
        *,
        resource_id: str,
        event_type: str,
        callback: Callable[[dict[str, Any]], None],
    ) -> Callable[[], None]:
        del resource_id, event_type, callback
        return lambda: None

    def perform_go2_action(
        self,
        *,
        api_id: int,
        parameter: dict[str, Any] | None = None,
        priority: int = 0,
        topic: str = "rt/api/sport/request",
    ) -> None:
        with self._lock:
            self.actions.append(
                {
                    "api_id": int(api_id),
                    "parameter": dict(parameter or {}),
                    "topic": str(topic or "rt/api/sport/request"),
                    "priority": int(priority),
                }
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
        del hz
        clamped = min(max(float(duration), 0.1), float(max_duration))
        self.publish_velocity(
            linear_x=linear_x,
            linear_y=linear_y,
            angular_z=angular_z,
        )
        self.publish_velocity()
        return clamped

    def publish_velocity(
        self,
        *,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        angular_z: float = 0.0,
    ) -> None:
        with self._lock:
            self.velocity_commands.append(
                {
                    "linear_x": float(linear_x),
                    "linear_y": float(linear_y),
                    "angular_z": float(angular_z),
                }
            )

    def stop(self) -> None:
        self.publish_velocity()

    def get_latest_image(
        self,
        resource_id: str | None = None,
        timeout: float = 2.0,
    ) -> ImageFrame | None:
        del resource_id, timeout
        return None

    def get_latest_rgbd(
        self,
        *,
        resource_id: str | None = None,
        timeout: float = 2.0,
        sync_slop_sec: float = 0.12,
        queue_size: int = 10,
    ) -> RGBDFrame | None:
        del resource_id, timeout, sync_slop_sec, queue_size
        return None

    def get_latest_intrinsics(
        self,
        resource_id: str | None = None,
        timeout: float = 0.05,
    ) -> CameraIntrinsics | None:
        del resource_id, timeout
        return None

    def get_transform(
        self,
        parent_frame: str,
        child_frame: str,
        timeout: float = 0.05,
    ) -> RobotTransform:
        del parent_frame, child_frame, timeout
        return RobotTransform(stamp_s=time.time())

    def get_battery_snapshot(self) -> BatterySnapshot | None:
        with self._lock:
            return self._battery

    def set_battery_snapshot(self, snapshot: BatterySnapshot) -> None:
        with self._lock:
            self._battery = snapshot
            callbacks = list(self._battery_callbacks)
        for callback in callbacks:
            callback(snapshot)

    def subscribe_battery(self, callback: Callable[[BatterySnapshot], None]) -> Callable[[], None]:
        with self._lock:
            self._battery_callbacks.append(callback)

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
        del timeout_sec
        goal = {
            "goal_id": str(goal_id),
            "pose": {"x": float(x), "y": float(y), "theta": float(theta)},
            "target_label": str(target_label or ""),
            "tool_name": str(tool_name or "navigation"),
            "blocking": bool(blocking),
            "policy": dict(policy or {}),
        }
        with self._lock:
            self.navigation_goals.append(goal)
        if blocking:
            return {"accepted": True, "outcome": "succeeded", "status_name": "SUCCEEDED"}
        return {"accepted": True}

    def follow_waypoints(
        self,
        *,
        goal_id: str,
        waypoints: list[dict[str, Any]],
        target_label: str = "",
        tool_name: str = "follow_waypoints",
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        goal = {
            "goal_id": str(goal_id),
            "waypoints": [dict(item) for item in waypoints],
            "target_label": str(target_label or ""),
            "tool_name": str(tool_name or "follow_waypoints"),
            "policy": dict(policy or {}),
        }
        with self._lock:
            self.navigation_goals.append(goal)
        return {"accepted": True}

    def cancel_navigation(self, *, goal_id: str | None = None) -> dict[str, Any]:
        event = {
            "event_type": "goal_result",
            "goal_id": str(goal_id or ""),
            "outcome": "canceled",
            "status_name": "CANCELED",
        }
        with self._lock:
            callbacks = list(self._navigation_callbacks)
        for callback in callbacks:
            callback(event)
        return {"canceled": True}

    def subscribe_navigation(self, callback: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        with self._lock:
            self._navigation_callbacks.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._navigation_callbacks:
                    self._navigation_callbacks.remove(callback)

        return unsubscribe

    def emit_navigation_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            callbacks = list(self._navigation_callbacks)
        for callback in callbacks:
            callback(dict(event))

    def dock_for_charging(
        self,
        *,
        approach_pose: dict[str, float],
        dock_timeout_sec: float = 60.0,
    ) -> dict[str, Any]:
        request = {
            "approach_pose": dict(approach_pose or {}),
            "dock_timeout_sec": float(dock_timeout_sec),
        }
        with self._lock:
            self.dock_requests.append(request)
        return {
            "success": True,
            "message": "Docked.",
            "charging_verification": "unknown",
        }

    def perform_spot_command(
        self,
        command: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        item = {"command": str(command or "").strip(), "params": dict(params or {})}
        with self._lock:
            self.spot_commands.append(item)
        return {"success": True, "message": f"{item['command']} completed."}

    def publish_voice_command(self, command: str) -> None:
        rendered = str(command or "").strip()
        if rendered:
            with self._lock:
                self.voice_commands.append(rendered)

    def publish_face_presence(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self.face_presence_snapshots.append(dict(snapshot or {}))
