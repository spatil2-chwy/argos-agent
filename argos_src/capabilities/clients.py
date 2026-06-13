"""Capability-scoped client protocols.

These protocols define the small provider-backed capability vocabulary used by
Argos tools and runtime services.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from argos_src.provider_api.models import (
    BatterySnapshot,
    CameraIntrinsics,
    ImageFrame,
    RGBDFrame,
    RobotTransform,
)


class MotionClient(Protocol):
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
        """Move with a short velocity command and then stop."""

    def publish_velocity(
        self,
        *,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        angular_z: float = 0.0,
    ) -> None:
        """Publish one velocity command sample."""

    def stop(self) -> None:
        """Stop base motion."""


class PostureClient(Protocol):
    def command_posture(self, posture: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run a posture command such as stand, sit, rest, or self_right."""


class EmbodimentClient(Protocol):
    def perform_action(
        self,
        action: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run an expressive embodied action."""


class CameraClient(Protocol):
    def get_latest_image(
        self,
        resource_id: str | None = None,
        timeout: float = 2.0,
    ) -> ImageFrame | None:
        """Return one decoded camera image."""

    def get_latest_rgbd(
        self,
        *,
        resource_id: str | None = None,
        timeout: float = 2.0,
        sync_slop_sec: float = 0.12,
        queue_size: int = 10,
    ) -> RGBDFrame | None:
        """Return one decoded color/depth pair."""

    def get_latest_intrinsics(
        self,
        resource_id: str | None = None,
        timeout: float = 0.05,
    ) -> CameraIntrinsics | None:
        """Return camera intrinsics."""


class TransformClient(Protocol):
    def get_transform(
        self,
        parent_frame: str,
        child_frame: str,
        timeout: float = 0.05,
    ) -> RobotTransform | Any:
        """Return a robot transform."""


class BatteryClient(Protocol):
    def get_battery_snapshot(self) -> BatterySnapshot | None:
        """Return latest battery telemetry."""

    def subscribe_battery(
        self,
        callback: Callable[[BatterySnapshot], None],
    ) -> Callable[[], None]:
        """Subscribe to battery updates."""


class NavigationClient(Protocol):
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
        """Start or run a map-frame navigation goal."""

    def follow_waypoints(
        self,
        *,
        goal_id: str,
        waypoints: list[dict[str, Any]],
        target_label: str = "",
        tool_name: str = "follow_waypoints",
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a waypoint-following navigation goal."""

    def cancel_navigation(self, *, goal_id: str | None = None) -> dict[str, Any]:
        """Cancel active navigation."""

    def subscribe_navigation(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> Callable[[], None]:
        """Subscribe to navigation progress/result events."""


class DockingClient(Protocol):
    def dock_for_charging(
        self,
        *,
        approach_pose: dict[str, float],
        dock_timeout_sec: float = 60.0,
    ) -> dict[str, Any]:
        """Run the provider's charging-dock sequence."""


class PresenceClient(Protocol):
    def publish_face_presence(self, snapshot: dict[str, Any]) -> None:
        """Publish a face-presence snapshot to outside observers."""


class VoiceCommandClient(Protocol):
    def publish_voice_command(self, command: str) -> None:
        """Publish a voice command to outside observers."""


__all__ = [
    "BatteryClient",
    "CameraClient",
    "DockingClient",
    "EmbodimentClient",
    "MotionClient",
    "NavigationClient",
    "PostureClient",
    "PresenceClient",
    "TransformClient",
    "VoiceCommandClient",
]
