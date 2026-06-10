"""Robot capability protocol used by the Argos agent."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from argos_src.robot_api.models import (
    BatterySnapshot,
    CameraIntrinsics,
    ImageFrame,
    RGBDFrame,
    RobotTransform,
)


class RobotClient(Protocol):
    """Transport-neutral robot operations expected by Argos."""

    def start(self) -> None:
        """Start transport threads or subscriptions."""

    def shutdown(self) -> None:
        """Release transport resources."""

    def perform_go2_action(
        self,
        *,
        api_id: int,
        parameter: dict[str, Any] | None = None,
        priority: int = 0,
        topic: str = "rt/api/sport/request",
    ) -> None:
        """Execute a Unitree Go2 sport action."""

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

    def get_latest_image(self, camera_topic: str, timeout: float = 2.0) -> ImageFrame | None:
        """Return one decoded camera image."""

    def get_latest_rgbd(
        self,
        *,
        color_topic: str,
        depth_topic: str,
        timeout: float = 2.0,
        sync_slop_sec: float = 0.12,
        queue_size: int = 10,
    ) -> RGBDFrame | None:
        """Return one decoded color/depth pair."""

    def get_latest_intrinsics(
        self,
        camera_info_topic: str,
        timeout: float = 0.05,
    ) -> CameraIntrinsics | None:
        """Return camera intrinsics."""

    def get_transform(
        self,
        parent_frame: str,
        child_frame: str,
        timeout: float = 0.05,
    ) -> RobotTransform | Any:
        """Return a robot transform."""

    def get_battery_snapshot(self) -> BatterySnapshot | None:
        """Return latest battery telemetry."""

    def subscribe_battery(self, callback: Callable[[BatterySnapshot], None]) -> Callable[[], None]:
        """Subscribe to battery updates and return an unsubscribe callback."""

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

    def subscribe_navigation(self, callback: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Subscribe to navigation progress/result events."""

    def dock_for_charging(
        self,
        *,
        approach_pose: dict[str, float],
        dock_timeout_sec: float = 60.0,
    ) -> dict[str, Any]:
        """Run the provider's charging-dock sequence."""

    def perform_spot_command(
        self,
        command: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a Spot capability command."""

    def publish_voice_command(self, command: str) -> None:
        """Optionally publish a voice command to outside observers."""

    def publish_face_presence(self, snapshot: dict[str, Any]) -> None:
        """Optionally publish a face-presence snapshot to outside observers."""
