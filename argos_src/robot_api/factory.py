"""RobotClient construction helpers."""

from __future__ import annotations

import os

from argos_src.robot_api.client import RobotClient
from argos_src.robot_api.fake import FakeRobotClient

DEFAULT_ROBOT_TRANSPORT = "zenoh"


def create_robot_client(
    *,
    transport: str | None = None,
    node_name: str = "argos_robot_client",
    key_prefix: str | None = None,
    connect_endpoints: list[str] | tuple[str, ...] | None = None,
) -> RobotClient:
    """Create a robot client for the selected transport.

    `zenoh` is the real/default deployment transport. `fake` is kept for tests
    and local unit-level checks only.
    """
    del node_name
    selected = str(
        transport
        or os.getenv("ARGOS_ROBOT_TRANSPORT", "")
        or DEFAULT_ROBOT_TRANSPORT
    ).strip().lower()
    if selected == "fake":
        return FakeRobotClient()
    if selected in {"zenoh", "bridge"}:
        from argos_src.robot_api.zenoh import ZenohRobotClient

        return ZenohRobotClient(
            key_prefix=key_prefix,
            connect_endpoints=connect_endpoints,
        )
    if selected in {"ros", "ros2", "legacy_ros2"}:
        raise ValueError(
            "ARGOS_ROBOT_TRANSPORT=ros2 is no longer supported by the "
            "Argos runtime factory. Run the ROS/SDK stack in the external "
            "robot provider and use ARGOS_ROBOT_TRANSPORT=zenoh."
        )
    raise ValueError(
        f"Unsupported ARGOS robot transport '{selected}'. "
        "Expected one of: zenoh, fake."
    )


__all__ = ["DEFAULT_ROBOT_TRANSPORT", "create_robot_client"]
