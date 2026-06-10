"""Transport-neutral robot capability API for Argos."""

from argos_src.robot_api.client import RobotClient
from argos_src.robot_api.fake import FakeRobotClient
from argos_src.robot_api.factory import create_robot_client
from argos_src.robot_api.errors import (
    RobotBridgeError,
    RobotBridgeTimeout,
    is_robot_provider_error,
)
from argos_src.robot_api.models import (
    BatterySnapshot,
    CameraIntrinsics,
    ImageFrame,
    RobotTransform,
    VelocityCommand,
)
from argos_src.robot_api.zenoh import ZenohRobotClient

__all__ = [
    "BatterySnapshot",
    "CameraIntrinsics",
    "FakeRobotClient",
    "ImageFrame",
    "RobotClient",
    "RobotBridgeError",
    "RobotBridgeTimeout",
    "RobotTransform",
    "VelocityCommand",
    "ZenohRobotClient",
    "create_robot_client",
    "is_robot_provider_error",
]
