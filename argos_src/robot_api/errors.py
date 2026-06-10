"""Robot provider error types shared by transport implementations."""

from __future__ import annotations


class RobotBridgeError(RuntimeError):
    """Raised when the robot provider rejects or fails a capability request."""


class RobotBridgeTimeout(RobotBridgeError):
    """Raised when the robot provider does not answer a request in time."""


def is_robot_provider_error(exc: BaseException) -> bool:
    """Whether an exception is an expected robot provider/capability failure."""
    return isinstance(exc, RobotBridgeError)


__all__ = [
    "RobotBridgeError",
    "RobotBridgeTimeout",
    "is_robot_provider_error",
]
