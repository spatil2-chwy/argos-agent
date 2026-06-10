"""Bridge process helpers for robot transports.

Bridge processes live outside the Argos agent runtime. They may import ROS,
Zenoh, Unitree SDKs, media libraries, or serial drivers, then expose the
transport-neutral capabilities defined in :mod:`argos_src.robot_api`.
"""

__all__ = []
