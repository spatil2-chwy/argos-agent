"""Plain Python data models for robot capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class ImageFrame:
    """One decoded camera frame."""

    image: np.ndarray
    topic: str = ""
    captured_at: float = 0.0
    stamp_s: float = 0.0


@dataclass(frozen=True)
class RGBDFrame:
    """One decoded color/depth pair."""

    color_image: np.ndarray
    depth_m: np.ndarray
    color_stamp_s: float = 0.0
    depth_stamp_s: float = 0.0

    @property
    def delta_ms(self) -> float:
        return abs(float(self.color_stamp_s) - float(self.depth_stamp_s)) * 1000.0


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    stamp_s: float = 0.0


@dataclass(frozen=True)
class BatterySnapshot:
    """Transport-neutral battery telemetry."""

    percentage: float
    current: float = 0.0
    power_supply_status: int = 0
    raw: Any = None


@dataclass(frozen=True)
class RobotTransform:
    """Minimal transform shape used by Argos control logic."""

    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    stamp_s: float = 0.0
    raw: Any = None


@dataclass(frozen=True)
class VelocityCommand:
    """Short robot base velocity command."""

    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0
    duration: float = 0.5
    hz: float = 10.0


BatteryCallback = Any
FacePresenceCallback = Any
VoiceCommandCallback = Any
OptionalImageFrame = Optional[ImageFrame]
OptionalRGBDFrame = Optional[RGBDFrame]
OptionalCameraIntrinsics = Optional[CameraIntrinsics]
