"""Pure helpers for estimating face bearing from color-camera intrinsics."""

from __future__ import annotations

import math
from typing import Any

from argos_src.provider_api.models import CameraIntrinsics


def face_center_px(face: dict[str, Any]) -> tuple[float, float] | None:
    """Return the bbox center for one face detection."""
    bbox = face.get("bbox") or {}
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["w"])
        h = float(bbox["h"])
    except Exception:
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    return x + (w / 2.0), y + (h / 2.0)


def estimate_robot_yaw_error_rad(
    face: dict[str, Any],
    *,
    intrinsics: CameraIntrinsics | None,
    camera_yaw_offset_rad: float = 0.0,
) -> float | None:
    """Estimate robot yaw needed to face a detected face.

    Positive yaw means turn left/counter-clockwise. Negative yaw means turn
    right/clockwise, matching the robot base yaw convention.
    """
    if intrinsics is None or intrinsics.fx <= 0.0:
        return None
    center = face_center_px(face)
    if center is None:
        return None
    center_x, _ = center
    image_bearing_right_rad = math.atan2(center_x - float(intrinsics.cx), float(intrinsics.fx))
    return float(camera_yaw_offset_rad) - image_bearing_right_rad
