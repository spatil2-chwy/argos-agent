"""Fast head-pose estimation from existing face landmarks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from argos_src.provider_api.models import CameraIntrinsics


LANDMARK_NAMES = (
    "left_eye",
    "right_eye",
    "nose",
    "mouth_left",
    "mouth_right",
)

# A compact generic face model in arbitrary units. The scale is irrelevant for
# rotation; the relative point layout is what solvePnP needs.
MODEL_POINTS = np.asarray(
    [
        (-30.0, -35.0, -30.0),  # left eye
        (30.0, -35.0, -30.0),  # right eye
        (0.0, 0.0, 0.0),  # nose
        (-22.0, 32.0, -25.0),  # mouth left
        (22.0, 32.0, -25.0),  # mouth right
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class HeadPoseObservation:
    """Yaw/pitch/roll estimate for one face."""

    success: bool
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    reason: str = ""
    reprojection_error_px: float | None = None


def _landmark_points(face: dict[str, Any]) -> np.ndarray | None:
    landmarks = face.get("landmarks") or {}
    points: list[tuple[float, float]] = []
    for name in LANDMARK_NAMES:
        raw = landmarks.get(name)
        if raw is None:
            return None
        try:
            x, y = raw
        except Exception:
            return None
        points.append((float(x), float(y)))
    return np.asarray(points, dtype=np.float64)


def _camera_matrix(intrinsics: CameraIntrinsics) -> np.ndarray | None:
    try:
        fx = float(intrinsics.fx)
        fy = float(intrinsics.fy)
        cx = float(intrinsics.cx)
        cy = float(intrinsics.cy)
    except Exception:
        return None
    if fx <= 0.0 or fy <= 0.0:
        return None
    return np.asarray(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_to_euler_deg(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(
        (rotation_matrix[0, 0] * rotation_matrix[0, 0])
        + (rotation_matrix[1, 0] * rotation_matrix[1, 0])
    )
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def estimate_head_pose(
    face: dict[str, Any],
    *,
    intrinsics: CameraIntrinsics | None,
) -> HeadPoseObservation:
    """Estimate head pose from five face landmarks and camera intrinsics."""
    if intrinsics is None:
        return HeadPoseObservation(success=False, reason="missing_intrinsics")
    image_points = _landmark_points(face)
    if image_points is None:
        return HeadPoseObservation(success=False, reason="missing_landmarks")
    camera_matrix = _camera_matrix(intrinsics)
    if camera_matrix is None:
        return HeadPoseObservation(success=False, reason="invalid_intrinsics")

    distortion = np.zeros((4, 1), dtype=np.float64)
    try:
        success, rotation_vec, translation_vec = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        return HeadPoseObservation(success=False, reason="solvepnp_failed")
    if not success:
        return HeadPoseObservation(success=False, reason="solvepnp_failed")

    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    yaw_deg, pitch_deg, roll_deg = _rotation_matrix_to_euler_deg(rotation_matrix)

    reprojection_error = None
    try:
        projected, _ = cv2.projectPoints(
            MODEL_POINTS,
            rotation_vec,
            translation_vec,
            camera_matrix,
            distortion,
        )
        projected_points = projected.reshape(-1, 2)
        reprojection_error = float(
            np.mean(np.linalg.norm(projected_points - image_points, axis=1))
        )
    except cv2.error:
        reprojection_error = None

    return HeadPoseObservation(
        success=True,
        yaw_deg=float(yaw_deg),
        pitch_deg=float(pitch_deg),
        roll_deg=float(roll_deg),
        reprojection_error_px=reprojection_error,
    )
