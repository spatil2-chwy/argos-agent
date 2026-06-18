from __future__ import annotations

import cv2
import numpy as np

from argos_src.face_recognition.attention_gate import (
    AttentionGateSettings,
    AttentionSmoothingSettings,
    FaceAttentionGate,
    estimate_head_pose,
)
from argos_src.face_recognition.attention_gate.head_pose import LANDMARK_NAMES, MODEL_POINTS
from argos_src.provider_api.models import CameraIntrinsics


def _face_from_points(points: np.ndarray):
    landmarks = {
        name: (float(point[0]), float(point[1]))
        for name, point in zip(LANDMARK_NAMES, points)
    }
    return {
        "bbox": {"x": 80, "y": 60, "w": 80, "h": 80},
        "landmarks": landmarks,
    }


def _frontal_face():
    intrinsics = CameraIntrinsics(
        fx=500.0,
        fy=500.0,
        cx=160.0,
        cy=120.0,
        width=320,
        height=240,
    )
    camera_matrix = np.asarray(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    projected, _ = cv2.projectPoints(
        MODEL_POINTS,
        np.zeros((3, 1), dtype=np.float64),
        np.asarray([[0.0], [0.0], [500.0]], dtype=np.float64),
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
    )
    return _face_from_points(projected.reshape(-1, 2)), intrinsics


def test_estimate_head_pose_from_existing_landmarks_is_frontal():
    face, intrinsics = _frontal_face()

    pose = estimate_head_pose(face, intrinsics=intrinsics)

    assert pose.success is True
    assert abs(pose.yaw_deg or 0.0) < 1.0
    assert abs(pose.pitch_deg or 0.0) < 1.0
    assert abs(pose.roll_deg or 0.0) < 1.0


def test_attention_gate_requires_smoothing_before_marking_attentive():
    face, intrinsics = _frontal_face()
    gate = FaceAttentionGate(
        AttentionGateSettings(
            min_face_area=100,
            min_confidence=0.2,
            smoothing=AttentionSmoothingSettings(
                window_sec=1.0,
                min_observations=2,
                hold_sec=0.0,
            ),
        )
    )

    first = gate.evaluate(
        face,
        image_shape=(240, 320, 3),
        intrinsics=intrinsics,
        track_id="person-1",
        now=10.0,
    )
    second = gate.evaluate(
        face,
        image_shape=(240, 320, 3),
        intrinsics=intrinsics,
        track_id="person-1",
        now=10.2,
    )

    assert first.raw_attentive is True
    assert first.attentive is False
    assert first.reason == "smoothing"
    assert second.attentive is True
    assert second.reason == "attentive"
