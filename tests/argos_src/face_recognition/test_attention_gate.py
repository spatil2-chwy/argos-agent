from __future__ import annotations

from argos_src.face_recognition.attention_gate import (
    AttentionGateSettings,
    AttentionSmoothingSettings,
    FaceAttentionGate,
)
from argos_src.face_recognition.attention_gate.models import HeadPoseObservation


class _Estimator:
    def __init__(self, pose: HeadPoseObservation):
        self.pose = pose
        self.calls = []

    def estimate(self, image, face):
        self.calls.append((image, face))
        return self.pose


def _face():
    return {
        "bbox": {"x": 80, "y": 60, "w": 80, "h": 80},
        "landmarks": {"nose": (120.0, 100.0)},
    }


def _face_with_bbox(x=80, y=60, w=80, h=80, **extra):
    return {
        "bbox": {"x": x, "y": y, "w": w, "h": h},
        "landmarks": {
            "nose": (
                float(x) + (float(w) / 2.0),
                float(y) + (float(h) / 2.0),
            ),
        },
        **extra,
    }


def test_attention_gate_uses_sixdrepnet_estimator_result():
    estimator = _Estimator(
        HeadPoseObservation(
            success=True,
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
        )
    )
    gate = FaceAttentionGate(
        AttentionGateSettings(
            min_face_area=100,
            smoothing=AttentionSmoothingSettings(
                window_sec=1.0,
                min_observations=1,
                hold_sec=0.0,
            ),
        ),
        head_pose_estimator=estimator,
    )

    result = gate.evaluate(
        object(),
        _face(),
        image_shape=(240, 320, 3),
        track_id="person-1",
        now=10.0,
    )

    assert result.attentive is True
    assert result.yaw_deg == 0.0
    assert len(estimator.calls) == 1


def test_attention_gate_requires_smoothing_before_marking_attentive():
    estimator = _Estimator(
        HeadPoseObservation(
            success=True,
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
        )
    )
    gate = FaceAttentionGate(
        AttentionGateSettings(
            min_face_area=100,
            smoothing=AttentionSmoothingSettings(
                window_sec=1.0,
                min_observations=2,
                hold_sec=0.0,
            ),
        ),
        head_pose_estimator=estimator,
    )

    first = gate.evaluate(
        object(),
        _face(),
        image_shape=(240, 320, 3),
        track_id="person-1",
        now=10.0,
    )
    second = gate.evaluate(
        object(),
        _face(),
        image_shape=(240, 320, 3),
        track_id="person-1",
        now=10.2,
    )

    assert first.raw_attentive is True
    assert first.attentive is False
    assert first.reason == "smoothing"
    assert second.attentive is True
    assert second.reason == "attentive"


def test_attention_gate_treats_configured_pose_limits_as_acceptance_bounds():
    estimator = _Estimator(
        HeadPoseObservation(
            success=True,
            yaw_deg=20.0,
            pitch_deg=0.0,
            roll_deg=0.0,
        )
    )
    gate = FaceAttentionGate(
        AttentionGateSettings(
            min_face_area=100,
            max_abs_yaw_deg=25.0,
            smoothing=AttentionSmoothingSettings(
                window_sec=1.0,
                min_observations=1,
                hold_sec=0.0,
            ),
        ),
        head_pose_estimator=estimator,
    )

    result = gate.evaluate(
        object(),
        _face(),
        image_shape=(240, 320, 3),
        track_id="person-1",
        now=10.0,
    )

    assert result.attentive is True
    assert result.reason == "attentive"
    assert result.confidence == 1.0


def test_attention_gate_rejects_pose_outside_configured_limits():
    estimator = _Estimator(
        HeadPoseObservation(
            success=True,
            yaw_deg=26.0,
            pitch_deg=0.0,
            roll_deg=0.0,
        )
    )
    gate = FaceAttentionGate(
        AttentionGateSettings(
            min_face_area=100,
            max_abs_yaw_deg=25.0,
            smoothing=AttentionSmoothingSettings(
                window_sec=1.0,
                min_observations=1,
                hold_sec=0.0,
            ),
        ),
        head_pose_estimator=estimator,
    )

    result = gate.evaluate(
        object(),
        _face(),
        image_shape=(240, 320, 3),
        track_id="person-1",
        now=10.0,
    )

    assert result.attentive is False
    assert result.reason == "head_pose_outside_threshold"


def test_attention_gate_reports_unavailable_model():
    estimator = _Estimator(HeadPoseObservation(success=False, reason="sixdrepnet_unavailable"))
    gate = FaceAttentionGate(
        AttentionGateSettings(min_face_area=100),
        head_pose_estimator=estimator,
    )

    result = gate.evaluate(
        object(),
        _face(),
        image_shape=(240, 320, 3),
        track_id="person-1",
        now=10.0,
    )

    assert result.attentive is False
    assert result.reason == "sixdrepnet_unavailable"


def test_attention_gate_uses_fixed_minimum_face_area():
    estimator = _Estimator(
        HeadPoseObservation(
            success=True,
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
        )
    )
    gate = FaceAttentionGate(
        AttentionGateSettings(
            min_face_area=2000,
        ),
        head_pose_estimator=estimator,
    )

    result = gate.evaluate(
        object(),
        _face_with_bbox(x=480, y=480, w=40, h=40),
        image_shape=(1000, 1000, 3),
        track_id="person-1",
        now=10.0,
    )

    assert result.attentive is False
    assert result.reason == "face_too_small"
    assert estimator.calls == []


def test_attention_gate_does_not_reject_off_center_faces():
    estimator = _Estimator(
        HeadPoseObservation(
            success=True,
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
        )
    )
    gate = FaceAttentionGate(
        AttentionGateSettings(
            min_face_area=100,
            smoothing=AttentionSmoothingSettings(
                window_sec=1.0,
                min_observations=1,
                hold_sec=0.0,
            ),
        ),
        head_pose_estimator=estimator,
    )

    result = gate.evaluate(
        object(),
        _face_with_bbox(x=900, y=900, w=80, h=80),
        image_shape=(1000, 1000, 3),
        track_id="person-1",
        now=10.0,
    )

    assert result.attentive is True
    assert result.reason == "attentive"
    assert len(estimator.calls) == 1


def test_attention_gate_uses_fixed_pitch_limit():
    estimator = _Estimator(
        HeadPoseObservation(
            success=True,
            yaw_deg=0.0,
            pitch_deg=28.0,
            roll_deg=0.0,
        )
    )
    gate = FaceAttentionGate(
        AttentionGateSettings(
            min_face_area=100,
            max_abs_pitch_deg=20.0,
            smoothing=AttentionSmoothingSettings(
                window_sec=1.0,
                min_observations=1,
                hold_sec=0.0,
            ),
        ),
        head_pose_estimator=estimator,
    )

    result = gate.evaluate(
        object(),
        _face_with_bbox(x=450, y=450, w=100, h=100),
        image_shape=(1000, 1000, 3),
        track_id="person-1",
        now=10.0,
    )

    assert result.attentive is False
    assert result.reason == "head_pose_outside_threshold"


def test_attention_gate_can_require_pitch_magnitude():
    estimator = _Estimator(
        HeadPoseObservation(
            success=True,
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
        )
    )
    gate = FaceAttentionGate(
        AttentionGateSettings(
            min_face_area=100,
            max_abs_pitch_deg=22.0,
            min_abs_pitch_deg=8.0,
            smoothing=AttentionSmoothingSettings(
                window_sec=1.0,
                min_observations=1,
                hold_sec=0.0,
            ),
        ),
        head_pose_estimator=estimator,
    )

    result = gate.evaluate(
        object(),
        _face_with_bbox(x=450, y=450, w=100, h=100),
        image_shape=(1000, 1000, 3),
        track_id="person-1",
        now=10.0,
    )

    assert result.attentive is False
    assert result.reason == "head_pose_outside_threshold"


def test_attention_gate_accepts_pitch_inside_configured_band():
    estimator = _Estimator(
        HeadPoseObservation(
            success=True,
            yaw_deg=0.0,
            pitch_deg=12.0,
            roll_deg=0.0,
        )
    )
    gate = FaceAttentionGate(
        AttentionGateSettings(
            min_face_area=100,
            max_abs_pitch_deg=22.0,
            min_abs_pitch_deg=8.0,
            smoothing=AttentionSmoothingSettings(
                window_sec=1.0,
                min_observations=1,
                hold_sec=0.0,
            ),
        ),
        head_pose_estimator=estimator,
    )

    result = gate.evaluate(
        object(),
        _face_with_bbox(x=450, y=450, w=100, h=100),
        image_shape=(1000, 1000, 3),
        track_id="person-1",
        now=10.0,
    )

    assert result.attentive is True
    assert result.reason == "attentive"
