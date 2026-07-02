"""Head-pose based attention gate for face detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from argos_src.face_recognition.attention_gate.models import (
    FaceAttentionObservation,
    HeadPoseObservation,
)
from argos_src.face_recognition.attention_gate.sixdrepnet import SixDRepNetHeadPoseEstimator


@dataclass(frozen=True)
class AttentionGateSettings:
    """Runtime knobs for deciding whether a visible face is attending."""

    enabled: bool = True
    min_face_area: int = 1300
    max_abs_yaw_deg: float = 20.0
    max_abs_pitch_deg: float = 18.0
    max_abs_roll_deg: float = 90.0
    min_abs_pitch_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.min_face_area < 1:
            raise ValueError("min_face_area must be >= 1")
        if self.max_abs_yaw_deg <= 0.0:
            raise ValueError("max_abs_yaw_deg must be > 0")
        if self.max_abs_pitch_deg <= 0.0:
            raise ValueError("max_abs_pitch_deg must be > 0")
        if self.max_abs_roll_deg <= 0.0:
            raise ValueError("max_abs_roll_deg must be > 0")
        if self.min_abs_pitch_deg < 0.0:
            raise ValueError("min_abs_pitch_deg must be >= 0")
        if self.min_abs_pitch_deg > self.max_abs_pitch_deg:
            raise ValueError("min_abs_pitch_deg must be <= max_abs_pitch_deg")


class FaceAttentionGate:
    """Evaluate whether a detected face appears to be attending to the robot."""

    def __init__(
        self,
        settings: AttentionGateSettings | None = None,
        *,
        head_pose_estimator: Any | None = None,
    ) -> None:
        self.settings = settings or AttentionGateSettings()
        self._head_pose_estimator = head_pose_estimator or SixDRepNetHeadPoseEstimator()

    def evaluate(
        self,
        image,
        face: dict[str, Any],
        *,
        image_shape: tuple[int, ...],
        track_id: str,
        now: float,
    ) -> FaceAttentionObservation:
        if not self.settings.enabled:
            return FaceAttentionObservation(
                attentive=True,
                confidence=1.0,
                reason="disabled",
                raw_attentive=True,
                raw_confidence=1.0,
            )

        raw_attentive, raw_confidence, reason, pose = self._evaluate_raw(
            image,
            face,
            image_shape=image_shape,
        )
        return FaceAttentionObservation(
            attentive=raw_attentive,
            confidence=raw_confidence,
            reason=reason,
            yaw_deg=pose.yaw_deg,
            pitch_deg=pose.pitch_deg,
            roll_deg=pose.roll_deg,
            raw_attentive=raw_attentive,
            raw_confidence=raw_confidence,
        )

    def _evaluate_raw(
        self,
        image,
        face: dict[str, Any],
        *,
        image_shape: tuple[int, ...],
    ):
        bbox = face.get("bbox") or {}
        width = float(bbox.get("w", 0.0) or 0.0)
        height = float(bbox.get("h", 0.0) or 0.0)
        area = int(max(0.0, width) * max(0.0, height))
        min_face_area = self._effective_min_face_area()
        if area < min_face_area:
            return False, 0.0, "face_too_small", _empty_pose()

        pose = self._head_pose_estimator.estimate(image, face)
        if not pose.success:
            return False, 0.0, pose.reason or "head_pose_unavailable", pose

        yaw_score = _axis_score(pose.yaw_deg, self.settings.max_abs_yaw_deg)
        pitch_score = _axis_band_score(
            pose.pitch_deg,
            min_abs_limit=self.settings.min_abs_pitch_deg,
            max_abs_limit=self.settings.max_abs_pitch_deg,
        )
        roll_score = _axis_score(pose.roll_deg, self.settings.max_abs_roll_deg)
        if min(yaw_score, pitch_score, roll_score) < 0.0:
            return False, 0.0, "head_pose_outside_threshold", pose

        return True, 1.0, "attentive", pose

    def _effective_min_face_area(self) -> int:
        return int(self.settings.min_face_area)


def _axis_score(value: float | None, limit: float) -> float:
    if value is None:
        return -1.0
    return 1.0 - (abs(float(value)) / float(limit))


def _axis_band_score(
    value: float | None,
    *,
    min_abs_limit: float,
    max_abs_limit: float,
) -> float:
    if value is None:
        return -1.0
    min_abs = max(0.0, float(min_abs_limit))
    max_abs = float(max_abs_limit)
    if min_abs <= 0.0:
        return _axis_score(value, max_abs)
    abs_value = abs(float(value))
    if abs_value < min_abs:
        return (abs_value / min_abs) - 1.0
    if abs_value > max_abs:
        return 1.0 - (abs_value / max_abs)
    span = max(max_abs - min_abs, 1e-6)
    lower_margin = (abs_value - min_abs) / span
    upper_margin = (max_abs - abs_value) / span
    return min(lower_margin, upper_margin)




def _empty_pose():
    return HeadPoseObservation(success=False)
