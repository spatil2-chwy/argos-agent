"""Head-pose based attention gate for face detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from argos_src.face_recognition.attention_gate.head_pose import estimate_head_pose
from argos_src.face_recognition.attention_gate.models import FaceAttentionObservation
from argos_src.face_recognition.attention_gate.smoothing import (
    AttentionSmoother,
    AttentionSmoothingSettings,
)
from argos_src.provider_api.models import CameraIntrinsics


@dataclass(frozen=True)
class AttentionGateSettings:
    """Runtime knobs for deciding whether a visible face is attending."""

    enabled: bool = True
    min_face_area: int = 1600
    max_abs_yaw_deg: float = 25.0
    max_abs_pitch_deg: float = 20.0
    max_abs_roll_deg: float = 35.0
    max_center_offset_ratio: float = 0.45
    min_confidence: float = 0.55
    smoothing: AttentionSmoothingSettings = AttentionSmoothingSettings()

    def __post_init__(self) -> None:
        if self.min_face_area < 1:
            raise ValueError("min_face_area must be >= 1")
        if self.max_abs_yaw_deg <= 0.0:
            raise ValueError("max_abs_yaw_deg must be > 0")
        if self.max_abs_pitch_deg <= 0.0:
            raise ValueError("max_abs_pitch_deg must be > 0")
        if self.max_abs_roll_deg <= 0.0:
            raise ValueError("max_abs_roll_deg must be > 0")
        if self.max_center_offset_ratio < 0.0:
            raise ValueError("max_center_offset_ratio must be >= 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")


class FaceAttentionGate:
    """Evaluate whether a detected face appears to be attending to the robot."""

    def __init__(self, settings: AttentionGateSettings | None = None) -> None:
        self.settings = settings or AttentionGateSettings()
        self._smoother = AttentionSmoother(self.settings.smoothing)

    def evaluate(
        self,
        face: dict[str, Any],
        *,
        image_shape: tuple[int, ...],
        intrinsics: CameraIntrinsics | None,
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
            face,
            image_shape=image_shape,
            intrinsics=intrinsics,
        )
        attentive, confidence = self._smoother.update(
            track_id=track_id,
            now=now,
            attentive=raw_attentive,
            confidence=raw_confidence,
        )
        if not attentive and raw_attentive:
            reason = "smoothing"
        return FaceAttentionObservation(
            attentive=attentive,
            confidence=confidence,
            reason=reason,
            yaw_deg=pose.yaw_deg,
            pitch_deg=pose.pitch_deg,
            roll_deg=pose.roll_deg,
            raw_attentive=raw_attentive,
            raw_confidence=raw_confidence,
        )

    def _evaluate_raw(
        self,
        face: dict[str, Any],
        *,
        image_shape: tuple[int, ...],
        intrinsics: CameraIntrinsics | None,
    ):
        bbox = face.get("bbox") or {}
        width = float(bbox.get("w", 0.0) or 0.0)
        height = float(bbox.get("h", 0.0) or 0.0)
        area = int(max(0.0, width) * max(0.0, height))
        if area < int(self.settings.min_face_area):
            return False, 0.0, "face_too_small", _empty_pose()

        center_score = self._center_score(face, image_shape=image_shape)
        if center_score is None:
            return False, 0.0, "invalid_bbox", _empty_pose()
        if center_score <= 0.0:
            return False, 0.0, "off_axis", _empty_pose()

        pose = estimate_head_pose(face, intrinsics=intrinsics)
        if not pose.success:
            return False, 0.0, pose.reason or "head_pose_unavailable", pose

        yaw_score = _axis_score(pose.yaw_deg, self.settings.max_abs_yaw_deg)
        pitch_score = _axis_score(pose.pitch_deg, self.settings.max_abs_pitch_deg)
        roll_score = _axis_score(pose.roll_deg, self.settings.max_abs_roll_deg)
        confidence = min(yaw_score, pitch_score, roll_score, center_score)
        attentive = confidence >= float(self.settings.min_confidence)
        if not attentive:
            reason = "head_pose_outside_threshold"
        else:
            reason = "attentive"
        return attentive, confidence, reason, pose

    def _center_score(
        self,
        face: dict[str, Any],
        *,
        image_shape: tuple[int, ...],
    ) -> float | None:
        try:
            image_h, image_w = image_shape[:2]
            bbox = face["bbox"]
            center_x = float(bbox["x"]) + (float(bbox["w"]) / 2.0)
            center_y = float(bbox["y"]) + (float(bbox["h"]) / 2.0)
        except Exception:
            return None
        if image_w <= 0 or image_h <= 0:
            return None
        half_diag = (((float(image_w) / 2.0) ** 2) + ((float(image_h) / 2.0) ** 2)) ** 0.5
        if half_diag <= 0.0:
            return None
        dx = center_x - (float(image_w) / 2.0)
        dy = center_y - (float(image_h) / 2.0)
        offset_ratio = ((dx * dx) + (dy * dy)) ** 0.5 / half_diag
        max_ratio = float(self.settings.max_center_offset_ratio)
        if max_ratio <= 0.0:
            return 1.0 if offset_ratio <= 0.0 else 0.0
        return max(0.0, 1.0 - (offset_ratio / max_ratio))


def _axis_score(value: float | None, limit: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, 1.0 - (abs(float(value)) / float(limit)))


def _empty_pose():
    from argos_src.face_recognition.attention_gate.head_pose import HeadPoseObservation

    return HeadPoseObservation(success=False)
