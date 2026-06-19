"""Head-pose based attention gate for face detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from argos_src.face_recognition.attention_gate.models import (
    FaceAttentionObservation,
    HeadPoseObservation,
)
from argos_src.face_recognition.attention_gate.smoothing import (
    AttentionSmoother,
    AttentionSmoothingSettings,
)
from argos_src.face_recognition.attention_gate.sixdrepnet import SixDRepNetHeadPoseEstimator


@dataclass(frozen=True)
class AttentionGateSettings:
    """Runtime knobs for deciding whether a visible face is attending."""

    enabled: bool = True
    min_face_area: int = 1600
    min_face_area_ratio: float = 0.0
    max_abs_yaw_deg: float = 25.0
    max_abs_pitch_deg: float = 20.0
    max_abs_roll_deg: float = 35.0
    distant_max_abs_yaw_deg: float = 25.0
    distant_max_abs_pitch_deg: float = 20.0
    distant_max_abs_roll_deg: float = 35.0
    near_face_area_ratio: float = 0.04
    distant_face_area_ratio: float = 0.012
    near_depth_m: float = 0.8
    distant_depth_m: float = 2.0
    max_center_offset_ratio: float = 0.45
    min_confidence: float = 0.55
    smoothing: AttentionSmoothingSettings = AttentionSmoothingSettings()

    def __post_init__(self) -> None:
        if self.min_face_area < 1:
            raise ValueError("min_face_area must be >= 1")
        if self.min_face_area_ratio < 0.0:
            raise ValueError("min_face_area_ratio must be >= 0")
        if self.max_abs_yaw_deg <= 0.0:
            raise ValueError("max_abs_yaw_deg must be > 0")
        if self.max_abs_pitch_deg <= 0.0:
            raise ValueError("max_abs_pitch_deg must be > 0")
        if self.max_abs_roll_deg <= 0.0:
            raise ValueError("max_abs_roll_deg must be > 0")
        if self.distant_max_abs_yaw_deg <= 0.0:
            raise ValueError("distant_max_abs_yaw_deg must be > 0")
        if self.distant_max_abs_pitch_deg <= 0.0:
            raise ValueError("distant_max_abs_pitch_deg must be > 0")
        if self.distant_max_abs_roll_deg <= 0.0:
            raise ValueError("distant_max_abs_roll_deg must be > 0")
        if self.near_face_area_ratio <= 0.0:
            raise ValueError("near_face_area_ratio must be > 0")
        if self.distant_face_area_ratio <= 0.0:
            raise ValueError("distant_face_area_ratio must be > 0")
        if self.near_face_area_ratio <= self.distant_face_area_ratio:
            raise ValueError(
                "near_face_area_ratio must be greater than distant_face_area_ratio"
            )
        if self.near_depth_m <= 0.0:
            raise ValueError("near_depth_m must be > 0")
        if self.distant_depth_m <= self.near_depth_m:
            raise ValueError("distant_depth_m must be greater than near_depth_m")
        if self.max_center_offset_ratio < 0.0:
            raise ValueError("max_center_offset_ratio must be >= 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")


class FaceAttentionGate:
    """Evaluate whether a detected face appears to be attending to the robot."""

    def __init__(
        self,
        settings: AttentionGateSettings | None = None,
        *,
        head_pose_estimator: Any | None = None,
    ) -> None:
        self.settings = settings or AttentionGateSettings()
        self._smoother = AttentionSmoother(self.settings.smoothing)
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
        image,
        face: dict[str, Any],
        *,
        image_shape: tuple[int, ...],
    ):
        bbox = face.get("bbox") or {}
        width = float(bbox.get("w", 0.0) or 0.0)
        height = float(bbox.get("h", 0.0) or 0.0)
        area = int(max(0.0, width) * max(0.0, height))
        min_face_area = self._effective_min_face_area(image_shape=image_shape)
        if area < min_face_area:
            return False, 0.0, "face_too_small", _empty_pose()

        center_score = self._center_score(face, image_shape=image_shape)
        if center_score is None:
            return False, 0.0, "invalid_bbox", _empty_pose()
        if center_score < 0.0:
            return False, 0.0, "off_axis", _empty_pose()

        pose = self._head_pose_estimator.estimate(image, face)
        if not pose.success:
            return False, 0.0, pose.reason or "head_pose_unavailable", pose

        limits = self._effective_pose_limits(
            face,
            area=area,
            image_shape=image_shape,
        )
        yaw_score = _axis_score(pose.yaw_deg, limits.yaw_deg)
        pitch_score = _axis_score(pose.pitch_deg, limits.pitch_deg)
        roll_score = _axis_score(pose.roll_deg, limits.roll_deg)
        if min(yaw_score, pitch_score, roll_score) < 0.0:
            confidence = _confidence_from_margin(
                min(yaw_score, pitch_score, roll_score, center_score),
                min_confidence=float(self.settings.min_confidence),
            )
            return False, confidence, "head_pose_outside_threshold", pose

        confidence = _confidence_from_margin(
            min(yaw_score, pitch_score, roll_score, center_score),
            min_confidence=float(self.settings.min_confidence),
        )
        return True, confidence, "attentive", pose

    def _effective_min_face_area(self, *, image_shape: tuple[int, ...]) -> int:
        threshold = int(self.settings.min_face_area)
        ratio = float(self.settings.min_face_area_ratio)
        if ratio <= 0.0:
            return threshold
        try:
            image_h, image_w = image_shape[:2]
        except Exception:
            return threshold
        frame_area = int(image_h) * int(image_w)
        if frame_area <= 0:
            return threshold
        return max(threshold, int(frame_area * ratio))

    def _effective_pose_limits(
        self,
        face: dict[str, Any],
        *,
        area: int,
        image_shape: tuple[int, ...],
    ) -> "_PoseLimits":
        distance_factor = self._distance_factor(
            face,
            area=area,
            image_shape=image_shape,
        )
        return _PoseLimits(
            yaw_deg=_lerp(
                self.settings.max_abs_yaw_deg,
                self.settings.distant_max_abs_yaw_deg,
                distance_factor,
            ),
            pitch_deg=_lerp(
                self.settings.max_abs_pitch_deg,
                self.settings.distant_max_abs_pitch_deg,
                distance_factor,
            ),
            roll_deg=_lerp(
                self.settings.max_abs_roll_deg,
                self.settings.distant_max_abs_roll_deg,
                distance_factor,
            ),
        )

    def _distance_factor(
        self,
        face: dict[str, Any],
        *,
        area: int,
        image_shape: tuple[int, ...],
    ) -> float:
        depth_m = _as_float(face.get("depth_m"))
        if depth_m is not None and depth_m > 0.0:
            return _clamp01(
                (depth_m - float(self.settings.near_depth_m))
                / (
                    float(self.settings.distant_depth_m)
                    - float(self.settings.near_depth_m)
                )
            )
        area_ratio = self._face_area_ratio(area=area, image_shape=image_shape)
        if area_ratio is None:
            return 0.0
        return _clamp01(
            (float(self.settings.near_face_area_ratio) - area_ratio)
            / (
                float(self.settings.near_face_area_ratio)
                - float(self.settings.distant_face_area_ratio)
            )
        )

    @staticmethod
    def _face_area_ratio(*, area: int, image_shape: tuple[int, ...]) -> float | None:
        try:
            image_h, image_w = image_shape[:2]
        except Exception:
            return None
        frame_area = int(image_h) * int(image_w)
        if frame_area <= 0:
            return None
        return max(0.0, float(area)) / float(frame_area)

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
        return 1.0 - (offset_ratio / max_ratio)


@dataclass(frozen=True)
class _PoseLimits:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _lerp(start: float, end: float, factor: float) -> float:
    bounded = _clamp01(factor)
    return float(start) + ((float(end) - float(start)) * bounded)


def _axis_score(value: float | None, limit: float) -> float:
    if value is None:
        return -1.0
    return 1.0 - (abs(float(value)) / float(limit))


def _confidence_from_margin(margin: float, *, min_confidence: float) -> float:
    bounded_margin = max(0.0, min(1.0, float(margin)))
    floor = max(0.0, min(1.0, float(min_confidence)))
    return floor + ((1.0 - floor) * bounded_margin)


def _empty_pose():
    return HeadPoseObservation(success=False)
