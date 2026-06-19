"""6DRepNet-backed head-pose estimation for face crops."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from argos_src.face_recognition.attention_gate.models import HeadPoseObservation


logger = logging.getLogger(__name__)


class SixDRepNetHeadPoseEstimator:
    """Lazy wrapper around the optional sixdrepnet package."""

    def __init__(self, model: Any | None = None) -> None:
        self._model = model
        self._load_failed = False

    def estimate(
        self,
        image: np.ndarray,
        face: dict[str, Any],
    ) -> HeadPoseObservation:
        crop = _crop_face(image, face)
        if crop is None:
            return HeadPoseObservation(success=False, reason="invalid_face_crop")
        model = self._ensure_model()
        if model is None:
            return HeadPoseObservation(success=False, reason="sixdrepnet_unavailable")
        try:
            pitch, yaw, roll = model.predict(crop)
        except Exception:
            logger.exception("6DRepNet head-pose inference failed")
            return HeadPoseObservation(success=False, reason="sixdrepnet_failed")
        try:
            return HeadPoseObservation(
                success=True,
                yaw_deg=float(_scalar(yaw)),
                pitch_deg=float(_scalar(pitch)),
                roll_deg=float(_scalar(roll)),
            )
        except Exception:
            return HeadPoseObservation(success=False, reason="invalid_sixdrepnet_output")

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        try:
            from sixdrepnet import SixDRepNet  # type: ignore
        except Exception as exc:
            self._load_failed = True
            logger.warning(
                "sixdrepnet is not available; attention gate cannot estimate head pose: %s",
                exc,
            )
            return None
        try:
            self._model = SixDRepNet()
        except Exception as exc:
            self._load_failed = True
            logger.warning("Failed to initialize SixDRepNet head-pose model: %s", exc)
            return None
        return self._model


def _crop_face(image: np.ndarray, face: dict[str, Any], padding_ratio: float = 0.25):
    if image is None or not hasattr(image, "shape"):
        return None
    try:
        height, width = image.shape[:2]
        bbox = face["bbox"]
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["w"])
        h = float(bbox["h"])
    except Exception:
        return None
    if width <= 0 or height <= 0 or w <= 0.0 or h <= 0.0:
        return None
    pad = max(w, h) * max(0.0, float(padding_ratio))
    x1 = max(0, int(round(x - pad)))
    y1 = max(0, int(round(y - pad)))
    x2 = min(width, int(round(x + w + pad)))
    y2 = min(height, int(round(y + h + pad)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size <= 0:
        return None
    return crop.copy()


def _scalar(value: Any) -> float:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size <= 0:
        raise ValueError("empty scalar")
    return float(array[0])
