"""Display overlay helpers for head pose and attention."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def draw_attention_overlay(
    image: np.ndarray,
    faces: list[dict[str, Any]],
) -> np.ndarray:
    """Return a copy of image with face boxes and head-pose axes overlaid."""
    if image is None or not hasattr(image, "copy"):
        return image
    annotated = image.copy()
    for face in faces:
        attention = face.get("attention")
        bbox = face.get("bbox") or {}
        try:
            x = int(bbox["x"])
            y = int(bbox["y"])
            w = int(bbox["w"])
            h = int(bbox["h"])
        except Exception:
            continue
        attentive = bool(getattr(attention, "attentive", False))
        color = (0, 220, 0) if attentive else (0, 180, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label = "attentive" if attentive else str(getattr(attention, "reason", "") or "not_attentive")
        name = str(face.get("recognized_name") or "").strip()
        if name:
            label = f"{name} {label}"
        cv2.putText(
            annotated,
            label,
            (x, max(15, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        if attention is not None:
            _draw_pose_axes(
                annotated,
                face=face,
                yaw_deg=getattr(attention, "yaw_deg", None),
                pitch_deg=getattr(attention, "pitch_deg", None),
                roll_deg=getattr(attention, "roll_deg", None),
                size=max(24.0, min(float(w), float(h)) * 0.55),
            )
    return annotated


def _draw_pose_axes(
    image: np.ndarray,
    *,
    face: dict[str, Any],
    yaw_deg: float | None,
    pitch_deg: float | None,
    roll_deg: float | None,
    size: float,
) -> None:
    if yaw_deg is None or pitch_deg is None or roll_deg is None:
        return
    origin = _axis_origin(face)
    if origin is None:
        return
    x0, y0 = origin
    yaw = -float(yaw_deg) * math.pi / 180.0
    pitch = float(pitch_deg) * math.pi / 180.0
    roll = float(roll_deg) * math.pi / 180.0

    x_axis = (
        x0 + size * (math.cos(yaw) * math.cos(roll)),
        y0 + size * (
            math.cos(pitch) * math.sin(roll)
            + math.cos(roll) * math.sin(pitch) * math.sin(yaw)
        ),
    )
    y_axis = (
        x0 + size * (-math.cos(yaw) * math.sin(roll)),
        y0 + size * (
            math.cos(pitch) * math.cos(roll)
            - math.sin(pitch) * math.sin(yaw) * math.sin(roll)
        ),
    )
    z_axis = (
        x0 + size * math.sin(yaw),
        y0 + size * (-math.cos(yaw) * math.sin(pitch)),
    )
    cv2.line(image, (int(x0), int(y0)), _point(x_axis), (0, 0, 255), 2)
    cv2.line(image, (int(x0), int(y0)), _point(y_axis), (0, 255, 0), 2)
    cv2.line(image, (int(x0), int(y0)), _point(z_axis), (255, 0, 0), 2)


def _axis_origin(face: dict[str, Any]) -> tuple[float, float] | None:
    landmarks = face.get("landmarks") or {}
    nose = landmarks.get("nose")
    if nose is not None:
        try:
            return float(nose[0]), float(nose[1])
        except Exception:
            pass
    bbox = face.get("bbox") or {}
    try:
        return (
            float(bbox["x"]) + (float(bbox["w"]) / 2.0),
            float(bbox["y"]) + (float(bbox["h"]) / 2.0),
        )
    except Exception:
        return None


def _point(point: tuple[float, float]) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))
