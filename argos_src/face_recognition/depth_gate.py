"""Pure helpers for depth-gating face detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LANDMARK_NAMES = (
    "left_eye",
    "right_eye",
    "nose",
    "mouth_left",
    "mouth_right",
)


@dataclass(frozen=True)
class DepthGateSettings:
    """Runtime settings for landmark-based face depth gating."""

    depth_topic: str = "/camera/aligned_depth_to_color/image_raw"
    sync_slop_sec: float = 0.12
    sync_queue_size: int = 10
    capture_timeout_sec: float = 1.5
    max_face_depth_m: float = 2.0
    min_valid_samples: int = 2
    patch_size: int = 3
    search_radius_px: int = 12
    max_valid_depth_m: float = 10.0

    def __post_init__(self) -> None:
        if self.sync_slop_sec <= 0.0:
            raise ValueError("sync_slop_sec must be > 0")
        if self.sync_queue_size < 1:
            raise ValueError("sync_queue_size must be >= 1")
        if self.capture_timeout_sec <= 0.0:
            raise ValueError("capture_timeout_sec must be > 0")
        if self.max_face_depth_m <= 0.0:
            raise ValueError("max_face_depth_m must be > 0")
        if self.min_valid_samples < 1:
            raise ValueError("min_valid_samples must be >= 1")
        if self.patch_size < 1 or self.patch_size % 2 == 0:
            raise ValueError("patch_size must be an odd integer >= 1")
        if self.search_radius_px < 0:
            raise ValueError("search_radius_px must be >= 0")
        if self.max_valid_depth_m <= 0.0:
            raise ValueError("max_valid_depth_m must be > 0")


@dataclass(frozen=True)
class DepthSample:
    """One sampled point from the aligned depth map."""

    label: str
    x: int
    y: int
    depth_m: float | None
    search_offset_px: int | None


@dataclass(frozen=True)
class FaceDepthObservation:
    """Depth estimate for one face detection."""

    depth_m: float | None
    accepted: bool
    valid_samples: int
    total_samples: int
    samples: tuple[DepthSample, ...]


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _sample_patch_at(
    aligned_depth_m: np.ndarray,
    *,
    x: int,
    y: int,
    patch_size: int,
    max_valid_depth_m: float,
) -> float | None:
    half = patch_size // 2
    y0 = max(0, y - half)
    y1 = min(aligned_depth_m.shape[0], y + half + 1)
    x0 = max(0, x - half)
    x1 = min(aligned_depth_m.shape[1], x + half + 1)
    patch = aligned_depth_m[y0:y1, x0:x1]

    valid = patch[np.isfinite(patch)]
    valid = valid[valid > 0.0]
    valid = valid[valid <= max_valid_depth_m]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _sample_aligned_depth(
    aligned_depth_m: np.ndarray,
    *,
    x: int,
    y: int,
    patch_size: int,
    search_radius_px: int,
    max_valid_depth_m: float,
) -> tuple[float | None, int | None]:
    direct = _sample_patch_at(
        aligned_depth_m,
        x=x,
        y=y,
        patch_size=patch_size,
        max_valid_depth_m=max_valid_depth_m,
    )
    if direct is not None:
        return direct, 0

    if search_radius_px == 0:
        return None, None

    y0 = max(0, y - search_radius_px)
    y1 = min(aligned_depth_m.shape[0], y + search_radius_px + 1)
    x0 = max(0, x - search_radius_px)
    x1 = min(aligned_depth_m.shape[1], x + search_radius_px + 1)
    window = aligned_depth_m[y0:y1, x0:x1]

    valid_mask = np.isfinite(window)
    valid_mask &= window > 0.0
    valid_mask &= window <= max_valid_depth_m
    if not np.any(valid_mask):
        return None, None

    ys, xs = np.nonzero(valid_mask)
    global_xs = xs + x0
    global_ys = ys + y0
    distances_sq = (global_xs - x) ** 2 + (global_ys - y) ** 2
    nearest_index = int(np.argmin(distances_sq))
    nearest_x = int(global_xs[nearest_index])
    nearest_y = int(global_ys[nearest_index])
    nearest_offset = int(round(float(np.sqrt(float(distances_sq[nearest_index])))))

    searched = _sample_patch_at(
        aligned_depth_m,
        x=nearest_x,
        y=nearest_y,
        patch_size=patch_size,
        max_valid_depth_m=max_valid_depth_m,
    )
    if searched is None:
        searched = float(window[ys[nearest_index], xs[nearest_index]])
    return searched, nearest_offset


def _build_sample_points(
    detection: dict[str, Any],
    *,
    width: int,
    height: int,
) -> list[tuple[str, int, int]]:
    points: list[tuple[str, int, int]] = []
    seen: set[tuple[int, int]] = set()

    landmarks = detection.get("landmarks") or {}
    for name in LANDMARK_NAMES:
        point = landmarks.get(name)
        if point is None or len(point) != 2:
            continue
        x = _clamp(int(round(float(point[0]))), 0, width - 1)
        y = _clamp(int(round(float(point[1]))), 0, height - 1)
        xy = (x, y)
        if xy in seen:
            continue
        seen.add(xy)
        points.append((name, x, y))

    bbox = detection["bbox"]
    cx = _clamp(int(round(float(bbox["x"] + bbox["w"] / 2.0))), 0, width - 1)
    cy = _clamp(int(round(float(bbox["y"] + bbox["h"] / 2.0))), 0, height - 1)
    center_xy = (cx, cy)
    if center_xy not in seen:
        points.append(("bbox_center", cx, cy))
    return points


def measure_face_depth(
    aligned_depth_m: np.ndarray,
    detection: dict[str, Any],
    settings: DepthGateSettings,
) -> FaceDepthObservation:
    """Estimate one face's depth from aligned depth using landmarks plus bbox center."""
    height, width = aligned_depth_m.shape[:2]
    sample_points = _build_sample_points(detection, width=width, height=height)
    samples: list[DepthSample] = []
    valid_depths: list[float] = []

    for label, x, y in sample_points:
        depth_m, offset_px = _sample_aligned_depth(
            aligned_depth_m,
            x=x,
            y=y,
            patch_size=settings.patch_size,
            search_radius_px=settings.search_radius_px,
            max_valid_depth_m=settings.max_valid_depth_m,
        )
        samples.append(
            DepthSample(
                label=label,
                x=x,
                y=y,
                depth_m=depth_m,
                search_offset_px=offset_px,
            )
        )
        if depth_m is not None:
            valid_depths.append(depth_m)

    depth_m = float(np.median(valid_depths)) if valid_depths else None
    accepted = (
        depth_m is not None
        and len(valid_depths) >= settings.min_valid_samples
        and depth_m <= settings.max_face_depth_m
    )
    return FaceDepthObservation(
        depth_m=depth_m,
        accepted=accepted,
        valid_samples=len(valid_depths),
        total_samples=len(samples),
        samples=tuple(samples),
    )


def filter_detections_by_depth(
    detections: list[dict[str, Any]],
    aligned_depth_m: np.ndarray,
    settings: DepthGateSettings,
) -> tuple[list[dict[str, Any]], int]:
    """Keep only face detections that pass the configured depth gate."""
    accepted: list[dict[str, Any]] = []
    rejected = 0

    for detection in detections:
        observation = measure_face_depth(aligned_depth_m, detection, settings)
        if not observation.accepted:
            rejected += 1
            continue

        enriched = dict(detection)
        enriched["depth_m"] = observation.depth_m
        enriched["depth_valid_samples"] = observation.valid_samples
        accepted.append(enriched)

    return accepted, rejected
