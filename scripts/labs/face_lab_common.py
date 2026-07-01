#!/usr/bin/env python3
"""Shared utilities for standalone Argos face diagnostic scripts."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from argos_src.face_recognition.depth_gate import DepthGateSettings
from argos_src.face_recognition.attention_gate import (
    AttentionGateSettings,
    AttentionSmoothingSettings,
)
from argos_src.face_recognition.face_recognition_service import (
    DEFAULT_FACE_ENROLLMENT_POLICY,
    FaceEnrollmentPolicy,
    FaceRecognitionService,
)
from argos_src.provider_api.factory import create_provider_client
from argos_src.profile_config import load_scenario_profile, resolve_repo_path

DEFAULT_PREVIEW_DIR = _REPO_ROOT / "scripts" / "labs" / "face_preview"


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        default="static_interaction",
        help="Argos profile name or YAML path. Default: static_interaction.",
    )
    parser.add_argument(
        "--camera-resource",
        default="",
        help="Override resources.face_camera.",
    )
    parser.add_argument(
        "--provider-transport",
        default="",
        help=(
            "Override the profile provider transport for lab runs. "
            "Use 'fake' for no-hardware smoke tests."
        ),
    )
    parser.add_argument(
        "--db-path",
        default="",
        help="Override face embedding DB path. Defaults to the selected profile.",
    )
    parser.add_argument(
        "--identity-db-path",
        default="",
        help="Override identity DB path. Defaults to the selected profile.",
    )
    parser.add_argument(
        "--recognition-threshold",
        type=float,
        default=None,
        help="Override face_recognition.recognition_threshold.",
    )
    parser.add_argument(
        "--recognition-margin-threshold",
        type=float,
        default=None,
        help="Override face_recognition.recognition_margin_threshold.",
    )
    parser.add_argument(
        "--disable-depth",
        action="store_true",
        help="Capture color only and skip the depth gate.",
    )
    parser.add_argument("--sync-slop-sec", type=float, default=None)
    parser.add_argument("--sync-queue-size", type=int, default=None)
    parser.add_argument("--capture-timeout-sec", type=float, default=None)
    parser.add_argument("--max-face-depth-m", type=float, default=None)
    parser.add_argument("--min-valid-samples", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--search-radius-px", type=int, default=None)
    parser.add_argument("--max-valid-depth-m", type=float, default=None)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )


def add_enrollment_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-face-area", type=int, default=None)
    parser.add_argument("--min-brightness", type=float, default=None)
    parser.add_argument("--max-brightness", type=float, default=None)
    parser.add_argument("--min-contrast", type=float, default=None)
    parser.add_argument("--min-embedding-similarity", type=float, default=None)


def build_enrollment_policy(args: argparse.Namespace) -> FaceEnrollmentPolicy:
    profile = load_scenario_profile(args.profile)
    profile_policy = profile.face_recognition.enrollment_policy
    policy = FaceEnrollmentPolicy(
        min_face_area=profile_policy.min_face_area,
        min_brightness=profile_policy.min_brightness,
        max_brightness=profile_policy.max_brightness,
        min_contrast=profile_policy.min_contrast,
        min_embedding_similarity=profile_policy.min_embedding_similarity,
    )
    replacements: dict[str, Any] = {}
    for attr in (
        "min_face_area",
        "min_brightness",
        "max_brightness",
        "min_contrast",
        "min_embedding_similarity",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            replacements[attr] = value
    return replace(policy, **replacements) if replacements else policy


def load_face_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_scenario_profile(args.profile)
    face = profile.face_recognition
    camera_resource_id = args.camera_resource or profile.resources.face_camera
    db_path = str(resolve_repo_path(args.db_path)) if args.db_path else face.db_path
    identity_db_path = (
        str(resolve_repo_path(args.identity_db_path))
        if args.identity_db_path
        else profile.identity_store.db_path
    )
    loop_interval_sec = float(face.loop_interval_sec)
    recognition_threshold = (
        float(args.recognition_threshold)
        if args.recognition_threshold is not None
        else float(face.recognition_threshold)
    )
    recognition_margin_threshold = (
        float(args.recognition_margin_threshold)
        if args.recognition_margin_threshold is not None
        else float(face.recognition_margin_threshold)
    )
    provider_transport = (
        str(args.provider_transport or "").strip()
        or profile.robot.bridge.transport
    )
    depth_settings = None
    if face.depth_gate.enabled and not args.disable_depth:
        depth_settings = DepthGateSettings(
            sync_slop_sec=(
                float(args.sync_slop_sec)
                if args.sync_slop_sec is not None
                else face.depth_gate.sync_slop_sec
            ),
            sync_queue_size=(
                int(args.sync_queue_size)
                if args.sync_queue_size is not None
                else face.depth_gate.sync_queue_size
            ),
            capture_timeout_sec=(
                float(args.capture_timeout_sec)
                if args.capture_timeout_sec is not None
                else face.depth_gate.capture_timeout_sec
            ),
            max_face_depth_m=(
                float(args.max_face_depth_m)
                if args.max_face_depth_m is not None
                else face.depth_gate.max_face_depth_m
            ),
            min_valid_samples=(
                int(args.min_valid_samples)
                if args.min_valid_samples is not None
                else face.depth_gate.min_valid_samples
            ),
            patch_size=(
                int(args.patch_size)
                if args.patch_size is not None
                else face.depth_gate.patch_size
            ),
            search_radius_px=(
                int(args.search_radius_px)
                if args.search_radius_px is not None
                else face.depth_gate.search_radius_px
            ),
            max_valid_depth_m=(
                float(args.max_valid_depth_m)
                if args.max_valid_depth_m is not None
                else face.depth_gate.max_valid_depth_m
            ),
        )
    return {
        "profile_name": profile.name,
        "camera_resource_id": camera_resource_id,
        "db_path": db_path,
        "identity_db_path": identity_db_path,
        "loop_interval_sec": loop_interval_sec,
        "recognition_threshold": recognition_threshold,
        "recognition_margin_threshold": recognition_margin_threshold,
        "depth_settings": depth_settings,
        "provider_transport": provider_transport,
        "provider_id": profile.robot.bridge.provider_id,
        "provider_resource_id": profile.robot.bridge.resource_id,
    }


def build_attention_gate_settings(profile: Any) -> AttentionGateSettings:
    attention_gate = profile.face_recognition.attention_gate
    return AttentionGateSettings(
        enabled=attention_gate.enabled,
        min_face_area=attention_gate.min_face_area,
        max_abs_yaw_deg=attention_gate.max_abs_yaw_deg,
        max_abs_pitch_deg=attention_gate.max_abs_pitch_deg,
        max_abs_roll_deg=attention_gate.max_abs_roll_deg,
        min_abs_pitch_deg=attention_gate.min_abs_pitch_deg,
        smoothing=AttentionSmoothingSettings(
            window_sec=attention_gate.smoothing_window_sec,
            min_observations=attention_gate.min_attentive_observations,
            hold_sec=attention_gate.hold_sec,
        ),
    )


def build_face_service(
    args: argparse.Namespace,
    *,
    enrollment_policy: FaceEnrollmentPolicy | None = None,
) -> tuple[FaceRecognitionService, dict[str, Any]]:
    config = load_face_runtime_config(args)
    profile = load_scenario_profile(args.profile)
    robot_client = create_provider_client(
        transport=config["provider_transport"],
        key_prefix=profile.robot.bridge.key_prefix,
        connect_endpoints=profile.robot.bridge.connect_endpoints,
        resource_id=profile.robot.bridge.resource_id,
        manifest=profile.manifest,
    )
    robot_client.start()
    service = FaceRecognitionService(
        db_path=config["db_path"],
        identity_db_path=config["identity_db_path"],
        recognition_threshold=config["recognition_threshold"],
        recognition_margin_threshold=config["recognition_margin_threshold"],
        robot_client=robot_client,
        camera_resource_id=config["camera_resource_id"],
        camera_yaw_offset_rad=(
            profile.face_recognition.owner_turn.camera_yaw_offset_rad
        ),
        depth_gate_settings=config["depth_settings"],
        attention_gate_settings=build_attention_gate_settings(profile),
        enrollment_policy=enrollment_policy,
    )
    config["robot_client"] = robot_client
    return service, config


def summarize_face(face: dict[str, Any], *, include_embedding: bool = False) -> dict[str, Any]:
    bbox = dict(face.get("bbox") or {})
    payload = {
        "bbox": {
            "x": int(bbox.get("x", 0) or 0),
            "y": int(bbox.get("y", 0) or 0),
            "w": int(bbox.get("w", 0) or 0),
            "h": int(bbox.get("h", 0) or 0),
        },
        "confidence": round(float(face.get("confidence", 0.0) or 0.0), 4),
    }
    if face.get("depth_m") is not None:
        payload["depth_m"] = round(float(face["depth_m"]), 4)
    if face.get("depth_valid_samples") is not None:
        payload["depth_valid_samples"] = int(face["depth_valid_samples"])
    if include_embedding and face.get("embedding") is not None:
        payload["embedding_dim"] = int(len(face["embedding"]))
    return payload


def _quality_checks(policy: FaceEnrollmentPolicy, metrics: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": "face_too_small",
            "passed": metrics.bbox_area >= policy.min_face_area,
            "value": metrics.bbox_area,
            "threshold": policy.min_face_area,
        },
        {
            "name": "face_clipped",
            "passed": not metrics.clipped,
            "value": metrics.clipped,
            "threshold": False,
        },
        {
            "name": "crop_valid",
            "passed": metrics.crop_valid,
            "value": metrics.crop_valid,
            "threshold": True,
        },
        {
            "name": "min_brightness",
            "passed": metrics.brightness >= policy.min_brightness,
            "value": metrics.brightness,
            "threshold": policy.min_brightness,
        },
        {
            "name": "max_brightness",
            "passed": metrics.brightness <= policy.max_brightness,
            "value": metrics.brightness,
            "threshold": policy.max_brightness,
        },
        {
            "name": "contrast",
            "passed": metrics.contrast >= policy.min_contrast,
            "value": metrics.contrast,
            "threshold": policy.min_contrast,
        },
    ]


def _landmark_point(face: dict[str, Any], name: str) -> tuple[float, float] | None:
    point = (face.get("landmarks") or {}).get(name)
    if point is None or len(point) != 2:
        return None
    return float(point[0]), float(point[1])


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return float(math.hypot(left[0] - right[0], left[1] - right[1]))


def _experimental_visibility_checks(face: dict[str, Any]) -> list[dict[str, Any]]:
    """Helper-only advisory checks for occlusion/pitch tuning experiments."""
    bbox = face.get("bbox") or {}
    width = float(bbox.get("w", 0.0) or 0.0)
    height = float(bbox.get("h", 0.0) or 0.0)
    area = width * height
    checks: list[dict[str, Any]] = [
        {
            "name": "recommended_face_area",
            "passed": area >= 4000.0,
            "value": area,
            "threshold": 4000.0,
            "note": "Advisory only: larger faces make pose and occlusion checks more reliable.",
        },
        {
            "name": "recommended_face_width",
            "passed": width >= 60.0,
            "value": width,
            "threshold": 60.0,
            "note": "Advisory only for 640x480 lab frames.",
        },
    ]

    left_eye = _landmark_point(face, "left_eye")
    right_eye = _landmark_point(face, "right_eye")
    nose = _landmark_point(face, "nose")
    mouth_left = _landmark_point(face, "mouth_left")
    mouth_right = _landmark_point(face, "mouth_right")
    if not all((left_eye, right_eye, nose, mouth_left, mouth_right)):
        checks.append(
            {
                "name": "experimental_landmark_geometry",
                "passed": False,
                "value": "missing",
                "threshold": "all five landmarks",
                "note": "Cannot estimate pitch/occlusion geometry without all landmarks.",
            }
        )
        return checks

    assert left_eye is not None
    assert right_eye is not None
    assert nose is not None
    assert mouth_left is not None
    assert mouth_right is not None
    eye_distance = max(_distance(left_eye, right_eye), 1e-6)
    mouth_distance = _distance(mouth_left, mouth_right)
    eye_mid = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    mouth_mid = ((mouth_left[0] + mouth_right[0]) / 2.0, (mouth_left[1] + mouth_right[1]) / 2.0)
    eye_to_mouth_vertical = max(abs(mouth_mid[1] - eye_mid[1]), 1e-6)
    nose_vertical_ratio = (nose[1] - eye_mid[1]) / eye_to_mouth_vertical
    mouth_width_ratio = mouth_distance / eye_distance
    mouth_tilt_ratio = abs(mouth_left[1] - mouth_right[1]) / eye_distance
    checks.extend(
        [
            {
                "name": "experimental_pitch_proxy",
                "passed": 0.25 <= nose_vertical_ratio <= 0.78,
                "value": nose_vertical_ratio,
                "threshold": "0.25..0.78",
                "note": "Advisory only: can flag some up/down head poses, but is not a true 3D pose estimate.",
            },
            {
                "name": "experimental_mouth_width_ratio",
                "passed": 0.45 <= mouth_width_ratio <= 1.65,
                "value": mouth_width_ratio,
                "threshold": "0.45..1.65",
                "note": "Advisory only: odd mouth geometry can indicate occlusion or bad landmarks.",
            },
            {
                "name": "experimental_mouth_tilt_ratio",
                "passed": mouth_tilt_ratio <= 0.35,
                "value": mouth_tilt_ratio,
                "threshold": 0.35,
                "note": "Advisory only: large mouth tilt can indicate roll, expression, or bad landmarks.",
            },
        ]
    )
    return checks


def describe_enrollment_face_quality(
    service: FaceRecognitionService,
    image: Any,
    face: dict[str, Any],
) -> dict[str, Any]:
    """Return helper-only enrollment-quality diagnostics using service internals."""
    policy = getattr(service, "_enrollment_policy", DEFAULT_FACE_ENROLLMENT_POLICY)
    metrics = service._measure_enrollment_face_quality(image, face)
    quality = service._assess_enrollment_face_quality(image, face, metrics)
    checks = _quality_checks(policy, metrics)
    experimental_checks = _experimental_visibility_checks(face)
    return {
        "accepted": quality.accepted,
        "reason": quality.reason,
        "guidance": quality.guidance,
        "metric": quality.metric,
        "metrics": asdict(metrics),
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
        "experimental_checks": experimental_checks,
        "experimental_failed_checks": [
            check["name"] for check in experimental_checks if not check["passed"]
        ],
        "experimental_note": (
            "These helper-only checks do not affect enrollment. Landmark geometry can miss "
            "real hand/hair occlusion because MTCNN may still infer plausible landmarks."
        ),
        "policy": asdict(policy),
    }


def save_preview_image(
    image: Any,
    *,
    output_dir: str | Path | None = None,
    prefix: str = "face_preview",
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save an enrollment preview image and adjacent JSON metadata."""
    if image is None or not hasattr(image, "shape"):
        return {}

    target_dir = Path(output_dir) if output_dir else DEFAULT_PREVIEW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_prefix = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(prefix or "face_preview")
    ).strip("_") or "face_preview"
    image_path = target_dir / f"{timestamp}_{safe_prefix}.png"
    metadata_path = target_dir / f"{timestamp}_{safe_prefix}.json"

    arr = np.asarray(image)
    if arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    else:
        arr = np.ascontiguousarray(arr)
    PILImage.fromarray(arr).save(image_path)

    metadata_payload = dict(metadata or {})
    metadata_payload.setdefault("image_path", str(image_path))
    metadata_payload.setdefault("image_shape", list(getattr(image, "shape", ())))
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "image_path": str(image_path),
        "metadata_path": str(metadata_path),
    }


def save_preview_data_url(
    data_url: str,
    *,
    output_dir: str | Path | None = None,
    prefix: str = "face_preview",
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save the exact PNG data URL that the display review would receive."""
    rendered = str(data_url or "").strip()
    marker = "base64,"
    if marker not in rendered:
        return {}

    target_dir = Path(output_dir) if output_dir else DEFAULT_PREVIEW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_prefix = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(prefix or "face_preview")
    ).strip("_") or "face_preview"
    image_path = target_dir / f"{timestamp}_{safe_prefix}.png"
    metadata_path = target_dir / f"{timestamp}_{safe_prefix}.json"

    encoded = rendered.split(marker, 1)[1]
    image_path.write_bytes(base64.b64decode(encoded))
    metadata_payload = dict(metadata or {})
    metadata_payload.setdefault("image_path", str(image_path))
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "image_path": str(image_path),
        "metadata_path": str(metadata_path),
    }


def json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
