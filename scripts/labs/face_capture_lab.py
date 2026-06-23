#!/usr/bin/env python3
"""Structured face perception capture lab.

This lab saves camera artifacts, production-style face predictions, and a
human-labeling template for later quantitative eval.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image as PILImage

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from argos_src.profile_config import load_scenario_profile
from scripts.labs.face_lab_common import (
    add_enrollment_policy_args,
    add_profile_args,
    build_enrollment_policy,
    build_face_service,
    configure_logging,
    describe_enrollment_face_quality,
    summarize_face,
)
from scripts.labs.perception_lab_common import (
    DEFAULT_LAB_ROOT,
    LabRunWriter,
    json_ready,
    write_json,
)


FACE_MODES = ("enrollment", "recognition", "attention", "depth", "all")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture structured face perception lab samples for eval."
    )
    add_profile_args(parser)
    add_enrollment_policy_args(parser)
    parser.add_argument("--mode", choices=FACE_MODES, default="all")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument(
        "--max-frame-wait-sec",
        type=float,
        default=0.0,
        help="Maximum wall-clock seconds to wait per frame. Default 0 waits indefinitely.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_LAB_ROOT),
        help="Root directory for lab runs. Default: var/labs.",
    )
    parser.add_argument("--run-id", default="")
    return parser


def _requested(mode: str, component: str) -> bool:
    if component == "face_detection":
        return True
    if mode == "all":
        return True
    return {
        "face_enrollment": "enrollment",
        "face_recognition": "recognition",
        "attention_gate": "attention",
        "depth_gate": "depth",
    }.get(component) == mode


def _save_image(path: Path, image: Any) -> str:
    arr = np.asarray(image)
    if arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    else:
        arr = np.ascontiguousarray(arr)
    PILImage.fromarray(arr).save(path)
    return str(path)


def _component_skip(reason: str) -> dict[str, Any]:
    return {"measured": False, "skipped_reason": reason}


def _recognition_predictions(service: Any, faces: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = []
    recognized_count = 0
    unknown_count = 0
    for index, face in enumerate(faces):
        match = service._recognize_face_match(face)
        if match is None:
            unknown_count += 1
        else:
            recognized_count += 1
        predictions.append(
            {
                "face_index": index,
                "match": (
                    {
                        "person_id": match["person_id"],
                        "name": match["name"],
                        "similarity": round(float(match["similarity"]), 4),
                    }
                    if match is not None
                    else None
                ),
            }
        )
    return {
        "measured": True,
        "recognized_count": recognized_count,
        "unknown_count": unknown_count,
        "predictions": predictions,
    }


def _enrollment_predictions(
    service: Any,
    image: Any,
    faces: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_face, multiple_people_visible = service._select_enrollment_face(faces)
    predictions = []
    accepted_selected = False
    for index, face in enumerate(faces):
        quality = describe_enrollment_face_quality(service, image, face)
        is_selected = selected_face is face
        accepted = bool(quality.get("accepted"))
        accepted_selected = accepted_selected or (is_selected and accepted)
        metrics = dict(quality.get("metrics") or {})
        predictions.append(
            {
                "face_index": index,
                "selected_for_enrollment": is_selected,
                "accepted": accepted,
                "reason": quality.get("reason", ""),
                "failed_checks": quality.get("failed_checks", []),
                "metrics": {
                    "bbox_area": metrics.get("bbox_area", 0),
                    "brightness": metrics.get("brightness", 0.0),
                    "contrast": metrics.get("contrast", 0.0),
                    "sharpness": metrics.get("sharpness", 0.0),
                    "eye_tilt": metrics.get("eye_tilt", 0.0),
                    "nose_center_offset": metrics.get("nose_center_offset", 0.0),
                },
                "policy": quality.get("policy", {}),
            }
        )
    return {
        "measured": True,
        "accepted": accepted_selected,
        "multiple_people_visible": multiple_people_visible,
        "predictions": predictions,
    }


def _attention_predictions(
    service: Any,
    image: Any,
    image_shape: tuple[int, ...],
    faces: list[dict[str, Any]],
    now: float,
) -> dict[str, Any]:
    gate = getattr(service, "_attention_gate", None)
    settings = getattr(gate, "settings", None)
    if gate is None or settings is None:
        return _component_skip("attention_gate_unavailable")
    if not bool(getattr(settings, "enabled", False)):
        return _component_skip("attention_gate_disabled")

    predictions = []
    attentive_count = 0
    for index, face in enumerate(faces):
        track_id = str(face.get("recognized_person_id") or f"lab-face-{index}")
        observation = gate.evaluate(
            image,
            face,
            image_shape=image_shape,
            track_id=track_id,
            now=now,
        )
        if observation.attentive:
            attentive_count += 1
        predictions.append(
            {
                "face_index": index,
                "attentive": bool(observation.attentive),
                "confidence": round(float(observation.confidence), 4),
                "reason": observation.reason,
                "yaw_deg": observation.yaw_deg,
                "pitch_deg": observation.pitch_deg,
                "roll_deg": observation.roll_deg,
                "raw_attentive": bool(observation.raw_attentive),
                "raw_confidence": round(float(observation.raw_confidence), 4),
            }
        )
    return {
        "measured": True,
        "attentive_count": attentive_count,
        "predictions": predictions,
        "thresholds": {
            "max_abs_yaw_deg": getattr(settings, "max_abs_yaw_deg", None),
            "max_abs_pitch_deg": getattr(settings, "max_abs_pitch_deg", None),
            "max_abs_roll_deg": getattr(settings, "max_abs_roll_deg", None),
            "min_confidence": getattr(settings, "min_confidence", None),
        },
    }


def _label_template(sample: dict[str, Any]) -> dict[str, Any]:
    components = sample.get("components") or {}
    labels: dict[str, Any] = {}
    if "face_detection" in components:
        labels["actual_face_count"] = None
        labels["expected_bbox_ok"] = None
    if "face_enrollment" in components:
        labels["should_accept_for_enrollment"] = None
        labels["label_reason"] = None
    if "face_recognition" in components:
        labels["actual_person_id"] = None
        labels["recognition_correct"] = None
    if "depth_gate" in components:
        labels["should_pass_depth_gate"] = None
        labels["approx_distance_bucket"] = None
    if "attention_gate" in components:
        labels["should_be_attentive"] = None
        labels["attention_pose"] = None
    return {
        "sample_id": sample["sample_id"],
        "artifacts": sample.get("artifacts", {}),
        "prediction_summary": _prediction_summary(sample),
        "labels": labels,
    }


def _prediction_summary(sample: dict[str, Any]) -> dict[str, Any]:
    components = sample.get("components") or {}
    detection = components.get("face_detection") or {}
    recognition = components.get("face_recognition") or {}
    enrollment = components.get("face_enrollment") or {}
    depth = components.get("depth_gate") or {}
    attention = components.get("attention_gate") or {}
    first_match = None
    for item in recognition.get("predictions", []) or []:
        if item.get("match") is not None:
            first_match = item.get("match")
            break
    return {
        "detected_count": detection.get("detected_count"),
        "enrollment_accepted": enrollment.get("accepted"),
        "recognized_count": recognition.get("recognized_count"),
        "first_match": first_match,
        "depth_gate_accepted": depth.get("accepted"),
        "attentive_count": attention.get("attentive_count"),
    }


def _capture_one(
    *,
    service: Any,
    config: dict[str, Any],
    writer: LabRunWriter,
    mode: str,
    timeout: float,
    max_frame_wait_sec: float,
    frame_index: int,
) -> dict[str, Any]:
    sample_id = f"frame_{frame_index:04d}"
    started_at = time.monotonic()
    attempts = 0
    image = None
    depth_m = None
    while image is None:
        attempts += 1
        image, depth_m = service._capture_for_recognition(
            config["camera_resource_id"],
            timeout=timeout,
        )
        elapsed = time.monotonic() - started_at
        if image is not None:
            break
        if max_frame_wait_sec > 0.0 and elapsed >= max_frame_wait_sec:
            break
        time.sleep(0.02)

    base_sample: dict[str, Any] = {
        "sample_id": sample_id,
        "captured_at_unix_s": round(time.time(), 3),
        "capture": {
            "success": image is not None,
            "attempts": attempts,
            "wait_s": round(time.monotonic() - started_at, 3),
        },
        "artifacts": {},
        "components": {},
    }
    if image is None:
        for component in (
            "face_detection",
            "face_enrollment",
            "face_recognition",
            "depth_gate",
            "attention_gate",
        ):
            if _requested(mode, component):
                base_sample["components"][component] = _component_skip("capture_failed")
        return base_sample

    image_path = writer.artifacts_dir / f"{sample_id}.png"
    base_sample["artifacts"]["image_path"] = _save_image(image_path, image)
    if depth_m is not None:
        depth_path = writer.artifacts_dir / f"{sample_id}_depth.npy"
        np.save(depth_path, np.asarray(depth_m))
        base_sample["artifacts"]["depth_path"] = str(depth_path)
    base_sample["image_shape"] = list(image.shape)

    prepared = service._prepare_faces_for_recognition_result(image, depth_m)
    faces = list(prepared.faces)
    base_sample["components"]["face_detection"] = {
        "measured": True,
        "detected_count": int(prepared.detected_count),
        "usable_face_count": len(faces),
        "rejected_count": int(prepared.rejected_count),
        "preparation_reason": prepared.reason,
        "faces": [summarize_face(face, include_embedding=True) for face in faces],
    }

    if _requested(mode, "face_enrollment"):
        base_sample["components"]["face_enrollment"] = (
            _enrollment_predictions(service, image, faces)
            if faces
            else {
                "measured": True,
                "accepted": False,
                "predictions": [],
                "reason": prepared.reason or "no_detection",
            }
        )
    if _requested(mode, "face_recognition"):
        base_sample["components"]["face_recognition"] = _recognition_predictions(
            service,
            faces,
        )
    if _requested(mode, "depth_gate"):
        depth_settings = config.get("depth_settings")
        if depth_settings is None:
            base_sample["components"]["depth_gate"] = _component_skip("depth_gate_disabled")
        else:
            base_sample["components"]["depth_gate"] = {
                "measured": True,
                "accepted": prepared.reason != "depth_rejected",
                "rejected_count": int(prepared.rejected_count),
                "reason": prepared.reason,
                "thresholds": asdict(depth_settings),
                "faces": [
                    {
                        "face_index": index,
                        "depth_m": face.get("depth_m"),
                        "depth_valid_samples": face.get("depth_valid_samples"),
                    }
                    for index, face in enumerate(faces)
                ],
            }
    if _requested(mode, "attention_gate"):
        base_sample["components"]["attention_gate"] = _attention_predictions(
            service,
            image,
            tuple(image.shape),
            faces,
            now=time.time(),
        )
    return base_sample


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    enrollment_policy = build_enrollment_policy(args)
    profile = load_scenario_profile(args.profile)
    service, config = build_face_service(args, enrollment_policy=enrollment_policy)
    writer = LabRunWriter(
        component="face",
        mode=args.mode,
        root=args.output_root,
        run_id=args.run_id or None,
    )
    depth_settings = config.get("depth_settings")
    attention_settings = getattr(getattr(service, "_attention_gate", None), "settings", None)
    writer.write_manifest(
        {
            "profile": config["profile_name"],
            "profile_arg": args.profile,
            "camera_resource_id": config["camera_resource_id"],
            "provider_transport": config["provider_transport"],
            "db_path": config["db_path"],
            "identity_db_path": config["identity_db_path"],
            "enabled_components": {
                "face_detection": True,
                "face_enrollment": _requested(args.mode, "face_enrollment"),
                "face_recognition": _requested(args.mode, "face_recognition"),
                "depth_gate": depth_settings is not None
                and _requested(args.mode, "depth_gate"),
                "attention_gate": bool(getattr(attention_settings, "enabled", False))
                and _requested(args.mode, "attention_gate"),
                "audio_detection": False,
            },
            "thresholds": {
                "recognition_threshold": config["recognition_threshold"],
                "depth_gate": asdict(depth_settings) if depth_settings is not None else None,
                "enrollment_policy": asdict(enrollment_policy),
                "attention_gate": asdict(attention_settings)
                if attention_settings is not None
                else None,
            },
            "requested_frames": int(args.frames),
            "interval_sec": float(args.interval_sec),
            "profile_depth_gate_enabled": bool(profile.face_recognition.depth_gate.enabled),
            "profile_attention_gate_enabled": bool(
                profile.face_recognition.attention_gate.enabled
            ),
        }
    )
    timeout = (
        depth_settings.capture_timeout_sec
        if depth_settings is not None
        else float(args.capture_timeout_sec or 1.5)
    )
    samples: list[dict[str, Any]] = []
    try:
        for frame_index in range(1, max(1, int(args.frames)) + 1):
            sample = _capture_one(
                service=service,
                config=config,
                writer=writer,
                mode=args.mode,
                timeout=float(timeout),
                max_frame_wait_sec=max(0.0, float(args.max_frame_wait_sec)),
                frame_index=frame_index,
            )
            writer.append_sample(sample, _label_template(sample))
            samples.append(sample)
            print(json_ready(_prediction_summary(sample)))
            if frame_index < int(args.frames):
                time.sleep(max(0.0, float(args.interval_sec)))
    finally:
        service.shutdown()
        robot_client = config.get("robot_client")
        if robot_client is not None:
            robot_client.shutdown()

    captured = sum(1 for sample in samples if sample.get("capture", {}).get("success"))
    writer.write_quick_summary(
        [
            "# Face Capture Lab",
            "",
            f"- run_dir: `{writer.run_dir}`",
            f"- mode: `{args.mode}`",
            f"- captured_frames: {captured}/{len(samples)}",
            f"- labels: `{writer.labels_path}`",
            "",
            "Edit `labels.todo.jsonl`, then run:",
            "",
            f"```bash\npoetry run python -m scripts.eval.perception_eval --run-dir {writer.run_dir}\n```",
        ]
    )
    write_json(writer.reports_dir / "quick_summary.json", {"captured": captured, "total": len(samples)})
    print(f"Wrote face lab run: {writer.run_dir}")
    return 0 if captured > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
