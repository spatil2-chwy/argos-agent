#!/usr/bin/env python3
"""Standalone face-registration lab.

Run from `argos_src`:

    source setup_shell.sh
    poetry run python -m argos_src.helpers.face_registration_lab --frames 5
    poetry run python -m argos_src.helpers.face_registration_lab --name "Jane Doe" --enroll

Useful tuning examples:

    poetry run python -m argos_src.helpers.face_registration_lab --min-contrast 12
    poetry run python -m argos_src.helpers.face_registration_lab --max-face-depth-m 2.5
    poetry run python -m argos_src.helpers.face_registration_lab --disable-depth

Default mode is a dry-run diagnostic: it captures frames, runs the same
registration preprocessing, and prints quality metrics without saving a person.
Pass `--enroll` to actually write a face reference to the configured DB.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from argos_src.helpers.face_lab_common import (
    add_enrollment_policy_args,
    add_profile_args,
    build_enrollment_policy,
    build_face_service,
    configure_logging,
    describe_enrollment_face_quality,
    json_print,
    summarize_face,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture live camera frames and test Argos face-registration preprocessing "
            "without starting the full realtime agent."
        )
    )
    add_profile_args(parser)
    add_enrollment_policy_args(parser)
    parser.add_argument(
        "--name",
        default="",
        help="Official name to use when --enroll is set.",
    )
    parser.add_argument(
        "--username",
        default="",
        help="Optional username to store with --enroll.",
    )
    parser.add_argument(
        "--enroll",
        action="store_true",
        help="Actually save an enrolled face. Without this flag, only diagnostics run.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Dry-run diagnostic frames to capture before enrollment. Default: 5.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.1,
        help="Delay between dry-run diagnostic frames. Default: 0.1.",
    )
    parser.add_argument(
        "--max-frame-wait-sec",
        type=float,
        default=0.0,
        help=(
            "Maximum wall-clock seconds to wait for each synced frame. "
            "Default 0 means wait until a frame arrives or Ctrl-C is pressed."
        ),
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print full per-face diagnostic internals instead of compact tuning output.",
    )
    return parser


def _agent_equivalent_for_quality(quality: dict[str, Any]) -> dict[str, Any]:
    accepted = bool(quality.get("accepted", False))
    reason = str(quality.get("reason", "") or "")
    guidance = str(quality.get("guidance", "") or "")
    if accepted:
        return {
            "status": "candidate_ok",
            "message": "This frame would pass per-frame registration quality.",
        }
    return {
        "status": "retry_quality",
        "failure_reason": reason,
        "message": guidance or "Please face me directly and hold still for a second.",
    }


def _agent_equivalent_for_prepare(prepared: Any) -> dict[str, Any]:
    reason = str(getattr(prepared, "reason", "") or "")
    if reason == "depth_rejected":
        return {
            "status": "retry_quality",
            "failure_reason": "depth_rejected",
            "message": (
                "I can see a face, but I need a closer face view with valid depth. "
                "Please stand within about two meters and face me."
            ),
        }
    if reason == "no_embedding":
        return {
            "status": "retry_quality",
            "failure_reason": "no_embedding",
            "message": (
                "I can see a face, but I couldn't encode it clearly. "
                "Please face me directly and hold still for a second."
            ),
        }
    return {
        "status": "retry_quality",
        "failure_reason": reason or "no_detection",
        "message": (
            "I couldn't get a stable face view. Please stand in front of me, "
            "look at the camera, and try again."
        ),
    }


def _diagnose_one_frame(
    service: Any,
    camera_topic: str,
    timeout: float,
    *,
    max_frame_wait_sec: float,
    details: bool,
) -> dict[str, Any]:
    started_at = time.monotonic()
    attempts = 0
    image = None
    depth_m = None
    while image is None:
        attempts += 1
        image, depth_m = service._capture_for_recognition(camera_topic, timeout=timeout)
        elapsed = time.monotonic() - started_at
        if image is not None:
            break
        if max_frame_wait_sec > 0.0 and elapsed >= max_frame_wait_sec:
            return {
                "captured": False,
                "failure_reason": "capture_failed",
                "capture_attempts": attempts,
                "capture_wait_s": round(elapsed, 3),
                "message": "No color frame or synced RGBD pair was captured.",
                "agent_equivalent": {
                    "status": "error",
                    "failure_reason": "capture_failed",
                    "message": (
                        "I couldn't get a clear camera view right now. "
                        "Please try again in a moment."
                    ),
                },
            }

    prepared = service._prepare_faces_for_recognition_result(image, depth_m)
    payload: dict[str, Any] = {
        "captured": True,
        "capture_attempts": attempts,
        "capture_wait_s": round(time.monotonic() - started_at, 3),
        "image_shape": list(image.shape),
        "depth_enabled": depth_m is not None,
        "preparation": {
            "reason": prepared.reason,
            "detected_count": prepared.detected_count,
            "rejected_count": prepared.rejected_count,
            "usable_face_count": len(prepared.faces),
        },
        "faces": [],
    }
    if not prepared.faces:
        payload["agent_equivalent"] = _agent_equivalent_for_prepare(prepared)
        return payload

    selected_face, multiple_people_visible = service._select_enrollment_face(prepared.faces)
    payload["multiple_people_visible"] = multiple_people_visible
    if multiple_people_visible:
        payload["agent_equivalent"] = {
            "status": "retry_single_face",
            "failure_reason": "multiple_faces",
            "message": (
                "I can still see more than one face. Please make sure you're the only "
                "person in view and try again."
            ),
        }

    for index, face in enumerate(prepared.faces):
        face_payload = summarize_face(face, include_embedding=True)
        face_payload["index"] = index
        quality = describe_enrollment_face_quality(
            service,
            image,
            face,
        )
        if details:
            face_payload["enrollment_quality"] = quality
        else:
            metrics = quality.get("metrics", {})
            face_payload["registration_check"] = {
                "accepted": quality.get("accepted"),
                "reason": quality.get("reason"),
                "failed_checks": quality.get("failed_checks", []),
                "brightness": round(float(metrics.get("brightness", 0.0) or 0.0), 2),
                "contrast": round(float(metrics.get("contrast", 0.0) or 0.0), 2),
                "sharpness": round(float(metrics.get("sharpness", 0.0) or 0.0), 2),
                "eye_tilt": round(float(metrics.get("eye_tilt", 0.0) or 0.0), 3),
                "nose_center_offset": round(
                    float(metrics.get("nose_center_offset", 0.0) or 0.0),
                    3,
                ),
                "bbox_area": int(metrics.get("bbox_area", 0) or 0),
                "policy": {
                    "min_brightness": quality["policy"]["min_brightness"],
                    "min_contrast": quality["policy"]["min_contrast"],
                    "min_sharpness": quality["policy"]["min_sharpness"],
                    "max_eye_tilt": quality["policy"]["max_eye_tilt"],
                    "max_nose_center_offset": quality["policy"]["max_nose_center_offset"],
                    "min_face_area": quality["policy"]["min_face_area"],
                },
            }
        match = service._recognize_face_match(face)
        agent_equivalent = _agent_equivalent_for_quality(quality)
        face_payload["existing_match"] = (
            {
                "person_id": match["person_id"],
                "name": match["name"],
                "similarity": round(float(match["similarity"]), 4),
            }
            if match is not None
            else None
        )
        if bool(quality.get("accepted")) and match is not None:
            agent_equivalent = {
                "status": "retry_already_known",
                "failure_reason": "already_known",
                "recognized_name": match["name"],
                "message": f"I think I already know you as {match['name']}.",
            }
        if details:
            face_payload["agent_equivalent"] = agent_equivalent
        else:
            face_payload["registration_check"]["agent_equivalent"] = agent_equivalent
        if selected_face is face:
            face_payload["selected_for_enrollment"] = True
        payload["faces"].append(face_payload)

    return payload


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.enroll and not str(args.name or "").strip():
        parser.error("--name is required when --enroll is set")

    configure_logging(args.verbose)
    enrollment_policy = build_enrollment_policy(args)
    service, config = build_face_service(args, enrollment_policy=enrollment_policy)
    depth_settings = config["depth_settings"]

    try:
        json_print(
            {
                "mode": "enroll" if args.enroll else "dry_run",
                "profile": config["profile_name"],
                "camera_topic": config["camera_topic"],
                "db_path": config["db_path"],
                "identity_db_path": config["identity_db_path"],
                "loop_interval_sec": config["loop_interval_sec"],
                "recognition_threshold": config["recognition_threshold"],
                "depth_gate": vars(depth_settings) if depth_settings is not None else None,
                "enrollment_policy": vars(enrollment_policy),
            }
        )

        timeout = (
            depth_settings.capture_timeout_sec
            if depth_settings is not None
            else float(args.capture_timeout_sec or 1.5)
        )
        diagnostics = []
        for index in range(max(0, int(args.frames))):
            frame_payload = _diagnose_one_frame(
                service,
                config["camera_topic"],
                timeout=timeout,
                max_frame_wait_sec=max(0.0, float(args.max_frame_wait_sec)),
                details=bool(args.details),
            )
            frame_payload["frame_index"] = index
            diagnostics.append(frame_payload)
            json_print(frame_payload)
            if index < args.frames - 1:
                time.sleep(max(0.0, float(args.sleep_sec)))

        if not args.enroll:
            accepted = sum(
                1
                for frame in diagnostics
                for face in frame.get("faces", [])
                if face.get("selected_for_enrollment")
                and (
                    face.get("enrollment_quality", {}).get("accepted")
                    or face.get("registration_check", {}).get("accepted")
                )
            )
            json_print(
                {
                    "summary": "dry_run_complete",
                    "accepted_selected_frames": accepted,
                    "required_stable_frames": 3,
                    "saved": False,
                }
            )
            return 0

        result = service.enroll_visible_person(
            official_name=args.name,
            username=args.username,
            camera_topic=config["camera_topic"],
        )
        json_print({"summary": "enrollment_result", "result": result})
        return 0 if result.get("success") else 2
    finally:
        service.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
