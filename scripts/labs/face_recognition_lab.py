#!/usr/bin/env python3
"""Standalone face-recognition lab.

Run from the repo root:

    source setup_shell.sh
    poetry run python -m scripts.labs.face_recognition_lab --once
    poetry run python -m scripts.labs.face_recognition_lab --loop --interval 0.5

Optional diagnostics:

    poetry run python -m scripts.labs.face_recognition_lab --once --include-enrollment-quality
    poetry run python -m scripts.labs.face_recognition_lab --once --disable-depth

This uses the same Argos face detection, optional depth gate, embedding extraction,
and face DB matching as the live system, but it does not start the realtime
agent, proactive events, audio, or LLM session.
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

from scripts.labs.face_lab_common import (
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
            "Run one-shot or looping face recognition without the full realtime agent."
        )
    )
    add_profile_args(parser)
    add_enrollment_policy_args(parser)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--once",
        action="store_true",
        help="Run one recognition capture. This is the default.",
    )
    mode.add_argument(
        "--loop",
        action="store_true",
        help="Keep recognizing until Ctrl-C.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Loop delay in seconds. Defaults to face_recognition.loop_interval_sec from the profile.",
    )
    parser.add_argument(
        "--include-enrollment-quality",
        action="store_true",
        help="Also print registration quality metrics for each detected face.",
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
    return parser


def _recognize_once(
    service: Any,
    *,
    camera_topic: str,
    timeout: float,
    include_enrollment_quality: bool,
    max_frame_wait_sec: float,
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
                "success": False,
                "failure_reason": "capture_failed",
                "capture_attempts": attempts,
                "capture_wait_s": round(elapsed, 3),
                "message": "No color frame or synced RGBD pair was captured.",
            }

    prepared = service._prepare_faces_for_recognition_result(image, depth_m)
    payload: dict[str, Any] = {
        "success": True,
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
        "recognized_count": 0,
        "unknown_count": 0,
    }
    if not prepared.faces:
        return payload

    for index, face in enumerate(prepared.faces):
        face_payload = summarize_face(face, include_embedding=True)
        face_payload["index"] = index
        match = service._recognize_face_match(face)
        if match is None:
            payload["unknown_count"] += 1
            face_payload["match"] = None
        else:
            payload["recognized_count"] += 1
            face_payload["match"] = {
                "person_id": match["person_id"],
                "name": match["name"],
                "similarity": round(float(match["similarity"]), 4),
            }
        if include_enrollment_quality:
            face_payload["enrollment_quality"] = describe_enrollment_face_quality(
                service,
                image,
                face,
            )
        payload["faces"].append(face_payload)

    return payload


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    enrollment_policy = build_enrollment_policy(args)
    service, config = build_face_service(args, enrollment_policy=enrollment_policy)
    depth_settings = config["depth_settings"]
    timeout = (
        depth_settings.capture_timeout_sec
        if depth_settings is not None
        else float(args.capture_timeout_sec or 1.5)
    )

    try:
        json_print(
            {
                "mode": "loop" if args.loop else "once",
                "profile": config["profile_name"],
                "camera_topic": config["camera_topic"],
                "db_path": config["db_path"],
                "identity_db_path": config["identity_db_path"],
                "loop_interval_sec": config["loop_interval_sec"],
                "recognition_threshold": config["recognition_threshold"],
                "depth_gate": vars(depth_settings) if depth_settings is not None else None,
                "include_enrollment_quality": bool(args.include_enrollment_quality),
            }
        )
        loop_interval_sec = (
            float(args.interval)
            if args.interval is not None
            else float(config["loop_interval_sec"])
        )
        while True:
            payload = _recognize_once(
                service,
                camera_topic=config["camera_topic"],
                timeout=timeout,
                include_enrollment_quality=args.include_enrollment_quality,
                max_frame_wait_sec=max(0.0, float(args.max_frame_wait_sec)),
            )
            payload["captured_at_unix_s"] = round(time.time(), 3)
            json_print(payload)
            if not args.loop:
                return 0 if payload.get("success") else 2
            time.sleep(max(0.0, loop_interval_sec))
    except KeyboardInterrupt:
        print("Stopped.")
        return 0
    finally:
        service.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
