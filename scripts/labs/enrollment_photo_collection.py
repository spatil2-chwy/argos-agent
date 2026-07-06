#!/usr/bin/env python3
"""Collect raw per-person camera photos for enrollment/model comparison."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image as PILImage

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.labs.enrollment_collection_common import (
    add_person_collection_args,
    create_display_runtime_for_profile,
    create_provider_for_resource,
    json_ready,
    load_profile,
    parse_camera_specs,
    resolve_collection_session,
    safe_path_part,
    write_session_manifest,
)
from scripts.labs.perception_lab_common import append_jsonl, write_json

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect raw RGB/RGBD photos for one person across configured camera resources."
        )
    )
    add_person_collection_args(parser)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--interval-sec", type=float, default=0.5)
    parser.add_argument("--timeout-sec", type=float, default=2.0)
    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Capture frames automatically using --interval-sec. "
            "Default is manual: press Enter before each photo."
        ),
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help=(
            "Camera alias/resource spec. Repeatable. Format alias=resource_id. "
            "Default: face_camera=<resources.face_camera from the selected profile>."
        ),
    )
    parser.add_argument(
        "--rgbd-camera",
        action="append",
        default=[],
        help=(
            "Camera alias or resource id to try as RGBD before falling back to RGB. "
            "Repeatable. Default: none."
        ),
    )
    parser.add_argument(
        "--skip-countdown",
        action="store_true",
        help="Skip the short display countdown before each camera starts.",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _save_image(path: Path, image: Any) -> str:
    arr = np.asarray(image)
    if arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    else:
        arr = np.ascontiguousarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(arr).save(path)
    return str(path)


def _capture_rgbd(client: Any, *, resource_id: str, timeout_sec: float):
    getter = getattr(client, "get_latest_rgbd", None)
    if not callable(getter):
        return None
    try:
        return getter(resource_id=resource_id, timeout=float(timeout_sec))
    except Exception as exc:
        logger.debug("RGBD capture failed for %s: %s", resource_id, exc)
        return None


def _capture_rgb(client: Any, *, resource_id: str, timeout_sec: float):
    getter = getattr(client, "get_latest_image", None)
    if not callable(getter):
        raise RuntimeError("Selected provider client does not support get_latest_image().")
    return getter(resource_id=resource_id, timeout=float(timeout_sec))


def _frame_metadata(frame: Any, *, capture_kind: str) -> dict[str, Any]:
    payload = {"capture_kind": capture_kind}
    for attr in ("resource_id", "captured_at", "stamp_s", "color_stamp_s", "depth_stamp_s"):
        if hasattr(frame, attr):
            payload[attr] = getattr(frame, attr)
    if capture_kind == "rgbd" and hasattr(frame, "delta_ms"):
        payload["delta_ms"] = getattr(frame, "delta_ms")
    return payload


def _capture_one(
    *,
    client: Any,
    camera_alias: str,
    resource_id: str,
    output_dir: Path,
    sample_id: str,
    timeout_sec: float,
    try_rgbd: bool,
) -> dict[str, Any]:
    started_at = time.monotonic()
    frame = _capture_rgbd(client, resource_id=resource_id, timeout_sec=timeout_sec) if try_rgbd else None
    capture_kind = "rgbd" if frame is not None else "rgb"
    if frame is None:
        try:
            frame = _capture_rgb(client, resource_id=resource_id, timeout_sec=timeout_sec)
        except Exception as exc:
            return {
                "sample_id": sample_id,
                "modality": "photo",
                "camera_alias": camera_alias,
                "camera_resource_id": resource_id,
                "capture": {
                    "success": False,
                    "wait_s": round(time.monotonic() - started_at, 3),
                    "failure_reason": "capture_failed",
                    "message": str(exc),
                },
                "artifacts": {},
            }
    if frame is None:
        return {
            "sample_id": sample_id,
            "modality": "photo",
            "camera_alias": camera_alias,
            "camera_resource_id": resource_id,
            "capture": {
                "success": False,
                "wait_s": round(time.monotonic() - started_at, 3),
                "failure_reason": "no_frame",
            },
            "artifacts": {},
        }

    image = getattr(frame, "color_image", None) if capture_kind == "rgbd" else getattr(frame, "image", frame)
    if image is None:
        return {
            "sample_id": sample_id,
            "modality": "photo",
            "camera_alias": camera_alias,
            "camera_resource_id": resource_id,
            "capture": {
                "success": False,
                "wait_s": round(time.monotonic() - started_at, 3),
                "failure_reason": "empty_image",
            },
            "artifacts": {},
        }

    image_path = output_dir / f"{sample_id}.png"
    artifacts = {"image_path": _save_image(image_path, image)}
    if capture_kind == "rgbd":
        depth = getattr(frame, "depth_m", None)
        if depth is not None:
            depth_path = output_dir / f"{sample_id}_depth_m.npy"
            np.save(depth_path, np.asarray(depth))
            artifacts["depth_m_path"] = str(depth_path)
    metadata_path = output_dir / f"{sample_id}.json"
    sample = {
        "sample_id": sample_id,
        "modality": "photo",
        "camera_alias": camera_alias,
        "camera_resource_id": resource_id,
        "captured_at_unix_s": round(time.time(), 3),
        "capture": {
            "success": True,
            "wait_s": round(time.monotonic() - started_at, 3),
            **_frame_metadata(frame, capture_kind=capture_kind),
        },
        "image": {
            "shape": list(getattr(image, "shape", ())),
            "dtype": str(getattr(image, "dtype", "")),
        },
        "artifacts": artifacts,
    }
    write_json(metadata_path, sample)
    sample["artifacts"]["metadata_path"] = str(metadata_path)
    return sample


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(bool(args.verbose))

    profile = load_profile(args.profile)
    cameras = parse_camera_specs(
        args.camera,
        default_resource_id=profile.resources.face_camera,
        default_alias="face_camera",
    )
    rgbd_targets = {
        safe_path_part(item)
        for item in (args.rgbd_camera or [])
        if str(item or "").strip()
    }
    session = resolve_collection_session(
        output_root=args.output_root,
        person_name=args.person_name,
        person_id=args.person_id,
        session_id=args.session_id,
    )
    session_dir = Path(session["session_dir"])
    photos_root = session_dir / "photos"
    photos_root.mkdir(parents=True, exist_ok=True)
    samples_path = session_dir / "photo_samples.jsonl"
    samples_path.write_text("", encoding="utf-8")

    display = create_display_runtime_for_profile(
        profile,
        disabled=bool(args.no_display),
        provider_transport=args.provider_transport,
    )
    write_session_manifest(
        session_dir=session_dir,
        filename="photo_manifest.json",
        payload={
            "collection_kind": "photos",
            "person_name": session["person_name"],
            "person_slug": session["person_slug"],
            "session_id": session["session_id"],
            "profile": profile.name,
            "profile_arg": args.profile,
            "cameras": cameras,
            "rgbd_targets": sorted(rgbd_targets),
            "requested_frames_per_camera": int(args.frames),
            "interval_sec": float(args.interval_sec),
            "timeout_sec": float(args.timeout_sec),
            "trigger_mode": "auto" if bool(args.auto) else "manual_enter",
        },
    )

    successes = 0
    total = max(1, int(args.frames)) * len(cameras)
    try:
        if display is not None:
            display.show_message(f"Photo collection: {session['person_name']}")
        for camera in cameras:
            alias = camera["alias"]
            resource_id = camera["resource_id"]
            camera_dir = photos_root / alias
            camera_dir.mkdir(parents=True, exist_ok=True)
            try_rgbd = alias in rgbd_targets or safe_path_part(resource_id) in rgbd_targets
            if display is not None:
                display.show_message(f"{alias}: ready")
                if not args.skip_countdown:
                    display.show_countdown(3)
            client = create_provider_for_resource(
                profile,
                resource_id=resource_id,
                provider_transport=args.provider_transport,
            )
            client.start()
            try:
                for index in range(1, max(1, int(args.frames)) + 1):
                    sample_id = f"{alias}_{index:04d}"
                    if not bool(args.auto):
                        if display is not None:
                            display.show_message(f"Ready photo {index}/{int(args.frames)}")
                        input(
                            f"[{alias} {index}/{int(args.frames)}] "
                            "Pose/change angle, then press Enter to capture.\n"
                        )
                    if display is not None:
                        display.show_subtitle(f"Capturing {alias} {index}/{int(args.frames)}")
                    sample = _capture_one(
                        client=client,
                        camera_alias=alias,
                        resource_id=resource_id,
                        output_dir=camera_dir,
                        sample_id=sample_id,
                        timeout_sec=float(args.timeout_sec),
                        try_rgbd=try_rgbd,
                    )
                    sample["person_name"] = session["person_name"]
                    sample["person_slug"] = session["person_slug"]
                    sample["session_id"] = session["session_id"]
                    append_jsonl(samples_path, json_ready(sample))
                    if sample.get("capture", {}).get("success"):
                        successes += 1
                    print(
                        {
                            "sample_id": sample_id,
                            "camera": alias,
                            "success": bool(sample.get("capture", {}).get("success")),
                            "capture_kind": sample.get("capture", {}).get("capture_kind"),
                        }
                    )
                    if bool(args.auto) and index < int(args.frames):
                        time.sleep(max(0.0, float(args.interval_sec)))
            finally:
                client.shutdown()
        if display is not None:
            display.show_message(f"Saved photos: {successes}/{total}")
    finally:
        if display is not None:
            display.shutdown()

    print(f"Wrote photo collection: {session_dir}")
    return 0 if successes > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
