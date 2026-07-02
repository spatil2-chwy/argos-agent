#!/usr/bin/env python3
"""Compare live Argos face embeddings with saved/reloaded image variants.

Run from the repo root:

    source setup_shell.sh
    poetry run python -m scripts.labs.face_embedding_parity_lab --target "Your Name"

This diagnostic does not enroll or modify the face DB. It captures one live
camera frame through the same FaceRecognitionService path as the agent, then
compares:

- direct live array embedding
- PNG saved with the historical PIL path and reloaded with cv2.imread
- PNG saved with cv2.imwrite and reloaded with cv2.imread
- RGB-corrected PIL PNG and reloaded with cv2.imread

If the saved/reloaded path scores much better than direct live, the issue is
almost certainly image channel handling or an eval DB built from channel-swapped
captures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image as PILImage

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.labs.face_lab_common import (  # noqa: E402
    add_enrollment_policy_args,
    add_profile_args,
    build_enrollment_policy,
    build_face_service,
    configure_logging,
    json_print,
    summarize_face,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one live face frame and compare direct-agent embeddings "
            "against saved/reloaded image variants."
        )
    )
    add_profile_args(parser)
    add_enrollment_policy_args(parser)
    parser.add_argument(
        "--target",
        default="",
        help="Optional person_id or case-insensitive name to highlight in scores.",
    )
    parser.add_argument(
        "--face-index",
        type=int,
        default=None,
        help=(
            "Usable face index to inspect. Defaults to the largest usable face "
            "by bbox area."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="scripts/labs/face_parity",
        help="Where to write frame variants and parity JSON.",
    )
    parser.add_argument(
        "--max-frame-wait-sec",
        type=float,
        default=5.0,
        help="Maximum wall-clock seconds to wait for a live frame.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of top DB scores to print for each variant.",
    )
    return parser


def _embedding_similarity(left: Any, right: Any) -> float:
    left_vec = np.asarray(left, dtype=np.float32).reshape(-1)
    right_vec = np.asarray(right, dtype=np.float32).reshape(-1)
    left_norm = float(np.linalg.norm(left_vec))
    right_norm = float(np.linalg.norm(right_vec))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    return float(np.dot(left_vec / left_norm, right_vec / right_norm))


def _bbox_area(face: dict[str, Any]) -> int:
    bbox = face.get("bbox") or {}
    return int(bbox.get("w", 0) or 0) * int(bbox.get("h", 0) or 0)


def _select_face(faces: list[dict[str, Any]], face_index: int | None) -> tuple[int, dict[str, Any]]:
    if not faces:
        raise ValueError("no usable faces")
    if face_index is not None:
        if face_index < 0 or face_index >= len(faces):
            raise ValueError(f"face-index {face_index} out of range for {len(faces)} face(s)")
        return face_index, faces[face_index]
    selected_index = max(range(len(faces)), key=lambda index: _bbox_area(faces[index]))
    return selected_index, faces[selected_index]


def _resolve_target_person_id(service: Any, target: str) -> str:
    rendered = str(target or "").strip()
    if not rendered:
        return ""
    people = service.db.list_all_people()
    for person in people:
        if rendered == str(person.get("person_id") or ""):
            return rendered
    lowered = rendered.casefold()
    matches = [
        str(person.get("person_id") or "")
        for person in people
        if str(person.get("name") or "").casefold() == lowered
    ]
    if len(matches) == 1:
        return matches[0]
    partial = [
        str(person.get("person_id") or "")
        for person in people
        if lowered in str(person.get("name") or "").casefold()
    ]
    if len(partial) == 1:
        return partial[0]
    return rendered


def _score_embedding(service: Any, embedding: np.ndarray, *, top_k: int) -> list[dict[str, Any]]:
    count = int(service.db.collection.count())
    if count <= 0:
        return []
    matches = service.db.recognize_face(
        face_embedding=embedding,
        threshold=-1.0,
        top_k=min(max(1, int(top_k)), count),
    )
    return [
        {
            "rank": index + 1,
            "person_id": str(match.get("person_id") or ""),
            "name": str(match.get("name") or ""),
            "similarity": round(float(match.get("similarity", 0.0) or 0.0), 4),
        }
        for index, match in enumerate(matches)
    ]


def _target_similarity(service: Any, embedding: np.ndarray, person_id: str) -> float | None:
    rendered = str(person_id or "").strip()
    if not rendered:
        return None
    record = service.db.get_person(rendered)
    if record is None or record.get("embedding") is None:
        return None
    return round(_embedding_similarity(embedding, record["embedding"]), 4)


def _variant_summary(
    service: Any,
    *,
    label: str,
    image: np.ndarray,
    reference_embedding: np.ndarray,
    face_bbox: dict[str, Any],
    target_person_id: str,
    top_k: int,
) -> dict[str, Any]:
    detection = dict(face_bbox)
    embedding = service.extract_face_embedding(image, detection)
    payload: dict[str, Any] = {
        "label": label,
        "image_shape": list(getattr(image, "shape", ())),
        "embedding_available": embedding is not None,
    }
    if embedding is None:
        return payload
    payload["similarity_to_direct_live_embedding"] = round(
        _embedding_similarity(reference_embedding, embedding),
        4,
    )
    target_score = _target_similarity(service, embedding, target_person_id)
    if target_score is not None:
        payload["target_similarity"] = target_score
    payload["top_scores"] = _score_embedding(service, embedding, top_k=top_k)
    try:
        prepared = service._prepare_faces_for_recognition_result(
            image,
            None,
            min_face_area=service._recognition_min_face_area(),
        )
        payload["redetected"] = {
            "reason": prepared.reason,
            "detected_count": prepared.detected_count,
            "rejected_count": prepared.rejected_count,
            "usable_face_count": len(prepared.faces),
        }
        if prepared.faces:
            redetected_index, redetected_face = _select_face(prepared.faces, None)
            redetected_embedding = np.asarray(redetected_face["embedding"], dtype=np.float32)
            redetected_payload: dict[str, Any] = {
                "selected_face_index": redetected_index,
                "selected_face": summarize_face(redetected_face, include_embedding=True),
                "similarity_to_direct_live_embedding": round(
                    _embedding_similarity(reference_embedding, redetected_embedding),
                    4,
                ),
                "top_scores": _score_embedding(
                    service,
                    redetected_embedding,
                    top_k=top_k,
                ),
            }
            redetected_target_score = _target_similarity(
                service,
                redetected_embedding,
                target_person_id,
            )
            if redetected_target_score is not None:
                redetected_payload["target_similarity"] = redetected_target_score
            payload["redetected"].update(redetected_payload)
    except Exception as exc:
        payload["redetected"] = {
            "failure_reason": "redetect_failed",
            "error": str(exc),
        }
    return payload


def _write_pil(path: Path, image: np.ndarray) -> None:
    arr = np.asarray(image)
    if arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    else:
        arr = np.ascontiguousarray(arr)
    PILImage.fromarray(arr).save(path)


def _capture_once(service: Any, camera_resource_id: str, timeout: float, max_wait: float):
    started_at = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        image, depth_m = service._capture_for_recognition(camera_resource_id, timeout=timeout)
        if image is not None:
            return image, depth_m, attempts, round(time.monotonic() - started_at, 3)
        if max_wait > 0.0 and (time.monotonic() - started_at) >= max_wait:
            raise TimeoutError("no live frame captured before max-frame-wait-sec")
        time.sleep(0.02)


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
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        image, depth_m, attempts, wait_s = _capture_once(
            service,
            config["camera_resource_id"],
            timeout,
            max(0.0, float(args.max_frame_wait_sec)),
        )
        prepared = service._prepare_faces_for_recognition_result(image, depth_m)
        if not prepared.faces:
            payload = {
                "success": False,
                "failure_reason": prepared.reason or "no_usable_faces",
                "detected_count": prepared.detected_count,
                "rejected_count": prepared.rejected_count,
                "rejection_details": prepared.rejection_details,
                "image_shape": list(getattr(image, "shape", ())),
            }
            json_print(payload)
            return 2

        selected_index, face = _select_face(prepared.faces, args.face_index)
        direct_embedding = np.asarray(face["embedding"], dtype=np.float32)
        target_person_id = _resolve_target_person_id(service, args.target)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = f"{timestamp}_face{selected_index}"

        raw_pil_path = output_dir / f"{stem}_raw_array_pil.png"
        cv2_path = output_dir / f"{stem}_cv2_imwrite.png"
        rgb_pil_path = output_dir / f"{stem}_bgr_to_rgb_pil.png"
        json_path = output_dir / f"{stem}_parity.json"

        _write_pil(raw_pil_path, image)
        cv2.imwrite(str(cv2_path), image)
        _write_pil(rgb_pil_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        variants = []
        for label, path in (
            ("direct_live", None),
            ("raw_array_saved_with_pil_then_cv2_imread", raw_pil_path),
            ("cv2_imwrite_then_cv2_imread", cv2_path),
            ("bgr_to_rgb_then_pil_save_then_cv2_imread", rgb_pil_path),
        ):
            if path is None:
                variant_payload: dict[str, Any] = {
                    "label": label,
                    "image_shape": list(getattr(image, "shape", ())),
                    "embedding_available": True,
                    "similarity_to_direct_live_embedding": 1.0,
                }
                target_score = _target_similarity(service, direct_embedding, target_person_id)
                if target_score is not None:
                    variant_payload["target_similarity"] = target_score
                variant_payload["top_scores"] = _score_embedding(
                    service,
                    direct_embedding,
                    top_k=args.top_k,
                )
            else:
                reloaded = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if reloaded is None:
                    variant_payload = {
                        "label": label,
                        "path": str(path),
                        "embedding_available": False,
                        "failure_reason": "cv2_imread_failed",
                    }
                else:
                    variant_payload = _variant_summary(
                        service,
                        label=label,
                        image=reloaded,
                        reference_embedding=direct_embedding,
                        face_bbox=face,
                        target_person_id=target_person_id,
                        top_k=args.top_k,
                    )
                    variant_payload["path"] = str(path)
            variants.append(variant_payload)

        payload = {
            "success": True,
            "profile": config["profile_name"],
            "camera_resource_id": config["camera_resource_id"],
            "db_path": config["db_path"],
            "identity_db_path": config["identity_db_path"],
            "recognition_threshold": config["recognition_threshold"],
            "recognition_margin_threshold": config["recognition_margin_threshold"],
            "target": args.target,
            "target_person_id": target_person_id,
            "capture": {
                "attempts": attempts,
                "wait_s": wait_s,
                "image_shape": list(getattr(image, "shape", ())),
                "dtype": str(getattr(image, "dtype", "")),
                "depth_enabled": depth_m is not None,
            },
            "preparation": {
                "reason": prepared.reason,
                "detected_count": prepared.detected_count,
                "rejected_count": prepared.rejected_count,
                "usable_face_count": len(prepared.faces),
                "rejection_details": prepared.rejection_details,
            },
            "selected_face_index": selected_index,
            "selected_face": summarize_face(face, include_embedding=True),
            "variants": variants,
            "output_json": str(json_path),
        }
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        json_print(payload)
        return 0
    finally:
        service.shutdown()
        robot_client = config.get("robot_client")
        if robot_client is not None:
            robot_client.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
