#!/usr/bin/env python3
"""Interactive owner-turn calibration lab.

Run from the repo root:

    source setup_shell.sh
    poetry run python -m scripts.labs.owner_turn_calibration_lab

Dry-run is the default. Add --move to send the same closed-loop owner turn
command used by the runtime after each Enter-triggered capture.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from argos_src.agent.owner_turn import (  # noqa: E402
    OwnerTurnController,
    OwnerTurnRequest,
    OwnerTurnSettings,
)
from argos_src.face_recognition.bearing import (  # noqa: E402
    estimate_robot_yaw_error_rad,
    face_center_px,
)
from argos_src.face_recognition.models import PersonContext  # noqa: E402
from argos_src.profile_config import load_scenario_profile  # noqa: E402
from scripts.labs.face_lab_common import (  # noqa: E402
    add_enrollment_policy_args,
    add_profile_args,
    build_enrollment_policy,
    build_face_service,
    configure_logging,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Press Enter to capture the face camera, compute owner-turn bearing, "
            "and optionally command the robot to center the recognized owner."
        )
    )
    add_profile_args(parser)
    add_enrollment_policy_args(parser)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Capture one sample and exit instead of waiting for repeated Enter presses.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="After each capture, run the production closed-loop owner-turn command.",
    )
    parser.add_argument(
        "--person-id",
        default="",
        help=(
            "Recognized person_id to center. Defaults to the current strict "
            "single recognized face target."
        ),
    )
    parser.add_argument(
        "--camera-yaw-offset-deg",
        type=float,
        default=None,
        help="Override face_recognition.owner_turn.camera_yaw_offset_rad for this run.",
    )
    parser.add_argument("--deadband-deg", type=float, default=None)
    parser.add_argument("--turn-gain", type=float, default=None)
    parser.add_argument("--max-turn-deg", type=float, default=None)
    parser.add_argument("--angular-speed-rad-s", type=float, default=None)
    parser.add_argument("--yaw-tolerance-deg", type=float, default=None)
    parser.add_argument("--max-duration-sec", type=float, default=None)
    parser.add_argument(
        "--max-frame-wait-sec",
        type=float,
        default=0.0,
        help="Maximum seconds to wait for a frame. Default 0 waits indefinitely.",
    )
    return parser


def _settings_from_profile(args: argparse.Namespace) -> OwnerTurnSettings:
    owner_turn = load_scenario_profile(args.profile).face_recognition.owner_turn
    settings = OwnerTurnSettings(
        enabled=True,
        deadband_deg=owner_turn.deadband_deg,
        turn_gain=owner_turn.turn_gain,
        max_turn_deg=owner_turn.max_turn_deg,
        angular_speed_rad_s=owner_turn.angular_speed_rad_s,
        command_hz=owner_turn.command_hz,
        delay_after_recording_sec=0.0,
        odom_frame=owner_turn.odom_frame,
        robot_frame=owner_turn.robot_frame,
        yaw_tolerance_deg=owner_turn.yaw_tolerance_deg,
        max_duration_sec=owner_turn.max_duration_sec,
        slow_zone_deg=owner_turn.slow_zone_deg,
        min_angular_speed_rad_s=owner_turn.min_angular_speed_rad_s,
    )
    replacements: dict[str, Any] = {}
    for arg_name, field_name in (
        ("deadband_deg", "deadband_deg"),
        ("turn_gain", "turn_gain"),
        ("max_turn_deg", "max_turn_deg"),
        ("angular_speed_rad_s", "angular_speed_rad_s"),
        ("yaw_tolerance_deg", "yaw_tolerance_deg"),
        ("max_duration_sec", "max_duration_sec"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            replacements[field_name] = value
    return replace(settings, **replacements) if replacements else settings


def _turn_plan(
    bearing_rad: float | None,
    settings: OwnerTurnSettings,
) -> dict[str, Any]:
    if bearing_rad is None:
        return {"status": "skip", "reason": "missing_bearing"}
    bearing = float(bearing_rad)
    deadband_rad = math.radians(float(settings.deadband_deg))
    if abs(bearing) <= deadband_rad:
        return {
            "status": "skip",
            "reason": "inside_deadband",
            "bearing_deg": round(math.degrees(bearing), 3),
            "deadband_deg": round(float(settings.deadband_deg), 3),
        }
    max_turn_rad = math.radians(float(settings.max_turn_deg))
    command_rad = max(
        -max_turn_rad,
        min(max_turn_rad, bearing * float(settings.turn_gain)),
    )
    return {
        "status": "turn",
        "bearing_deg": round(math.degrees(bearing), 3),
        "turn_gain": round(float(settings.turn_gain), 3),
        "command_deg": round(math.degrees(command_rad), 3),
        "direction": "left" if command_rad > 0.0 else "right",
    }


def _capture_frame(
    service: Any,
    *,
    camera_resource_id: str,
    timeout: float,
    max_frame_wait_sec: float,
) -> tuple[Any | None, Any | None, dict[str, Any]]:
    started_at = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        image, depth_m = service._capture_for_recognition(
            camera_resource_id,
            timeout=timeout,
        )
        elapsed = time.monotonic() - started_at
        if image is not None:
            return image, depth_m, {
                "capture_attempts": attempts,
                "capture_wait_s": round(elapsed, 3),
            }
        if max_frame_wait_sec > 0.0 and elapsed >= max_frame_wait_sec:
            return None, None, {
                "capture_attempts": attempts,
                "capture_wait_s": round(elapsed, 3),
                "failure_reason": "capture_failed",
            }
        time.sleep(0.02)


def _attention_payload(face: dict[str, Any]) -> dict[str, Any]:
    attention = face.get("attention")
    if attention is None:
        return {}
    return {
        "attentive": bool(getattr(attention, "attentive", False)),
        "raw_attentive": bool(getattr(attention, "raw_attentive", False)),
        "confidence": round(float(getattr(attention, "confidence", 0.0) or 0.0), 3),
        "reason": str(getattr(attention, "reason", "") or ""),
        "head_yaw_deg": _round_optional(getattr(attention, "yaw_deg", None)),
        "head_pitch_deg": _round_optional(getattr(attention, "pitch_deg", None)),
        "head_roll_deg": _round_optional(getattr(attention, "roll_deg", None)),
    }


def _round_optional(value: Any, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _face_payloads(
    *,
    service: Any,
    faces: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    settings: OwnerTurnSettings,
) -> list[dict[str, Any]]:
    intrinsics = service._get_camera_intrinsics()
    height, width = image_shape[:2]
    payloads: list[dict[str, Any]] = []
    for index, face in enumerate(faces):
        bbox = dict(face.get("bbox") or {})
        center = face_center_px(face)
        bearing_rad = estimate_robot_yaw_error_rad(
            face,
            intrinsics=intrinsics,
            camera_yaw_offset_rad=float(
                getattr(service, "_camera_yaw_offset_rad", 0.0) or 0.0
            ),
        )
        center_offset_ratio = None
        if center is not None and width:
            center_offset_ratio = (float(center[0]) - (float(width) / 2.0)) / (
                float(width) / 2.0
            )
        payload: dict[str, Any] = {
            "index": index,
            "bbox": {
                "x": int(bbox.get("x", 0) or 0),
                "y": int(bbox.get("y", 0) or 0),
                "w": int(bbox.get("w", 0) or 0),
                "h": int(bbox.get("h", 0) or 0),
            },
            "face_center_px": (
                [round(center[0], 1), round(center[1], 1)] if center is not None else None
            ),
            "image_size": [int(width), int(height)],
            "center_offset_ratio": _round_optional(center_offset_ratio),
            "bearing_deg": _round_optional(
                math.degrees(bearing_rad) if bearing_rad is not None else None
            ),
            "depth_m": _round_optional(face.get("depth_m")),
            "depth_valid_samples": face.get("depth_valid_samples"),
            "confidence": _round_optional(face.get("confidence")),
            "recognized_name": str(face.get("recognized_name") or ""),
            "turn_plan": _turn_plan(bearing_rad, settings),
        }
        attention_payload = _attention_payload(face)
        if attention_payload:
            payload["attention"] = attention_payload
        payloads.append(payload)
    return payloads


def _person_payload(person: PersonContext, settings: OwnerTurnSettings) -> dict[str, Any]:
    return {
        "person_id": person.person_id,
        "name": person.name,
        "confidence": round(float(person.confidence), 3),
        "depth_m": _round_optional(person.depth_m),
        "face_center_px": [
            _round_optional(person.face_center_x_px, 1),
            _round_optional(person.face_center_y_px, 1),
        ],
        "bearing_deg": _round_optional(
            math.degrees(person.bearing_rad) if person.bearing_rad is not None else None
        ),
        "attentive": bool(person.attentive),
        "attention_confidence": round(float(person.attention_confidence), 3),
        "turn_plan": _turn_plan(person.bearing_rad, settings),
    }


def _run_one_sample(
    *,
    service: Any,
    config: dict[str, Any],
    settings: OwnerTurnSettings,
    requested_person_id: str,
    timeout: float,
    max_frame_wait_sec: float,
) -> dict[str, Any]:
    image, depth_m, capture_payload = _capture_frame(
        service,
        camera_resource_id=config["camera_resource_id"],
        timeout=timeout,
        max_frame_wait_sec=max_frame_wait_sec,
    )
    payload: dict[str, Any] = {
        "captured_at_unix_s": round(time.time(), 3),
        "camera_resource_id": config["camera_resource_id"],
        "depth_enabled": bool(config["depth_settings"] is not None),
        "capture": capture_payload,
    }
    if image is None:
        payload["success"] = False
        return payload

    prepared = service._prepare_faces_for_recognition_result(
        image,
        depth_m,
        min_face_area=service._recognition_min_face_area(),
    )
    payload["success"] = True
    payload["preparation"] = {
        "reason": prepared.reason,
        "detected_count": prepared.detected_count,
        "rejected_count": prepared.rejected_count,
        "usable_face_count": len(prepared.faces),
    }
    if not prepared.faces:
        payload["faces"] = []
        payload["recognized_people"] = []
        return payload

    now = time.time()
    service._presence_cache.mark_faces_seen(now)
    should_record = service._presence_cache.should_record_interaction
    service._presence_cache.should_record_interaction = lambda *_args, **_kwargs: False
    try:
        persons, unknown_count, current_ids, analysis = service._build_scene_state(
            image=image,
            detected_faces=prepared.faces,
            image_shape=image.shape,
            now=now,
        )
    finally:
        service._presence_cache.should_record_interaction = should_record
    service._presence_cache.expire_inactive(current_ids, now)
    service._presence_cache.update(
        persons=persons,
        faces_detected=len(prepared.faces),
        unknown_count=unknown_count,
        attentive_unknown_count=analysis.attentive_unknown_count,
        attention_target=analysis.attention_target,
        primary_attention_target=analysis.primary_attention_target,
        social_scene=analysis.social_scene,
        now=now,
    )

    payload["faces"] = _face_payloads(
        service=service,
        faces=prepared.faces,
        image_shape=image.shape,
        settings=settings,
    )
    payload["recognized_people"] = [
        _person_payload(person, settings) for person in persons
    ]
    target = service.get_face_turn_target(requested_person_id or None)
    payload["selected_target"] = None
    if target is not None:
        payload["selected_target"] = {
            "person_id": target.person_id,
            "name": target.name,
            "confidence": round(float(target.confidence), 3),
            "depth_m": _round_optional(target.depth_m),
            "bearing_deg": round(math.degrees(float(target.bearing_rad)), 3),
            "turn_plan": _turn_plan(target.bearing_rad, settings),
        }
    payload["centered_person_offset_hint"] = _offset_hint(payload, service)
    return payload


def _offset_hint(payload: dict[str, Any], service: Any) -> dict[str, Any] | None:
    """Suggest an offset only when there is one unambiguous selected target."""
    target = payload.get("selected_target")
    if not target:
        return None
    bearing_deg = target.get("bearing_deg")
    if bearing_deg is None:
        return None
    current_offset_rad = float(getattr(service, "_camera_yaw_offset_rad", 0.0) or 0.0)
    suggested_offset_rad = current_offset_rad - math.radians(float(bearing_deg))
    return {
        "use_only_if_person_is_physically_centered": True,
        "current_camera_yaw_offset_deg": round(math.degrees(current_offset_rad), 3),
        "suggested_camera_yaw_offset_deg": round(
            math.degrees(suggested_offset_rad),
            3,
        ),
    }


def _print_sample(payload: dict[str, Any]) -> None:
    print("")
    if not payload.get("success"):
        print(
            "capture failed "
            f"after {payload.get('capture', {}).get('capture_wait_s', 0.0)}s"
        )
        return
    prep = payload.get("preparation", {})
    print(
        "capture ok: "
        f"usable={prep.get('usable_face_count', 0)} "
        f"detected={prep.get('detected_count', 0)} "
        f"rejected={prep.get('rejected_count', 0)} "
        f"depth={'on' if payload.get('depth_enabled') else 'off'}"
    )
    for face in payload.get("faces", []):
        plan = face.get("turn_plan", {})
        attention = face.get("attention", {})
        print(
            "face "
            f"#{face['index']} center={face.get('face_center_px')} "
            f"bearing={face.get('bearing_deg')}deg "
            f"depth={face.get('depth_m')} "
            f"attention={attention.get('reason', 'n/a')} "
            f"plan={plan.get('status')} "
            f"{plan.get('direction', '')} {plan.get('command_deg', '')}".rstrip()
        )
    for person in payload.get("recognized_people", []):
        plan = person.get("turn_plan", {})
        print(
            "recognized "
            f"{person['name']} ({person['person_id']}) "
            f"bearing={person.get('bearing_deg')}deg "
            f"attentive={person.get('attentive')} "
            f"plan={plan.get('status')} "
            f"{plan.get('direction', '')} {plan.get('command_deg', '')}".rstrip()
        )
    target = payload.get("selected_target")
    if target:
        plan = target.get("turn_plan", {})
        print(
            "selected target: "
            f"{target['name']} ({target['person_id']}) "
            f"bearing={target.get('bearing_deg')}deg "
            f"plan={plan.get('status')} "
            f"{plan.get('direction', '')} {plan.get('command_deg', '')}".rstrip()
        )
    else:
        print("selected target: none")
    hint = payload.get("centered_person_offset_hint")
    if hint:
        print(
            "if that person is physically centered, try "
            "camera_yaw_offset_deg="
            f"{hint['suggested_camera_yaw_offset_deg']}"
        )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    settings = _settings_from_profile(args)
    enrollment_policy = build_enrollment_policy(args)
    service, config = build_face_service(args, enrollment_policy=enrollment_policy)
    if args.camera_yaw_offset_deg is not None:
        service._camera_yaw_offset_rad = math.radians(float(args.camera_yaw_offset_deg))

    depth_settings = config["depth_settings"]
    timeout = (
        depth_settings.capture_timeout_sec
        if depth_settings is not None
        else float(args.capture_timeout_sec or 1.5)
    )
    controller: OwnerTurnController | None = None
    try:
        print(
            "owner turn calibration lab "
            f"profile={config['profile_name']} "
            f"camera={config['camera_resource_id']} "
            f"move={'on' if args.move else 'off'}"
        )
        print(
            "settings "
            f"camera_yaw_offset_deg={math.degrees(service._camera_yaw_offset_rad):.3f} "
            f"deadband_deg={settings.deadband_deg:.3f} "
            f"turn_gain={settings.turn_gain:.3f} "
            f"max_turn_deg={settings.max_turn_deg:.3f}"
        )
        if args.move:
            controller = OwnerTurnController(
                connector=config["robot_client"],
                face_service=service,
                nav_state=None,
                recording_state_provider=lambda: False,
                settings=settings,
            )

        sample_index = 0
        while True:
            if not args.once:
                command = input(
                    "\nPress Enter to capture/center, or q then Enter to quit: "
                )
                if command.strip().lower() in {"q", "quit", "exit"}:
                    return 0
            sample_index += 1
            payload = _run_one_sample(
                service=service,
                config=config,
                settings=settings,
                requested_person_id=str(args.person_id or "").strip(),
                timeout=timeout,
                max_frame_wait_sec=max(0.0, float(args.max_frame_wait_sec)),
            )
            payload["sample_index"] = sample_index
            _print_sample(payload)
            target = payload.get("selected_target")
            if args.move and target and controller is not None:
                controller._execute_request(
                    OwnerTurnRequest(
                        person_id=str(target["person_id"]),
                        req_id=f"owner-turn-lab-{sample_index}",
                        owner_source="lab",
                    )
                )
                print("move attempt finished")
            elif args.move:
                print("move skipped: no recognized selected target")
            if args.once:
                return 0 if payload.get("success") else 2
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    finally:
        if controller is not None:
            controller.shutdown()
        service.shutdown()
        robot_client = config.get("robot_client") if "config" in locals() else None
        if robot_client is not None:
            robot_client.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
