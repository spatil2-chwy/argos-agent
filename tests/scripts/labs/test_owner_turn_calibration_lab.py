from __future__ import annotations

import math
import types

import pytest

from argos_src.agent.owner_turn import OwnerTurnSettings
from argos_src.provider_api.models import CameraIntrinsics
from scripts.labs.owner_turn_calibration_lab import (
    _calibration_target,
    _face_payloads,
    _offset_hint,
    _turn_plan,
)


def test_turn_plan_applies_deadband_gain_and_clamp() -> None:
    settings = OwnerTurnSettings(
        enabled=True,
        deadband_deg=3.0,
        turn_gain=0.8,
        max_turn_deg=10.0,
    )

    assert _turn_plan(math.radians(2.0), settings)["status"] == "skip"

    plan = _turn_plan(math.radians(-20.0), settings)

    assert plan["status"] == "turn"
    assert plan["direction"] == "right"
    assert plan["command_deg"] == pytest.approx(-10.0)


def test_face_payload_uses_camera_yaw_offset_for_bearing() -> None:
    service = types.SimpleNamespace(
        _camera_yaw_offset_rad=math.radians(5.0),
        _get_camera_intrinsics=lambda: CameraIntrinsics(
            fx=100.0,
            fy=100.0,
            cx=50.0,
            cy=50.0,
            width=100,
            height=100,
        ),
    )

    payload = _face_payloads(
        service=service,
        faces=[{"bbox": {"x": 40, "y": 10, "w": 20, "h": 20}}],
        image_shape=(100, 100, 3),
        settings=OwnerTurnSettings(enabled=True),
    )[0]

    assert payload["bearing_deg"] == pytest.approx(5.0)
    assert payload["turn_plan"]["command_deg"] == pytest.approx(5.0)


def test_offset_hint_subtracts_current_bearing_from_offset() -> None:
    service = types.SimpleNamespace(_camera_yaw_offset_rad=math.radians(5.0))
    payload = {"calibration_target": {"bearing_deg": 2.0}}

    hint = _offset_hint(payload, service)

    assert hint is not None
    assert hint["suggested_camera_yaw_offset_deg"] == pytest.approx(3.0)


def test_calibration_target_falls_back_to_single_detected_face() -> None:
    target = _calibration_target(
        {
            "selected_target": None,
            "faces": [
                {
                    "index": 0,
                    "bearing_deg": -4.0,
                    "turn_plan": {"status": "turn", "direction": "right"},
                }
            ],
        }
    )

    assert target is not None
    assert target["kind"] == "detected_face"
    assert target["bearing_deg"] == pytest.approx(-4.0)


def test_calibration_target_rejects_multiple_detected_faces() -> None:
    assert (
        _calibration_target(
            {
                "selected_target": None,
                "faces": [
                    {"index": 0, "bearing_deg": -4.0},
                    {"index": 1, "bearing_deg": 3.0},
                ],
            }
        )
        is None
    )
