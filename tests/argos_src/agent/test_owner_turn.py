from __future__ import annotations

import types
import math

import pytest

from argos_src.agent.owner_turn import (
    CMD_VEL_TOPIC,
    OwnerTurnController,
    OwnerTurnRequest,
    OwnerTurnSettings,
)
from argos_src.robot_api.motion import motion_lock_for_topic


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += max(0.0, float(duration))


class _FakeConnector:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def publish_velocity(
        self,
        *,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        angular_z: float = 0.0,
    ) -> None:
        self.messages.append(
            {
                "linear_x": linear_x,
                "linear_y": linear_y,
                "angular_z": angular_z,
            }
        )


class _ClosedLoopConnector(_FakeConnector):
    def __init__(self) -> None:
        super().__init__()
        self.yaw = 0.0
        self.last_angular_z = 0.0

    def publish_velocity(
        self,
        *,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        angular_z: float = 0.0,
    ) -> None:
        super().publish_velocity(
            linear_x=linear_x,
            linear_y=linear_y,
            angular_z=angular_z,
        )
        self.last_angular_z = float(angular_z)

    def advance(self, duration: float) -> None:
        self.yaw += self.last_angular_z * max(0.0, float(duration))

    def get_transform(self, target_frame: str, source_frame: str):
        assert target_frame == "odom"
        assert source_frame == "base_link"
        return types.SimpleNamespace(
            transform=types.SimpleNamespace(
                rotation=types.SimpleNamespace(
                    x=0.0,
                    y=0.0,
                    z=math.sin(self.yaw / 2.0),
                    w=math.cos(self.yaw / 2.0),
                )
            )
        )


def _make_controller(*, bearing_rad: float, recording_active=False, nav_active=False):
    clock = _FakeClock()
    connector = _FakeConnector()
    face_service = types.SimpleNamespace(
        get_face_turn_target=lambda _person_id: types.SimpleNamespace(
            person_id="person-1",
            name="Alex",
            bearing_rad=bearing_rad,
            timestamp=100.0,
            confidence=0.9,
            depth_m=1.0,
        )
    )
    nav_state = types.SimpleNamespace(
        get_active_goal=lambda: {"goal_id": "nav"} if nav_active else None
    )
    controller = OwnerTurnController(
        connector=connector,
        face_service=face_service,
        nav_state=nav_state,
        recording_state_provider=lambda: bool(recording_active),
        settings=OwnerTurnSettings(
            enabled=True,
            deadband_deg=3.0,
            turn_gain=1.0,
            max_turn_deg=25.0,
            angular_speed_rad_s=0.8,
            command_hz=50.0,
            delay_after_recording_sec=0.0,
        ),
        time_fn=clock.time,
        sleep_fn=clock.sleep,
    )
    return controller, connector


def test_owner_turn_skips_when_transform_unavailable():
    controller, connector = _make_controller(bearing_rad=-0.10)
    try:
        controller._execute_request(OwnerTurnRequest(person_id="person-1", req_id="rt-1"))
    finally:
        controller.shutdown()

    assert connector.messages == []


def test_owner_turn_skips_inside_deadband():
    controller, connector = _make_controller(bearing_rad=0.02)
    try:
        controller._execute_request(OwnerTurnRequest(person_id="person-1", req_id="rt-1"))
    finally:
        controller.shutdown()

    assert connector.messages == []


def test_owner_turn_closed_loop_stops_near_target_yaw():
    clock = _FakeClock()
    connector = _ClosedLoopConnector()
    face_service = types.SimpleNamespace(
        get_face_turn_target=lambda _person_id: types.SimpleNamespace(
            person_id="person-1",
            name="Alex",
            bearing_rad=-0.10,
            timestamp=100.0,
            confidence=0.9,
            depth_m=1.0,
        )
    )

    def sleep(duration: float) -> None:
        connector.advance(duration)
        clock.sleep(duration)

    controller = OwnerTurnController(
        connector=connector,
        face_service=face_service,
        nav_state=None,
        recording_state_provider=lambda: False,
        settings=OwnerTurnSettings(
            enabled=True,
            deadband_deg=3.0,
            turn_gain=1.0,
            max_turn_deg=25.0,
            angular_speed_rad_s=0.8,
            command_hz=50.0,
            delay_after_recording_sec=0.0,
            yaw_tolerance_deg=1.5,
            max_duration_sec=1.0,
            slow_zone_deg=8.0,
            min_angular_speed_rad_s=0.25,
        ),
        time_fn=clock.time,
        sleep_fn=sleep,
    )
    try:
        controller._execute_request(OwnerTurnRequest(person_id="person-1", req_id="rt-1"))
    finally:
        controller.shutdown()

    assert connector.messages
    assert any(message["angular_z"] < 0.0 for message in connector.messages)
    assert connector.messages[-1]["angular_z"] == 0.0
    assert connector.yaw == pytest.approx(-0.10, abs=math.radians(1.5))


def test_owner_turn_releases_motion_lock_when_stop_publish_fails(monkeypatch):
    clock = _FakeClock()
    connector = _ClosedLoopConnector()
    face_service = types.SimpleNamespace(
        get_face_turn_target=lambda _person_id: types.SimpleNamespace(
            person_id="person-1",
            name="Alex",
            bearing_rad=-0.10,
            timestamp=100.0,
            confidence=0.9,
            depth_m=1.0,
        )
    )

    def sleep(duration: float) -> None:
        connector.advance(duration)
        clock.sleep(duration)

    controller = OwnerTurnController(
        connector=connector,
        face_service=face_service,
        nav_state=None,
        recording_state_provider=lambda: False,
        settings=OwnerTurnSettings(
            enabled=True,
            deadband_deg=3.0,
            turn_gain=1.0,
            max_turn_deg=25.0,
            angular_speed_rad_s=0.8,
            command_hz=50.0,
            delay_after_recording_sec=0.0,
            yaw_tolerance_deg=1.5,
            max_duration_sec=1.0,
            slow_zone_deg=8.0,
            min_angular_speed_rad_s=0.25,
        ),
        time_fn=clock.time,
        sleep_fn=sleep,
    )
    monkeypatch.setattr(
        controller,
        "_publish_stop",
        lambda: (_ for _ in ()).throw(RuntimeError("stop failed")),
    )

    try:
        controller._execute_request(OwnerTurnRequest(person_id="person-1", req_id="rt-1"))
        lock = motion_lock_for_topic(CMD_VEL_TOPIC)
        acquired = lock.acquire(blocking=False)
        try:
            assert acquired is True
        finally:
            if acquired:
                lock.release()
        assert controller._motion_lock_acquired is False
    finally:
        controller.shutdown()


def test_owner_turn_skips_when_recording_restarted():
    controller, connector = _make_controller(bearing_rad=-0.10, recording_active=True)
    try:
        controller._execute_request(OwnerTurnRequest(person_id="person-1", req_id="rt-1"))
    finally:
        controller.shutdown()

    assert connector.messages == []


def test_owner_turn_skips_when_navigation_active():
    controller, connector = _make_controller(bearing_rad=-0.10, nav_active=True)
    try:
        controller._execute_request(OwnerTurnRequest(person_id="person-1", req_id="rt-1"))
    finally:
        controller.shutdown()

    assert connector.messages == []
