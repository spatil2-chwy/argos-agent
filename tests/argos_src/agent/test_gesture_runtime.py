from dataclasses import replace
import time
from types import SimpleNamespace

from argos_src.agent.gesture_runtime import (
    GESTURE_PRESETS,
    GESTURE_STATE_IDLE,
    GESTURE_STATE_LISTENING,
    GestureRuntime,
    resolve_gesture_preset_name,
)


LISTENING_PARAMETER = {"x": 0.0, "y": -0.22, "z": 0.0}
IDLE_LEFT_PARAMETER = {"x": -0.45, "y": 0.0, "z": 0.0}
IDLE_RIGHT_PARAMETER = {"x": 0.45, "y": 0.0, "z": 0.0}
IDLE_PARAMETERS = [IDLE_LEFT_PARAMETER, IDLE_RIGHT_PARAMETER]
STATE_PARAMETERS = [LISTENING_PARAMETER, IDLE_LEFT_PARAMETER, IDLE_RIGHT_PARAMETER]
GO2_BALANCE_STAND_API_ID = 1004
GO2_ENTER_POSE_MODE_API_ID = 1003
GO2_ENABLE_POSE_API_ID = 1028
GO2_STATE_POSE_API_ID = 1007


class _FakeConnector:
    def __init__(self):
        self.messages = []

    def perform_go2_action(self, *, api_id, parameter=None, priority=0):
        self.messages.append(
            {
                "api_id": int(api_id),
                "parameter": dict(parameter or {}),
                "priority": int(priority),
            }
        )


class _FakeEngagement:
    def __init__(self, state="idle"):
        self.state = state

    def snapshot(self):
        return SimpleNamespace(state=self.state)


def _wait_for(predicate, *, timeout=1.5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_resolve_auto_gesture_preset_for_go2():
    resolved = resolve_gesture_preset_name(
        robot_family="unitree_go2",
        preset="auto",
    )

    assert resolved == "go2_pose_v1"


def test_resolve_robot_family_gesture_preset_alias_for_go2():
    resolved = resolve_gesture_preset_name(
        robot_family="unitree_go2",
        preset="unitree_go2",
    )

    assert resolved == "go2_pose_v1"


def test_idle_pose_enters_mode_and_publishes_when_engagement_is_idle():
    connector = _FakeConnector()
    engagement = _FakeEngagement(state="idle")
    runtime = GestureRuntime(
        connector=connector,
        engagement=engagement,
        preset_name="go2_pose_v1",
    )
    try:
        assert _wait_for(
            lambda: any(msg["api_id"] == 1003 for msg in connector.messages)
        )
        assert _wait_for(
            lambda: any(
                msg["api_id"] == 1007
                and msg["parameter"] in IDLE_PARAMETERS
                for msg in connector.messages
            )
        )
    finally:
        runtime.shutdown()


def test_idle_pose_sends_balance_stand_before_pose_commands():
    connector = _FakeConnector()
    engagement = _FakeEngagement(state="idle")
    runtime = GestureRuntime(
        connector=connector,
        engagement=engagement,
        preset_name="go2_pose_v1",
    )
    try:
        assert _wait_for(
            lambda: len(connector.messages) >= 4
            and [
                msg["api_id"]
                for msg in connector.messages[:4]
            ]
            == [
                GO2_BALANCE_STAND_API_ID,
                GO2_ENTER_POSE_MODE_API_ID,
                GO2_ENABLE_POSE_API_ID,
                GO2_STATE_POSE_API_ID,
            ]
        )
    finally:
        runtime.shutdown()


def test_idle_pose_alternates_every_interval(monkeypatch):
    preset = GESTURE_PRESETS["go2_pose_v1"]
    fast_idle_state = replace(preset.states[GESTURE_STATE_IDLE], interval_sec=0.05)
    monkeypatch.setitem(
        GESTURE_PRESETS,
        "go2_pose_v1",
        replace(
            preset,
            states={**preset.states, GESTURE_STATE_IDLE: fast_idle_state},
        ),
    )

    connector = _FakeConnector()
    engagement = _FakeEngagement(state="idle")
    runtime = GestureRuntime(
        connector=connector,
        engagement=engagement,
        preset_name="go2_pose_v1",
    )
    try:
        assert _wait_for(
            lambda: any(
                msg["api_id"] == 1007
                and msg["parameter"] == IDLE_LEFT_PARAMETER
                for msg in connector.messages
            )
        )
        assert _wait_for(
            lambda: any(
                msg["api_id"] == 1007
                and msg["parameter"] == IDLE_RIGHT_PARAMETER
                for msg in connector.messages
            )
        )
    finally:
        runtime.shutdown()


def test_listening_pose_takes_precedence_while_recording_active():
    connector = _FakeConnector()
    engagement = _FakeEngagement(state="idle")
    runtime = GestureRuntime(
        connector=connector,
        engagement=engagement,
        preset_name="go2_pose_v1",
    )
    try:
        assert _wait_for(
            lambda: any(
                msg["api_id"] == 1007
                and msg["parameter"] in IDLE_PARAMETERS
                for msg in connector.messages
            )
        )

        before_count = len(connector.messages)
        runtime.set_recording_active(True)
        assert _wait_for(
            lambda: any(
                msg["api_id"] == 1007
                and msg["parameter"] == LISTENING_PARAMETER
                for msg in connector.messages[before_count:]
            )
        )
    finally:
        runtime.shutdown()


def test_listening_pose_is_skipped_when_nodding_state_disabled():
    connector = _FakeConnector()
    engagement = _FakeEngagement(state="idle")
    runtime = GestureRuntime(
        connector=connector,
        engagement=engagement,
        preset_name="go2_pose_v1",
        enabled_states=(GESTURE_STATE_IDLE,),
    )
    try:
        assert _wait_for(
            lambda: any(
                msg["api_id"] == 1007
                and msg["parameter"] in IDLE_PARAMETERS
                for msg in connector.messages
            )
        )

        before_count = len(connector.messages)
        runtime.set_recording_active(True)
        time.sleep(0.2)

        assert not any(
            msg["api_id"] == 1007
            and msg["parameter"] == LISTENING_PARAMETER
            for msg in connector.messages[before_count:]
        )
    finally:
        runtime.shutdown()


def test_idle_pose_is_skipped_when_tilt_state_disabled():
    connector = _FakeConnector()
    engagement = _FakeEngagement(state="idle")
    runtime = GestureRuntime(
        connector=connector,
        engagement=engagement,
        preset_name="go2_pose_v1",
        enabled_states=(GESTURE_STATE_LISTENING,),
    )
    try:
        time.sleep(0.2)

        assert not any(
            msg["api_id"] == 1007
            and msg["parameter"] in IDLE_PARAMETERS
            for msg in connector.messages
        )
    finally:
        runtime.shutdown()


def test_recording_stop_returns_to_idle_only_when_engagement_is_idle():
    connector = _FakeConnector()
    engagement = _FakeEngagement(state="idle")
    runtime = GestureRuntime(
        connector=connector,
        engagement=engagement,
        preset_name="go2_pose_v1",
    )
    try:
        runtime.set_recording_active(True)
        assert _wait_for(
            lambda: any(
                msg["api_id"] == 1007
                and msg["parameter"] == LISTENING_PARAMETER
                for msg in connector.messages
            )
        )

        before_count = len(connector.messages)
        runtime.set_recording_active(False)
        assert _wait_for(
            lambda: any(
                msg["api_id"] == 1007
                and msg["parameter"] in IDLE_PARAMETERS
                for msg in connector.messages[before_count:]
            )
        )
    finally:
        runtime.shutdown()


def test_non_idle_non_recording_state_exits_pose_mode():
    connector = _FakeConnector()
    engagement = _FakeEngagement(state="idle")
    runtime = GestureRuntime(
        connector=connector,
        engagement=engagement,
        preset_name="go2_pose_v1",
    )
    try:
        assert _wait_for(
            lambda: any(
                msg["api_id"] == 1007
                and msg["parameter"] in IDLE_PARAMETERS
                for msg in connector.messages
            )
        )

        before_count = len(connector.messages)
        engagement.state = "engaged"
        assert _wait_for(
            lambda: len(connector.messages) >= before_count + 2
            and connector.messages[-2]["api_id"] == 1028
            and connector.messages[-2]["parameter"] == {"data": False}
            and connector.messages[-1]["api_id"] == 1002
        )
    finally:
        runtime.shutdown()


def test_shutdown_sends_best_effort_exit_pose_commands():
    connector = _FakeConnector()
    engagement = _FakeEngagement(state="idle")
    runtime = GestureRuntime(
        connector=connector,
        engagement=engagement,
        preset_name="go2_pose_v1",
    )
    assert _wait_for(
        lambda: any(
            msg["api_id"] == 1007
            and msg["parameter"] in IDLE_PARAMETERS
            for msg in connector.messages
        )
    )

    runtime.shutdown()

    assert connector.messages[-2]["api_id"] == 1028
    assert connector.messages[-2]["parameter"] == {"data": False}
    assert connector.messages[-1]["api_id"] == 1002


def test_runtime_uses_idle_and_listening_states_only():
    connector = _FakeConnector()
    engagement = _FakeEngagement(state="idle")
    runtime = GestureRuntime(
        connector=connector,
        engagement=engagement,
        preset_name="go2_pose_v1",
    )
    try:
        runtime.set_recording_active(True)
        assert _wait_for(
            lambda: any(
                msg["api_id"] == 1007
                and msg["parameter"] == LISTENING_PARAMETER
                for msg in connector.messages
            )
        )
        runtime.set_recording_active(False)
        assert _wait_for(
            lambda: any(
                msg["api_id"] == 1007
                and msg["parameter"] in IDLE_PARAMETERS
                for msg in connector.messages
            )
        )

        state_messages = [
            msg["parameter"]
            for msg in connector.messages
            if msg["api_id"] == 1007
        ]
        assert state_messages
        assert all(
            parameter in STATE_PARAMETERS
            for parameter in state_messages
        )
        assert GESTURE_STATE_IDLE == "idle"
        assert GESTURE_STATE_LISTENING == "listening"
    finally:
        runtime.shutdown()
