from __future__ import annotations

from contextlib import nullcontext
import json
import math
import time
from pathlib import Path

import pytest

from argos_src.agent.control.tool_runtime import ToolRuntime
from argos_src.nav_support.locations import LocationStore, NavigationState
from argos_src.nav_support.timeouts import (
    DOCK_ALIGNMENT_TIMEOUT_SEC,
    charging_tool_timeout_sec,
    estimate_navigation_timeout_sec,
)
from argos_src.provider_api.fake import FakeProviderClient
from argos_src.provider_api.models import BatterySnapshot, RobotTransform
from argos_src.runtime.battery_state import BatteryStateCache
from argos_src.tools.execution import tool_execution_context
from argos_src.tools.unitree_go2.navigation.toolset import (
    CancelNavigationTool,
    ChargingDockTool,
    NavigateToLocationBlockingTool,
    NavigateToReturnPointBlockingTool,
    process_navigation_event,
)


class _PoseRobot(FakeProviderClient):
    def __init__(
        self,
        *,
        x: float = 0.0,
        y: float = 0.0,
        stamp_s: float = 0.0,
        transform_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self._x = x
        self._y = y
        self._stamp_s = stamp_s
        self._transform_error = transform_error
        self.canceled_goal_ids: list[str] = []

    def get_transform(
        self,
        parent_frame: str,
        child_frame: str,
        timeout: float = 0.05,
    ) -> RobotTransform:
        del parent_frame, child_frame, timeout
        if self._transform_error is not None:
            raise self._transform_error
        return RobotTransform(
            translation=(self._x, self._y, 0.0),
            stamp_s=self._stamp_s,
        )

    def cancel_navigation(self, *, goal_id: str | None = None) -> dict:
        self.canceled_goal_ids.append(str(goal_id or ""))
        return super().cancel_navigation(goal_id=goal_id)


class _NavigationTimeoutRobot(_PoseRobot):
    def navigate_to_pose(self, **kwargs) -> dict:
        self.navigation_attempts = getattr(self, "navigation_attempts", 0) + 1
        self.timed_out_goal_id = str(kwargs["goal_id"])
        raise TimeoutError("navigation timed out")


class _UncancelableNavigationTimeoutRobot(_NavigationTimeoutRobot):
    def cancel_navigation(self, *, goal_id: str | None = None) -> dict:
        self.canceled_goal_ids.append(str(goal_id or ""))
        raise TimeoutError("cancel timed out")


class _AmbiguousCancelNavigationTimeoutRobot(_NavigationTimeoutRobot):
    def cancel_navigation(self, *, goal_id: str | None = None) -> dict:
        self.canceled_goal_ids.append(str(goal_id or ""))
        return {}


class _NonDictCancelNavigationTimeoutRobot(_NavigationTimeoutRobot):
    def cancel_navigation(self, *, goal_id: str | None = None):
        self.canceled_goal_ids.append(str(goal_id or ""))
        return True


class _RetryableCancelNavigationTimeoutRobot(_NavigationTimeoutRobot):
    def cancel_navigation(self, *, goal_id: str | None = None) -> dict:
        self.canceled_goal_ids.append(str(goal_id or ""))
        if len(self.canceled_goal_ids) == 1:
            return {}
        return {"canceled": True}


def _state(tmp_path: Path) -> NavigationState:
    return NavigationState(LocationStore(tmp_path / "locations.json"))


@pytest.mark.parametrize(
    ("distance_m", "expected_timeout_sec"),
    [
        (0.0, 30.0),
        (1.0, 40.0),
        (5.0, 80.0),
        (27.0, 300.0),
        (100.0, 1030.0),
    ],
)
def test_estimate_navigation_timeout_is_distance_derived(
    distance_m: float,
    expected_timeout_sec: float,
) -> None:
    assert estimate_navigation_timeout_sec(distance_m) == expected_timeout_sec


@pytest.mark.parametrize("distance_m", [-1.0, math.inf, -math.inf, math.nan])
def test_estimate_navigation_timeout_rejects_invalid_distance(
    distance_m: float,
) -> None:
    with pytest.raises(ValueError):
        estimate_navigation_timeout_sec(distance_m)


def test_saved_location_uses_distance_based_timeout(tmp_path: Path) -> None:
    robot = _PoseRobot(x=1.0, y=2.0)
    state = _state(tmp_path)
    state.location_store.set("destination", {"x": 4.0, "y": 6.0, "theta": 0.0})

    result = json.loads(
        NavigateToLocationBlockingTool(robot_client=robot, state=state)._run(
            "destination"
        )
    )

    assert result["success"] is True
    assert robot.navigation_goals[-1]["timeout_sec"] == 80.0


def test_return_point_uses_distance_based_timeout(tmp_path: Path) -> None:
    robot = _PoseRobot(x=3.0, y=4.0)
    state = _state(tmp_path)
    state.set_return_point("start", {"x": 0.0, "y": 0.0, "theta": 0.0})

    result = json.loads(
        NavigateToReturnPointBlockingTool(robot_client=robot, state=state)._run("start")
    )

    assert result["success"] is True
    assert robot.navigation_goals[-1]["timeout_sec"] == 80.0


def test_blocking_navigation_fails_closed_when_pose_lookup_fails(tmp_path: Path) -> None:
    robot = _PoseRobot(transform_error=RuntimeError("transform unavailable"))
    state = _state(tmp_path)
    state.location_store.set("destination", {"x": 2.0, "y": 0.0, "theta": 0.0})

    result = json.loads(
        NavigateToLocationBlockingTool(robot_client=robot, state=state)._run(
            "destination"
        )
    )

    assert result["success"] is False
    assert "transform unavailable" in result["message"]
    assert robot.navigation_goals == []


def test_blocking_navigation_fails_closed_when_current_pose_is_stale(
    tmp_path: Path,
) -> None:
    robot = _PoseRobot(x=1.0, y=0.0, stamp_s=time.time() - 10.0)
    state = _state(tmp_path)
    state.location_store.set("destination", {"x": 2.0, "y": 0.0, "theta": 0.0})

    result = json.loads(
        NavigateToLocationBlockingTool(robot_client=robot, state=state)._run(
            "destination"
        )
    )

    assert result["success"] is False
    assert "stale" in result["message"]
    assert robot.navigation_goals == []


@pytest.mark.parametrize(
    "coords",
    [
        {"x": math.nan, "y": 0.0, "theta": 0.0},
        {"x": 0.0, "y": math.inf, "theta": 0.0},
        {"x": 0.0, "y": 0.0, "theta": math.nan},
        {"x": 1_000_000_000.0, "y": 0.0, "theta": 0.0},
    ],
)
def test_blocking_navigation_rejects_unsupported_target_before_motion(
    tmp_path: Path,
    coords: dict[str, float],
) -> None:
    robot = _PoseRobot()
    state = _state(tmp_path)
    state.location_store.set("destination", coords)

    result = json.loads(
        NavigateToLocationBlockingTool(robot_client=robot, state=state)._run(
            "destination"
        )
    )

    assert result["success"] is False
    assert robot.navigation_goals == []


def test_blocking_navigation_timeout_cancels_provider_goal(tmp_path: Path) -> None:
    robot = _NavigationTimeoutRobot()
    state = _state(tmp_path)
    state.location_store.set("destination", {"x": 2.0, "y": 0.0, "theta": 0.0})

    result = json.loads(
        NavigateToLocationBlockingTool(robot_client=robot, state=state)._run(
            "destination"
        )
    )

    assert result["success"] is False
    assert robot.canceled_goal_ids == [robot.timed_out_goal_id]
    assert state.get_active_goal() is None


def test_failed_timeout_cancellation_keeps_goal_active(tmp_path: Path) -> None:
    robot = _UncancelableNavigationTimeoutRobot()
    state = _state(tmp_path)
    state.location_store.set("destination", {"x": 2.0, "y": 0.0, "theta": 0.0})

    result = json.loads(
        NavigateToLocationBlockingTool(robot_client=robot, state=state)._run(
            "destination"
        )
    )

    assert result["success"] is False
    assert robot.canceled_goal_ids == [robot.timed_out_goal_id]
    assert state.get_active_goal()["goal_id"] == robot.timed_out_goal_id
    assert "cancellation could not be confirmed" in result["message"]

    retry = json.loads(
        NavigateToLocationBlockingTool(robot_client=robot, state=state)._run(
            "destination"
        )
    )

    assert retry["status"] == "blocked"
    assert robot.navigation_attempts == 1
    assert state.get_active_goal()["goal_id"] == robot.timed_out_goal_id


@pytest.mark.parametrize(
    "robot_type",
    [_AmbiguousCancelNavigationTimeoutRobot, _NonDictCancelNavigationTimeoutRobot],
)
def test_ambiguous_timeout_cancellation_keeps_goal_active(
    tmp_path: Path,
    robot_type: type[_NavigationTimeoutRobot],
) -> None:
    robot = robot_type()
    state = _state(tmp_path)
    state.location_store.set("destination", {"x": 2.0, "y": 0.0, "theta": 0.0})

    result = json.loads(
        NavigateToLocationBlockingTool(robot_client=robot, state=state)._run(
            "destination"
        )
    )

    assert result["success"] is False
    assert robot.canceled_goal_ids == [robot.timed_out_goal_id]
    assert state.has_unconfirmed_active_goal() is True


def test_cancel_tool_can_clear_an_unconfirmed_goal_on_retry(tmp_path: Path) -> None:
    robot = _RetryableCancelNavigationTimeoutRobot()
    state = _state(tmp_path)
    state.location_store.set("destination", {"x": 2.0, "y": 0.0, "theta": 0.0})
    json.loads(
        NavigateToLocationBlockingTool(robot_client=robot, state=state)._run(
            "destination"
        )
    )

    result = json.loads(CancelNavigationTool(robot_client=robot, state=state)._run())

    assert result["success"] is True
    assert result["status"] == "canceled"
    assert robot.canceled_goal_ids == [
        robot.timed_out_goal_id,
        robot.timed_out_goal_id,
    ]
    assert state.get_active_goal() is None


def test_explicit_internal_timeout_override_is_preserved(tmp_path: Path) -> None:
    robot = _PoseRobot(transform_error=AssertionError("pose lookup should be skipped"))
    state = _state(tmp_path)
    state.location_store.set("destination", {"x": 2.0, "y": 0.0, "theta": 0.0})
    tool = NavigateToLocationBlockingTool(
        robot_client=robot,
        state=state,
        timeout_sec=42.0,
    )

    result = json.loads(tool._run("destination"))

    assert result["success"] is True
    assert robot.navigation_goals[-1]["timeout_sec"] == 42.0


def test_explicit_internal_timeout_override_is_not_capped(tmp_path: Path) -> None:
    robot = _PoseRobot(transform_error=AssertionError("pose lookup should be skipped"))
    state = _state(tmp_path)
    state.location_store.set("destination", {"x": 2.0, "y": 0.0, "theta": 0.0})
    tool = NavigateToLocationBlockingTool(
        robot_client=robot,
        state=state,
        timeout_sec=600.0,
    )

    result = json.loads(tool._run("destination"))

    assert result["success"] is True
    assert robot.navigation_goals[-1]["timeout_sec"] == 600.0


def test_blocking_navigation_schema_hides_operational_timeout(tmp_path: Path) -> None:
    robot = _PoseRobot()
    state = _state(tmp_path)

    location_schema = ToolRuntime.build_schema(
        NavigateToLocationBlockingTool(robot_client=robot, state=state)
    )
    return_schema = ToolRuntime.build_schema(
        NavigateToReturnPointBlockingTool(robot_client=robot, state=state)
    )

    assert "timeout_sec" not in location_schema["parameters"]["properties"]
    assert "timeout_sec" not in return_schema["parameters"]["properties"]


def test_low_battery_blocks_before_pose_lookup_or_navigation(tmp_path: Path) -> None:
    robot = _PoseRobot(transform_error=AssertionError("pose lookup should be skipped"))
    state = _state(tmp_path)
    state.location_store.set("destination", {"x": 2.0, "y": 0.0, "theta": 0.0})
    battery = BatteryStateCache(robot, low_battery_pct=10.0)
    robot.set_battery_snapshot(BatterySnapshot(percentage=5.0))

    result = json.loads(
        NavigateToLocationBlockingTool(
            robot_client=robot,
            state=state,
            battery=battery,
        )._run("destination")
    )

    assert result["success"] is False
    assert result["status"] == "blocked"
    assert robot.navigation_goals == []
    assert state.get_active_goal() is None


class _OutcomeRobot(_PoseRobot):
    def __init__(self, result: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self._result = dict(result)

    def navigate_to_pose(self, **kwargs) -> dict:
        super().navigate_to_pose(**kwargs)
        return dict(self._result)


class _UncancelableDockRobot(_PoseRobot):
    def dock_for_charging(self, **_kwargs) -> dict:
        raise TimeoutError("alignment timed out")

    def cancel_charging_dock(self) -> dict:
        self.dock_cancel_requests += 1
        return {}


@pytest.mark.parametrize("outcome", ["aborted", "canceled"])
def test_explicit_terminal_failure_does_not_send_extra_cancel(
    tmp_path: Path,
    outcome: str,
) -> None:
    robot = _OutcomeRobot({"accepted": True, "outcome": outcome})
    state = _state(tmp_path)
    state.location_store.set("destination", {"x": 1.0, "y": 0.0, "theta": 0.0})

    result = json.loads(
        NavigateToLocationBlockingTool(robot_client=robot, state=state)._run(
            "destination"
        )
    )

    assert result["success"] is False
    assert robot.canceled_goal_ids == []
    assert state.get_active_goal() is None


def test_missing_terminal_outcome_is_canceled(tmp_path: Path) -> None:
    robot = _OutcomeRobot({"accepted": True})
    state = _state(tmp_path)
    state.location_store.set("destination", {"x": 1.0, "y": 0.0, "theta": 0.0})

    result = json.loads(
        NavigateToLocationBlockingTool(robot_client=robot, state=state)._run(
            "destination"
        )
    )

    assert result["success"] is False
    assert len(robot.canceled_goal_ids) == 1
    assert state.get_active_goal() is None


def test_charging_uses_dynamic_approach_then_fixed_alignment(tmp_path: Path) -> None:
    robot = _PoseRobot(x=0.0, y=0.0)
    state = _state(tmp_path)
    state.location_store.set("charge_dock", {"x": 3.0, "y": 4.0, "theta": 0.5})
    tool = ChargingDockTool(robot_client=robot, nav_state=state)
    watchdog_timeouts: list[float] = []

    with tool_execution_context(
        set_timeout=watchdog_timeouts.append,
        side_effect_guard=lambda: nullcontext(True),
    ):
        result = json.loads(tool._run())

    assert charging_tool_timeout_sec(
        80.0,
        alignment_timeout_sec=DOCK_ALIGNMENT_TIMEOUT_SEC,
    ) == 160.0
    assert watchdog_timeouts == [160.0]
    assert result["success"] is True
    assert robot.navigation_goals[-1]["timeout_sec"] == 80.0
    assert robot.navigation_goals[-1]["policy"]["source"] == "charging_dock"
    assert robot.dock_requests == [
        {
            "approach_pose": {
                "x": 3.0,
                "y": 4.0,
                "theta": 0.5,
                "frame_id": "map",
            },
            "dock_timeout_sec": DOCK_ALIGNMENT_TIMEOUT_SEC,
            "alignment_only": True,
        }
    ]


def test_low_battery_does_not_block_charging_approach(tmp_path: Path) -> None:
    robot = _PoseRobot()
    state = _state(tmp_path)
    state.location_store.set("charge_dock", {"x": 1.0, "y": 0.0, "theta": 0.0})
    battery = BatteryStateCache(robot, low_battery_pct=10.0)
    robot.set_battery_snapshot(BatterySnapshot(percentage=5.0))

    result = json.loads(
        ChargingDockTool(
            robot_client=robot,
            nav_state=state,
            battery=battery,
        )._run()
    )

    assert result["success"] is True
    assert len(robot.navigation_goals) == 1
    assert len(robot.dock_requests) == 1


def test_failed_charging_approach_skips_alignment(tmp_path: Path) -> None:
    robot = _OutcomeRobot({"accepted": True, "outcome": "aborted"})
    state = _state(tmp_path)
    state.location_store.set("charge_dock", {"x": 1.0, "y": 0.0, "theta": 0.0})

    result = json.loads(ChargingDockTool(robot_client=robot, nav_state=state)._run())

    assert result["success"] is False
    assert robot.dock_requests == []


def test_charging_watchdog_claim_before_alignment_prevents_dock_motion(
    tmp_path: Path,
) -> None:
    robot = _PoseRobot()
    state = _state(tmp_path)
    state.location_store.set("charge_dock", {"x": 1.0, "y": 0.0, "theta": 0.0})
    side_effect_decisions = iter((True, False))

    with tool_execution_context(
        set_timeout=lambda _timeout_sec: None,
        side_effect_guard=lambda: nullcontext(next(side_effect_decisions)),
    ):
        result = json.loads(
            ChargingDockTool(robot_client=robot, nav_state=state)._run()
        )

    assert result["success"] is False
    assert result["status"] == "canceled"
    assert len(robot.navigation_goals) == 1
    assert robot.dock_requests == []
    assert state.has_active_dock_alignment() is False


def test_unconfirmed_alignment_cancel_blocks_conflicting_navigation(
    tmp_path: Path,
) -> None:
    robot = _UncancelableDockRobot()
    state = _state(tmp_path)
    state.location_store.set("charge_dock", {"x": 1.0, "y": 0.0, "theta": 0.0})
    state.location_store.set("destination", {"x": 2.0, "y": 0.0, "theta": 0.0})

    dock_result = json.loads(
        ChargingDockTool(robot_client=robot, nav_state=state)._run()
    )
    navigation_count = len(robot.navigation_goals)
    blocked_result = json.loads(
        NavigateToLocationBlockingTool(robot_client=robot, state=state)._run(
            "destination"
        )
    )

    assert dock_result["success"] is False
    assert state.has_active_dock_alignment() is True
    assert robot.dock_cancel_requests == 1
    assert blocked_result["status"] == "blocked"
    assert len(robot.navigation_goals) == navigation_count


def test_unknown_async_goal_result_keeps_motion_suppressed(tmp_path: Path) -> None:
    state = _state(tmp_path)
    goal = state.begin_goal(
        goal_id="nav-unknown",
        tool_name="navigate_to_location",
        target_label="destination",
    )

    process_navigation_event(
        state=state,
        event={
            "event_type": "goal_result",
            "goal_id": goal["goal_id"],
            "outcome": "timeout",
        },
        on_nav_event=None,
    )

    assert state.get_active_goal()["goal_id"] == goal["goal_id"]
    assert state.has_unconfirmed_active_goal() is True


def test_active_dock_alignment_suppresses_face_turn_motion(tmp_path: Path) -> None:
    state = _state(tmp_path)

    state.begin_dock_alignment()

    assert state.allows_proactive_face_attention() is False
    assert state.active_goal_allows_auto_interrupt() is False
    assert state.active_goal_allows_passive_listen() is False
    assert state.build_interaction_context() == {
        "nav_active": True,
        "nav_source": "charging_dock",
        "nav_interruptible": False,
        "nav_passive_listen_allowed": False,
    }
