"""Transport-neutral navigation tools for Unitree Go2 profiles."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, List, Optional, Type

from pydantic import BaseModel, Field

from argos_src.runtime.battery_state import (
    BatteryStateCache,
    LOW_BATTERY_NAVIGATION_MSG,
)
from argos_src.nav_support.locations import (
    CHARGE_DOCK_LOCATION_NAME,
    CHARGING_DOCK_NAVIGATION_POLICY,
    FOCUSED_NAVIGATION_POLICY,
    INTERRUPTIBLE_NAVIGATION_POLICY,
    LocationStore,
    NavigationPolicy,
    NavigationState,
)
from argos_src.nav_support.timeouts import (
    DOCK_ALIGNMENT_TIMEOUT_SEC,
    charging_tool_timeout_sec,
    estimate_navigation_timeout_sec,
    navigation_tool_timeout_sec,
)
from argos_src.tools.base import BaseTool
from argos_src.tools.common.tool_response import build_tool_response, tool_response_json
from argos_src.tools.execution import (
    guard_tool_side_effect_start,
    set_tool_execution_timeout,
)

MAX_RELATIVE_TF_AGE_SEC = 3.0
LOCALIZE_NEAR_DISTANCE_M = 3.0
LOCALIZE_APPROXIMATE_DISTANCE_M = 4.0
MAP_FRAME = "map"
ROBOT_FRAME = "base_link"
NavEventSink = Callable[[dict[str, Any]], None]
UNCONFIRMED_NAVIGATION_CANCEL_MSG = (
    "Previous robot-motion cancellation is unconfirmed. "
    "Retry cancel_navigation before starting another navigation goal."
)


def _nav_tool_json(
    *,
    success: bool,
    status: str,
    message: str,
    eventual: bool = False,
    result_source: str = "immediate",
    **extra: Any,
) -> str:
    return tool_response_json(
        success=success,
        status=status,
        message=message,
        eventual=eventual,
        result_source=result_source,
        **extra,
    )


def _battery_blocks_nav(battery: Optional[BatteryStateCache]) -> Optional[str]:
    if battery is None:
        return None
    if battery.should_block_general_navigation():
        return battery.navigation_block_message()
    return None


def _policy_payload(policy: NavigationPolicy) -> dict[str, Any]:
    return {
        "source": policy.source,
        "interruptible": bool(policy.interruptible),
        "passive_listen_allowed": bool(policy.passive_listen_allowed),
    }


def _yaw_from_quaternion(rotation: Any) -> float:
    if isinstance(rotation, dict):
        x = float(rotation.get("x", 0.0) or 0.0)
        y = float(rotation.get("y", 0.0) or 0.0)
        z = float(rotation.get("z", 0.0) or 0.0)
        w = float(rotation.get("w", 1.0) or 1.0)
    else:
        values = tuple(rotation or (0.0, 0.0, 0.0, 1.0))
        x, y, z, w = (float(values[i]) for i in range(4))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _transform_pose(transform: Any) -> tuple[float, float, float, float]:
    translation = getattr(transform, "translation", None)
    rotation = getattr(transform, "rotation", None)
    stamp_s = float(getattr(transform, "stamp_s", 0.0) or 0.0)
    if translation is None and hasattr(transform, "transform"):
        nested = transform.transform
        t = nested.translation
        r = nested.rotation
        translation = (t.x, t.y, getattr(t, "z", 0.0))
        rotation = (r.x, r.y, r.z, r.w)
        stamp = getattr(getattr(transform, "header", None), "stamp", None)
        if stamp_s <= 0.0 and stamp is not None:
            stamp_s = float(getattr(stamp, "sec", 0) or 0) + (
                float(getattr(stamp, "nanosec", 0) or 0) / 1e9
            )
    if isinstance(translation, dict):
        x = float(translation.get("x", 0.0) or 0.0)
        y = float(translation.get("y", 0.0) or 0.0)
    else:
        values = tuple(translation or (0.0, 0.0, 0.0))
        x = float(values[0])
        y = float(values[1])
    return x, y, _yaw_from_quaternion(rotation or (0.0, 0.0, 0.0, 1.0)), stamp_s


def _validate_pose(*, x: float, y: float, theta: float, label: str) -> None:
    if not all(math.isfinite(float(value)) for value in (x, y, theta)):
        raise ValueError(f"{label} pose must contain finite x, y, and theta values.")


def _resolve_navigation_timeout_sec(
    *,
    robot_client: Any,
    x: float,
    y: float,
    theta: float,
    timeout_sec: float | None = None,
) -> float:
    _validate_pose(x=x, y=y, theta=theta, label="Target")
    if timeout_sec is not None:
        rendered_timeout_sec = float(timeout_sec)
        navigation_tool_timeout_sec(rendered_timeout_sec)
        return rendered_timeout_sec
    current_x, current_y, current_yaw, stamp_s = _transform_pose(
        robot_client.get_transform(MAP_FRAME, ROBOT_FRAME)
    )
    _validate_pose(
        x=current_x,
        y=current_y,
        theta=current_yaw,
        label="Current",
    )
    if stamp_s > 0.0 and time.time() - stamp_s > MAX_RELATIVE_TF_AGE_SEC:
        raise ValueError("Current pose is stale.")
    distance_m = math.hypot(float(x) - current_x, float(y) - current_y)
    return estimate_navigation_timeout_sec(distance_m)


def _cancel_active_navigation(robot_client: Any, state: NavigationState) -> bool:
    active_goal = state.get_active_goal()
    if not active_goal:
        return True
    goal_id = str(active_goal.get("goal_id", "") or "")
    if not goal_id:
        return False
    canceled = _cancel_provider_goal_best_effort(robot_client, goal_id)
    if canceled:
        state.clear_goal_if_active(goal_id)
    else:
        state.mark_active_goal_cancel_unconfirmed(goal_id)
    return canceled


def _cancel_active_dock_alignment(robot_client: Any, state: NavigationState) -> bool:
    if not state.has_active_dock_alignment():
        return True
    try:
        result = robot_client.cancel_charging_dock()
    except Exception:
        result = None
    canceled = bool(isinstance(result, dict) and result.get("canceled") is True)
    if canceled:
        state.clear_dock_alignment()
    return canceled


def _cancel_ambiguous_navigation_result(
    *,
    robot_client: Any,
    state: NavigationState,
    goal_id: str,
    result: dict[str, Any],
    reason: str,
) -> tuple[bool, str]:
    if _cancel_provider_goal_best_effort(robot_client, goal_id):
        state.clear_goal_if_active(goal_id)
        return False, _result_error(result, f"{reason} and was canceled.")
    state.mark_active_goal_cancel_unconfirmed(goal_id)
    return False, _result_error(
        result,
        f"{reason} and cancellation could not be confirmed.",
    )


def _accepted(result: dict[str, Any]) -> bool:
    if "accepted" in result:
        return bool(result.get("accepted"))
    if "success" in result:
        return bool(result.get("success"))
    return True


def _result_error(result: dict[str, Any], fallback: str) -> str:
    for key in ("message", "error", "error_msg", "detail"):
        value = str(result.get(key, "") or "").strip()
        if value:
            return value
    return fallback


class _NavigationCancelHandle:
    def __init__(self, robot_client: Any, goal_id: str) -> None:
        self._robot_client = robot_client
        self._goal_id = goal_id

    def cancel_goal_async(self) -> dict[str, Any]:
        return self._robot_client.cancel_navigation(goal_id=self._goal_id)


def _begin_provider_goal(
    *,
    robot_client: Any,
    state: NavigationState,
    goal_id: str,
    tool_name: str,
    target_label: str,
    waypoint_names: Optional[list[str]] = None,
    policy: NavigationPolicy = INTERRUPTIBLE_NAVIGATION_POLICY,
) -> dict[str, Any]:
    return state.begin_goal(
        goal_id=goal_id,
        tool_name=tool_name,
        target_label=target_label,
        handle=_NavigationCancelHandle(robot_client, goal_id),
        waypoint_names=waypoint_names,
        policy=policy,
    )


def _cancel_provider_goal_best_effort(robot_client: Any, goal_id: str) -> bool:
    try:
        result = robot_client.cancel_navigation(goal_id=goal_id)
    except Exception:
        return False
    return bool(isinstance(result, dict) and result.get("canceled", False))


def _unconfirmed_navigation_blocks_start(state: NavigationState) -> bool:
    return state.has_unconfirmed_active_goal() or state.has_active_dock_alignment()


def start_navigation_to_saved_location(
    *,
    robot_client: Any,
    state: NavigationState,
    location_name: str,
    battery: Optional[BatteryStateCache] = None,
    tool_name: str = "navigate_to_location",
    policy: NavigationPolicy = INTERRUPTIBLE_NAVIGATION_POLICY,
) -> dict[str, Any]:
    if _unconfirmed_navigation_blocks_start(state):
        return build_tool_response(
            success=False,
            status="blocked",
            message=UNCONFIRMED_NAVIGATION_CANCEL_MSG,
            location_name=location_name,
        )
    blocked = _battery_blocks_nav(battery)
    if blocked and location_name != CHARGE_DOCK_LOCATION_NAME:
        return build_tool_response(
            success=False,
            status="blocked",
            message=blocked,
            location_name=location_name,
        )
    coords = state.location_store.get(location_name)
    if coords is None:
        known = ", ".join(state.location_store.names()) or "none saved yet"
        return build_tool_response(
            success=False,
            status="error",
            message=f"Location '{location_name}' not found. Known locations: {known}.",
            location_name=location_name,
        )

    goal_id = state.new_goal_id()
    try:
        result = robot_client.navigate_to_pose(
            goal_id=goal_id,
            x=coords["x"],
            y=coords["y"],
            theta=coords["theta"],
            target_label=location_name,
            tool_name=tool_name,
            blocking=False,
            policy=_policy_payload(policy),
        )
    except Exception as exc:
        return build_tool_response(
            success=False,
            status="error",
            message=str(exc),
            location_name=location_name,
        )
    if not _accepted(result):
        return build_tool_response(
            success=False,
            status="error",
            message=_result_error(result, "Navigation provider rejected the goal."),
            location_name=location_name,
        )
    _begin_provider_goal(
        robot_client=robot_client,
        state=state,
        goal_id=goal_id,
        tool_name=tool_name,
        target_label=location_name,
        policy=policy,
    )
    return build_tool_response(
        success=True,
        status="started",
        message=(
            f"Navigation started to {location_name}. Do not assume arrival yet. "
            "Wait for a NAV_EVENT goal_result with outcome=succeeded before saying you reached it."
        ),
        eventual=True,
        result_source="deferred_event",
        location_name=location_name,
        goal_id=goal_id,
    )


def process_navigation_event(
    *,
    state: NavigationState,
    event: dict[str, Any],
    on_nav_event: Optional[NavEventSink],
) -> None:
    routed_event = dict(event)
    event_type = str(routed_event.get("event_type", "") or "")
    goal_id = str(routed_event.get("goal_id", "") or "")
    active_goal = state.get_active_goal()
    if active_goal and str(active_goal.get("goal_id", "") or "") == goal_id:
        routed_event.setdefault("tool_name", active_goal.get("tool_name", ""))
        routed_event.setdefault("target_label", active_goal.get("target_label", ""))
    if event_type == "waypoint_reached":
        index = int(event.get("waypoint_index", 0) or 0)
        zero_based = max(0, index - 1)
        if goal_id and not state.mark_waypoint_reported(goal_id, zero_based):
            return
    if event_type == "goal_result" and goal_id:
        outcome = str(routed_event.get("outcome", "") or "").lower()
        if outcome in {"succeeded", "aborted", "canceled"}:
            state.clear_goal_if_active(goal_id)
        else:
            state.mark_active_goal_cancel_unconfirmed(goal_id)
    if on_nav_event is not None:
        try:
            on_nav_event(routed_event)
        except Exception:
            return


def navigate_to_pose_blocking(
    *,
    robot_client: Any,
    state: NavigationState,
    x: float,
    y: float,
    theta: float,
    timeout_sec: float | None = None,
    watchdog_timeout_fn: Callable[[float], float] = navigation_tool_timeout_sec,
    battery: Optional[BatteryStateCache] = None,
    skip_low_battery_check: bool = False,
    tool_name: str = "navigation_blocking",
    target_label: str = "custom_pose",
    policy: NavigationPolicy = FOCUSED_NAVIGATION_POLICY,
) -> tuple[bool, str]:
    if _unconfirmed_navigation_blocks_start(state):
        return False, UNCONFIRMED_NAVIGATION_CANCEL_MSG
    if not skip_low_battery_check and _battery_blocks_nav(battery):
        return False, LOW_BATTERY_NAVIGATION_MSG

    try:
        resolved_timeout_sec = _resolve_navigation_timeout_sec(
            robot_client=robot_client,
            x=x,
            y=y,
            theta=theta,
            timeout_sec=timeout_sec,
        )
        set_tool_execution_timeout(watchdog_timeout_fn(resolved_timeout_sec))
    except Exception as exc:
        return False, f"Could not calculate a safe navigation timeout: {exc}"

    with guard_tool_side_effect_start() as side_effect_allowed:
        if not side_effect_allowed:
            return False, "Navigation was canceled before motion started."
        goal_id = state.new_goal_id()
        _begin_provider_goal(
            robot_client=robot_client,
            state=state,
            goal_id=goal_id,
            tool_name=tool_name,
            target_label=target_label,
            policy=policy,
        )
    try:
        result = robot_client.navigate_to_pose(
            goal_id=goal_id,
            x=x,
            y=y,
            theta=theta,
            target_label=target_label,
            tool_name=tool_name,
            blocking=True,
            timeout_sec=resolved_timeout_sec,
            policy=_policy_payload(policy),
        )
    except Exception as exc:
        cancel_confirmed = _cancel_provider_goal_best_effort(robot_client, goal_id)
        if cancel_confirmed:
            state.clear_goal_if_active(goal_id)
            return False, str(exc)
        state.mark_active_goal_cancel_unconfirmed(goal_id)
        return False, f"{exc}. Navigation cancellation could not be confirmed."
    outcome = str(result.get("outcome", "") or "").lower()
    accepted = result.get("accepted")
    if accepted is False:
        state.clear_goal_if_active(goal_id)
        return False, _result_error(result, "Navigation provider rejected the goal.")
    if accepted is not True:
        return _cancel_ambiguous_navigation_result(
            robot_client=robot_client,
            state=state,
            goal_id=goal_id,
            result=result,
            reason="Navigation returned no explicit acceptance",
        )
    if outcome == "succeeded":
        state.clear_goal_if_active(goal_id)
        return True, ""
    if outcome in {"aborted", "canceled"}:
        state.clear_goal_if_active(goal_id)
        return False, _result_error(result, f"Navigation {outcome}.")
    return _cancel_ambiguous_navigation_result(
        robot_client=robot_client,
        state=state,
        goal_id=goal_id,
        result=result,
        reason="Navigation returned no explicit terminal outcome",
    )


class NavigateToLocationInput(BaseModel):
    location_name: str = Field(
        ...,
        description="Exact name of a saved location (e.g. room_maddie, spot_x).",
    )


class NavigateRelativeInput(BaseModel):
    forward_m: float = Field(
        ...,
        description="Distance to move along the robot's current heading in meters.",
    )
    left_m: float = Field(
        default=0.0,
        description="Distance to move to the robot's left in meters.",
    )
    delta_theta_rad: float = Field(
        default=0.0,
        description="Change in yaw at the goal in radians.",
    )


class FollowWaypointsInput(BaseModel):
    location_names: List[str] = Field(
        ...,
        description="Ordered list of exact saved location names to visit.",
    )


class LocalizeCurrentLocationInput(BaseModel):
    data: Optional[str] = Field(default=None)


class MarkReturnPointInput(BaseModel):
    label: str = Field(
        default="assignment_start",
        description=(
            "Temporary return-point label for the current task. Use assignment_start "
            "unless the user gave a more specific return name."
        ),
    )


class NavigateToReturnPointInput(BaseModel):
    label: str = Field(
        default="assignment_start",
        description="Temporary return-point label previously created with mark_return_point.",
    )


class SaveCurrentLocationInput(BaseModel):
    name: str = Field(
        ...,
        description="Name to persist for the robot's current location.",
    )


class CancelNavigationInput(BaseModel):
    data: Optional[str] = Field(default=None)


class StopPatrolInput(BaseModel):
    cancel_navigation: bool = Field(default=True)


class ChargingDockInput(BaseModel):
    data: Optional[str] = Field(default=None)


class NavigateToLocationTool(BaseTool):
    name: str = "navigate_to_location"
    description: str = (
        "Start non-blocking navigation to a saved location and return immediately. "
        "Use mainly for patrol/background movement when a later NAV_EVENT will report arrival. "
        "For human requests that need arrival before speaking, inspecting, or using another tool, "
        "use navigate_to_location_blocking instead."
    )
    args_schema: Type[BaseModel] = NavigateToLocationInput
    robot_client: Any = Field(exclude=True)
    state: NavigationState = Field(exclude=True)
    on_nav_event: Optional[NavEventSink] = Field(default=None, exclude=True)
    battery: Optional[BatteryStateCache] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, location_name: str) -> str:
        payload = start_navigation_to_saved_location(
            robot_client=self.robot_client,
            state=self.state,
            location_name=location_name,
            battery=self.battery,
            tool_name=self.name,
            policy=INTERRUPTIBLE_NAVIGATION_POLICY,
        )
        return tool_response_json(**payload)


class NavigateRelativeTool(BaseTool):
    name: str = "navigate_relative"
    description: str = "Navigate relative to current pose."
    args_schema: Type[BaseModel] = NavigateRelativeInput
    robot_client: Any = Field(exclude=True)
    state: NavigationState = Field(exclude=True)
    on_nav_event: Optional[NavEventSink] = Field(default=None, exclude=True)
    battery: Optional[BatteryStateCache] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        forward_m: float,
        left_m: float = 0.0,
        delta_theta_rad: float = 0.0,
    ) -> str:
        if _unconfirmed_navigation_blocks_start(self.state):
            return _nav_tool_json(
                success=False,
                status="blocked",
                message=UNCONFIRMED_NAVIGATION_CANCEL_MSG,
            )
        blocked = _battery_blocks_nav(self.battery)
        if blocked:
            return _nav_tool_json(
                success=False,
                status="blocked",
                message=blocked,
                forward_m=forward_m,
                left_m=left_m,
                delta_theta_rad=delta_theta_rad,
            )
        try:
            pose_x, pose_y, yaw, stamp_s = _transform_pose(
                self.robot_client.get_transform(MAP_FRAME, ROBOT_FRAME)
            )
        except Exception as exc:
            return _nav_tool_json(
                success=False,
                status="error",
                message=f"Could not get current pose for relative navigation: {exc}.",
                forward_m=forward_m,
                left_m=left_m,
                delta_theta_rad=delta_theta_rad,
            )
        if stamp_s > 0.0:
            tf_age_sec = time.time() - stamp_s
            if tf_age_sec > MAX_RELATIVE_TF_AGE_SEC:
                return _nav_tool_json(
                    success=False,
                    status="error",
                    message=(
                        f"Pose is {tf_age_sec:.1f}s stale - relative navigation would be inaccurate "
                        f"(limit {MAX_RELATIVE_TF_AGE_SEC}s)."
                    ),
                    forward_m=forward_m,
                    left_m=left_m,
                    delta_theta_rad=delta_theta_rad,
                )
        dx = forward_m * math.cos(yaw) - left_m * math.sin(yaw)
        dy = forward_m * math.sin(yaw) + left_m * math.cos(yaw)
        target_label = (
            f"forward={forward_m:.2f}m,left={left_m:.2f}m,delta_theta={delta_theta_rad:.2f}rad"
        )
        goal_id = self.state.new_goal_id()
        try:
            result = self.robot_client.navigate_to_pose(
                goal_id=goal_id,
                x=pose_x + dx,
                y=pose_y + dy,
                theta=yaw + delta_theta_rad,
                target_label=target_label,
                tool_name=self.name,
                blocking=False,
                policy=_policy_payload(INTERRUPTIBLE_NAVIGATION_POLICY),
            )
        except Exception as exc:
            return _nav_tool_json(
                success=False,
                status="error",
                message=str(exc),
                forward_m=forward_m,
                left_m=left_m,
                delta_theta_rad=delta_theta_rad,
            )
        if not _accepted(result):
            return _nav_tool_json(
                success=False,
                status="error",
                message=_result_error(result, "Navigation provider rejected the goal."),
                forward_m=forward_m,
                left_m=left_m,
                delta_theta_rad=delta_theta_rad,
            )
        _begin_provider_goal(
            robot_client=self.robot_client,
            state=self.state,
            goal_id=goal_id,
            tool_name=self.name,
            target_label=target_label,
            policy=INTERRUPTIBLE_NAVIGATION_POLICY,
        )
        return _nav_tool_json(
            success=True,
            status="started",
            message=(
                f"Navigation started relative: forward={forward_m:.2f}m, "
                f"left={left_m:.2f}m, delta_theta={delta_theta_rad:.2f}rad. "
                "Wait for NAV_EVENT goal_result with outcome=succeeded before saying movement is complete."
            ),
            eventual=True,
            result_source="deferred_event",
            forward_m=forward_m,
            left_m=left_m,
            delta_theta_rad=delta_theta_rad,
            goal_id=goal_id,
        )


class FollowWaypointsTool(BaseTool):
    name: str = "follow_waypoints"
    description: str = "Visit a sequence of named locations in order."
    args_schema: Type[BaseModel] = FollowWaypointsInput
    robot_client: Any = Field(exclude=True)
    state: NavigationState = Field(exclude=True)
    on_nav_event: Optional[NavEventSink] = Field(default=None, exclude=True)
    battery: Optional[BatteryStateCache] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, location_names: List[str]) -> str:
        if _unconfirmed_navigation_blocks_start(self.state):
            return _nav_tool_json(
                success=False,
                status="blocked",
                message=UNCONFIRMED_NAVIGATION_CANCEL_MSG,
                location_names=location_names,
            )
        if not location_names:
            return _nav_tool_json(success=False, status="error", message="No waypoints given.")
        blocked = _battery_blocks_nav(self.battery)
        if blocked and any(name != CHARGE_DOCK_LOCATION_NAME for name in location_names):
            return _nav_tool_json(
                success=False,
                status="blocked",
                message=blocked,
                location_names=location_names,
            )
        waypoints: list[dict[str, Any]] = []
        missing: list[str] = []
        for name in location_names:
            coords = self.state.location_store.get(name)
            if coords is None:
                missing.append(name)
            else:
                waypoints.append(
                    {
                        "name": name,
                        "x": float(coords["x"]),
                        "y": float(coords["y"]),
                        "theta": float(coords["theta"]),
                        "frame_id": "map",
                    }
                )
        if missing:
            known = ", ".join(self.state.location_store.names()) or "none"
            return _nav_tool_json(
                success=False,
                status="error",
                message=f"Unknown location(s): {', '.join(missing)}. Known: {known}.",
                location_names=location_names,
                missing=missing,
            )
        goal_id = self.state.new_goal_id()
        target_label = ", ".join(location_names)
        try:
            result = self.robot_client.follow_waypoints(
                goal_id=goal_id,
                waypoints=waypoints,
                target_label=target_label,
                tool_name=self.name,
                policy=_policy_payload(INTERRUPTIBLE_NAVIGATION_POLICY),
            )
        except Exception as exc:
            return _nav_tool_json(
                success=False,
                status="error",
                message=str(exc),
                location_names=location_names,
            )
        if not _accepted(result):
            return _nav_tool_json(
                success=False,
                status="error",
                message=_result_error(result, "Navigation provider rejected the route."),
                location_names=location_names,
            )
        _begin_provider_goal(
            robot_client=self.robot_client,
            state=self.state,
            goal_id=goal_id,
            tool_name=self.name,
            target_label=target_label,
            waypoint_names=list(location_names),
            policy=INTERRUPTIBLE_NAVIGATION_POLICY,
        )
        return _nav_tool_json(
            success=True,
            status="started",
            message=(
                f"Waypoint navigation started: {', '.join(location_names)}. "
                "Use waypoint_reached NAV_EVENT updates for progress and wait for final goal_result."
            ),
            eventual=True,
            result_source="deferred_event",
            location_names=location_names,
            goal_id=goal_id,
        )


class CancelNavigationTool(BaseTool):
    name: str = "cancel_navigation"
    description: str = "Cancel the current navigation."
    args_schema: Type[BaseModel] = CancelNavigationInput
    robot_client: Any = Field(exclude=True)
    state: NavigationState = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _capture_resume_mission(self) -> None:
        active = self.state.get_active_goal()
        if not active:
            return
        tool_name = str(active.get("tool_name", ""))
        if tool_name == "navigate_to_location":
            target = str(active.get("target_label", "")).strip()
            if target:
                self.state.save_interrupted_mission(
                    {"mission_type": "navigate_to_location", "location_name": target}
                )
        elif tool_name == "follow_waypoints":
            names = list(active.get("waypoint_names") or [])
            reported = active.get("reported_waypoint_indices") or set()
            next_index = (max(reported) + 1) if isinstance(reported, set) and reported else 0
            remaining = names[next_index:]
            if remaining:
                self.state.save_interrupted_mission(
                    {"mission_type": "follow_waypoints", "remaining_waypoints": remaining}
                )

    def _run(self, data: Optional[str] = None) -> str:
        del data
        self._capture_resume_mission()
        active = self.state.get_active_goal()
        goal_id = str((active or {}).get("goal_id", "") or "")
        if not goal_id:
            if self.state.has_active_dock_alignment():
                if _cancel_active_dock_alignment(self.robot_client, self.state):
                    return _nav_tool_json(
                        success=True,
                        status="canceled",
                        message="Charging alignment canceled.",
                    )
                return _nav_tool_json(
                    success=False,
                    status="error",
                    message="Charging alignment cancellation could not be confirmed.",
                )
            return _nav_tool_json(
                success=False,
                status="error",
                message="No navigation in progress.",
            )
        if _cancel_active_navigation(self.robot_client, self.state):
            self.state.take_last_goal_handle()
            return _nav_tool_json(success=True, status="canceled", message="Navigation canceled.")
        return _nav_tool_json(
            success=False,
            status="error",
            message="Cancel failed: provider cancellation could not be confirmed.",
        )


class StopPatrolTool(BaseTool):
    name: str = "stop_patrol"
    description: str = "Stop the autonomous patrol loop."
    args_schema: Type[BaseModel] = StopPatrolInput
    robot_client: Any = Field(exclude=True)
    state: NavigationState = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, cancel_navigation: bool = True) -> str:
        self.state.stop_patrol()
        if cancel_navigation:
            active = self.state.get_active_goal()
            goal_id = str((active or {}).get("goal_id", "") or "")
            if goal_id:
                if _cancel_active_navigation(self.robot_client, self.state):
                    self.state.take_last_goal_handle()
                    return _nav_tool_json(
                        success=True,
                        status="completed",
                        message="Patrol stopped and active navigation cancelled.",
                    )
                return _nav_tool_json(
                    success=False,
                    status="error",
                    message=(
                        "Patrol stopped, but active navigation cancellation "
                        "could not be confirmed."
                    ),
                )
            if self.state.has_active_dock_alignment():
                if _cancel_active_dock_alignment(self.robot_client, self.state):
                    return _nav_tool_json(
                        success=True,
                        status="completed",
                        message="Patrol stopped and charging alignment canceled.",
                    )
                return _nav_tool_json(
                    success=False,
                    status="error",
                    message=(
                        "Patrol stopped, but charging alignment cancellation "
                        "could not be confirmed."
                    ),
                )
        return _nav_tool_json(success=True, status="completed", message="Patrol stopped.")


class LocalizeCurrentLocationTool(BaseTool):
    name: str = "localize_current_location"
    description: str = (
        "Answer where the robot currently is by comparing its current pose to saved locations. "
        "Use for questions like 'where are you?', 'what location are you at?', or "
        "'are you near X?'. This is read-only: it does not save a location or mark a return point."
    )
    args_schema: Type[BaseModel] = LocalizeCurrentLocationInput
    robot_client: Any = Field(exclude=True)
    state: NavigationState = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, data: Optional[str] = None) -> str:
        del data
        try:
            x, y, yaw, _stamp_s = _transform_pose(
                self.robot_client.get_transform(MAP_FRAME, ROBOT_FRAME)
            )
        except Exception as exc:
            return _nav_tool_json(
                success=False,
                status="error",
                message=f"Could not localize current location: {exc}.",
            )

        pose = {"x": x, "y": y, "theta": yaw}
        saved_locations = self.state.location_store.get_all()
        if not saved_locations:
            return _nav_tool_json(
                success=True,
                status="completed",
                message="Current pose is known, but no saved locations exist yet.",
                result_source="immediate",
                confidence="unknown",
                nearest_location="",
                distance_m=None,
                pose=pose,
                saved_location_count=0,
            )

        nearest_name = ""
        nearest_distance = float("inf")
        for name, coords in saved_locations.items():
            distance = math.hypot(float(coords["x"]) - x, float(coords["y"]) - y)
            if distance < nearest_distance:
                nearest_name = str(name)
                nearest_distance = distance

        rounded_distance = round(nearest_distance, 3)
        if nearest_distance <= LOCALIZE_NEAR_DISTANCE_M:
            confidence = "near"
            message = f"I am near {nearest_name}."
        elif nearest_distance <= LOCALIZE_APPROXIMATE_DISTANCE_M:
            confidence = "approximate"
            message = f"I am closest to {nearest_name}, about {nearest_distance:.1f} meters away."
        else:
            confidence = "unknown"
            message = (
                f"I am not confidently at a saved location. "
                f"The closest saved location is {nearest_name}, about {nearest_distance:.1f} meters away."
            )

        return _nav_tool_json(
            success=True,
            status="completed",
            message=message,
            result_source="immediate",
            confidence=confidence,
            nearest_location=nearest_name,
            distance_m=rounded_distance,
            pose=pose,
            saved_location_count=len(saved_locations),
        )


class MarkReturnPointTool(BaseTool):
    name: str = "mark_return_point"
    description: str = (
        "Remember the robot's current pose as a temporary return point for the active task. "
        "Use this before leaving the user for inspection/report-back missions. "
        "This does not create a persistent saved location."
    )
    args_schema: Type[BaseModel] = MarkReturnPointInput
    robot_client: Any = Field(exclude=True)
    state: NavigationState = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, label: str = "assignment_start") -> str:
        rendered_label = str(label or "").strip() or "assignment_start"
        try:
            x, y, yaw, _stamp_s = _transform_pose(
                self.robot_client.get_transform(MAP_FRAME, ROBOT_FRAME)
            )
        except Exception as exc:
            return _nav_tool_json(
                success=False,
                status="error",
                message=f"Could not mark return point: {exc}.",
                label=rendered_label,
            )
        self.state.set_return_point(rendered_label, {"x": x, "y": y, "theta": yaw})
        return _nav_tool_json(
            success=True,
            status="completed",
            message=f"Marked temporary return point '{rendered_label}'.",
            result_source="immediate",
            label=rendered_label,
            data={"x": x, "y": y, "theta": yaw},
        )


class NavigateToReturnPointBlockingTool(BaseTool):
    name: str = "navigate_to_return_point_blocking"
    description: str = (
        "Navigate back to a temporary return point created with mark_return_point and wait "
        "for arrival. Use this before the final spoken report for inspection/report-back missions."
    )
    args_schema: Type[BaseModel] = NavigateToReturnPointInput
    robot_client: Any = Field(exclude=True)
    state: NavigationState = Field(exclude=True)
    battery: Optional[BatteryStateCache] = Field(default=None, exclude=True)
    timeout_sec: float | None = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def cancel_active_execution(self) -> bool:
        return _cancel_active_navigation(self.robot_client, self.state)

    def _run(self, label: str = "assignment_start") -> str:
        rendered_label = str(label or "").strip() or "assignment_start"
        if _unconfirmed_navigation_blocks_start(self.state):
            return _nav_tool_json(
                success=False,
                status="blocked",
                message=UNCONFIRMED_NAVIGATION_CANCEL_MSG,
                label=rendered_label,
            )
        coords = self.state.get_return_point(rendered_label)
        if coords is None:
            return _nav_tool_json(
                success=False,
                status="error",
                message=(
                    f"Return point '{rendered_label}' is not marked. "
                    "Call mark_return_point before leaving if the task requires returning."
                ),
                label=rendered_label,
            )
        ok, detail = navigate_to_pose_blocking(
            robot_client=self.robot_client,
            state=self.state,
            x=coords["x"],
            y=coords["y"],
            theta=coords["theta"],
            timeout_sec=self.timeout_sec,
            battery=self.battery,
            tool_name=self.name,
            target_label=f"return_point:{rendered_label}",
            policy=FOCUSED_NAVIGATION_POLICY,
        )
        if not ok:
            return _nav_tool_json(
                success=False,
                status="error",
                message=f"Return navigation failed: {detail}",
                label=rendered_label,
            )
        return _nav_tool_json(
            success=True,
            status="completed",
            message=f"Returned to '{rendered_label}'.",
            label=rendered_label,
        )


class SaveCurrentLocationTool(BaseTool):
    name: str = "save_current_location"
    description: str = (
        "Persist the robot's current pose as a named saved location. Use when the user asks "
        "to save, remember, mark, or name this spot for future navigation. "
        "For temporary task return points, use mark_return_point instead."
    )
    args_schema: Type[BaseModel] = SaveCurrentLocationInput
    robot_client: Any = Field(exclude=True)
    state: NavigationState = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, name: str) -> str:
        rendered_name = str(name or "").strip()
        if not rendered_name:
            return _nav_tool_json(
                success=False,
                status="error",
                message="A saved-location name is required.",
                name=name,
            )
        try:
            x, y, yaw, _stamp_s = _transform_pose(
                self.robot_client.get_transform(MAP_FRAME, ROBOT_FRAME)
            )
        except Exception as exc:
            return _nav_tool_json(
                success=False,
                status="error",
                message=f"Could not save current location: {exc}.",
                name=rendered_name,
            )
        self.state.location_store.set(rendered_name, {"x": x, "y": y, "theta": yaw})
        return _nav_tool_json(
            success=True,
            status="completed",
            message=f"Saved current location as '{rendered_name}'.",
            result_source="immediate",
            data={"x": x, "y": y, "theta": yaw},
            name=rendered_name,
        )


class NavigateToLocationBlockingTool(BaseTool):
    name: str = "navigate_to_location_blocking"
    description: str = (
        "Navigate to a saved location and wait for final arrival before continuing. "
        "Use this for normal human-requested navigation and before capture_scene when inspecting a place."
    )
    args_schema: Type[BaseModel] = NavigateToLocationInput
    robot_client: Any = Field(exclude=True)
    state: NavigationState = Field(exclude=True)
    battery: Optional[BatteryStateCache] = Field(default=None, exclude=True)
    timeout_sec: float | None = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def cancel_active_execution(self) -> bool:
        return _cancel_active_navigation(self.robot_client, self.state)

    def _run(self, location_name: str) -> str:
        if _unconfirmed_navigation_blocks_start(self.state):
            return _nav_tool_json(
                success=False,
                status="blocked",
                message=UNCONFIRMED_NAVIGATION_CANCEL_MSG,
                location_name=location_name,
            )
        blocked = _battery_blocks_nav(self.battery)
        if blocked and location_name != CHARGE_DOCK_LOCATION_NAME:
            return _nav_tool_json(
                success=False,
                status="blocked",
                message=blocked,
                location_name=location_name,
            )
        coords = self.state.location_store.get(location_name)
        if coords is None:
            known = ", ".join(self.state.location_store.names()) or "none saved yet"
            return _nav_tool_json(
                success=False,
                status="error",
                message=f"Location '{location_name}' not found. Known locations: {known}.",
                location_name=location_name,
            )
        ok, detail = navigate_to_pose_blocking(
            robot_client=self.robot_client,
            state=self.state,
            x=coords["x"],
            y=coords["y"],
            theta=coords["theta"],
            timeout_sec=self.timeout_sec,
            battery=self.battery,
            tool_name=self.name,
            target_label=location_name,
            policy=FOCUSED_NAVIGATION_POLICY,
        )
        if not ok:
            return _nav_tool_json(
                success=False,
                status="error",
                message=f"Navigation failed: {detail}",
                location_name=location_name,
            )
        return _nav_tool_json(
            success=True,
            status="completed",
            message=f"Arrived at {location_name}.",
            location_name=location_name,
        )


class ChargingDockTool(BaseTool):
    name: str = "charging_dock"
    description: str = (
        "Dock on the charging station using the saved `charge_dock` approach pose "
        "and the robot provider's charging sequence."
    )
    args_schema: Type[BaseModel] = ChargingDockInput
    robot_client: Any = Field(exclude=True)
    nav_state: NavigationState = Field(exclude=True)
    battery: Optional[BatteryStateCache] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def cancel_active_execution(self) -> bool:
        if self.nav_state.has_active_dock_alignment():
            return _cancel_active_dock_alignment(self.robot_client, self.nav_state)
        return _cancel_active_navigation(self.robot_client, self.nav_state)

    def _restore_patrol_after_failure(self, paused_patrol: Optional[dict[str, Any]]) -> None:
        if paused_patrol is None:
            return
        try:
            self.nav_state.resume_paused_patrol()
        except Exception:
            pass

    def _run(self, data: Optional[str] = None) -> str:
        del data
        if _unconfirmed_navigation_blocks_start(self.nav_state):
            return tool_response_json(
                success=False,
                status="blocked",
                message=UNCONFIRMED_NAVIGATION_CANCEL_MSG,
                robot_state_after="unknown",
            )
        coords = self.nav_state.location_store.get(CHARGE_DOCK_LOCATION_NAME)
        if coords is None:
            return tool_response_json(
                success=False,
                status="error",
                message=(
                    f"Location '{CHARGE_DOCK_LOCATION_NAME}' is not saved. "
                    "Save the approach pose near the charging station with "
                    "save_current_location(name='charge_dock'), then retry."
                ),
                robot_state_after="unknown",
            )
        paused_patrol = self.nav_state.pause_patrol()
        ok, detail = navigate_to_pose_blocking(
            robot_client=self.robot_client,
            state=self.nav_state,
            x=coords["x"],
            y=coords["y"],
            theta=coords["theta"],
            battery=self.battery,
            skip_low_battery_check=True,
            tool_name=self.name,
            target_label=CHARGE_DOCK_LOCATION_NAME,
            policy=CHARGING_DOCK_NAVIGATION_POLICY,
            watchdog_timeout_fn=charging_tool_timeout_sec,
        )
        if not ok:
            if not _unconfirmed_navigation_blocks_start(self.nav_state):
                self._restore_patrol_after_failure(paused_patrol)
            return tool_response_json(
                success=False,
                status="error",
                message=f"Charging approach navigation failed: {detail}",
                robot_state_after="unknown",
            )
        with guard_tool_side_effect_start() as side_effect_allowed:
            if not side_effect_allowed:
                return tool_response_json(
                    success=False,
                    status="canceled",
                    message="Charging was canceled before final alignment started.",
                    robot_state_after="unknown",
                )
            self.nav_state.begin_dock_alignment()
        try:
            result = self.robot_client.dock_for_charging(
                approach_pose={
                    "x": float(coords["x"]),
                    "y": float(coords["y"]),
                    "theta": float(coords["theta"]),
                    "frame_id": "map",
                },
                dock_timeout_sec=DOCK_ALIGNMENT_TIMEOUT_SEC,
                alignment_only=True,
            )
        except Exception as exc:
            cancel_confirmed = _cancel_active_dock_alignment(
                self.robot_client,
                self.nav_state,
            )
            return tool_response_json(
                success=False,
                status="error",
                message=(
                    str(exc)
                    if cancel_confirmed
                    else f"{exc}. Charging alignment cancellation could not be confirmed."
                ),
                robot_state_after="unknown",
            )
        if result.get("success") is not True:
            cancel_confirmed = _cancel_active_dock_alignment(
                self.robot_client,
                self.nav_state,
            )
            return tool_response_json(
                success=False,
                status="error",
                message=(
                    _result_error(result, "Charging dock sequence failed.")
                    if cancel_confirmed
                    else (
                        "Charging dock sequence failed and alignment cancellation "
                        "could not be confirmed."
                    )
                ),
                robot_state_after="unknown",
            )
        self.nav_state.clear_dock_alignment()
        verification = str(result.get("charging_verification", "unknown") or "unknown")
        message = str(result.get("message", "") or "").strip()
        if not message:
            message = "Reached the charging approach and completed final dock alignment."
        return tool_response_json(
            success=True,
            status="completed",
            message=message,
            robot_state_after=str(result.get("robot_state_after", "damp_rest") or "damp_rest"),
            charging_verification=verification,
        )


def get_navigation_tools(
    robot_client: Any,
    *,
    location_store: LocationStore,
    state: NavigationState,
    on_nav_event: Optional[NavEventSink] = None,
    battery_cache: Optional[BatteryStateCache] = None,
) -> list[BaseTool]:
    del location_store
    return [
        NavigateToLocationTool(
            robot_client=robot_client,
            state=state,
            on_nav_event=on_nav_event,
            battery=battery_cache,
        ),
        NavigateToLocationBlockingTool(
            robot_client=robot_client,
            state=state,
            battery=battery_cache,
        ),
        LocalizeCurrentLocationTool(robot_client=robot_client, state=state),
        MarkReturnPointTool(robot_client=robot_client, state=state),
        NavigateToReturnPointBlockingTool(
            robot_client=robot_client,
            state=state,
            battery=battery_cache,
        ),
        SaveCurrentLocationTool(robot_client=robot_client, state=state),
        NavigateRelativeTool(
            robot_client=robot_client,
            state=state,
            on_nav_event=on_nav_event,
            battery=battery_cache,
        ),
        FollowWaypointsTool(
            robot_client=robot_client,
            state=state,
            on_nav_event=on_nav_event,
            battery=battery_cache,
        ),
        CancelNavigationTool(robot_client=robot_client, state=state),
        StopPatrolTool(robot_client=robot_client, state=state),
        ChargingDockTool(
            robot_client=robot_client,
            nav_state=state,
            battery=battery_cache,
        ),
    ]


__all__ = [
    "CancelNavigationInput",
    "CancelNavigationTool",
    "ChargingDockInput",
    "ChargingDockTool",
    "FollowWaypointsInput",
    "FollowWaypointsTool",
    "LocalizeCurrentLocationInput",
    "LocalizeCurrentLocationTool",
    "MarkReturnPointInput",
    "MarkReturnPointTool",
    "NavigateRelativeInput",
    "NavigateRelativeTool",
    "NavigateToReturnPointBlockingTool",
    "NavigateToReturnPointInput",
    "NavigateToLocationInput",
    "NavigateToLocationBlockingTool",
    "NavigateToLocationTool",
    "SaveCurrentLocationInput",
    "SaveCurrentLocationTool",
    "StopPatrolInput",
    "StopPatrolTool",
    "get_navigation_tools",
    "navigate_to_pose_blocking",
    "process_navigation_event",
]
