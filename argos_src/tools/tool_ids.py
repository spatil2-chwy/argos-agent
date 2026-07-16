"""Built-in capability-style tool identifiers."""

from __future__ import annotations

from typing import Iterable

ROBOT_FAMILY_UNITREE_GO2 = "unitree_go2"
ROBOT_FAMILY_SPOT = "spot"
SUPPORTED_ROBOT_FAMILIES = (
    ROBOT_FAMILY_UNITREE_GO2,
    ROBOT_FAMILY_SPOT,
)

GO2_ACTION_TOOL_NAMES = (
    "go2_damp",
    "go2_balance_stand",
    "go2_stop_move",
    "go2_sit",
    "go2_hello",
    "go2_stretch",
    "go2_content",
    "go2_dance1",
    "go2_dance2",
    "go2_scrape",
    "go2_front_jump",
    "go2_front_pounce",
    "go2_finger_heart",
    "go2_bow_down",
    "go2_look_up",
    "go2_left_tilt",
    "go2_right_tilt",
)

NAVIGATION_TOOL_NAMES = (
    "navigate_to_location",
    "navigate_to_location_blocking",
    "navigate_relative",
    "follow_waypoints",
    "cancel_navigation",
    "stop_patrol",
    "localize_current_location",
    "mark_return_point",
    "navigate_to_return_point_blocking",
    "save_current_location",
    "charging_dock",
)

SINGLETON_TOOL_NAMES = (
    "move_robot",
    "capture_scene",
    "enroll_visible_person",
    "resolve_employee_identity",
)

SPOT_SYSTEM_TOOL_NAMES = (
    "spot_claim",
    "spot_release",
    "spot_power_on",
    "spot_power_off",
)

SPOT_MOBILITY_TOOL_NAMES = (
    "spot_stand",
    "spot_sit",
    "spot_stop",
    "spot_self_right",
    "spot_rollover",
    "spot_set_stand_height",
    "spot_reset_body_pose",
)

MEMORY_TOOL_NAMES = (
    "search_memory_semantic",
)

TOOL_RUNTIME_BY_ID_BY_FAMILY: dict[str, dict[str, str]] = {
    "motion.move_robot": {
        ROBOT_FAMILY_UNITREE_GO2: "move_robot",
        ROBOT_FAMILY_SPOT: "move_robot",
    },
    "vision.capture_scene": {
        ROBOT_FAMILY_UNITREE_GO2: "capture_scene",
    },
    "identity.enroll_visible_person": {
        ROBOT_FAMILY_UNITREE_GO2: "enroll_visible_person",
    },
    "identity.resolve_employee_identity": {
        ROBOT_FAMILY_UNITREE_GO2: "resolve_employee_identity",
    },
    "navigation.navigate_to_location": {
        ROBOT_FAMILY_UNITREE_GO2: "navigate_to_location",
    },
    "navigation.navigate_to_location_blocking": {
        ROBOT_FAMILY_UNITREE_GO2: "navigate_to_location_blocking",
    },
    "navigation.navigate_relative": {
        ROBOT_FAMILY_UNITREE_GO2: "navigate_relative",
    },
    "navigation.follow_waypoints": {
        ROBOT_FAMILY_UNITREE_GO2: "follow_waypoints",
    },
    "navigation.cancel": {
        ROBOT_FAMILY_UNITREE_GO2: "cancel_navigation",
    },
    "navigation.stop_patrol": {
        ROBOT_FAMILY_UNITREE_GO2: "stop_patrol",
    },
    "navigation.localize_current_location": {
        ROBOT_FAMILY_UNITREE_GO2: "localize_current_location",
    },
    "navigation.mark_return_point": {
        ROBOT_FAMILY_UNITREE_GO2: "mark_return_point",
    },
    "navigation.navigate_to_return_point_blocking": {
        ROBOT_FAMILY_UNITREE_GO2: "navigate_to_return_point_blocking",
    },
    "navigation.save_current_location": {
        ROBOT_FAMILY_UNITREE_GO2: "save_current_location",
    },
    "dock.charging": {
        ROBOT_FAMILY_UNITREE_GO2: "charging_dock",
    },
    "posture.rest": {
        ROBOT_FAMILY_UNITREE_GO2: "go2_damp",
    },
    "posture.stand": {
        ROBOT_FAMILY_UNITREE_GO2: "go2_balance_stand",
        ROBOT_FAMILY_SPOT: "spot_stand",
    },
    "posture.stop": {
        ROBOT_FAMILY_UNITREE_GO2: "go2_stop_move",
        ROBOT_FAMILY_SPOT: "spot_stop",
    },
    "posture.sit": {
        ROBOT_FAMILY_UNITREE_GO2: "go2_sit",
        ROBOT_FAMILY_SPOT: "spot_sit",
    },
    "posture.self_right": {
        ROBOT_FAMILY_SPOT: "spot_self_right",
    },
    "posture.rollover": {
        ROBOT_FAMILY_SPOT: "spot_rollover",
    },
    "posture.set_stand_height": {
        ROBOT_FAMILY_SPOT: "spot_set_stand_height",
    },
    "posture.reset_body_pose": {
        ROBOT_FAMILY_SPOT: "spot_reset_body_pose",
    },
    "spot.system.claim": {
        ROBOT_FAMILY_SPOT: "spot_claim",
    },
    "spot.system.release": {
        ROBOT_FAMILY_SPOT: "spot_release",
    },
    "spot.system.power_on": {
        ROBOT_FAMILY_SPOT: "spot_power_on",
    },
    "spot.system.power_off": {
        ROBOT_FAMILY_SPOT: "spot_power_off",
    },
    "memory.search_semantic": {
        ROBOT_FAMILY_UNITREE_GO2: "search_memory_semantic",
        ROBOT_FAMILY_SPOT: "search_memory_semantic",
    },
}

for _action_name in GO2_ACTION_TOOL_NAMES:
    _short_name = _action_name.removeprefix("go2_")
    TOOL_RUNTIME_BY_ID_BY_FAMILY[f"embodiment.unitree_go2.{_short_name}"] = {
        ROBOT_FAMILY_UNITREE_GO2: _action_name,
    }

BUILT_IN_TOOL_NAMES = tuple(sorted(TOOL_RUNTIME_BY_ID_BY_FAMILY))


def _validate_robot_family(robot_family: str | None) -> str:
    raw_family = str(robot_family or "").strip()
    if raw_family in SUPPORTED_ROBOT_FAMILIES:
        return raw_family
    raise ValueError(
        "robot_family is required and must be one of: "
        + ", ".join(SUPPORTED_ROBOT_FAMILIES)
    )


def resolve_builtin_tool_name(name: str, *, robot_family: str | None = None) -> str:
    """Resolve one capability-style tool id to the current runtime tool name."""
    raw_name = str(name or "").strip()
    by_family = TOOL_RUNTIME_BY_ID_BY_FAMILY.get(raw_name)
    if by_family is None:
        raise ValueError(f"Unknown built-in tool id: {raw_name}")
    family = _validate_robot_family(robot_family)
    runtime_name = by_family.get(family)
    if runtime_name is None:
        raise ValueError(
            f"Built-in tool id '{raw_name}' is not available for robot family '{family}'."
        )
    return runtime_name


def resolve_builtin_tool_names(
    enabled_tool_ids: Iterable[str],
    *,
    robot_family: str | None = None,
) -> tuple[str, ...]:
    """Resolve built-in capability-style tool ids to runtime tool names."""
    resolved: list[str] = []
    seen: set[str] = set()
    errors: list[str] = []

    for raw_name in enabled_tool_ids:
        try:
            runtime_name = resolve_builtin_tool_name(
                str(raw_name),
                robot_family=robot_family,
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if runtime_name in seen:
            continue
        seen.add(runtime_name)
        resolved.append(runtime_name)

    if errors:
        raise ValueError("; ".join(errors))
    return tuple(resolved)


def required_capability_ids_for_tool_id(
    name: str,
    *,
    robot_family: str | None = None,
) -> tuple[str, ...]:
    """Return provider capabilities needed to enable one built-in tool id."""
    raw_name = str(name or "").strip()
    runtime_name = resolve_builtin_tool_name(raw_name, robot_family=robot_family)

    if raw_name.startswith("motion."):
        return ("motion.velocity",)
    if raw_name.startswith("vision."):
        return ("camera.rgb",)
    if raw_name.startswith("identity.enroll"):
        return ("camera.rgb",)
    if raw_name.startswith("identity.resolve"):
        return ()
    if raw_name.startswith("navigation."):
        if runtime_name in {
            "navigate_relative",
            "localize_current_location",
            "mark_return_point",
            "navigate_to_return_point_blocking",
            "save_current_location",
        }:
            return ("navigation.goal", "transform.lookup")
        return ("navigation.goal",)
    if raw_name.startswith("dock."):
        return ("dock.charging",)
    if raw_name.startswith("posture."):
        return ("posture.command",)
    if raw_name.startswith("embodiment."):
        return ("embodiment.action",)
    if raw_name.startswith("spot.system."):
        return ("posture.command",)
    if raw_name.startswith("memory."):
        return ()
    return ()


__all__ = [
    "BUILT_IN_TOOL_NAMES",
    "GO2_ACTION_TOOL_NAMES",
    "MEMORY_TOOL_NAMES",
    "NAVIGATION_TOOL_NAMES",
    "ROBOT_FAMILY_SPOT",
    "ROBOT_FAMILY_UNITREE_GO2",
    "SINGLETON_TOOL_NAMES",
    "SPOT_MOBILITY_TOOL_NAMES",
    "SPOT_SYSTEM_TOOL_NAMES",
    "SUPPORTED_ROBOT_FAMILIES",
    "TOOL_RUNTIME_BY_ID_BY_FAMILY",
    "required_capability_ids_for_tool_id",
    "resolve_builtin_tool_name",
    "resolve_builtin_tool_names",
]
