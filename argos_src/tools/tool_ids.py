"""Built-in tool identifiers for robot-family-specific scenario profiles."""

from __future__ import annotations

from collections import defaultdict
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
    "get_current_location",
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

LEGACY_BUILT_IN_TOOL_NAMES = tuple(
    list(GO2_ACTION_TOOL_NAMES) + list(NAVIGATION_TOOL_NAMES) + list(SINGLETON_TOOL_NAMES)
)

RUNTIME_TOOL_NAME_BY_CANONICAL = {
    **{f"unitree_go2.actions.{name}": name for name in GO2_ACTION_TOOL_NAMES},
    "unitree_go2.locomotion.move_robot": "move_robot",
    "unitree_go2.vision.capture_scene": "capture_scene",
    "unitree_go2.vision.enroll_visible_person": "enroll_visible_person",
    "unitree_go2.vision.resolve_employee_identity": "resolve_employee_identity",
    "unitree_go2.navigation.navigate_to_location": "navigate_to_location",
    "unitree_go2.navigation.navigate_to_location_blocking": "navigate_to_location_blocking",
    "unitree_go2.navigation.navigate_relative": "navigate_relative",
    "unitree_go2.navigation.follow_waypoints": "follow_waypoints",
    "unitree_go2.navigation.cancel_navigation": "cancel_navigation",
    "unitree_go2.navigation.stop_patrol": "stop_patrol",
    "unitree_go2.navigation.get_current_location": "get_current_location",
    "unitree_go2.navigation.charging_dock": "charging_dock",
    "spot.system.claim": "spot_claim",
    "spot.system.release": "spot_release",
    "spot.system.power_on": "spot_power_on",
    "spot.system.power_off": "spot_power_off",
    "spot.mobility.stand": "spot_stand",
    "spot.mobility.sit": "spot_sit",
    "spot.mobility.stop": "spot_stop",
    "spot.mobility.self_right": "spot_self_right",
    "spot.mobility.rollover": "spot_rollover",
    "spot.mobility.set_stand_height": "spot_set_stand_height",
    "spot.mobility.reset_body_pose": "spot_reset_body_pose",
    "spot.locomotion.move_robot": "move_robot",
}

ROBOT_FAMILY_BY_CANONICAL = {
    canonical_name: (
        ROBOT_FAMILY_SPOT if canonical_name.startswith("spot.") else ROBOT_FAMILY_UNITREE_GO2
    )
    for canonical_name in RUNTIME_TOOL_NAME_BY_CANONICAL
}

CANONICAL_TOOL_NAME_BY_LEGACY: dict[str, str] = {}
TOOL_NAME_ALIASES: dict[str, str] = {}

_canonical_by_runtime_name: dict[str, set[str]] = defaultdict(set)
for canonical_name, runtime_name in RUNTIME_TOOL_NAME_BY_CANONICAL.items():
    TOOL_NAME_ALIASES[canonical_name] = runtime_name
    _canonical_by_runtime_name[runtime_name].add(canonical_name)

for legacy_name in LEGACY_BUILT_IN_TOOL_NAMES:
    if legacy_name in CANONICAL_TOOL_NAME_BY_LEGACY:
        continue
    if legacy_name in GO2_ACTION_TOOL_NAMES:
        canonical_name = f"unitree_go2.actions.{legacy_name}"
    elif legacy_name == "move_robot":
        canonical_name = "unitree_go2.locomotion.move_robot"
    elif legacy_name == "capture_scene":
        canonical_name = "unitree_go2.vision.capture_scene"
    elif legacy_name == "enroll_visible_person":
        canonical_name = "unitree_go2.vision.enroll_visible_person"
    elif legacy_name == "resolve_employee_identity":
        canonical_name = "unitree_go2.vision.resolve_employee_identity"
    else:
        canonical_name = f"unitree_go2.navigation.{legacy_name}"
    CANONICAL_TOOL_NAME_BY_LEGACY[legacy_name] = canonical_name
    TOOL_NAME_ALIASES[legacy_name] = RUNTIME_TOOL_NAME_BY_CANONICAL[canonical_name]

AMBIGUOUS_LEGACY_TOOL_NAMES = frozenset(
    sorted(
        runtime_name
        for runtime_name, canonical_names in _canonical_by_runtime_name.items()
        if len(canonical_names) > 1 and runtime_name not in CANONICAL_TOOL_NAME_BY_LEGACY
    )
)

BUILT_IN_TOOL_NAMES = tuple(sorted(TOOL_NAME_ALIASES))


def _validate_robot_family(robot_family: str | None) -> str | None:
    if robot_family is None:
        return None
    raw_family = str(robot_family).strip()
    if raw_family in SUPPORTED_ROBOT_FAMILIES:
        return raw_family
    raise ValueError(
        "Unsupported robot family: "
        f"{robot_family}. Expected one of: {', '.join(SUPPORTED_ROBOT_FAMILIES)}"
    )


def canonical_tool_name_for(name: str) -> str:
    """Resolve an input tool id to its canonical dotted name."""
    raw_name = str(name).strip()
    if raw_name in RUNTIME_TOOL_NAME_BY_CANONICAL:
        return raw_name
    canonical = CANONICAL_TOOL_NAME_BY_LEGACY.get(raw_name)
    if canonical is not None:
        return canonical
    if raw_name in AMBIGUOUS_LEGACY_TOOL_NAMES:
        canonical_names = sorted(_canonical_by_runtime_name[raw_name])
        raise ValueError(
            f"Ambiguous built-in tool name '{raw_name}'. Use one of: "
            + ", ".join(canonical_names)
        )
    raise ValueError(f"Unknown built-in tool name: {raw_name}")


def resolve_builtin_tool_name(name: str, *, robot_family: str | None = None) -> str:
    """Resolve one built-in tool id to the current runtime tool name."""
    family = _validate_robot_family(robot_family)
    canonical_name = canonical_tool_name_for(name)
    canonical_family = ROBOT_FAMILY_BY_CANONICAL[canonical_name]
    if family is not None and canonical_family != family:
        raise ValueError(
            f"Built-in tool '{name}' belongs to robot family '{canonical_family}', "
            f"not '{family}'."
        )
    return RUNTIME_TOOL_NAME_BY_CANONICAL[canonical_name]


def resolve_builtin_tool_names(
    enabled_tool_names: Iterable[str],
    *,
    robot_family: str | None = None,
) -> tuple[str, ...]:
    """Resolve built-in tool ids to the current runtime tool names."""
    resolved: list[str] = []
    seen: set[str] = set()
    errors: list[str] = []

    for raw_name in enabled_tool_names:
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
