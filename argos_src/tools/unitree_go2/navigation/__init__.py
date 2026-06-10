"""Lazy exports for Unitree Go2 navigation tools."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "CancelNavigationInput",
    "CancelNavigationTool",
    "ChargingDockInput",
    "ChargingDockTool",
    "FollowWaypointsInput",
    "FollowWaypointsTool",
    "GetCurrentLocationInput",
    "GetCurrentLocationTool",
    "NavigateRelativeInput",
    "NavigateRelativeTool",
    "NavigateToLocationInput",
    "NavigateToLocationBlockingTool",
    "NavigateToLocationTool",
    "StopPatrolInput",
    "StopPatrolTool",
    "get_navigation_tools",
    "navigate_to_pose_blocking",
    "process_navigation_event",
]

_LAZY_EXPORTS = {
    "CancelNavigationInput": (
        "argos_src.tools.unitree_go2.navigation.cancel_navigation",
        "CancelNavigationInput",
    ),
    "CancelNavigationTool": (
        "argos_src.tools.unitree_go2.navigation.cancel_navigation",
        "CancelNavigationTool",
    ),
    "ChargingDockInput": (
        "argos_src.tools.unitree_go2.navigation.charging_dock",
        "ChargingDockInput",
    ),
    "ChargingDockTool": (
        "argos_src.tools.unitree_go2.navigation.charging_dock",
        "ChargingDockTool",
    ),
    "FollowWaypointsInput": (
        "argos_src.tools.unitree_go2.navigation.follow_waypoints",
        "FollowWaypointsInput",
    ),
    "FollowWaypointsTool": (
        "argos_src.tools.unitree_go2.navigation.follow_waypoints",
        "FollowWaypointsTool",
    ),
    "GetCurrentLocationInput": (
        "argos_src.tools.unitree_go2.navigation.get_current_location",
        "GetCurrentLocationInput",
    ),
    "GetCurrentLocationTool": (
        "argos_src.tools.unitree_go2.navigation.get_current_location",
        "GetCurrentLocationTool",
    ),
    "NavigateRelativeInput": (
        "argos_src.tools.unitree_go2.navigation.navigate_relative",
        "NavigateRelativeInput",
    ),
    "NavigateRelativeTool": (
        "argos_src.tools.unitree_go2.navigation.navigate_relative",
        "NavigateRelativeTool",
    ),
    "NavigateToLocationInput": (
        "argos_src.tools.unitree_go2.navigation.navigate_to_location",
        "NavigateToLocationInput",
    ),
    "NavigateToLocationTool": (
        "argos_src.tools.unitree_go2.navigation.navigate_to_location",
        "NavigateToLocationTool",
    ),
    "NavigateToLocationBlockingTool": (
        "argos_src.tools.unitree_go2.navigation.navigate_to_location_blocking",
        "NavigateToLocationBlockingTool",
    ),
    "StopPatrolInput": (
        "argos_src.tools.unitree_go2.navigation.stop_patrol",
        "StopPatrolInput",
    ),
    "StopPatrolTool": (
        "argos_src.tools.unitree_go2.navigation.stop_patrol",
        "StopPatrolTool",
    ),
    "get_navigation_tools": (
        "argos_src.tools.unitree_go2.navigation.toolset",
        "get_navigation_tools",
    ),
    "navigate_to_pose_blocking": (
        "argos_src.tools.unitree_go2.navigation.toolset",
        "navigate_to_pose_blocking",
    ),
    "process_navigation_event": (
        "argos_src.tools.unitree_go2.navigation.toolset",
        "process_navigation_event",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
