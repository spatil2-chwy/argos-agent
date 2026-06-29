"""Lazy exports for Go2 tool helpers and registries."""

from __future__ import annotations

from importlib import import_module

from .tool_ids import (
    BUILT_IN_TOOL_NAMES,
    NAVIGATION_TOOL_NAMES,
    ROBOT_FAMILY_SPOT,
    ROBOT_FAMILY_UNITREE_GO2,
    SPOT_MOBILITY_TOOL_NAMES,
    SPOT_SYSTEM_TOOL_NAMES,
    SUPPORTED_ROBOT_FAMILIES,
    MEMORY_TOOL_NAMES,
    resolve_builtin_tool_name,
    resolve_builtin_tool_names,
)

__all__ = [
    "BUILT_IN_TOOL_NAMES",
    "NAVIGATION_TOOL_NAMES",
    "ROBOT_FAMILY_SPOT",
    "ROBOT_FAMILY_UNITREE_GO2",
    "SPOT_MOBILITY_TOOL_NAMES",
    "SPOT_SYSTEM_TOOL_NAMES",
    "SUPPORTED_ROBOT_FAMILIES",
    "build_builtin_tools",
    "build_knowledge_tools",
    "get_capture_scene_tool",
    "get_chewy_knowledge_tool",
    "get_enroll_visible_person_tool",
    "get_go2_action_tools",
    "get_memory_query_tools",
    "get_move_robot_tool",
    "get_resolve_employee_identity_tool",
    "get_spot_tools",
    "MEMORY_TOOL_NAMES",
    "resolve_builtin_tool_name",
    "resolve_builtin_tool_names",
]


_LAZY_EXPORTS = {
    "build_builtin_tools": ("argos_src.tools.registry", "build_builtin_tools"),
    "build_knowledge_tools": ("argos_src.tools.registry", "build_knowledge_tools"),
    "get_capture_scene_tool": (
        "argos_src.tools.unitree_go2.vision.capture_scene",
        "get_capture_scene_tool",
    ),
    "get_chewy_knowledge_tool": (
        "argos_src.tools.common.knowledge",
        "get_chewy_knowledge_tool",
    ),
    "get_enroll_visible_person_tool": (
        "argos_src.tools.unitree_go2.vision.enroll_visible_person",
        "get_enroll_visible_person_tool",
    ),
    "get_go2_action_tools": (
        "argos_src.tools.unitree_go2.actions",
        "get_go2_action_tools",
    ),
    "get_memory_query_tools": (
        "argos_src.tools.common.memory",
        "get_memory_query_tools",
    ),
    "get_move_robot_tool": (
        "argos_src.tools.unitree_go2.locomotion.move_robot",
        "get_move_robot_tool",
    ),
    "get_resolve_employee_identity_tool": (
        "argos_src.tools.unitree_go2.vision.resolve_employee_identity",
        "get_resolve_employee_identity_tool",
    ),
    "get_spot_tools": ("argos_src.tools.spot.registry", "get_spot_tools"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
