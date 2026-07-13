"""Built-in tool registry for Go2 scenario profiles."""

from __future__ import annotations

from typing import Iterable

from argos_src.nav_support.locations import LocationStore, NavigationState
from argos_src.tools.base import BaseTool
from argos_src.tools.common.knowledge.whoami_query import build_knowledge_tool
from argos_src.tools.common.memory import get_memory_query_tools
from argos_src.tools.spot import get_spot_tools
from argos_src.tools.tool_ids import (
    MEMORY_TOOL_NAMES,
    NAVIGATION_TOOL_NAMES,
    ROBOT_FAMILY_SPOT,
    resolve_builtin_tool_names,
)
from argos_src.tools.unitree_go2.actions import GO2_ACTION_TOOL_NAMES, get_go2_action_tools
from argos_src.tools.unitree_go2.locomotion.move_robot import get_move_robot_tool
from argos_src.tools.unitree_go2.navigation import get_navigation_tools
from argos_src.tools.unitree_go2.vision.capture_scene import get_capture_scene_tool
from argos_src.tools.unitree_go2.vision.enroll_visible_person import (
    get_enroll_visible_person_tool,
)
from argos_src.tools.unitree_go2.vision.resolve_employee_identity import (
    get_resolve_employee_identity_tool,
)


def build_builtin_tools(
    *,
    robot_family: str,
    enabled_tool_ids: Iterable[str],
    robot_client,
    face_service,
    identity_memory_client,
    location_store: LocationStore,
    nav_state: NavigationState,
    on_nav_event,
    battery_cache,
    default_camera_resource: str,
    display_runtime=None,
    memory_provider=None,
) -> list[BaseTool]:
    """Build selected built-in tools for the selected robot family."""
    requested = resolve_builtin_tool_names(
        enabled_tool_ids,
        robot_family=robot_family,
    )
    requested_set = set(requested)

    tools: list[BaseTool] = []

    if robot_family == ROBOT_FAMILY_SPOT:
        tools.extend(
            get_spot_tools(
                robot_client=robot_client,
                runtime_tool_names=requested,
            )
        )
        memory_names = [name for name in MEMORY_TOOL_NAMES if name in requested_set]
        if memory_names and memory_provider is not None:
            tools.extend(
                get_memory_query_tools(
                    memory_provider,
                    runtime_tool_names=memory_names,
                )
            )
        return tools

    action_names = [name for name in GO2_ACTION_TOOL_NAMES if name in requested_set]
    tools.extend(get_go2_action_tools(robot_client, runtime_tool_names=action_names))

    if "move_robot" in requested_set:
        tools.append(get_move_robot_tool(robot_client))
    if "capture_scene" in requested_set:
        tools.append(
            get_capture_scene_tool(
                face_service,
                default_camera_resource=default_camera_resource,
            )
        )
    if "enroll_visible_person" in requested_set and face_service is not None:
        tools.append(
            get_enroll_visible_person_tool(
                face_service,
                identity_memory_client=identity_memory_client,
                default_camera_resource=default_camera_resource,
                display_runtime=display_runtime,
            )
        )
    if (
        "resolve_employee_identity" in requested_set
        and identity_memory_client is not None
    ):
        tools.append(
            get_resolve_employee_identity_tool(identity_memory_client)
        )

    memory_names = [name for name in MEMORY_TOOL_NAMES if name in requested_set]
    if memory_names and memory_provider is not None:
        tools.extend(
            get_memory_query_tools(
                memory_provider,
                runtime_tool_names=memory_names,
            )
        )

    nav_requested = [name for name in NAVIGATION_TOOL_NAMES if name in requested_set]
    if nav_requested:
        navigation_tools = get_navigation_tools(
            robot_client,
            location_store=location_store,
            state=nav_state,
            on_nav_event=on_nav_event,
            battery_cache=battery_cache,
        )
        tools.extend(
            tool for tool in navigation_tools if getattr(tool, "name", "") in requested_set
        )

    return tools


def build_knowledge_tools(entries) -> list[BaseTool]:
    """Build configured knowledge-base tools from profile config."""
    tools: list[BaseTool] = []
    for entry in entries:
        tool = build_knowledge_tool(entry)
        if tool is not None:
            tools.append(tool)
    return tools
