"""Spot-specific built-in tool registry."""

from __future__ import annotations

from typing import Iterable

from argos_src.tools.base import BaseTool
from argos_src.tools.spot.locomotion.move_robot import get_move_robot_tool
from argos_src.tools.spot.mobility.toolset import (
    SPOT_MOBILITY_TOOL_NAMES,
    get_spot_mobility_tools,
)
from argos_src.tools.spot.system.toolset import (
    SPOT_SYSTEM_TOOL_NAMES,
    get_spot_system_tools,
)


def get_spot_tools(*, robot_client, runtime_tool_names: Iterable[str]) -> list[BaseTool]:
    """Build selected Spot tools from the runtime tool names."""
    requested = {str(name) for name in runtime_tool_names}
    tools: list[BaseTool] = []

    system_names = [name for name in SPOT_SYSTEM_TOOL_NAMES if name in requested]
    tools.extend(get_spot_system_tools(robot_client, runtime_tool_names=system_names))

    mobility_names = [name for name in SPOT_MOBILITY_TOOL_NAMES if name in requested]
    tools.extend(
        get_spot_mobility_tools(robot_client, runtime_tool_names=mobility_names)
    )

    if "move_robot" in requested:
        tools.append(get_move_robot_tool(robot_client))

    return tools
