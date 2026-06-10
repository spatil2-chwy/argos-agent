"""One LangChain tool per Go2 provider action capability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Type

from pydantic import BaseModel, Field

from argos_src.observability.observability import (
    LatencyLogger,
    get_request_context,
    perf_now,
)
from argos_src.tools.base import BaseTool
from argos_src.tools.common.tool_response import tool_response_json

from .catalog import GO2_ACTIONS

API_ID_TO_POSTURE = {
    1005: "damp_rest",
    1004: "standing",
    1006: "standing",
    1009: "sitting",
    1016: "standing",
    1017: "standing",
    1020: "standing",
    1022: "standing",
    1023: "standing",
    1029: "standing",
    1030: "standing",
    1031: "standing",
    1036: "standing",
    1007: "standing",
    1002: "standing",
    1003: "standing",
    1028: "standing",
}


@dataclass(frozen=True)
class Go2ActionCommandSpec:
    api_id: int
    parameter: dict[str, Any]
    priority: int = 0


_ENTER_POSE_MODE_COMMANDS = (
    Go2ActionCommandSpec(api_id=1003, parameter={}, priority=0),
    Go2ActionCommandSpec(api_id=1028, parameter={"data": True}, priority=0),
)

_EXIT_POSE_MODE_COMMANDS = (
    Go2ActionCommandSpec(api_id=1028, parameter={"data": False}, priority=0),
    Go2ActionCommandSpec(api_id=1002, parameter={}, priority=0),
)

_POSE_ACTION_COMMAND_BY_TOOL_NAME = {
    "go2_bow_down": Go2ActionCommandSpec(
        api_id=1007,
        parameter={"x": 0.0, "y": 0.30, "z": 0.0},
        priority=1,
    ),
    "go2_look_up": Go2ActionCommandSpec(
        api_id=1007,
        parameter={"x": 0.0, "y": -0.20, "z": 0.0},
        priority=1,
    ),
    "go2_left_tilt": Go2ActionCommandSpec(
        api_id=1007,
        parameter={"x": 0.25, "y": 0.0, "z": 0.0},
        priority=1,
    ),
    "go2_right_tilt": Go2ActionCommandSpec(
        api_id=1007,
        parameter={"x": -0.25, "y": 0.0, "z": 0.0},
        priority=1,
    ),
}


class _Go2ActionToolInput(BaseModel):
    """No input required; the action is fixed per tool."""

    pass


class Go2ActionTool(BaseTool):
    """Publishes to Go2 /webrtc_req topic with a fixed api_id. One tool instance per action."""

    name: str = ""
    description: str = ""
    robot_client: Any = Field(exclude=True)
    api_id: int = 0
    args_schema: Type[BaseModel] = _Go2ActionToolInput
    latency_logger: LatencyLogger = Field(
        default_factory=lambda: LatencyLogger("action"),
        exclude=True,
    )

    class Config:
        arbitrary_types_allowed = True

    def _run(self) -> str:
        if not self.api_id:
            return tool_response_json(
                success=False,
                status="error",
                message="api_id not set",
                robot_state_after="unknown",
            )
        try:
            commands = self._commands_for_tool()
            self._publish_commands(commands)
            req_ctx = get_request_context()
            transcript_perf = req_ctx.get("transcript_perf_s")
            if transcript_perf:
                self.latency_logger.timing(
                    "tool_dispatch_s",
                    perf_now() - transcript_perf,
                    tool=self.name,
                    req_id=req_ctx.get("req_id"),
                )
            robot_state_after = API_ID_TO_POSTURE.get(commands[-1].api_id, "unchanged")
            return tool_response_json(
                success=True,
                status="completed",
                message=f"{self.name} completed.",
                robot_state_after=robot_state_after,
                api_id=self.api_id,
                command_count=len(commands),
            )
        except Exception as exc:
            return tool_response_json(
                success=False,
                status="error",
                message=str(exc),
                robot_state_after="unknown",
                api_id=self.api_id,
            )

    def _commands_for_tool(self) -> tuple[Go2ActionCommandSpec, ...]:
        pose_command = _POSE_ACTION_COMMAND_BY_TOOL_NAME.get(self.name)
        if pose_command is not None:
            return _ENTER_POSE_MODE_COMMANDS + (pose_command,) + _EXIT_POSE_MODE_COMMANDS
        return (Go2ActionCommandSpec(api_id=self.api_id, parameter={}, priority=0),)

    def _publish_commands(self, commands: tuple[Go2ActionCommandSpec, ...]) -> None:
        for command in commands:
            self.robot_client.perform_go2_action(
                api_id=int(command.api_id),
                parameter=dict(command.parameter),
                priority=int(command.priority),
            )


def get_go2_action_tools(
    robot_client: Any,
    enabled_tool_names: Iterable[str] | None = None,
) -> List[BaseTool]:
    """Build one LangChain tool per selected Go2 action."""
    allowed = (
        None
        if enabled_tool_names is None
        else {str(name) for name in enabled_tool_names}
    )
    tools: List[BaseTool] = []
    for tool_name, description, api_id in GO2_ACTIONS:
        if allowed is not None and tool_name not in allowed:
            continue
        tools.append(
            Go2ActionTool(
                name=tool_name,
                description=description,
                robot_client=robot_client,
                api_id=api_id,
            )
        )
    return tools
