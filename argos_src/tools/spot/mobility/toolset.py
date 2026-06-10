"""Spot posture and mobility tools backed by RobotClient capabilities."""

from __future__ import annotations

from typing import Any, Iterable, List, Type

from pydantic import BaseModel, Field

from argos_src.observability.observability import (
    LatencyLogger,
    get_request_context,
    perf_now,
)
from argos_src.tools.base import BaseTool
from argos_src.tools.common.tool_response import tool_response_json
from argos_src.tools.spot.system.toolset import NoArgsInput


SPOT_TRIGGER_MOBILITY_SPECS = (
    (
        "spot_stand",
        "Stand Spot up into its normal neutral posture.",
        "stand",
        "standing",
    ),
    (
        "spot_sit",
        "Sit Spot down into a calm lower posture and hold still.",
        "sit",
        "sitting",
    ),
    (
        "spot_stop",
        "Immediately stop Spot motion.",
        "stop",
        "standing",
    ),
    (
        "spot_self_right",
        "Command Spot to self-right or recover posture.",
        "self_right",
        "standing",
    ),
    (
        "spot_rollover",
        "Command Spot into the rollover battery-change pose.",
        "rollover",
        "rollover",
    ),
)

SPOT_MOBILITY_TOOL_NAMES = tuple(
    [spec[0] for spec in SPOT_TRIGGER_MOBILITY_SPECS]
    + ["spot_set_stand_height", "spot_reset_body_pose"]
)


class SpotCapabilityTool(BaseTool):
    name: str = ""
    description: str = ""
    robot_client: Any = Field(exclude=True)
    command: str = ""
    robot_state_after: str = "unchanged"
    args_schema: Type[BaseModel] = NoArgsInput
    latency_logger: LatencyLogger = Field(
        default_factory=lambda: LatencyLogger("action"),
        exclude=True,
    )

    class Config:
        arbitrary_types_allowed = True

    def _call(self, params: dict[str, Any] | None = None) -> str:
        try:
            result = self.robot_client.perform_spot_command(self.command, params or {})
            req_ctx = get_request_context()
            transcript_perf = req_ctx.get("transcript_perf_s")
            if transcript_perf:
                self.latency_logger.timing(
                    "tool_dispatch_s",
                    perf_now() - transcript_perf,
                    tool=self.name,
                    req_id=req_ctx.get("req_id"),
                )
            success = bool(result.get("success", result.get("ok", True)))
            message = str(result.get("message", "") or "").strip()
            return tool_response_json(
                success=success,
                status="completed" if success else "error",
                message=message or f"{self.name} completed.",
                robot_state_after=self.robot_state_after if success else "unknown",
                command=self.command,
                **dict(params or {}),
            )
        except Exception as exc:
            return tool_response_json(
                success=False,
                status="error",
                message=str(exc),
                robot_state_after="unknown",
                command=self.command,
                **dict(params or {}),
            )

    def _run(self) -> str:
        return self._call()


class SpotSetStandHeightInput(BaseModel):
    height: float = Field(
        ...,
        description="Relative stand height adjustment in meters.",
    )


class SpotSetStandHeightTool(SpotCapabilityTool):
    name: str = "spot_set_stand_height"
    description: str = "Adjust Spot standing height relative to neutral."
    command: str = "set_stand_height"
    robot_state_after: str = "standing"
    args_schema: Type[BaseModel] = SpotSetStandHeightInput

    def _run(self, height: float) -> str:
        return self._call({"height": float(height)})


class SpotResetBodyPoseTool(SpotCapabilityTool):
    name: str = "spot_reset_body_pose"
    description: str = "Reset Spot body pose to a neutral centered standing pose."
    command: str = "reset_body_pose"
    robot_state_after: str = "standing"


def get_spot_mobility_tools(
    robot_client: Any,
    enabled_tool_names: Iterable[str] | None = None,
) -> List[BaseTool]:
    allowed = None if enabled_tool_names is None else {str(name) for name in enabled_tool_names}
    tools: List[BaseTool] = []
    for tool_name, description, command, robot_state_after in SPOT_TRIGGER_MOBILITY_SPECS:
        if allowed is not None and tool_name not in allowed:
            continue
        tools.append(
            SpotCapabilityTool(
                name=tool_name,
                description=description,
                robot_client=robot_client,
                command=command,
                robot_state_after=robot_state_after,
            )
        )
    if allowed is None or "spot_set_stand_height" in allowed:
        tools.append(SpotSetStandHeightTool(robot_client=robot_client))
    if allowed is None or "spot_reset_body_pose" in allowed:
        tools.append(SpotResetBodyPoseTool(robot_client=robot_client))
    return tools
