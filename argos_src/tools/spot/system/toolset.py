"""Spot lease and power control tools backed by RobotClient capabilities."""

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


class NoArgsInput(BaseModel):
    pass


SPOT_TRIGGER_TOOL_SPECS = (
    (
        "spot_claim",
        "Claim Spot's control lease so this session can command the robot.",
        "claim",
        "claimed",
    ),
    (
        "spot_release",
        "Release Spot's control lease.",
        "release",
        "released",
    ),
    (
        "spot_power_on",
        "Turn on Spot motor power.",
        "power_on",
        "powered_on",
    ),
    (
        "spot_power_off",
        "Turn off Spot motor power.",
        "power_off",
        "powered_off",
    ),
)

SPOT_SYSTEM_TOOL_NAMES = tuple(spec[0] for spec in SPOT_TRIGGER_TOOL_SPECS)


class SpotCommandTool(BaseTool):
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

    def _run(self) -> str:
        try:
            result = self.robot_client.perform_spot_command(self.command)
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
            )
        except Exception as exc:
            return tool_response_json(
                success=False,
                status="error",
                message=str(exc),
                robot_state_after="unknown",
                command=self.command,
            )


def get_spot_system_tools(
    robot_client: Any,
    enabled_tool_names: Iterable[str] | None = None,
) -> List[BaseTool]:
    allowed = None if enabled_tool_names is None else {str(name) for name in enabled_tool_names}
    tools: List[BaseTool] = []
    for tool_name, description, command, robot_state_after in SPOT_TRIGGER_TOOL_SPECS:
        if allowed is not None and tool_name not in allowed:
            continue
        tools.append(
            SpotCommandTool(
                name=tool_name,
                description=description,
                robot_client=robot_client,
                command=command,
                robot_state_after=robot_state_after,
            )
        )
    return tools
