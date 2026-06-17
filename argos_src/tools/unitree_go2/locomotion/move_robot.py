"""Tool to move the Go2 robot through provider-backed motion capabilities."""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel, Field

from argos_src.observability.observability import (
    LatencyLogger,
    get_request_context,
    perf_now,
)
from argos_src.tools.base import BaseTool
from argos_src.tools.common.tool_response import tool_response_json

CMD_VEL_HZ = 10
CMD_VEL_MAX_DURATION = 10.0


class MoveRobotInput(BaseModel):
    linear_x: float = Field(
        default=0.0,
        description="Forward (+) / backward (-) velocity in m/s. Gentle walk: 0.3, normal: 0.5, run: 0.8. Max 1.0.",
    )
    linear_y: float = Field(
        default=0.0,
        description="Strafe left (+) / right (-) in m/s. Typically 0.0–0.3.",
    )
    angular_z: float = Field(
        default=0.0,
        description=(
            "Rotate counter-clockwise (+) / clockwise (-) in rad/s. "
            "Gentle turn: 0.5, sharp turn: 1.0, spin: 1.5. "
        ),
    )
    duration: float = Field(
        default=0.5,
        description=(
            "How long to move in seconds (0.1–10.0). "
            "Quick nudge: 0.3, walk across room: 1.5–2.0, "
            "full spin: about 4.0."
        ),
    )


class MoveRobotTool(BaseTool):
    """Move with short velocity commands for the requested duration, then stop."""

    name: str = "move_robot"
    description: str = (
        "Move the robot with a short local velocity command: step forward or back, strafe, turn, or spin. "
        "Use this for nearby repositioning and expressive motion, not for mapped navigation to named places or meter-accurate travel. "
        "Set linear_x for forward/backward speed, linear_y for strafing, angular_z for turning/spinning, and duration in seconds. "
        "For a full spin, use about 4 seconds at spin speed. "
        "Use slower speeds for casual movement (walk, come here) and faster for energetic movement (run to me, charge). "
        "Always sends a stop command after the duration so the robot does not keep moving."
    )
    args_schema: Type[BaseModel] = MoveRobotInput
    robot_client: Any = Field(exclude=True)
    latency_logger: LatencyLogger = Field(
        default_factory=lambda: LatencyLogger("action"),
        exclude=True,
    )

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        angular_z: float = 0.0,
        duration: float = 0.5,
    ) -> str:
        try:
            req_ctx = get_request_context()

            def _on_first_publish() -> None:
                transcript_perf = req_ctx.get("transcript_perf_s")
                if transcript_perf:
                    self.latency_logger.timing(
                        "tool_dispatch_s",
                        perf_now() - transcript_perf,
                        tool=self.name,
                        req_id=req_ctx.get("req_id"),
                    )

            _on_first_publish()
            clamped_duration = self.robot_client.move_velocity(
                linear_x=linear_x,
                linear_y=linear_y,
                angular_z=angular_z,
                duration=duration,
                hz=CMD_VEL_HZ,
                max_duration=CMD_VEL_MAX_DURATION,
            )
            return tool_response_json(
                success=True,
                status="completed",
                message=(
                    "Movement command sent and stop command issued after the requested duration."
                ),
                robot_state_after="standing",
                linear_x=linear_x,
                linear_y=linear_y,
                angular_z=angular_z,
                duration=clamped_duration,
            )
        except Exception as exc:
            return tool_response_json(
                success=False,
                status="error",
                message=str(exc),
                robot_state_after="unknown",
            )


def get_move_robot_tool(robot_client: Any) -> MoveRobotTool:
    return MoveRobotTool(robot_client=robot_client)
