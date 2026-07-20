"""Robot arbitration state reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from argos_src.agent.control.observers import safe_transition
from argos_src.agent.control.types import (
    RobotArbitrationState,
    StateAxis,
    StateTransition,
)


@dataclass(frozen=True)
class RobotArbitrationDecision:
    """Decision returned by robot arbitration policy helpers."""

    allowed: bool
    state: RobotArbitrationState
    reason: str = ""
    fields: dict[str, Any] = field(default_factory=dict)


def decide_idle_patrol_resume(
    *,
    nav_state: Any,
    coalescer: Any,
    battery_cache: Any = None,
) -> RobotArbitrationDecision:
    """Decide whether the robot may resume patrol after engagement returns idle."""
    if nav_state is None or coalescer is None:
        return RobotArbitrationDecision(
            allowed=False,
            state=RobotArbitrationState.PATROL_SUPPRESSED,
            reason="missing_navigation_or_coalescer",
        )
    patrol = nav_state.get_patrol()
    if not patrol.get("enabled", False):
        return RobotArbitrationDecision(
            allowed=False,
            state=RobotArbitrationState.PATROL_SUPPRESSED,
            reason="patrol_disabled",
        )
    if nav_state.get_active_goal() is not None:
        return RobotArbitrationDecision(
            allowed=False,
            state=RobotArbitrationState.PATROL_SUPPRESSED,
            reason="active_navigation_goal",
        )
    if battery_cache is not None:
        if battery_cache.should_block_general_navigation():
            return RobotArbitrationDecision(
                allowed=False,
                state=RobotArbitrationState.BATTERY_LOW_BLOCKING,
                reason="battery_blocks_navigation",
            )
    target = str(patrol.get("awaiting_target", "")).strip()
    if not target:
        return RobotArbitrationDecision(
            allowed=False,
            state=RobotArbitrationState.PATROL_SUPPRESSED,
            reason="missing_patrol_target",
        )
    return RobotArbitrationDecision(
        allowed=True,
        state=RobotArbitrationState.PATROL_ALLOWED,
        fields={"target_label": target},
    )


def decide_proactive_face_attention(
    *,
    engagement_state: Any,
    nav_state: Any,
    recording_active: bool,
    human_turn_active: bool = False,
) -> RobotArbitrationDecision:
    """Decide whether vision may emit a proactive face interaction."""
    state_name = str(getattr(engagement_state, "value", engagement_state) or "")
    if state_name != "idle":
        return RobotArbitrationDecision(
            allowed=False,
            state=RobotArbitrationState.FACE_ATTENTION_SUPPRESSED,
            reason="engagement_not_idle",
            fields={"engagement_state": state_name},
        )
    if recording_active:
        return RobotArbitrationDecision(
            allowed=False,
            state=RobotArbitrationState.FACE_ATTENTION_SUPPRESSED,
            reason="recording_active",
        )
    if human_turn_active:
        return RobotArbitrationDecision(
            allowed=False,
            state=RobotArbitrationState.FACE_ATTENTION_SUPPRESSED,
            reason="human_turn_active",
        )
    if nav_state is not None and not nav_state.allows_proactive_face_attention():
        return RobotArbitrationDecision(
            allowed=False,
            state=RobotArbitrationState.FACE_ATTENTION_SUPPRESSED,
            reason="navigation_blocks_face_attention",
        )
    return RobotArbitrationDecision(
        allowed=True,
        state=RobotArbitrationState.FACE_ATTENTION_ALLOWED,
        reason="proactive_face_attention_allowed",
    )


def emit_robot_arbitration(
    host: Any,
    state: RobotArbitrationState | str,
    *,
    trigger: str,
    reason: str = "",
    fields: dict[str, Any] | None = None,
) -> None:
    """Emit a robot-arbitration transition through a runtime-like host."""
    if host is None:
        return
    new_state = state.value if isinstance(state, RobotArbitrationState) else str(state)
    old_state = str(getattr(host, "_robot_arbitration_state", "") or "")
    if old_state == new_state:
        return
    try:
        setattr(host, "_robot_arbitration_state", new_state)
    except Exception:
        pass
    safe_transition(
        getattr(host, "_state_observer", None),
        StateTransition(
            axis=StateAxis.ROBOT_ARBITRATION,
            old_state=old_state or "unknown",
            new_state=new_state,
            trigger=trigger,
            reason=reason,
            fields=dict(fields or {}),
        ),
    )
