"""Family-specific deterministic startup preparation for robot agent sessions."""

from __future__ import annotations

import logging
from typing import Any

from argos_src.profile_config import ScenarioProfile


logger = logging.getLogger(__name__)

SPOT_PREPARE_SEQUENCE = (
    ("claim", "claim"),
    ("power_on", "power_on"),
    ("stand", "stand"),
)


class RobotStartupPreparationError(RuntimeError):
    """Raised when deterministic robot startup preparation cannot reach readiness."""


def prepare_robot_for_agent_session(
    robot_client,
    *,
    scenario_profile: ScenarioProfile,
) -> list[dict[str, Any]]:
    """Run deterministic startup preparation for the selected robot family."""
    startup = scenario_profile.startup
    if not startup.prepare_robot:
        logger.info(
            "Robot startup preparation disabled for profile '%s'.",
            scenario_profile.name,
        )
        return []

    if scenario_profile.robot_family == "spot":
        return _prepare_spot_for_agent_session(
            robot_client,
            timeout_sec=startup.service_timeout_sec,
            fail_on_error=startup.fail_on_prepare_error,
        )

    logger.info(
        "No deterministic startup sequence configured for robot family '%s'.",
        scenario_profile.robot_family,
    )
    return []


def derive_initial_robot_posture(
    *,
    scenario_profile: ScenarioProfile,
    startup_steps: list[dict[str, Any]],
) -> str:
    """Infer the safest initial posture assumption after startup preparation."""
    if scenario_profile.robot_family != "spot":
        return "standing"

    if not scenario_profile.startup.prepare_robot:
        return "unknown"

    for step in reversed(startup_steps):
        if str(step.get("step", "")).strip() != "stand":
            continue
        return "standing" if bool(step.get("goal_satisfied")) else "unknown"

    return "unknown"


def _prepare_spot_for_agent_session(
    robot_client,
    *,
    timeout_sec: float,
    fail_on_error: bool,
) -> list[dict[str, Any]]:
    """Prepare Spot for an interactive agent session."""
    del timeout_sec
    results: list[dict[str, Any]] = []
    ready_for_interaction = False
    for step_name, command in SPOT_PREPARE_SEQUENCE:
        logger.info("Spot startup: running %s capability command=%s", step_name, command)
        try:
            payload = robot_client.perform_spot_command(command)
        except Exception as exc:
            message = f"Spot startup step '{step_name}' failed: {exc}"
            logger.warning(message)
            results.append(
                {
                    "step": step_name,
                    "command": command,
                    "success": False,
                    "goal_satisfied": False,
                    "message": str(exc),
                }
            )
            continue

        success = bool(payload.get("success", payload.get("ok", True)))
        message = str(payload.get("message", "") or "").strip()
        goal_satisfied = _spot_step_goal_satisfied(
            step_name=step_name,
            success=success,
            message=message,
        )
        if step_name == "stand" and goal_satisfied:
            ready_for_interaction = True
        if not success:
            error = (
                f"Spot startup step '{step_name}' reported failure"
                + (f": {message}" if message else ".")
            )
            logger.warning(error)
        else:
            logger.info(
                "Spot startup step '%s' succeeded%s",
                step_name,
                f": {message}" if message else "",
            )
        results.append(
            {
                "step": step_name,
                "command": command,
                "success": success,
                "goal_satisfied": goal_satisfied,
                "message": message,
            }
        )

    if fail_on_error and not ready_for_interaction:
        failure_summary = "; ".join(
            f"{item['step']}={item['message'] or 'failed'}" for item in results
        )
        raise RobotStartupPreparationError(
            "Spot startup preparation did not reach a ready standing state. "
            + failure_summary
        )
    return results


def _spot_step_goal_satisfied(*, step_name: str, success: bool, message: str) -> bool:
    """Whether a startup step achieved its intended state, even if not freshly changed."""
    if success:
        return True
    lowered = str(message or "").strip().lower()
    if not lowered:
        return False
    if step_name == "claim":
        return "already claimed" in lowered or "already has lease" in lowered
    if step_name == "power_on":
        return "already powered on" in lowered or "already on" in lowered
    if step_name == "stand":
        return "already standing" in lowered or "is standing" in lowered
    return False
