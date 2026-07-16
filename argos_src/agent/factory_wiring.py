"""Callback coordination helpers for the Argos realtime agent factory."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from argos_src.agent.control.robot_arbitration import (
    decide_idle_patrol_resume,
    emit_robot_arbitration,
)
from argos_src.agent.control.types import RobotArbitrationState
from argos_src.provider_api.errors import is_provider_error

logger = logging.getLogger(__name__)


class FactoryRuntimeWireup:
    """Own the mutable callback wiring used while the runtime is assembled."""

    def __init__(
        self,
        *,
        robot_client: Any,
        nav_state: Any = None,
        battery_cache: Any = None,
        format_navigation_event: Callable[[dict[str, Any]], str],
    ) -> None:
        self._robot_client = robot_client
        self._nav_state = nav_state
        self._battery_cache = battery_cache
        self._format_navigation_event = format_navigation_event
        self._agent: Any = None
        self._coalescer: Any = None
        self._patrol_bridge: Any = None

    def bind_agent(self, agent: Any) -> None:
        self._agent = agent

    def bind_coalescer(self, coalescer: Any) -> None:
        self._coalescer = coalescer

    def bind_patrol_bridge(self, patrol_bridge: Any) -> None:
        self._patrol_bridge = patrol_bridge

    def bind_battery_cache(self, battery_cache: Any) -> None:
        self._battery_cache = battery_cache

    def _start_patrol_navigation(self, target: str, *, trigger: str) -> bool:
        rendered_target = str(target or "").strip()
        if not rendered_target or self._nav_state is None:
            return False
        try:
            from argos_src.tools.unitree_go2.navigation.toolset import (
                start_navigation_to_saved_location,
            )

            result = start_navigation_to_saved_location(
                robot_client=self._robot_client,
                state=self._nav_state,
                location_name=rendered_target,
                battery=self._battery_cache,
                tool_name="patrol_navigation",
            )
        except Exception:
            logger.exception(
                "Failed to start patrol navigation trigger=%s target=%s",
                trigger,
                rendered_target,
            )
            return False
        if not bool(result.get("success", False)):
            logger.warning(
                "Patrol navigation not started trigger=%s target=%s status=%s message=%s",
                trigger,
                rendered_target,
                result.get("status"),
                result.get("message"),
            )
            return False
        logger.info(
            "Patrol navigation started trigger=%s target=%s goal_id=%s",
            trigger,
            rendered_target,
            result.get("goal_id"),
        )
        return True

    def on_idle_entered(self) -> None:
        agent = self._agent
        if agent is not None:
            agent.flush_preference_segments(reason="idle")
            display_mode = getattr(agent, "_set_display_mode_async", None)
            if callable(display_mode):
                display_mode("idle", force=True)
        try:
            decision = decide_idle_patrol_resume(
                nav_state=self._nav_state,
                coalescer=self._coalescer,
                battery_cache=self._battery_cache,
            )
        except Exception:
            logger.exception("Failed to check robot arbitration before patrol resume")
            emit_robot_arbitration(
                agent,
                RobotArbitrationState.PATROL_SUPPRESSED,
                trigger="idle_entered",
                reason="arbitration_failed",
            )
            return
        emit_robot_arbitration(
            agent,
            decision.state,
            trigger="idle_entered",
            reason=decision.reason,
            fields=decision.fields,
        )
        if not decision.allowed:
            return
        target = str(decision.fields.get("target_label", "") or "").strip()
        self._start_patrol_navigation(target, trigger="idle_entered")

    def publish_voice_cmd(self, cmd: str) -> None:
        agent = self._agent
        if agent is not None:
            try:
                agent.note_local_voice_command(cmd)
            except Exception:
                logger.exception("Failed to mark local voice command=%s", cmd)
        try:
            publisher = getattr(self._robot_client, "publish_voice_command", None)
            if callable(publisher):
                publisher(cmd)
        except Exception as exc:
            if is_provider_error(exc):
                logger.warning("Robot provider voice command publish failed cmd=%s: %s", cmd, exc)
            else:
                logger.exception("Failed to publish voice command=%s", cmd)
        if agent is not None:
            handler = getattr(agent, "handle_voice_command", None)
            if callable(handler):
                handler(cmd)

    def notify_charging_ready(self, pct: float) -> None:
        if self._coalescer is None:
            return
        resumed_patrol = None
        if self._nav_state is not None:
            resumed_patrol = self._nav_state.resume_paused_patrol()
        resume_suffix = ""
        if resumed_patrol is not None:
            next_target = str(resumed_patrol.get("awaiting_target", "")).strip()
            if next_target:
                resume_suffix = (
                    " Patrol was paused for charging and will resume from "
                    f"{next_target} after you return to idle."
                )
        if self._battery_cache is not None and self._battery_cache.can_self_charge():
            text = (
                f"BATTERY_EVENT: You were in damp (rest) on the charger and have been charging. "
                f"Battery is now about {pct:.0f}% - high enough to leave the dock and work again. "
                f"Stand up: exit damp/rest, then you're cleared for normal tasks and movement.{resume_suffix} "
                "Reply with a short in-character line along the lines of 'I'm ready to explore again!'."
            )
        else:
            text = (
                f"BATTERY_EVENT: Battery is now about {pct:.0f}% and high enough "
                "for normal tasks and movement again. "
                "Reply with a short in-character line along the lines of 'I'm ready to continue!'."
            )
        self._coalescer.submit(
            text=text,
            metadata={
                "internal": True,
                "internal_event": "battery",
                "source": "battery_state",
            },
        )

    def submit_nav_event(self, event: dict[str, Any]) -> None:
        if self._coalescer is not None and event.get("tool_name") != "patrol_navigation":
            self._coalescer.submit(
                text=self._format_navigation_event(event),
                metadata={
                    "internal": True,
                    "internal_event": "navigation",
                    "source": "navigation",
                    "event_type": event.get("event_type", ""),
                    "goal_id": event.get("goal_id", ""),
                },
            )
        if self._patrol_bridge is not None:
            self._patrol_bridge.on_nav_event(event)

    def maybe_start_startup_patrol(
        self,
        *,
        startup_patrol_route: list[str],
        navigation_runtime_store: Any,
        startup_delay_sec: float,
    ) -> None:
        if (
            not startup_patrol_route
            or self._nav_state is None
            or navigation_runtime_store is None
        ):
            return

        missing = [
            name
            for name in startup_patrol_route
            if navigation_runtime_store.get(name) is None
        ]
        if missing:
            known = ", ".join(navigation_runtime_store.names()) or "none"
            raise ValueError(
                f"Unknown startup patrol location(s): {', '.join(missing)}. Known: {known}."
            )

        self._nav_state.start_patrol(startup_patrol_route)
        first_target = startup_patrol_route[0]

        def emit_startup_patrol_event_after_delay() -> None:
            time.sleep(startup_delay_sec)
            self._start_patrol_navigation(first_target, trigger="startup_patrol")

        threading.Thread(
            target=emit_startup_patrol_event_after_delay,
            daemon=True,
        ).start()

        agent = self._agent
        runtime_logger = getattr(agent, "logger", logger)
        runtime_logger.info(
            "Startup patrol initialized with route: "
            + ", ".join(startup_patrol_route)
            + f" (first hop delayed {startup_delay_sec:.0f}s)"
        )
