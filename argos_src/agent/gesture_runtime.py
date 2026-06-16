"""Embodied gesture runtime for profile-driven Argos robot poses."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Optional

from argos_src.provider_api.errors import is_provider_error
from argos_src.tools.tool_ids import ROBOT_FAMILY_UNITREE_GO2


logger = logging.getLogger(__name__)

GO2_BALANCE_STAND_API_ID = 1004

GESTURE_STATE_IDLE = "idle"
GESTURE_STATE_LISTENING = "listening"

ERROR_LOG_THROTTLE_SEC = 5.0
RETRY_DELAY_SEC = 1.0
REQUEST_ID_START = 10000
STATE_POLL_SEC = 0.25


@dataclass(frozen=True)
class GestureCommandSpec:
    api_id: int
    parameter: dict[str, Any]
    priority: int = 0


@dataclass(frozen=True)
class GestureStateSpec:
    name: str
    interval_sec: float
    api_id: int
    parameters: tuple[dict[str, Any], ...]
    priority: int = 1


@dataclass(frozen=True)
class GesturePreset:
    name: str
    enter_commands: tuple[GestureCommandSpec, ...]
    exit_commands: tuple[GestureCommandSpec, ...]
    states: dict[str, GestureStateSpec]


GO2_POSE_V1_PRESET = GesturePreset(
    name="go2_pose_v1",
    enter_commands=(
        GestureCommandSpec(api_id=GO2_BALANCE_STAND_API_ID, parameter={}, priority=0),
        GestureCommandSpec(api_id=1003, parameter={}, priority=0),
        GestureCommandSpec(api_id=1028, parameter={"data": True}, priority=0),
    ),
    exit_commands=(
        GestureCommandSpec(api_id=1028, parameter={"data": False}, priority=0),
        GestureCommandSpec(api_id=1002, parameter={}, priority=0),
    ),
    states={
        GESTURE_STATE_IDLE: GestureStateSpec(
            name=GESTURE_STATE_IDLE,
            interval_sec=3.0,
            api_id=1007,
            parameters=(
                {"x": -0.45, "y": 0.0, "z": 0.0},
                {"x": 0.45, "y": 0.0, "z": 0.0},
            ),
            priority=1,
        ),
        GESTURE_STATE_LISTENING: GestureStateSpec(
            name=GESTURE_STATE_LISTENING,
            interval_sec=1.0 / 0.6,
            api_id=1007,
            parameters=({"x": 0.0, "y": -0.22, "z": 0.0},),
            priority=1,
        ),
    },
)

GESTURE_PRESETS: dict[str, GesturePreset] = {
    GO2_POSE_V1_PRESET.name: GO2_POSE_V1_PRESET,
}

AUTO_GESTURE_PRESET_BY_ROBOT_FAMILY: dict[str, str] = {
    ROBOT_FAMILY_UNITREE_GO2: GO2_POSE_V1_PRESET.name,
}


def resolve_gesture_preset_name(
    *,
    robot_family: str,
    preset: Optional[str],
) -> Optional[str]:
    """Resolve a configured gesture preset name for a robot family."""
    robot_family = str(robot_family or "").strip()
    raw_preset = str(preset or "auto").strip() or "auto"
    if raw_preset == "auto":
        return AUTO_GESTURE_PRESET_BY_ROBOT_FAMILY.get(robot_family)
    if raw_preset in AUTO_GESTURE_PRESET_BY_ROBOT_FAMILY:
        return AUTO_GESTURE_PRESET_BY_ROBOT_FAMILY.get(raw_preset)
    if raw_preset not in GESTURE_PRESETS:
        available = ", ".join(
            sorted({*GESTURE_PRESETS, *AUTO_GESTURE_PRESET_BY_ROBOT_FAMILY})
        )
        raise ValueError(
            f"Unknown gesture preset '{raw_preset}'. Available presets: {available}"
        )
    return raw_preset


class GestureRuntime:
    """Owns background idle and listening gesture publishing for the realtime Argos agent."""

    def __init__(
        self,
        *,
        connector: Any,
        engagement: Any,
        preset_name: str,
        enabled_states: Optional[set[str] | frozenset[str] | tuple[str, ...]] = None,
        request_id_start: int = REQUEST_ID_START,
    ) -> None:
        preset = GESTURE_PRESETS.get(str(preset_name or "").strip())
        if preset is None:
            raise ValueError(f"Unknown gesture preset '{preset_name}'.")

        self.connector = connector
        self._engagement = engagement
        self._preset = preset
        if enabled_states is None:
            self._enabled_states = frozenset(preset.states)
        else:
            self._enabled_states = frozenset(
                str(state).strip() for state in enabled_states if str(state).strip()
            )

        self._condition = threading.Condition(threading.RLock())
        self._stop_requested = False
        self._recording_active = False
        self._pose_mode_active = False
        self._active_state_name: Optional[str] = None
        self._next_publish_at = 0.0
        self._next_request_id = int(request_id_start)
        self._next_error_log_by_key: dict[str, float] = {}
        self._state_cycle_index_by_name: dict[str, int] = {}

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    @property
    def preset_name(self) -> str:
        return self._preset.name

    def set_recording_active(self, active: bool) -> None:
        with self._condition:
            self._recording_active = bool(active)
            self._condition.notify_all()

    def shutdown(self) -> None:
        with self._condition:
            self._recording_active = False
            self._stop_requested = True
            self._deactivate_locked(reason="shutdown")
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        while True:
            with self._condition:
                if self._stop_requested:
                    return

                desired_state = self._desired_state_locked()
                if desired_state != self._active_state_name:
                    self._active_state_name = desired_state
                    self._next_publish_at = 0.0

                if desired_state is None:
                    self._deactivate_locked(reason="inactive")
                    self._condition.wait()
                    continue

                if not self._pose_mode_active and not self._enter_pose_mode_locked():
                    self._condition.wait(timeout=RETRY_DELAY_SEC)
                    continue

                now = time.monotonic()
                if now >= self._next_publish_at:
                    publish_delay = self._publish_state_locked(desired_state)
                    self._next_publish_at = time.monotonic() + publish_delay

                wait_for = max(0.0, self._next_publish_at - time.monotonic())
                self._condition.wait(timeout=min(wait_for, STATE_POLL_SEC))

    def _enter_pose_mode_locked(self) -> bool:
        if self._pose_mode_active:
            return True
        command_index = -1
        command: GestureCommandSpec | None = None
        try:
            for command_index, command in enumerate(self._preset.enter_commands):
                self._publish_command_locked(command)
        except Exception as exc:
            self._pose_mode_active = False
            self._log_throttled(
                "enter_pose_mode",
                "Gesture runtime failed to enter pose mode.",
                exc=exc,
                context=self._command_log_context(
                    command,
                    command_index=command_index,
                    phase="enter",
                ),
            )
            return False
        self._pose_mode_active = True
        self._next_publish_at = 0.0
        return True

    def _desired_state_locked(self) -> Optional[str]:
        if self._recording_active:
            if GESTURE_STATE_LISTENING in self._enabled_states:
                return GESTURE_STATE_LISTENING
            return None
        try:
            snapshot = self._engagement.snapshot()
        except Exception:
            self._log_throttled(
                "engagement_snapshot",
                "Gesture runtime failed to read engagement state.",
                exc_info=True,
            )
            return None
        if (
            getattr(snapshot, "state", None) == GESTURE_STATE_IDLE
            and GESTURE_STATE_IDLE in self._enabled_states
        ):
            return GESTURE_STATE_IDLE
        return None

    def _deactivate_locked(self, *, reason: str) -> None:
        if not self._pose_mode_active:
            self._active_state_name = None
            self._next_publish_at = 0.0
            return
        command_index = -1
        command: GestureCommandSpec | None = None
        try:
            for command_index, command in enumerate(self._preset.exit_commands):
                self._publish_command_locked(command)
        except Exception as exc:
            self._log_throttled(
                "exit_pose_mode",
                "Gesture runtime failed to exit pose mode cleanly.",
                exc=exc,
                context=self._command_log_context(
                    command,
                    command_index=command_index,
                    phase="exit",
                    reason=reason,
                ),
            )
        finally:
            self._pose_mode_active = False
            self._active_state_name = None
            self._next_publish_at = 0.0

    def _publish_state_locked(self, state_name: str) -> float:
        state = self._preset.states[state_name]
        cycle_index = self._state_cycle_index_by_name.get(state_name, 0)
        parameter = state.parameters[cycle_index % len(state.parameters)]
        self._state_cycle_index_by_name[state_name] = cycle_index + 1
        try:
            self._publish_command_locked(
                GestureCommandSpec(
                    api_id=state.api_id,
                    parameter=parameter,
                    priority=state.priority,
                )
            )
        except Exception as exc:
            self._log_throttled(
                f"publish_state:{state_name}",
                f"Gesture runtime failed to publish {state_name} state.",
                exc=exc,
                context={
                    "preset": self._preset.name,
                    "phase": "state",
                    "state": state_name,
                    "cycle_index": cycle_index,
                    "api_id": state.api_id,
                    "priority": state.priority,
                    "parameter": parameter,
                },
            )
            return RETRY_DELAY_SEC
        return max(state.interval_sec, 0.0)

    def _publish_command_locked(self, command: GestureCommandSpec) -> None:
        self.connector.perform_go2_action(
            api_id=int(command.api_id),
            parameter=dict(command.parameter),
            priority=int(command.priority),
        )

    def _next_message_id_locked(self) -> int:
        msg_id = self._next_request_id
        self._next_request_id += 1
        return msg_id

    def _command_log_context(
        self,
        command: GestureCommandSpec | None,
        *,
        command_index: int,
        phase: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "preset": self._preset.name,
            "phase": phase,
            "command_index": command_index,
        }
        if reason is not None:
            context["reason"] = reason
        if command is not None:
            context.update(
                {
                    "api_id": command.api_id,
                    "priority": command.priority,
                    "parameter": command.parameter,
                }
            )
        return context

    def _log_throttled(
        self,
        key: str,
        message: str,
        *,
        exc: BaseException | None = None,
        exc_info: bool = False,
        context: dict[str, Any] | None = None,
    ) -> None:
        now = time.monotonic()
        next_allowed = self._next_error_log_by_key.get(key, 0.0)
        if now < next_allowed:
            return
        self._next_error_log_by_key[key] = now + ERROR_LOG_THROTTLE_SEC
        suffix = f" context={context}" if context else ""
        if exc is not None and is_provider_error(exc):
            logger.warning("%s%s provider_error=%s", message, suffix, exc)
            return
        logger.error("%s%s", message, suffix, exc_info=exc_info or exc is not None)
