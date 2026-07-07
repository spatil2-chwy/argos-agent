"""Pure engagement-state transition rules."""

from __future__ import annotations

from dataclasses import dataclass, field
import enum

from argos_src.agent.control.types import ControlAction, EngagementMode


class EngagementTrigger(str, enum.Enum):
    FACE_OR_WAKE = "face_or_wake"
    HUMAN_INPUT = "human_input"
    AGENT_OUTPUT_STARTED = "agent_output_started"
    AGENT_DONE = "agent_done"
    ALERT_TIMEOUT = "alert_timeout"
    COOLDOWN_TIMEOUT = "cooldown_timeout"
    PLAYBACK_FALLBACK = "playback_fallback"
    PLAYBACK_TERMINAL = "playback_terminal"


@dataclass(frozen=True)
class EngagementDecision:
    old_state: str
    new_state: str
    reason: str
    actions: tuple[ControlAction, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> bool:
        return self.old_state != self.new_state


def reduce_engagement(
    state: str,
    trigger: EngagementTrigger | str,
    *,
    has_reply: bool = False,
) -> EngagementDecision:
    """Return the next engagement state plus declarative side-effect actions."""
    current = _mode_value(state)
    rendered_trigger = (
        trigger.value if isinstance(trigger, EngagementTrigger) else str(trigger or "")
    )

    if rendered_trigger == EngagementTrigger.FACE_OR_WAKE.value:
        if current == EngagementMode.IDLE.value:
            return EngagementDecision(
                old_state=current,
                new_state=EngagementMode.ALERT.value,
                reason="face_detected",
                actions=(
                    ControlAction("publish_voice_command", {"command": "stop"}),
                    ControlAction("cancel_active_navigation"),
                ),
            )
        return _unchanged(current, rendered_trigger)

    if rendered_trigger == EngagementTrigger.HUMAN_INPUT.value:
        if current in {
            EngagementMode.IDLE.value,
            EngagementMode.ALERT.value,
            EngagementMode.COOLDOWN.value,
        }:
            actions: tuple[ControlAction, ...] = ()
            if current in {EngagementMode.IDLE.value, EngagementMode.COOLDOWN.value}:
                actions = (
                    ControlAction("publish_voice_command", {"command": "stop"}),
                    ControlAction("cancel_active_navigation"),
                )
            return EngagementDecision(
                old_state=current,
                new_state=EngagementMode.ENGAGED.value,
                reason="human_input",
                actions=actions,
            )
        return _unchanged(current, rendered_trigger)

    if rendered_trigger == EngagementTrigger.AGENT_OUTPUT_STARTED.value:
        if current in {EngagementMode.ALERT.value, EngagementMode.ENGAGED.value}:
            return EngagementDecision(
                old_state=current,
                new_state=EngagementMode.SPEAKING.value,
                reason="agent_output_started",
            )
        if current == EngagementMode.SPEAKING.value:
            return _unchanged(current, rendered_trigger)
        return _unchanged(current, rendered_trigger)

    if rendered_trigger == EngagementTrigger.AGENT_DONE.value:
        if has_reply:
            if current in {EngagementMode.ALERT.value, EngagementMode.ENGAGED.value}:
                return EngagementDecision(
                    old_state=current,
                    new_state=EngagementMode.SPEAKING.value,
                    reason="agent_done_with_reply",
                    actions=(ControlAction("await_playback_terminal"),),
                )
            if current == EngagementMode.SPEAKING.value:
                return EngagementDecision(
                    old_state=current,
                    new_state=current,
                    reason="agent_done_with_reply",
                    actions=(ControlAction("await_playback_terminal"),),
                )
            return _unchanged(current, rendered_trigger)
        if current in {EngagementMode.ALERT.value, EngagementMode.ENGAGED.value}:
            return EngagementDecision(
                old_state=current,
                new_state=EngagementMode.COOLDOWN.value,
                reason="agent_done",
            )
        return _unchanged(current, rendered_trigger)

    if rendered_trigger == EngagementTrigger.ALERT_TIMEOUT.value:
        if current == EngagementMode.ALERT.value:
            return EngagementDecision(
                old_state=current,
                new_state=EngagementMode.IDLE.value,
                reason="timeout",
                actions=(ControlAction("force_flush_coalescer"), ControlAction("notify_idle_entered")),
            )
        return _unchanged(current, rendered_trigger)

    if rendered_trigger == EngagementTrigger.COOLDOWN_TIMEOUT.value:
        if current == EngagementMode.COOLDOWN.value:
            return EngagementDecision(
                old_state=current,
                new_state=EngagementMode.IDLE.value,
                reason="timeout",
                actions=(ControlAction("notify_idle_entered"),),
            )
        return _unchanged(current, rendered_trigger)

    if rendered_trigger == EngagementTrigger.PLAYBACK_FALLBACK.value:
        if current in {EngagementMode.ENGAGED.value, EngagementMode.SPEAKING.value}:
            return EngagementDecision(
                old_state=current,
                new_state=EngagementMode.COOLDOWN.value,
                reason="fallback",
            )
        return _unchanged(current, rendered_trigger)

    if rendered_trigger == EngagementTrigger.PLAYBACK_TERMINAL.value:
        if current in {EngagementMode.ENGAGED.value, EngagementMode.SPEAKING.value}:
            return EngagementDecision(
                old_state=current,
                new_state=EngagementMode.COOLDOWN.value,
                reason="playback_terminal",
            )
        return _unchanged(current, rendered_trigger)

    return _unchanged(current, rendered_trigger)


def decision_has_action(decision: EngagementDecision, kind: str) -> bool:
    return any(action.kind == kind for action in decision.actions)


def _unchanged(state: str, trigger: str) -> EngagementDecision:
    return EngagementDecision(old_state=state, new_state=state, reason=trigger)


def _mode_value(state: str) -> str:
    rendered = str(state or "").strip()
    if not rendered:
        return EngagementMode.IDLE.value
    return rendered
