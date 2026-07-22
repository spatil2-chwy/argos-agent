"""Watchdog runtime for stalled realtime turns."""

from __future__ import annotations

import time
from typing import Any

from argos_src.agent.realtime_turns import (
    TURN_PHASE_CANCELED,
    TURN_PHASE_MODEL_DONE,
    TURN_PHASE_PLAYING,
    TURN_PHASE_RESPONSE_REQUESTED,
    TURN_PHASE_WAITING_FIRST_AUDIO,
)
LONG_RUNNING_TOOL_TIMEOUTS_SEC = {
    "enroll_visible_person": 45.0,
}


class TurnWatchdogRuntime:
    """Cancel or recover turns that stop making progress."""

    def __init__(self, host: Any) -> None:
        self._host = host

    def loop(
        self,
        *,
        poll_s: float,
        response_timeout_s: float,
        playback_timeout_s: float,
    ) -> None:
        while not self._host._stop_event.wait(poll_s):
            self.poll_once(
                response_timeout_s=response_timeout_s,
                playback_timeout_s=playback_timeout_s,
            )

    def poll_once(
        self,
        *,
        response_timeout_s: float,
        playback_timeout_s: float,
        now: float | None = None,
    ) -> None:
        host = self._host
        current_time = time.time() if now is None else float(now)
        with host._turn_lock:
            turns = list(host._turns_by_req_id.values())
        for turn in turns:
            if host._is_turn_terminal(turn):
                continue
            if turn.phase in {TURN_PHASE_RESPONSE_REQUESTED, TURN_PHASE_WAITING_FIRST_AUDIO}:
                started_at = turn.response_requested_at or turn.phase_updated_at
                response_state = turn.response_outputs.get(turn.response_id)
                if response_state is not None and not response_state.response_done:
                    started_at = max(started_at, response_state.last_progress_at)
                if current_time - started_at >= response_timeout_s:
                    host.logger.warning(
                        "Realtime response watchdog cancel req_id=%s phase=%s",
                        turn.req_id,
                        turn.phase,
                    )
                    host._terminate_turn(turn, TURN_PHASE_CANCELED, "response_timeout")
                    continue
            if turn.pending_tool_calls > 0:
                active_call_id = str(
                    getattr(turn, "active_tool_call_id", "") or ""
                )
                deadline_at = self._tool_deadline_at(
                    turn,
                    default_timeout_s=response_timeout_s,
                )
                if current_time >= deadline_at and self._claim_tool_timeout(
                    turn,
                    now=current_time,
                    default_timeout_s=response_timeout_s,
                    expected_call_id=active_call_id,
                ):
                    host.logger.warning(
                        "Realtime tool watchdog cancel req_id=%s call_id=%s tool=%s",
                        turn.req_id,
                        turn.active_tool_call_id,
                        turn.active_tool_name,
                    )
                    host._terminate_turn(turn, TURN_PHASE_CANCELED, "tool_timeout")
                    self._cancel_active_tool(turn)
                    continue
            if (
                turn.phase in {TURN_PHASE_PLAYING, TURN_PHASE_MODEL_DONE}
                and turn.response_finished.is_set()
                and not turn.playback_finished.is_set()
            ):
                progress_at = (
                    turn.last_playback_progress_at
                    or turn.audio_started_at
                    or turn.phase_updated_at
                )
                effective_timeout = max(
                    playback_timeout_s,
                    float(getattr(host.realtime_profile, "silence_grace_period", 0.0)) + 5.0,
                )
                if current_time - progress_at >= effective_timeout:
                    host.logger.warning(
                        "Realtime playback stall forcing completion req_id=%s response_id=%s",
                        turn.req_id,
                        turn.response_id,
                    )
                    host._force_complete_stalled_playback(turn, reason="stall_timeout")

    @staticmethod
    def _tool_deadline_at(turn: Any, *, default_timeout_s: float) -> float:
        started_at = float(getattr(turn, "active_tool_started_at", 0.0) or 0.0)
        explicit_deadline = float(
            getattr(turn, "active_tool_deadline_at", 0.0) or 0.0
        )
        if started_at > 0.0:
            if explicit_deadline > 0.0:
                return explicit_deadline
            tool_name = str(getattr(turn, "active_tool_name", "") or "").strip()
            timeout_s = max(
                float(default_timeout_s),
                float(LONG_RUNNING_TOOL_TIMEOUTS_SEC.get(tool_name, 0.0)),
            )
            return started_at + timeout_s
        progress_at = float(getattr(turn, "last_tool_progress_at", 0.0) or 0.0)
        return max(progress_at, float(turn.phase_updated_at)) + float(default_timeout_s)

    def _claim_tool_timeout(
        self,
        turn: Any,
        *,
        now: float,
        default_timeout_s: float,
        expected_call_id: str,
    ) -> bool:
        """Recheck the active call under the turn lock before timing it out."""
        with self._host._turn_lock:
            if self._host._is_turn_terminal(turn):
                return False
            if turn.pending_tool_calls <= 0:
                return False
            if str(getattr(turn, "active_tool_call_id", "") or "") != expected_call_id:
                return False
            return now >= self._tool_deadline_at(
                turn,
                default_timeout_s=default_timeout_s,
            )

    def _cancel_active_tool(self, turn: Any) -> None:
        tool_name = str(getattr(turn, "active_tool_name", "") or "").strip()
        tool = (getattr(self._host, "_tool_registry", {}) or {}).get(tool_name)
        cancel = getattr(tool, "cancel_active_execution", None)
        if not callable(cancel):
            return
        try:
            cancel()
        except Exception:
            self._host.logger.exception(
                "Tool watchdog cancellation failed req_id=%s tool=%s",
                turn.req_id,
                tool_name,
            )
