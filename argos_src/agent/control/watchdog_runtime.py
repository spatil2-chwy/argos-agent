"""Watchdog runtime for stalled realtime turns."""

from __future__ import annotations

import time
from typing import Any

from argos_src.agent.realtime_turns import (
    TURN_PHASE_CANCELED,
    TURN_PHASE_PLAYING,
    TURN_PHASE_RESPONSE_REQUESTED,
    TURN_PHASE_WAITING_FIRST_AUDIO,
    TURN_PHASE_WAITING_TOOLS,
)


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
                if current_time - started_at >= response_timeout_s:
                    host.logger.warning(
                        "Realtime response watchdog cancel req_id=%s phase=%s",
                        turn.req_id,
                        turn.phase,
                    )
                    host._terminate_turn(turn, TURN_PHASE_CANCELED, "response_timeout")
                    continue
            if turn.phase == TURN_PHASE_WAITING_TOOLS and turn.pending_tool_calls > 0:
                started_at = turn.phase_updated_at
                if current_time - started_at >= response_timeout_s:
                    host.logger.warning(
                        "Realtime tool watchdog cancel req_id=%s pending_tool_calls=%s",
                        turn.req_id,
                        turn.pending_tool_calls,
                    )
                    host._terminate_turn(turn, TURN_PHASE_CANCELED, "tool_timeout")
                    continue
            if (
                turn.phase == TURN_PHASE_PLAYING
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
