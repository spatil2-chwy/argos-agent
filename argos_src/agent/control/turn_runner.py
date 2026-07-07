"""Turn runner for preparing and settling realtime turns."""

from __future__ import annotations

from typing import Any

from argos_src.agent.realtime_turns import QueuedTurn, TURN_PHASE_FINALIZED
from argos_src.observability.observability import clear_request_context, set_request_context


class TurnRunner:
    """Prepare one queued turn, request a response, and wait for settlement."""

    def __init__(self, host: Any) -> None:
        self._host = host

    def run(self, turn: QueuedTurn) -> None:
        host = self._host
        with host._turn_lock:
            host._active_turn = turn
            host._turns_by_req_id[turn.req_id] = turn
            host._clear_playback_tracking_locked()
        host.logger.info("Starting turn req_id=%s kind=%s", turn.req_id, turn.kind)

        set_request_context(
            req_id=turn.req_id,
            speech_end_perf_s=turn.speech_end_perf_s,
            speech_end_unix_s=turn.speech_end_unix_s,
            transcript_perf_s=turn.transcript_perf_s,
        )
        try:
            host._maybe_rotate_history_for_turn(turn)
            if turn.kind == "text":
                host._append_text_message_item(
                    turn,
                    turn.input_text,
                    role="system" if turn.source_is_internal else "user",
                )
            if turn.pending_internal_text:
                host._append_text_message_item(
                    turn,
                    turn.pending_internal_text,
                    role="system",
                )
            host._send_response_create(turn)
            self.wait_for_settled(turn)
            if not host._is_turn_terminal(turn):
                host._complete_turn_success(turn)
            if turn.phase == TURN_PHASE_FINALIZED:
                host._maybe_capture_voice_reference(turn)
                host._maybe_note_preference_turn(turn)
        finally:
            host.logger.info(
                "Finished turn req_id=%s phase=%s finalized_reason=%s audio_started=%s pending_tool_calls=%s pending_response_requests=%s",
                turn.req_id,
                turn.phase,
                turn.finalized_reason,
                turn.audio_started,
                turn.pending_tool_calls,
                turn.pending_response_requests,
            )
            clear_request_context()
            with host._turn_lock:
                if host._active_turn is turn:
                    host._active_turn = None

    def wait_for_settled(self, turn: QueuedTurn) -> None:
        host = self._host
        while not host._stop_event.is_set():
            if turn.response_finished.wait(timeout=0.1) and turn.playback_finished.wait(timeout=0.1):
                return
            if host._is_turn_terminal(turn):
                return
