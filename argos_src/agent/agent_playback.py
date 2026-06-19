"""Playback and interruption helpers for the Argos agent runtime."""

from __future__ import annotations

import time

from argos_src.agent.realtime_turns import TURN_PHASE_CANCELED, QueuedTurn


class RealtimeAgentPlaybackMixin:
    def _wait_for_playback_and_complete(
        self,
        turn: QueuedTurn,
        stream_id: str,
    ) -> None:
        while not self._stop_event.is_set() and not turn.interrupted and not turn.playback_finished.is_set():
            if (
                self._playback_buffer.buffered_frames() > 0
                or turn.pending_tool_calls > 0
                or turn.pending_response_requests > 0
                or not turn.response_finished.is_set()
            ):
                time.sleep(0.02)
                continue
            break
        if turn.playback_finished.is_set():
            return
        if turn.interrupted:
            turn.playback_finished.set()
            return
        rendered_stream_id = str(stream_id or turn.response_id or "").strip()
        self._input_suppressed_until_s = max(
            float(getattr(self, "_input_suppressed_until_s", 0.0) or 0.0),
            time.time() + 0.8,
        )
        self.engagement.on_playback_event(
            "playback_completed",
            turn.req_id,
            stream_id=rendered_stream_id,
        )
        display_mode = getattr(self, "_set_display_mode_async", None)
        if callable(display_mode):
            display_mode("idle")
        turn.playback_finished.set()

    def _force_complete_stalled_playback(self, turn: QueuedTurn, *, reason: str) -> None:
        if turn.playback_finished.is_set():
            return
        with self._turn_lock:
            if self._playback_req_id == turn.req_id:
                self._playback_buffer.clear()
                self._clear_playback_tracking_locked()
        self._input_suppressed_until_s = max(
            float(getattr(self, "_input_suppressed_until_s", 0.0) or 0.0),
            time.time() + 0.8,
        )
        self.engagement.on_playback_event(
            "playback_stopped",
            turn.req_id,
            stream_id=turn.response_id,
        )
        display_mode = getattr(self, "_set_display_mode_async", None)
        if callable(display_mode):
            display_mode("idle")
        turn.playback_finished.set()

    def interrupt_current_response(self, *, reason: str) -> None:
        """Stop local playback and align server-side conversation with what was heard."""
        with self._turn_lock:
            turn = self._active_turn
        if turn is None or self._is_turn_terminal(turn):
            return
        self._terminate_turn(
            turn,
            TURN_PHASE_CANCELED,
            reason,
            send_cancel=True,
            clear_playback=True,
            truncate_playback=turn.audio_started,
        )
