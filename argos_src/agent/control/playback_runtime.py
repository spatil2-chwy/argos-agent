"""Playback completion and interruption runtime helpers."""

from __future__ import annotations

import time
from typing import Any

from argos_src.agent.realtime_turns import TURN_PHASE_CANCELED, QueuedTurn
from argos_src.agent.control.observers import safe_transition
from argos_src.agent.control.types import PlaybackState, StateAxis, StateTransition


def _exchange_fields(host: Any, turn: QueuedTurn) -> dict[str, Any]:
    fields_fn = getattr(host, "_exchange_log_fields", None)
    if callable(fields_fn):
        return dict(fields_fn(turn))
    return {}


class PlaybackRuntime:
    """Coordinate local playback completion, stall recovery, and interruption."""

    def __init__(self, host: Any) -> None:
        self._host = host

    def transition(
        self,
        state: PlaybackState | str,
        *,
        trigger: str,
        req_id: str = "",
        stream_id: str = "",
        reason: str = "",
    ) -> None:
        host = self._host
        new_state = state.value if isinstance(state, PlaybackState) else str(state)
        old_state = str(getattr(host, "_playback_state", PlaybackState.IDLE.value) or "")
        if old_state == new_state:
            return
        host._playback_state = new_state
        safe_transition(
            getattr(host, "_state_observer", None),
            StateTransition(
                axis=StateAxis.PLAYBACK,
                old_state=old_state,
                new_state=new_state,
                trigger=trigger,
                req_id=req_id,
                stream_id=stream_id,
                reason=reason,
            ),
        )

    def wait_for_playback_and_complete(self, turn: QueuedTurn, stream_id: str) -> None:
        host = self._host
        while (
            not host._stop_event.is_set()
            and not turn.interrupted
            and not turn.playback_finished.is_set()
        ):
            if (
                host._playback_buffer.buffered_frames() > 0
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
        host._input_suppressed_until_s = max(
            float(getattr(host, "_input_suppressed_until_s", 0.0) or 0.0),
            time.time() + 0.8,
        )
        host.engagement.on_playback_event(
            "playback_completed",
            turn.req_id,
            stream_id=rendered_stream_id,
        )
        latency = getattr(host, "_latency", None)
        if latency is not None:
            latency.emit(
                event="playback_completed",
                req_id=turn.req_id,
                stream_id=rendered_stream_id,
                **_exchange_fields(host, turn),
            )
        self.transition(
            PlaybackState.COMPLETED,
            trigger="playback_completed",
            req_id=turn.req_id,
            stream_id=rendered_stream_id,
        )
        display_mode = getattr(host, "_set_display_mode_async", None)
        if callable(display_mode):
            display_mode("idle")
        turn.playback_finished.set()

    def wait_for_intermediate_playback(self, turn: QueuedTurn, stream_id: str) -> None:
        """Finish one audible segment without finalizing its active tool turn."""
        host = self._host
        rendered_stream_id = str(stream_id or "").strip()
        while not host._stop_event.is_set() and not turn.interrupted:
            if turn.response_finished.is_set() or turn.playback_completion_armed:
                return
            with host._turn_lock:
                owns_playback = (
                    host._playback_req_id == turn.req_id
                    and str(host._playback_stream_id or "").strip()
                    == rendered_stream_id
                )
            if not owns_playback:
                return
            if host._playback_buffer.buffered_frames() > 0:
                time.sleep(0.02)
                continue
            break
        if host._stop_event.is_set() or turn.interrupted:
            return
        if turn.response_finished.is_set() or turn.playback_completion_armed:
            return
        with host._turn_lock:
            if (
                host._playback_req_id != turn.req_id
                or str(host._playback_stream_id or "").strip()
                != rendered_stream_id
            ):
                return
            host._clear_playback_tracking_locked()
        host._input_suppressed_until_s = max(
            float(getattr(host, "_input_suppressed_until_s", 0.0) or 0.0),
            time.time() + 0.8,
        )
        host.engagement.on_playback_event(
            "playback_segment_completed",
            turn.req_id,
            stream_id=rendered_stream_id,
        )
        latency = getattr(host, "_latency", None)
        if latency is not None:
            latency.emit(
                event="playback_segment_completed",
                req_id=turn.req_id,
                stream_id=rendered_stream_id,
                **_exchange_fields(host, turn),
            )
        self.transition(
            PlaybackState.IDLE,
            trigger="intermediate_playback_completed",
            req_id=turn.req_id,
            stream_id=rendered_stream_id,
        )
        display_mode = getattr(host, "_set_display_mode_async", None)
        if callable(display_mode):
            display_mode("thinking")

    def force_complete_stalled_playback(self, turn: QueuedTurn, *, reason: str) -> None:
        host = self._host
        if turn.playback_finished.is_set():
            return
        with host._turn_lock:
            if host._playback_req_id == turn.req_id:
                host._playback_buffer.clear()
                host._clear_playback_tracking_locked()
        host._input_suppressed_until_s = max(
            float(getattr(host, "_input_suppressed_until_s", 0.0) or 0.0),
            time.time() + 0.8,
        )
        host.engagement.on_playback_event(
            "playback_stopped",
            turn.req_id,
            stream_id=turn.response_id,
        )
        latency = getattr(host, "_latency", None)
        if latency is not None:
            latency.emit(
                event="playback_stopped",
                req_id=turn.req_id,
                stream_id=turn.response_id,
                terminal_reason=reason,
                **_exchange_fields(host, turn),
            )
        self.transition(
            PlaybackState.FORCE_COMPLETED,
            trigger="playback_force_complete",
            req_id=turn.req_id,
            stream_id=turn.response_id,
            reason=reason,
        )
        display_mode = getattr(host, "_set_display_mode_async", None)
        if callable(display_mode):
            display_mode("idle")
        turn.playback_finished.set()

    def interrupt_current_response(self, *, reason: str) -> None:
        """Stop local playback and align server-side conversation with what was heard."""
        host = self._host
        with host._turn_lock:
            turn = host._active_turn
        if turn is None or host._is_turn_terminal(turn):
            return
        self.transition(
            PlaybackState.STOPPED_TRUNCATED,
            trigger="interrupt_current_response",
            req_id=turn.req_id,
            stream_id=turn.response_id,
            reason=reason,
        )
        host._terminate_turn(
            turn,
            TURN_PHASE_CANCELED,
            reason,
            send_cancel=True,
            clear_playback=True,
            truncate_playback=turn.audio_started,
        )
