from __future__ import annotations

from types import SimpleNamespace
import logging
import threading

from argos_src.agent.control.turn_runner import TurnRunner
from argos_src.agent.realtime_turns import (
    QueuedTurn,
    TURN_PHASE_CANCELED,
    TURN_PHASE_FINALIZED,
    TURN_PHASE_PREPARING_HISTORY,
)


class _Host:
    def __init__(self) -> None:
        self.logger = logging.getLogger("test.turn_runner")
        self._stop_event = threading.Event()
        self._turn_lock = threading.RLock()
        self._active_turn = None
        self._turns_by_req_id = {}
        self._stop_reason = ""
        self.rotated = []
        self.text_items = []
        self.response_creates = []
        self.voice_refs = []
        self.preferences = []
        self.phases = []
        self.terminated = []

    def _clear_playback_tracking_locked(self):
        return

    def _maybe_rotate_history_for_turn(self, turn):
        self.rotated.append(turn.req_id)

    def _set_turn_phase(self, turn, phase, *, trigger="set_turn_phase"):
        self.phases.append((turn.req_id, phase, trigger))
        turn.phase = phase

    def _append_text_message_item(self, turn, text, *, role):
        self.text_items.append((turn.req_id, text, role))

    def _send_response_create(self, turn):
        self.response_creates.append(turn.req_id)
        turn.response_finished.set()
        turn.playback_finished.set()

    def _is_turn_terminal(self, turn):
        return turn is None or bool(getattr(turn, "finalized", False))

    def _complete_turn_success(self, turn):
        turn.finalized = True
        turn.finalized_reason = "completed"
        turn.phase = TURN_PHASE_FINALIZED

    def _terminate_turn(self, turn, phase, reason, **kwargs):
        self.terminated.append((turn.req_id, phase, reason, kwargs))
        turn.finalized = True
        turn.finalized_reason = reason
        turn.phase = phase
        turn.response_finished.set()
        turn.playback_finished.set()

    def _maybe_capture_voice_reference(self, turn):
        self.voice_refs.append(turn.req_id)

    def _maybe_note_preference_turn(self, turn):
        self.preferences.append(turn.req_id)


def _turn(req_id: str = "rt-turn") -> QueuedTurn:
    return QueuedTurn(
        kind="text",
        req_id=req_id,
        speech_end_perf_s=0.0,
        speech_end_unix_s=0.0,
        transcript_perf_s=0.0,
        input_text="SYSTEM_EVENT",
        source_is_internal=True,
        pending_internal_text="PENDING_EVENT",
    )


def test_turn_runner_prepares_text_turn_and_finalizes_success() -> None:
    host = _Host()
    runner = TurnRunner(host)
    turn = _turn()

    runner.run(turn)

    assert host.rotated == [turn.req_id]
    assert host.phases[0] == (
        turn.req_id,
        TURN_PHASE_PREPARING_HISTORY,
        "turn_runner_prepare",
    )
    assert host.text_items == [
        (turn.req_id, "SYSTEM_EVENT", "system"),
        (turn.req_id, "PENDING_EVENT", "system"),
    ]
    assert host.response_creates == [turn.req_id]
    assert turn.finalized is True
    assert turn.finalized_reason == "completed"
    assert host.voice_refs == [turn.req_id]
    assert host.preferences == [turn.req_id]
    assert host._active_turn is None


def test_turn_runner_clears_active_turn_on_exception() -> None:
    host = _Host()
    runner = TurnRunner(host)
    turn = _turn("rt-error")

    def fail(_turn):
        raise RuntimeError("boom")

    host._send_response_create = fail

    try:
        runner.run(turn)
    except RuntimeError:
        pass

    assert host._active_turn is None
    assert host._turns_by_req_id[turn.req_id] is turn


def test_turn_runner_cancels_unsettled_turn_when_runtime_stops() -> None:
    host = _Host()
    runner = TurnRunner(host)
    turn = _turn("rt-websocket-closed")

    def stop_runtime(_turn):
        host.response_creates.append(_turn.req_id)
        host._stop_reason = "websocket_closed"
        host._stop_event.set()

    host._send_response_create = stop_runtime

    runner.run(turn)

    assert host.response_creates == [turn.req_id]
    assert turn.finalized is True
    assert turn.finalized_reason == "websocket_closed"
    assert turn.phase == TURN_PHASE_CANCELED
    assert host.terminated == [
        (
            turn.req_id,
            TURN_PHASE_CANCELED,
            "websocket_closed",
            {"send_cancel": False},
        )
    ]
    assert host.voice_refs == []
    assert host.preferences == []
    assert host._active_turn is None
