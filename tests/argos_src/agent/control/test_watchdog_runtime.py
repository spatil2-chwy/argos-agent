from __future__ import annotations

from types import SimpleNamespace
import logging
import threading

from argos_src.agent.control.watchdog_runtime import TurnWatchdogRuntime
from argos_src.agent.realtime_turns import (
    QueuedTurn,
    TURN_PHASE_CANCELED,
    TURN_PHASE_MODEL_DONE,
    TURN_PHASE_PLAYING,
    TURN_PHASE_RESPONSE_REQUESTED,
    TURN_PHASE_WAITING_TOOLS,
)


class _Host:
    def __init__(self) -> None:
        self.logger = logging.getLogger("test.watchdog_runtime")
        self._turn_lock = threading.RLock()
        self._turns_by_req_id = {}
        self.realtime_profile = SimpleNamespace(silence_grace_period=0.0)
        self.terminated = []
        self.force_completed = []

    def _is_turn_terminal(self, turn):
        return turn is None or bool(getattr(turn, "finalized", False))

    def _terminate_turn(self, turn, phase, reason):
        turn.phase = phase
        turn.finalized = True
        turn.response_finished.set()
        turn.playback_finished.set()
        self.terminated.append((turn.req_id, phase, reason))

    def _force_complete_stalled_playback(self, turn, *, reason):
        turn.playback_finished.set()
        self.force_completed.append((turn.req_id, reason))


def _turn(req_id: str = "rt-watchdog") -> QueuedTurn:
    return QueuedTurn(
        kind="audio",
        req_id=req_id,
        speech_end_perf_s=0.0,
        speech_end_unix_s=0.0,
        transcript_perf_s=0.0,
    )


def test_watchdog_cancels_stalled_response_turn() -> None:
    host = _Host()
    runtime = TurnWatchdogRuntime(host)
    turn = _turn()
    turn.phase = TURN_PHASE_RESPONSE_REQUESTED
    turn.response_requested_at = 1.0
    host._turns_by_req_id[turn.req_id] = turn

    runtime.poll_once(response_timeout_s=2.0, playback_timeout_s=10.0, now=4.0)

    assert host.terminated == [(turn.req_id, TURN_PHASE_CANCELED, "response_timeout")]


def test_watchdog_cancels_stalled_tool_turn() -> None:
    host = _Host()
    runtime = TurnWatchdogRuntime(host)
    turn = _turn()
    turn.phase = TURN_PHASE_WAITING_TOOLS
    turn.phase_updated_at = 1.0
    turn.pending_tool_calls = 1
    host._turns_by_req_id[turn.req_id] = turn

    runtime.poll_once(response_timeout_s=2.0, playback_timeout_s=10.0, now=4.0)

    assert host.terminated == [(turn.req_id, TURN_PHASE_CANCELED, "tool_timeout")]


def test_watchdog_allows_long_running_enrollment_tool() -> None:
    host = _Host()
    runtime = TurnWatchdogRuntime(host)
    turn = _turn()
    turn.phase = TURN_PHASE_WAITING_TOOLS
    turn.phase_updated_at = 1.0
    turn.pending_tool_calls = 1
    turn.pending_call_ids.add("call-enroll")
    turn.pending_tool_names_by_call_id["call-enroll"] = "enroll_visible_person"
    host._turns_by_req_id[turn.req_id] = turn

    runtime.poll_once(response_timeout_s=12.0, playback_timeout_s=10.0, now=20.0)

    assert host.terminated == []
    assert turn.finalized is False


def test_watchdog_cancels_enrollment_after_extended_timeout() -> None:
    host = _Host()
    runtime = TurnWatchdogRuntime(host)
    turn = _turn()
    turn.phase = TURN_PHASE_WAITING_TOOLS
    turn.phase_updated_at = 1.0
    turn.pending_tool_calls = 1
    turn.pending_call_ids.add("call-enroll")
    turn.pending_tool_names_by_call_id["call-enroll"] = "enroll_visible_person"
    host._turns_by_req_id[turn.req_id] = turn

    runtime.poll_once(response_timeout_s=12.0, playback_timeout_s=10.0, now=47.0)

    assert host.terminated == [(turn.req_id, TURN_PHASE_CANCELED, "tool_timeout")]


def test_watchdog_force_completes_stalled_playback() -> None:
    host = _Host()
    runtime = TurnWatchdogRuntime(host)
    turn = _turn()
    turn.phase = TURN_PHASE_PLAYING
    turn.response_id = "resp-playback"
    turn.last_playback_progress_at = 1.0
    turn.response_finished.set()
    host._turns_by_req_id[turn.req_id] = turn

    runtime.poll_once(response_timeout_s=2.0, playback_timeout_s=3.0, now=7.0)

    assert host.force_completed == [(turn.req_id, "stall_timeout")]


def test_watchdog_force_completes_stalled_playback_after_model_done() -> None:
    host = _Host()
    runtime = TurnWatchdogRuntime(host)
    turn = _turn()
    turn.phase = TURN_PHASE_MODEL_DONE
    turn.response_id = "resp-playback"
    turn.last_playback_progress_at = 1.0
    turn.response_finished.set()
    host._turns_by_req_id[turn.req_id] = turn

    runtime.poll_once(response_timeout_s=2.0, playback_timeout_s=3.0, now=7.0)

    assert host.force_completed == [(turn.req_id, "stall_timeout")]
