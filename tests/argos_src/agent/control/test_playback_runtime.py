from __future__ import annotations

from types import SimpleNamespace
import threading

from argos_src.agent.control.playback_runtime import PlaybackRuntime
from argos_src.agent.realtime_turns import PlaybackBuffer, QueuedTurn, TURN_PHASE_CANCELED


class _Engagement:
    def __init__(self) -> None:
        self.events = []

    def on_playback_event(self, event, req_id, *, stream_id=""):
        self.events.append((event, req_id, stream_id))


class _Host:
    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._turn_lock = threading.RLock()
        self._playback_buffer = PlaybackBuffer()
        self._playback_state = "idle"
        self._input_suppressed_until_s = 0.0
        self._playback_req_id = ""
        self._playback_stream_id = ""
        self._active_turn = None
        self.engagement = _Engagement()
        self.display_modes = []
        self.terminated = []
        self.transitions = []
        self._state_observer = SimpleNamespace(
            transition=lambda transition: self.transitions.append(transition)
        )

    def _set_display_mode_async(self, mode):
        self.display_modes.append(mode)

    def _clear_playback_tracking_locked(self):
        self._playback_req_id = ""
        self._playback_stream_id = ""

    def _is_turn_terminal(self, turn):
        return turn is None or bool(getattr(turn, "finalized", False))

    def _terminate_turn(self, turn, phase, reason, **kwargs):
        turn.phase = phase
        turn.finalized_reason = reason
        self.terminated.append((turn.req_id, phase, reason, kwargs))


def _turn(req_id: str = "rt-playback") -> QueuedTurn:
    return QueuedTurn(
        kind="audio",
        req_id=req_id,
        speech_end_perf_s=0.0,
        speech_end_unix_s=0.0,
        transcript_perf_s=0.0,
    )


def test_playback_runtime_marks_completed_when_response_done_and_buffer_empty() -> None:
    host = _Host()
    runtime = PlaybackRuntime(host)
    turn = _turn()
    turn.response_id = "resp-1"
    turn.response_finished.set()

    runtime.wait_for_playback_and_complete(turn, "")

    assert turn.playback_finished.is_set()
    assert host.engagement.events == [("playback_completed", turn.req_id, "resp-1")]
    assert host.display_modes == ["idle"]
    assert [(t.axis, t.new_state, t.req_id) for t in host.transitions] == [
        ("playback", "completed", turn.req_id)
    ]


def test_intermediate_playback_completion_reopens_the_active_turn() -> None:
    host = _Host()
    runtime = PlaybackRuntime(host)
    turn = _turn()
    host._playback_req_id = turn.req_id
    host._playback_stream_id = "resp-preamble"
    host._playback_state = "playing"

    runtime.wait_for_intermediate_playback(turn, "resp-preamble")

    assert turn.playback_finished.is_set() is False
    assert host._playback_req_id == ""
    assert host._playback_stream_id == ""
    assert host.engagement.events == [
        ("playback_segment_completed", turn.req_id, "resp-preamble")
    ]
    assert host.display_modes == ["thinking"]
    assert host.transitions[-1].new_state == "idle"


def test_playback_runtime_force_completes_stalled_playback() -> None:
    host = _Host()
    runtime = PlaybackRuntime(host)
    turn = _turn()
    turn.response_id = "resp-stall"
    host._playback_req_id = turn.req_id

    runtime.force_complete_stalled_playback(turn, reason="stall_timeout")

    assert turn.playback_finished.is_set()
    assert host._playback_req_id == ""
    assert host.engagement.events == [("playback_stopped", turn.req_id, "resp-stall")]
    assert host.display_modes == ["idle"]
    assert host.transitions[-1].new_state == "force_completed"


def test_playback_runtime_interrupts_active_turn() -> None:
    host = _Host()
    runtime = PlaybackRuntime(host)
    turn = _turn()
    turn.audio_started = True
    host._active_turn = turn

    runtime.interrupt_current_response(reason="voice_command")

    assert host.terminated == [
        (
            turn.req_id,
            TURN_PHASE_CANCELED,
            "voice_command",
            {
                "send_cancel": True,
                "clear_playback": True,
                "truncate_playback": True,
            },
        )
    ]
    assert host.transitions[-1].new_state == "stopped_truncated"
