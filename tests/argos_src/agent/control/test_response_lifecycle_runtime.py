from __future__ import annotations

from types import SimpleNamespace
import threading

from argos_src.agent.control.response_lifecycle_runtime import ResponseLifecycleRuntime
from argos_src.agent.realtime_turns import (
    QueuedTurn,
    TURN_PHASE_REQUESTING_FOLLOWUP,
)


class _Host:
    def __init__(self) -> None:
        self._turn_lock = threading.RLock()
        self._response_id_to_req_id = {}
        self.logger = SimpleNamespace(
            warning=lambda *_args, **_kwargs: None,
            exception=lambda *_args, **_kwargs: None,
        )
        self.deleted = []
        self.forgotten = []
        self.phases = []
        self.followups = []

    def _transport_host(self):
        return self

    def _send_event(self, payload):
        self.deleted.append(payload)

    def _forget_history_item(self, turn, item_id):
        self.forgotten.append((turn.req_id, item_id))

    def _set_turn_phase(self, turn, phase, *, trigger="set_turn_phase"):
        turn.phase = phase
        self.phases.append((phase, trigger))

    def _send_response_create(self, turn):
        self.followups.append(turn.req_id)


def _turn() -> QueuedTurn:
    return QueuedTurn(
        kind="audio",
        req_id="rt-response",
        speech_end_perf_s=0.0,
        speech_end_unix_s=0.0,
        transcript_perf_s=0.0,
    )


def test_response_lifecycle_retries_no_audio_and_deletes_silent_item() -> None:
    host = _Host()
    runtime = ResponseLifecycleRuntime(host)
    turn = _turn()
    turn.response_id = "resp-1"
    turn.assistant_transcript = "text only"
    host._response_id_to_req_id["resp-1"] = turn.req_id

    did_retry = runtime.retry_no_audio_response(
        turn,
        {"output": [{"id": "asst-1", "type": "message"}]},
    )

    assert did_retry is True
    assert turn.response_id == ""
    assert turn.assistant_transcript == ""
    assert "resp-1" not in host._response_id_to_req_id
    assert host.deleted == [{"type": "conversation.item.delete", "item_id": "asst-1"}]
    assert host.forgotten == [(turn.req_id, "asst-1")]
    assert host.phases == [(TURN_PHASE_REQUESTING_FOLLOWUP, "no_audio_retry")]
    assert host.followups == [turn.req_id]
