from __future__ import annotations

from argos_src.agent.control.types import StateAxis, StateTransition
from argos_src.observability.state_observer import StructuredStateObserver
from argos_src.observability.state_report import parse_line


def test_structured_state_observer_emits_parseable_state_rows(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "latency.log"
    monkeypatch.setenv("GO2_LATENCY_LOG_PATH", str(log_path))
    monkeypatch.setenv("GO2_LATENCY_CONSOLE", "0")

    observer = StructuredStateObserver()
    observer.transition(
        StateTransition(
            axis=StateAxis.TURN,
            old_state="committed",
            new_state="response_requested",
            trigger="set_turn_phase",
            req_id="rt-1",
            fields={"pending_response_requests": 1},
        )
    )
    observer.ignored(
        axis=StateAxis.COALESCER,
        trigger="timer_flush",
        reason="recording_active",
    )

    rows = [parse_line(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    assert rows[0]["component"] == "state"
    assert rows[0]["event"] == "transition"
    assert rows[0]["axis"] == "turn"
    assert rows[0]["new_state"] == "response_requested"
    assert rows[0]["pending_response_requests"] == "1"
    assert rows[1]["event"] == "ignored"
    assert rows[1]["ignored_reason"] == "recording_active"
