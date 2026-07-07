from argos_src.agent.control.types import CaptureState, TurnState
from argos_src.agent.realtime_turns import (
    TURN_PHASE_COMMITTED,
    TURN_PHASE_RESPONSE_REQUESTED,
    TURN_PHASE_WAITING_FIRST_AUDIO,
)


def test_realtime_turn_phase_constants_use_typed_turn_axis_values() -> None:
    assert TURN_PHASE_COMMITTED == TurnState.COMMITTED.value
    assert TURN_PHASE_RESPONSE_REQUESTED == TurnState.RESPONSE_REQUESTED.value
    assert TURN_PHASE_WAITING_FIRST_AUDIO == TurnState.WAITING_FIRST_OUTPUT.value


def test_capture_state_names_are_dashboard_stable() -> None:
    assert CaptureState.RECORDING.value == "recording"
    assert CaptureState.COMMITTING.value == "committing"
    assert CaptureState.COMMITTED.value == "committed"
