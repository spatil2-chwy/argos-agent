from argos_src.observability.state_report import parse_line, summarize_state_rows


def test_state_report_summarizes_transitions_and_ignored_events() -> None:
    rows = [
        parse_line(
            "ts=2026-07-07 10:00:00.000 | component=state | event=transition | "
            "axis=turn | old_state=committed | new_state=response_requested | "
            "trigger=set_turn_phase | req_id=rt-1"
        ),
        parse_line(
            "ts=2026-07-07 10:00:00.001 | component=state | event=transition | "
            "axis=turn | old_state=committed | new_state=response_requested | "
            "trigger=set_turn_phase | req_id=rt-2"
        ),
        parse_line(
            "ts=2026-07-07 10:00:00.002 | component=state | event=ignored | "
            "axis=coalescer | trigger=timer_flush | ignored_reason=recording_active"
        ),
        parse_line(
            "ts=2026-07-07 10:00:00.003 | component=realtime | event=response_create"
        ),
    ]

    summary = summarize_state_rows(rows)

    assert summary["transitions"][
        "turn:committed->response_requested:set_turn_phase"
    ] == 2
    assert summary["ignored"]["coalescer:timer_flush:recording_active"] == 1
    assert summary["ignored_reasons"]["recording_active"] == 1
