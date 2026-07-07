from argos_src.observability.dashboard_data import (
    build_dashboard_snapshot,
    parse_latency_line,
    read_latency_rows,
)


def _row(line: str) -> dict[str, str]:
    return parse_latency_line(line)


def test_dashboard_snapshot_groups_sessions_interactions_and_state() -> None:
    rows = [
        _row(
            "ts=2026-07-07 10:00:00.000 | component=realtime | "
            "event=recording_started | session_id=s-1 | req_id=rt-1"
        ),
        _row(
            "ts=2026-07-07 10:00:00.100 | component=state | event=transition | "
            "session_id=s-1 | req_id=rt-1 | axis=capture | old_state=idle | "
            "new_state=recording | trigger=recording_started"
        ),
        _row(
            "ts=2026-07-07 10:00:00.420 | component=realtime | "
            "metric=first_audio_latency_s | duration_s=0.320 | session_id=s-1 | req_id=rt-1"
        ),
        _row(
            "ts=2026-07-07 10:00:00.500 | component=tool | event=tool_result | "
            "tool=capture_scene | session_id=s-1 | req_id=rt-1"
        ),
        _row(
            "ts=2026-07-07 10:00:00.600 | component=realtime | event=response_usage | "
            "estimated_cost_usd=0.00100000 | session_total_cost_usd=0.00300000 | "
            "session_id=s-1 | req_id=rt-1"
        ),
        _row(
            "ts=2026-07-07 10:00:01.000 | component=state | event=ignored | "
            "session_id=s-1 | axis=coalescer | trigger=timer_flush | "
            "ignored_reason=recording_active"
        ),
    ]

    snapshot = build_dashboard_snapshot(rows, source="sample.log")

    assert snapshot["source"] == "sample.log"
    assert snapshot["summary"]["session_count"] == 1
    assert snapshot["summary"]["interaction_count"] == 1
    assert snapshot["summary"]["first_audio_latency_p50_s"] == 0.32
    assert snapshot["summary"]["latest_session_total_cost_usd"] == 0.003
    assert snapshot["summary"]["state_axis_counts"]["capture"] == 1
    assert snapshot["summary"]["ignored_reason_counts"]["recording_active"] == 1

    interaction = snapshot["interactions"][0]
    assert interaction["req_id"] == "rt-1"
    assert interaction["status"] == "complete"
    assert interaction["first_audio_latency_s"] == 0.32
    assert interaction["tools"] == {"capture_scene": 1}
    assert interaction["state_transitions"] == [
        {
            "axis": "capture",
            "old_state": "idle",
            "new_state": "recording",
            "trigger": "recording_started",
            "ts": "2026-07-07 10:00:00.100",
        }
    ]
    assert snapshot["system_events"][0]["ignored_reason"] == "recording_active"


def test_dashboard_snapshot_marks_error_interactions() -> None:
    rows = [
        _row(
            "ts=2026-07-07 10:00:00.000 | component=realtime | "
            "event=response_create | req_id=rt-err"
        ),
        _row(
            "ts=2026-07-07 10:00:00.100 | component=realtime | "
            "event=response_failed | req_id=rt-err"
        ),
        _row(
            "ts=2026-07-07 10:00:00.200 | component=realtime | "
            "event=websocket_error"
        ),
    ]

    snapshot = build_dashboard_snapshot(rows)

    assert snapshot["interactions"][0]["status"] == "error"
    assert snapshot["summary"]["error_count"] == 2
    assert [error["label"] for error in snapshot["errors"]] == [
        "response_failed",
        "websocket_error",
    ]


def test_dashboard_snapshot_reports_latest_cost_not_max_cost() -> None:
    rows = [
        _row(
            "ts=2026-07-07 10:00:00.000 | component=realtime | event=response_usage | "
            "session_id=s-old | req_id=rt-old | session_total_cost_usd=1.00000000"
        ),
        _row(
            "ts=2026-07-07 10:05:00.000 | component=realtime | event=response_usage | "
            "session_id=s-new | req_id=rt-new | session_total_cost_usd=0.05000000"
        ),
    ]

    snapshot = build_dashboard_snapshot(rows)

    assert snapshot["summary"]["latest_session_total_cost_usd"] == 0.05


def test_read_latency_rows_returns_empty_for_missing_log(tmp_path) -> None:
    assert read_latency_rows(tmp_path / "missing.log") == []
