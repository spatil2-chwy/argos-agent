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
            "ts=2026-07-07 10:00:00.450 | component=realtime | event=tool_call_requested | "
            "tool=capture_scene | call_id=call-1 | tool_arguments_json={\"mode\":\"quick\"} | "
            "session_id=s-1 | req_id=rt-1"
        ),
        _row(
            "ts=2026-07-07 10:00:00.500 | component=tool | event=tool_result | "
            "tool=capture_scene | call_id=call-1 | tool_success=True | "
            "tool_result_preview={\"success\":true} | session_id=s-1 | req_id=rt-1"
        ),
        _row(
            "ts=2026-07-07 10:00:00.600 | component=realtime | event=response_usage | "
            "estimated_cost_usd=0.00100000 | session_total_cost_usd=0.00300000 | "
            "session_id=s-1 | req_id=rt-1"
        ),
        _row(
            "ts=2026-07-07 10:00:00.700 | component=realtime | event=exchange_context | "
            "audio_score=0.620 | audio_runner_up_score=0.210 | audio_score_margin=0.410 | "
            "face_match_status=rejected | face_match_reason=below_threshold | "
            "face_match_name=Alice | face_match_person_id=person-1 | "
            "face_score=0.420 | face_score_threshold=0.600 | "
            "face_runner_up_score=0.310 | face_score_margin=0.110 | "
            "face_margin_threshold=0.200 | "
            "owner_source=audio | session_id=s-1 | req_id=rt-1"
        ),
        _row(
            "ts=2026-07-07 10:00:00.760 | component=identity_memory | "
            "event=adaptive_biometric_update | session_id=s-1 | req_id=rt-1 | "
            "biometric_update_modality=voice | biometric_update_person_id=person-1 | "
            "biometric_update_accepted=True | biometric_update_status=updated | "
            "biometric_update_reason=updated | biometric_update_sample_count=2 | "
            "biometric_update_target_sample_count=5 | biometric_update_similarity=0.910"
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
    assert snapshot["summary"]["raw_session_count"] == 1
    assert snapshot["summary"]["exchange_count"] == 1
    assert snapshot["summary"]["interaction_count"] == 1
    assert snapshot["summary"]["conversation_segment_count"] == 1
    assert snapshot["summary"]["first_audio_latency_p50_s"] == 0.32
    assert snapshot["summary"]["first_audio_latency_max_s"] == 0.32
    assert snapshot["summary"]["latest_session_total_cost_usd"] == 0.003
    assert snapshot["summary"]["total_logged_cost_usd"] == 0.003
    assert snapshot["summary"]["state_axis_counts"]["capture"] == 1
    assert snapshot["summary"]["ignored_reason_counts"]["recording_active"] == 1

    interaction = snapshot["interactions"][0]
    assert snapshot["exchanges"][0]["exchange_id"] == "rt-1"
    assert interaction["conversation_segment_id"] == "s-1:conversation:1"
    assert interaction["owner_key"] == "anonymous"
    assert interaction["req_id"] == "rt-1"
    assert interaction["status"] == "complete"
    assert interaction["first_audio_latency_s"] == 0.32
    assert interaction["costs"]["estimated_exchange_cost_usd"] == 0.001
    identity_stage = next(stage for stage in interaction["lifecycle"] if stage["key"] == "identity")
    assert identity_stage["details"]["audio_score"] == "0.620"
    assert identity_stage["details"]["audio_runner_up_score"] == "0.210"
    assert identity_stage["details"]["audio_score_margin"] == "0.410"
    assert identity_stage["details"]["face_match_status"] == "rejected"
    assert identity_stage["details"]["face_match_reason"] == "below_threshold"
    assert identity_stage["details"]["face_score"] == "0.420"
    assert identity_stage["details"]["face_score_margin"] == "0.110"
    biometric_stage = next(stage for stage in interaction["lifecycle"] if stage["key"] == "biometric_update")
    assert biometric_stage["details"]["biometric_update_modality"] == "voice"
    assert biometric_stage["details"]["biometric_update_accepted"] == "True"
    assert biometric_stage["details"]["biometric_update_sample_count"] == "2"
    assert biometric_stage["details"]["biometric_update_target_sample_count"] == "5"
    assert biometric_stage["details"]["biometric_update_similarity"] == "0.910"
    assert interaction["tools"] == {"capture_scene": 1}
    assert interaction["state_by_axis"] == [
        {
            "axis": "capture",
            "transitions": [
                {
                    "axis": "capture",
                    "old_state": "idle",
                    "new_state": "recording",
                    "trigger": "recording_started",
                    "ts": "2026-07-07 10:00:00.100",
                }
            ],
            "ignored": [],
        }
    ]
    assert interaction["tool_calls"] == [
        {
            "call_id": "call-1",
            "tool": "capture_scene",
            "requested_at": "2026-07-07 10:00:00.450",
            "finished_at": "2026-07-07 10:00:00.500",
            "arguments_json": "{\"mode\":\"quick\"}",
            "result_preview": "{\"success\":true}",
            "success": "True",
        }
    ]
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


def test_dashboard_snapshot_groups_consecutive_exchanges_by_owner_segment() -> None:
    rows = [
        _row(
            "ts=2026-07-07 10:00:00.000 | component=realtime | event=exchange_context | "
            "run_id=run-live | exchange_id=ex-1 | exchange_index=1 | req_id=rt-1 | "
            "turn_kind=human_audio | owner_id=person-a | owner_source=face"
        ),
        _row(
            "ts=2026-07-07 10:00:00.200 | component=realtime | metric=first_audio_latency_s | "
            "duration_s=0.200 | run_id=run-live | exchange_id=ex-1 | exchange_index=1 | "
            "req_id=rt-1 | turn_kind=human_audio"
        ),
        _row(
            "ts=2026-07-07 10:01:00.000 | component=realtime | event=exchange_context | "
            "run_id=run-live | exchange_id=ex-2 | exchange_index=2 | req_id=rt-2 | "
            "turn_kind=human_audio | owner_id=person-a | owner_source=audio"
        ),
        _row(
            "ts=2026-07-07 10:02:00.000 | component=realtime | event=exchange_context | "
            "run_id=run-live | exchange_id=ex-3 | exchange_index=3 | req_id=rt-3 | "
            "turn_kind=human_audio | owner_source=unknown"
        ),
        _row(
            "ts=2026-07-07 10:03:00.000 | component=realtime | event=exchange_context | "
            "run_id=run-live | exchange_id=ex-4 | exchange_index=4 | req_id=rt-4 | "
            "turn_kind=human_audio | owner_id=person-b | owner_source=audio"
        ),
    ]

    snapshot = build_dashboard_snapshot(rows)

    assert snapshot["summary"]["conversation_segment_count"] == 3
    segments = sorted(
        snapshot["conversation_segments"],
        key=lambda segment: segment["segment_index"],
    )
    assert [
        (segment["owner_key"], segment["exchange_ids"], segment["boundary_reason"])
        for segment in segments
    ] == [
        ("owner:person-a", ["ex-1", "ex-2"], "session_start"),
        ("anonymous", ["ex-3"], "owner_handoff"),
        ("owner:person-b", ["ex-4"], "owner_handoff"),
    ]
    assert segments[0]["owner_source_counts"] == {"face": 1, "audio": 1}
    assert segments[0]["avg_first_audio_latency_s"] == 0.2
    exchanges = sorted(snapshot["exchanges"], key=lambda exchange: exchange["exchange_index"])
    assert [exchange["conversation_segment_index"] for exchange in exchanges] == [1, 2, 1, 1]


def test_dashboard_snapshot_merges_req_rows_with_missing_session_id() -> None:
    rows = [
        _row(
            "ts=2026-07-07 10:00:00.000 | component=realtime | "
            "event=audio_commit | req_id=rt-split"
        ),
        _row(
            "ts=2026-07-07 10:00:00.100 | component=realtime | "
            "event=response_create | req_id=rt-split"
        ),
        _row(
            "ts=2026-07-07 10:00:00.300 | component=realtime | "
            "event=response_usage | session_id=sess-live | req_id=rt-split"
        ),
    ]

    snapshot = build_dashboard_snapshot(rows)

    assert snapshot["summary"]["exchange_count"] == 1
    assert len(snapshot["sessions"]) == 1
    assert snapshot["sessions"][0]["session_id"] == "sess-live"
    assert snapshot["sessions"][0]["exchange_count"] == 1
    assert snapshot["sessions"][0]["session_total_cost_usd"] is None
    exchange = snapshot["exchanges"][0]
    assert exchange["req_id"] == "rt-split"
    assert exchange["status"] == "complete"
    assert [stage["key"] for stage in exchange["lifecycle"]] == [
        "audio_commit",
        "model_requested",
        "response_usage",
    ]


def test_dashboard_snapshot_merges_state_rows_by_request_exchange_mapping() -> None:
    rows = [
        _row(
            "ts=2026-07-07 10:00:00.000 | component=realtime | "
            "event=recording_started | run_id=run-live | session_id=sess-live | "
            "exchange_id=ex-live | exchange_index=1 | turn_kind=human_audio"
        ),
        _row(
            "ts=2026-07-07 10:00:01.000 | component=realtime | event=audio_commit | "
            "run_id=run-live | session_id=sess-live | exchange_id=ex-live | "
            "exchange_index=1 | req_id=rt-live | turn_kind=human_audio"
        ),
        _row(
            "ts=2026-07-07 10:00:01.050 | component=state | event=transition | "
            "req_id=rt-live | axis=engagement | old_state=idle | "
            "new_state=engaged | trigger=human_input"
        ),
        _row(
            "ts=2026-07-07 10:00:02.000 | component=realtime | event=exchange_complete | "
            "run_id=run-live | session_id=sess-live | exchange_id=ex-live | "
            "exchange_index=1 | req_id=rt-live | terminal_status=complete | "
            "turn_kind=human_audio"
        ),
    ]

    snapshot = build_dashboard_snapshot(rows)

    assert snapshot["summary"]["session_count"] == 1
    assert snapshot["summary"]["raw_session_count"] == 1
    assert snapshot["summary"]["exchange_count"] == 1
    exchange = snapshot["exchanges"][0]
    assert exchange["exchange_id"] == "ex-live"
    assert exchange["session_id"] == "run-live"
    assert exchange["status"] == "complete"
    assert exchange["state_transitions"] == [
        {
            "axis": "engagement",
            "old_state": "idle",
            "new_state": "engaged",
            "trigger": "human_input",
            "ts": "2026-07-07 10:00:01.050",
        }
    ]


def test_dashboard_snapshot_attaches_legacy_recording_rows_to_next_audio_turn() -> None:
    rows = [
        _row(
            "ts=2026-07-07 10:00:00.000 | component=realtime | "
            "event=recording_started | admission_reason=wake_word"
        ),
        _row(
            "ts=2026-07-07 10:00:01.000 | component=realtime | event=speech_end"
        ),
        _row(
            "ts=2026-07-07 10:00:01.100 | component=realtime | "
            "event=audio_commit | req_id=rt-legacy"
        ),
    ]

    snapshot = build_dashboard_snapshot(rows)

    exchange = snapshot["exchanges"][0]
    assert exchange["req_id"] == "rt-legacy"
    assert [stage["key"] for stage in exchange["lifecycle"]] == [
        "recording",
        "speech_end",
        "audio_commit",
    ]


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
    assert snapshot["sessions"][0]["error_count"] == 2
    assert [error["label"] for error in snapshot["errors"]] == [
        "response_failed",
        "websocket_error",
    ]


def test_dashboard_snapshot_exposes_server_error_details() -> None:
    rows = [
        _row(
            "ts=2026-07-07 10:00:00.000 | component=realtime | "
            "event=response_create | req_id=rt-server"
        ),
        _row(
            "ts=2026-07-07 10:00:00.100 | component=realtime | "
            "event=exchange_terminal | req_id=rt-server | terminal_status=error | "
            "terminal_reason=server_error | error_source=openai_realtime | "
            "error_type=invalid_request_error | error_message=Some other realtime request failed | "
            "server_error_type=invalid_request_error | "
            "server_error_message=Some other realtime request failed"
        ),
    ]

    snapshot = build_dashboard_snapshot(rows)

    interaction = snapshot["interactions"][0]
    assert interaction["status"] == "error"
    assert interaction["context"]["error_source"] == "openai_realtime"
    assert interaction["context"]["error_type"] == "invalid_request_error"
    assert interaction["context"]["error_message"] == "Some other realtime request failed"
    assert interaction["context"]["server_error_type"] == "invalid_request_error"
    assert interaction["context"]["server_error_message"] == "Some other realtime request failed"
    terminal_stage = next(stage for stage in interaction["lifecycle"] if stage["key"] == "exchange_terminal")
    assert terminal_stage["details"]["error_source"] == "openai_realtime"
    assert terminal_stage["details"]["error_type"] == "invalid_request_error"
    assert terminal_stage["details"]["error_message"] == "Some other realtime request failed"
    assert terminal_stage["details"]["server_error_type"] == "invalid_request_error"
    assert terminal_stage["details"]["server_error_message"] == "Some other realtime request failed"


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


def test_dashboard_snapshot_sums_latest_session_costs_for_cost_to_date() -> None:
    rows = [
        _row(
            "ts=2026-07-07 10:00:00.000 | component=realtime | event=response_usage | "
            "run_id=run-a | session_id=s-a | exchange_id=ex-a | req_id=rt-a | "
            "turn_kind=human_audio | session_total_cost_usd=0.10000000"
        ),
        _row(
            "ts=2026-07-07 10:01:00.000 | component=realtime | event=response_usage | "
            "run_id=run-a | session_id=s-a | exchange_id=ex-a2 | req_id=rt-a2 | "
            "turn_kind=human_audio | session_total_cost_usd=0.15000000"
        ),
        _row(
            "ts=2026-07-07 10:05:00.000 | component=realtime | event=response_usage | "
            "run_id=run-b | session_id=s-b | exchange_id=ex-b | req_id=rt-b | "
            "turn_kind=human_audio | session_total_cost_usd=0.05000000"
        ),
    ]

    snapshot = build_dashboard_snapshot(rows)

    assert snapshot["summary"]["latest_session_total_cost_usd"] == 0.05
    assert snapshot["summary"]["total_logged_cost_usd"] == 0.2
    costs_by_session = {
        session["session_id"]: session["session_total_cost_usd"]
        for session in snapshot["sessions"]
    }
    assert costs_by_session == {"run-a": 0.15, "run-b": 0.05}


def test_dashboard_snapshot_counts_only_exchange_sessions_in_primary_list() -> None:
    rows = [
        _row(
            "ts=2026-07-07 10:00:00.000 | component=state | event=transition | "
            "session_id=sess-startup | axis=session | old_state=stopped | "
            "new_state=ready | trigger=session.updated"
        ),
        _row(
            "ts=2026-07-07 10:00:05.000 | component=realtime | event=audio_commit | "
            "run_id=run-live | session_id=sess-live | exchange_id=ex-live | "
            "req_id=rt-live | turn_kind=human_audio"
        ),
    ]

    snapshot = build_dashboard_snapshot(rows)

    assert snapshot["summary"]["session_count"] == 1
    assert snapshot["summary"]["raw_session_count"] == 2
    assert [session["session_id"] for session in snapshot["sessions"]] == ["run-live"]


def test_read_latency_rows_returns_empty_for_missing_log(tmp_path) -> None:
    assert read_latency_rows(tmp_path / "missing.log") == []
