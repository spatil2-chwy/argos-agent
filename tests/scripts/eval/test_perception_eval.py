from __future__ import annotations

from pathlib import Path

import pytest

from scripts.eval.perception_eval import evaluate_runs, write_eval_outputs
from scripts.labs.perception_lab_common import append_jsonl, write_json


def _metric(result, component: str, metric: str):
    for row in result["metrics"]:
        if row["component"] == component and row["metric"] == metric:
            return row
    raise AssertionError(f"missing metric {component}.{metric}")


def _write_run(run_dir: Path, samples: list[dict], labels: list[dict]) -> None:
    run_dir.mkdir(parents=True)
    write_json(run_dir / "run_manifest.json", {"profile": "test"})
    for sample in samples:
        append_jsonl(run_dir / "samples.jsonl", sample)
    for label in labels:
        append_jsonl(run_dir / "labels.todo.jsonl", label)


def test_evaluate_perception_metrics_from_labeled_fixture(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    _write_run(
        run_dir,
        [
            {
                "sample_id": "s1",
                "artifacts": {"image_path": "s1.png"},
                "components": {
                    "face_detection": {"measured": True, "detected_count": 1},
                    "face_enrollment": {"measured": True, "accepted": True},
                    "face_recognition": {
                        "measured": True,
                        "predictions": [
                            {
                                "face_index": 0,
                                "match": {
                                    "person_id": "alice",
                                    "name": "Alice",
                                    "similarity": 0.82,
                                },
                            }
                        ],
                    },
                    "depth_gate": {"measured": True, "accepted": True},
                    "attention_gate": {
                        "measured": True,
                        "predictions": [
                            {"attentive": True, "yaw_deg": 5.0, "pitch_deg": 0.0}
                        ],
                    },
                    "audio_detection": {
                        "measured": True,
                        "speech_detected": True,
                        "raw_frame_rms": {"rms_p90": 600.0},
                    },
                },
            },
            {
                "sample_id": "s2",
                "artifacts": {"image_path": "s2.png"},
                "components": {
                    "face_detection": {"measured": True, "detected_count": 1},
                    "face_enrollment": {"measured": True, "accepted": True},
                    "face_recognition": {
                        "measured": True,
                        "predictions": [
                            {
                                "face_index": 0,
                                "match": {
                                    "person_id": "bob",
                                    "name": "Bob",
                                    "similarity": 0.7,
                                },
                            }
                        ],
                    },
                    "depth_gate": {"measured": True, "accepted": True},
                    "attention_gate": {
                        "measured": True,
                        "predictions": [
                            {"attentive": True, "yaw_deg": 8.0, "pitch_deg": 0.0}
                        ],
                    },
                    "audio_detection": {
                        "measured": True,
                        "speech_detected": True,
                        "raw_frame_rms": {"rms_p90": 650.0},
                    },
                },
            },
            {
                "sample_id": "s3",
                "artifacts": {"image_path": "s3.png"},
                "components": {
                    "face_detection": {"measured": True, "detected_count": 0},
                    "face_enrollment": {"measured": True, "accepted": False},
                    "face_recognition": {"measured": True, "predictions": []},
                    "depth_gate": {"measured": True, "accepted": False},
                    "attention_gate": {"measured": True, "predictions": []},
                    "audio_detection": {
                        "measured": True,
                        "speech_detected": False,
                        "raw_frame_rms": {"rms_p90": 50.0},
                    },
                },
            },
        ],
        [
            {
                "sample_id": "s1",
                "labels": {
                    "actual_face_count": 1,
                    "should_accept_for_enrollment": "yes",
                    "actual_person_id": "alice",
                    "should_pass_depth_gate": "yes",
                    "should_be_attentive": "yes",
                    "contains_speech": "yes",
                },
            },
            {
                "sample_id": "s2",
                "labels": {
                    "actual_face_count": 0,
                    "should_accept_for_enrollment": "no",
                    "actual_person_id": "unknown",
                    "should_pass_depth_gate": "no",
                    "approx_distance_bucket": "too_far",
                    "should_be_attentive": "no",
                    "contains_speech": "no",
                },
            },
            {
                "sample_id": "s3",
                "labels": {
                    "actual_face_count": 0,
                    "should_accept_for_enrollment": "no",
                    "actual_person_id": "unknown",
                    "should_pass_depth_gate": "no",
                    "approx_distance_bucket": "invalid_depth",
                    "should_be_attentive": "no",
                    "contains_speech": "no",
                    "speech_quality": "noisy",
                },
            },
        ],
    )

    result = evaluate_runs([run_dir])

    assert _metric(result, "face_detection", "recall")["value"] == 1.0
    assert _metric(result, "face_detection", "false_positive_rate")["value"] == 0.5
    assert _metric(result, "face_enrollment", "bad_frame_reject_rate")["value"] == 0.5
    assert _metric(result, "face_recognition", "known_person_recall")["value"] == 1.0
    assert _metric(result, "face_recognition", "unknown_false_accept_rate")["value"] == 0.5
    assert _metric(result, "depth_gate", "too_far_reject_rate")["value"] == 0.0
    assert _metric(result, "attention_gate", "non_attentive_false_open_rate")["value"] == 0.5
    assert _metric(result, "audio_detection", "silence_noise_false_positive_rate")["value"] == 0.5
    assert any(row["component"] == "face_recognition" for row in result["threshold_sweeps"])
    assert len(result["failures"]) >= 4


def test_evaluate_reports_disabled_components_as_not_measured(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_disabled"
    _write_run(
        run_dir,
        [
            {
                "sample_id": "s1",
                "components": {
                    "depth_gate": {
                        "measured": False,
                        "skipped_reason": "depth_gate_disabled",
                    }
                },
            }
        ],
        [{"sample_id": "s1", "labels": {}}],
    )

    result = evaluate_runs([run_dir])

    assert result["not_measured"] == {"depth_gate:depth_gate_disabled": 1}


def test_evaluate_rejects_label_for_unknown_sample(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad_run"
    _write_run(
        run_dir,
        [{"sample_id": "s1", "components": {}}],
        [{"sample_id": "missing", "labels": {}}],
    )

    with pytest.raises(ValueError, match="unknown sample_id"):
        evaluate_runs([run_dir])


def test_write_eval_outputs_creates_expected_files(tmp_path: Path) -> None:
    result = {
        "sample_count": 0,
        "labeled_sample_count": 0,
        "metrics": [],
        "failures": [],
        "threshold_sweeps": [],
        "not_measured": {},
        "targets": {},
    }

    write_eval_outputs(result, tmp_path)

    assert (tmp_path / "eval_report.md").exists()
    assert (tmp_path / "eval_report.json").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "failures.csv").exists()
    assert (tmp_path / "threshold_sweeps.csv").exists()
