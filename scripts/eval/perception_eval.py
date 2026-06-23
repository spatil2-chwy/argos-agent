#!/usr/bin/env python3
"""Evaluate structured perception lab runs against human labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.labs.perception_lab_common import (
    REPO_ROOT,
    make_run_id,
    read_jsonl,
    write_json,
    yes_no_label,
)


DEFAULT_TARGETS = {
    "face_detection.recall": 0.90,
    "face_recognition.known_person_recall": 0.90,
    "face_recognition.unknown_false_accept_rate": 0.02,
    "face_enrollment.bad_frame_reject_rate": 0.95,
    "depth_gate.too_far_reject_rate": 0.95,
    "attention_gate.non_attentive_false_open_rate": 0.05,
    "audio_detection.speech_recall": 0.95,
    "audio_detection.silence_noise_false_positive_rate": 0.02,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate perception lab samples against human labels."
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to var/eval/perception/<eval_id>.",
    )
    return parser


def _label_payload(row: dict[str, Any]) -> dict[str, Any]:
    labels = row.get("labels", row)
    if not isinstance(labels, dict):
        raise ValueError(f"Label row {row.get('sample_id', '<missing>')} has non-object labels")
    return labels


def _has_label(labels: dict[str, Any], key: str) -> bool:
    return key in labels and labels.get(key) not in (None, "")


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _bool_label(labels: dict[str, Any], key: str) -> bool | None:
    return yes_no_label(labels.get(key))


def _load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest_path = run_dir / "run_manifest.json"
    samples_path = run_dir / "samples.jsonl"
    labels_path = run_dir / "labels.jsonl"
    if not labels_path.exists():
        labels_path = run_dir / "labels.todo.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples file: {samples_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = read_jsonl(samples_path)
    labels = read_jsonl(labels_path)
    label_map: dict[str, dict[str, Any]] = {}
    sample_ids = {str(sample.get("sample_id", "")) for sample in samples}
    for row in labels:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"{labels_path} contains a label row without sample_id")
        if sample_id not in sample_ids:
            raise ValueError(f"{labels_path} contains label for unknown sample_id={sample_id}")
        label_map[sample_id] = row
    return manifest, samples, label_map


def _iter_labeled(
    samples: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    label_key: str,
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for sample in samples:
        sample_id = str(sample.get("sample_id", ""))
        row = labels.get(sample_id)
        if row is None:
            continue
        payload = _label_payload(row)
        if _has_label(payload, label_key):
            yield sample, payload


def _component(sample: dict[str, Any], name: str) -> dict[str, Any]:
    component = (sample.get("components") or {}).get(name) or {}
    return component if isinstance(component, dict) else {}


def _metric(
    rows: list[dict[str, Any]],
    component: str,
    name: str,
    value: float | None,
    numerator: int | None = None,
    denominator: int | None = None,
) -> None:
    rows.append(
        {
            "component": component,
            "metric": name,
            "value": "" if value is None else round(float(value), 6),
            "numerator": "" if numerator is None else int(numerator),
            "denominator": "" if denominator is None else int(denominator),
            "target": DEFAULT_TARGETS.get(f"{component}.{name}", ""),
            "passed": _passed(f"{component}.{name}", value),
        }
    )


def _passed(metric_key: str, value: float | None) -> str:
    if value is None or metric_key not in DEFAULT_TARGETS:
        return ""
    target = DEFAULT_TARGETS[metric_key]
    if "false" in metric_key:
        return "yes" if value <= target else "no"
    return "yes" if value >= target else "no"


def evaluate_runs(run_dirs: list[Path]) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    labels: dict[str, dict[str, Any]] = {}
    duplicate_counts: dict[str, int] = {}
    for run_dir in run_dirs:
        manifest, run_samples, run_labels = _load_run(run_dir)
        manifests.append(manifest)
        for sample in run_samples:
            original_id = str(sample.get("sample_id", ""))
            scoped_id = f"{run_dir.name}:{original_id}"
            duplicate_counts[scoped_id] = duplicate_counts.get(scoped_id, 0) + 1
            copied = dict(sample)
            copied["sample_id"] = scoped_id
            copied["run_dir"] = str(run_dir)
            samples.append(copied)
            if original_id in run_labels:
                row = dict(run_labels[original_id])
                row["sample_id"] = scoped_id
                labels[scoped_id] = row

    metrics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sweeps: list[dict[str, Any]] = []

    _eval_face_detection(samples, labels, metrics, failures)
    _eval_face_enrollment(samples, labels, metrics, failures)
    _eval_face_recognition(samples, labels, metrics, failures, sweeps)
    _eval_depth_gate(samples, labels, metrics, failures)
    _eval_attention_gate(samples, labels, metrics, failures, sweeps)
    _eval_audio_detection(samples, labels, metrics, failures, sweeps)

    not_measured = _not_measured_components(samples)
    return {
        "run_dirs": [str(path) for path in run_dirs],
        "manifests": manifests,
        "sample_count": len(samples),
        "labeled_sample_count": len(labels),
        "metrics": metrics,
        "failures": failures,
        "threshold_sweeps": sweeps,
        "not_measured": not_measured,
        "targets": DEFAULT_TARGETS,
    }


def _eval_face_detection(samples, labels, metrics, failures) -> None:
    total_present = detected_present = 0
    total_absent = false_positive = 0
    count_total = count_correct = 0
    for sample, row in _iter_labeled(samples, labels, "actual_face_count"):
        actual = int(row.get("actual_face_count") or 0)
        predicted = int(_component(sample, "face_detection").get("detected_count", 0) or 0)
        if actual > 0:
            total_present += 1
            if predicted > 0:
                detected_present += 1
            else:
                failures.append(_failure(sample, "face_detection", "missed_face", actual, predicted))
        else:
            total_absent += 1
            if predicted > 0:
                false_positive += 1
                failures.append(_failure(sample, "face_detection", "false_positive_face", actual, predicted))
        count_total += 1
        if actual == predicted:
            count_correct += 1
    _metric(metrics, "face_detection", "recall", _ratio(detected_present, total_present), detected_present, total_present)
    _metric(metrics, "face_detection", "false_positive_rate", _ratio(false_positive, total_absent), false_positive, total_absent)
    _metric(metrics, "face_detection", "count_accuracy", _ratio(count_correct, count_total), count_correct, count_total)


def _eval_face_enrollment(samples, labels, metrics, failures) -> None:
    good_total = good_accept = 0
    bad_total = bad_reject = 0
    for sample, row in _iter_labeled(samples, labels, "should_accept_for_enrollment"):
        expected = _bool_label(row, "should_accept_for_enrollment")
        if expected is None:
            continue
        predicted = bool(_component(sample, "face_enrollment").get("accepted", False))
        if expected:
            good_total += 1
            if predicted:
                good_accept += 1
            else:
                failures.append(_failure(sample, "face_enrollment", "false_reject", True, predicted))
        else:
            bad_total += 1
            if not predicted:
                bad_reject += 1
            else:
                failures.append(_failure(sample, "face_enrollment", "false_accept", False, predicted))
    _metric(metrics, "face_enrollment", "good_frame_accept_rate", _ratio(good_accept, good_total), good_accept, good_total)
    _metric(metrics, "face_enrollment", "bad_frame_reject_rate", _ratio(bad_reject, bad_total), bad_reject, bad_total)
    _metric(metrics, "face_enrollment", "false_reject_rate", _ratio(good_total - good_accept, good_total), good_total - good_accept, good_total)
    _metric(metrics, "face_enrollment", "false_accept_rate", _ratio(bad_total - bad_reject, bad_total), bad_total - bad_reject, bad_total)


def _predicted_person(sample: dict[str, Any], threshold: float | None = None) -> tuple[str, float]:
    best_id = "unknown"
    best_score = 0.0
    for item in (_component(sample, "face_recognition").get("predictions") or []):
        match = item.get("match")
        if not match:
            continue
        score = float(match.get("similarity", 0.0) or 0.0)
        if score > best_score:
            best_score = score
            best_id = str(match.get("person_id") or "unknown")
    if threshold is not None and best_score < threshold:
        return "unknown", best_score
    return best_id, best_score


def _eval_face_recognition(samples, labels, metrics, failures, sweeps) -> None:
    total = correct = 0
    known_total = known_correct = 0
    unknown_total = unknown_rejected = false_accept = 0
    known_wrong_identity = 0
    labeled_pairs: list[tuple[dict[str, Any], str]] = []
    for sample, row in _iter_labeled(samples, labels, "actual_person_id"):
        actual = str(row.get("actual_person_id") or "unknown").strip() or "unknown"
        predicted, _score = _predicted_person(sample)
        override = yes_no_label(row.get("recognition_correct"))
        is_correct = override if override is not None else predicted == actual
        total += 1
        correct += int(bool(is_correct))
        labeled_pairs.append((sample, actual))
        if actual == "unknown":
            unknown_total += 1
            if predicted == "unknown":
                unknown_rejected += 1
            else:
                false_accept += 1
                failures.append(_failure(sample, "face_recognition", "unknown_false_accept", actual, predicted))
        else:
            known_total += 1
            if predicted == actual:
                known_correct += 1
            elif predicted != "unknown":
                known_wrong_identity += 1
                failures.append(_failure(sample, "face_recognition", "wrong_identity", actual, predicted))
            else:
                failures.append(_failure(sample, "face_recognition", "known_false_reject", actual, predicted))
    _metric(metrics, "face_recognition", "top1_accuracy", _ratio(correct, total), correct, total)
    _metric(metrics, "face_recognition", "known_person_recall", _ratio(known_correct, known_total), known_correct, known_total)
    _metric(metrics, "face_recognition", "unknown_reject_rate", _ratio(unknown_rejected, unknown_total), unknown_rejected, unknown_total)
    _metric(metrics, "face_recognition", "unknown_false_accept_rate", _ratio(false_accept, unknown_total), false_accept, unknown_total)
    _metric(metrics, "face_recognition", "false_identity_rate", _ratio(known_wrong_identity, known_total), known_wrong_identity, known_total)
    for threshold in _float_range(0.4, 0.9, 0.05):
        known_ok = known_n = unknown_fp = unknown_n = 0
        for sample, actual in labeled_pairs:
            predicted, _score = _predicted_person(sample, threshold=threshold)
            if actual == "unknown":
                unknown_n += 1
                unknown_fp += int(predicted != "unknown")
            else:
                known_n += 1
                known_ok += int(predicted == actual)
        sweeps.append(
            {
                "component": "face_recognition",
                "parameter": "recognition_threshold",
                "threshold": round(threshold, 3),
                "known_person_recall": _ratio(known_ok, known_n),
                "unknown_false_accept_rate": _ratio(unknown_fp, unknown_n),
            }
        )


def _eval_depth_gate(samples, labels, metrics, failures) -> None:
    pass_total = pass_ok = fail_total = fail_ok = 0
    too_far_total = too_far_reject = 0
    invalid_total = invalid_reject = 0
    for sample, row in _iter_labeled(samples, labels, "should_pass_depth_gate"):
        expected = _bool_label(row, "should_pass_depth_gate")
        if expected is None:
            continue
        predicted = bool(_component(sample, "depth_gate").get("accepted", False))
        bucket = str(row.get("approx_distance_bucket") or "").strip().lower()
        if expected:
            pass_total += 1
            pass_ok += int(predicted)
            if not predicted:
                failures.append(_failure(sample, "depth_gate", "false_reject", True, predicted))
        else:
            fail_total += 1
            fail_ok += int(not predicted)
            if predicted:
                failures.append(_failure(sample, "depth_gate", "false_accept", False, predicted))
        if bucket == "too_far":
            too_far_total += 1
            too_far_reject += int(not predicted)
        if bucket == "invalid_depth":
            invalid_total += 1
            invalid_reject += int(not predicted)
    _metric(metrics, "depth_gate", "valid_accept_rate", _ratio(pass_ok, pass_total), pass_ok, pass_total)
    _metric(metrics, "depth_gate", "reject_rate", _ratio(fail_ok, fail_total), fail_ok, fail_total)
    _metric(metrics, "depth_gate", "too_far_reject_rate", _ratio(too_far_reject, too_far_total), too_far_reject, too_far_total)
    _metric(metrics, "depth_gate", "invalid_depth_reject_rate", _ratio(invalid_reject, invalid_total), invalid_reject, invalid_total)


def _attention_predicted(sample: dict[str, Any], yaw_threshold: float | None = None) -> bool:
    predictions = _component(sample, "attention_gate").get("predictions") or []
    if yaw_threshold is None:
        return any(bool(item.get("attentive")) for item in predictions)
    for item in predictions:
        yaw = item.get("yaw_deg")
        if yaw is None:
            continue
        if abs(float(yaw)) <= yaw_threshold:
            return True
    return False


def _eval_attention_gate(samples, labels, metrics, failures, sweeps) -> None:
    attentive_total = attentive_ok = 0
    non_total = non_false_open = 0
    tp = fp = 0
    labeled_pairs: list[tuple[dict[str, Any], bool]] = []
    for sample, row in _iter_labeled(samples, labels, "should_be_attentive"):
        expected = _bool_label(row, "should_be_attentive")
        if expected is None:
            continue
        predicted = _attention_predicted(sample)
        labeled_pairs.append((sample, expected))
        if expected:
            attentive_total += 1
            attentive_ok += int(predicted)
            tp += int(predicted)
            if not predicted:
                failures.append(_failure(sample, "attention_gate", "false_reject", True, predicted))
        else:
            non_total += 1
            non_false_open += int(predicted)
            fp += int(predicted)
            if predicted:
                failures.append(_failure(sample, "attention_gate", "false_open", False, predicted))
    _metric(metrics, "attention_gate", "attentive_recall", _ratio(attentive_ok, attentive_total), attentive_ok, attentive_total)
    _metric(metrics, "attention_gate", "non_attentive_false_open_rate", _ratio(non_false_open, non_total), non_false_open, non_total)
    _metric(metrics, "attention_gate", "precision", _ratio(tp, tp + fp), tp, tp + fp)
    for threshold in (10, 15, 20, 25, 30, 35, 40):
        yes_ok = yes_n = no_fp = no_n = 0
        for sample, expected in labeled_pairs:
            predicted = _attention_predicted(sample, yaw_threshold=float(threshold))
            if expected:
                yes_n += 1
                yes_ok += int(predicted)
            else:
                no_n += 1
                no_fp += int(predicted)
        sweeps.append(
            {
                "component": "attention_gate",
                "parameter": "max_abs_yaw_deg_proxy",
                "threshold": threshold,
                "attentive_recall": _ratio(yes_ok, yes_n),
                "non_attentive_false_open_rate": _ratio(no_fp, no_n),
            }
        )


def _eval_audio_detection(samples, labels, metrics, failures, sweeps) -> None:
    speech_total = speech_ok = 0
    silence_total = false_positive = 0
    quiet_total = quiet_miss = 0
    labeled_pairs: list[tuple[dict[str, Any], bool]] = []
    for sample, row in _iter_labeled(samples, labels, "contains_speech"):
        expected = _bool_label(row, "contains_speech")
        if expected is None:
            continue
        predicted = bool(_component(sample, "audio_detection").get("speech_detected", False))
        labeled_pairs.append((sample, expected))
        quality = str(row.get("speech_quality") or "").strip().lower()
        if expected:
            speech_total += 1
            speech_ok += int(predicted)
            if quality == "quiet":
                quiet_total += 1
                quiet_miss += int(not predicted)
            if not predicted:
                failures.append(_failure(sample, "audio_detection", "missed_speech", True, predicted))
        else:
            silence_total += 1
            false_positive += int(predicted)
            if predicted:
                failures.append(_failure(sample, "audio_detection", "false_positive_audio", False, predicted))
    _metric(metrics, "audio_detection", "speech_recall", _ratio(speech_ok, speech_total), speech_ok, speech_total)
    _metric(metrics, "audio_detection", "silence_noise_false_positive_rate", _ratio(false_positive, silence_total), false_positive, silence_total)
    _metric(metrics, "audio_detection", "quiet_speech_miss_rate", _ratio(quiet_miss, quiet_total), quiet_miss, quiet_total)
    for threshold in (100, 200, 300, 400, 500, 700, 900):
        speech_ok_t = speech_n = silence_fp_t = silence_n = 0
        for sample, expected in labeled_pairs:
            rms = (
                _component(sample, "audio_detection")
                .get("raw_frame_rms", {})
                .get("rms_p90", 0.0)
            )
            predicted = float(rms or 0.0) >= float(threshold)
            if expected:
                speech_n += 1
                speech_ok_t += int(predicted)
            else:
                silence_n += 1
                silence_fp_t += int(predicted)
        sweeps.append(
            {
                "component": "audio_detection",
                "parameter": "rms_p90_proxy_threshold",
                "threshold": threshold,
                "speech_recall": _ratio(speech_ok_t, speech_n),
                "silence_noise_false_positive_rate": _ratio(silence_fp_t, silence_n),
            }
        )


def _failure(sample: dict[str, Any], component: str, reason: str, expected: Any, predicted: Any) -> dict[str, Any]:
    return {
        "sample_id": sample.get("sample_id", ""),
        "run_dir": sample.get("run_dir", ""),
        "component": component,
        "reason": reason,
        "expected": expected,
        "predicted": predicted,
        "artifacts": sample.get("artifacts", {}),
    }


def _float_range(start: float, stop: float, step: float) -> Iterable[float]:
    value = start
    while value <= stop + 1e-9:
        yield round(value, 10)
        value += step


def _not_measured_components(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        for component, payload in (sample.get("components") or {}).items():
            if isinstance(payload, dict) and not payload.get("measured", False):
                reason = str(payload.get("skipped_reason") or "not_measured")
                key = f"{component}:{reason}"
                counts[key] = counts.get(key, 0) + 1
    return counts


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Perception Eval Report",
        "",
        f"- sample_count: {result['sample_count']}",
        f"- labeled_sample_count: {result['labeled_sample_count']}",
        "",
        "## Metrics",
        "",
        "| Component | Metric | Value | Target | Passed |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in result["metrics"]:
        value = row["value"] if row["value"] != "" else "n/a"
        target = row["target"] if row["target"] != "" else ""
        lines.append(
            f"| {row['component']} | {row['metric']} | {value} | {target} | {row['passed']} |"
        )
    lines.extend(["", "## Top Failures", ""])
    for failure in result["failures"][:25]:
        lines.append(
            f"- `{failure['sample_id']}` {failure['component']} {failure['reason']} "
            f"expected={failure['expected']} predicted={failure['predicted']}"
        )
    if not result["failures"]:
        lines.append("- none")
    lines.extend(["", "## Not Measured", ""])
    if result["not_measured"]:
        for key, count in sorted(result["not_measured"].items()):
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_eval_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "eval_report.json", result)
    (output_dir / "eval_report.md").write_text(_render_report(result), encoding="utf-8")
    _write_csv(
        output_dir / "metrics.csv",
        result["metrics"],
        ["component", "metric", "value", "numerator", "denominator", "target", "passed"],
    )
    _write_csv(
        output_dir / "failures.csv",
        result["failures"],
        ["sample_id", "run_dir", "component", "reason", "expected", "predicted", "artifacts"],
    )
    sweep_fields = sorted(
        {
            key
            for row in result["threshold_sweeps"]
            for key in row.keys()
        }
    )
    _write_csv(output_dir / "threshold_sweeps.csv", result["threshold_sweeps"], sweep_fields)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    run_dirs = [Path(path).expanduser().resolve() for path in args.run_dir]
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = REPO_ROOT / "var" / "eval" / "perception" / make_run_id("eval")
    result = evaluate_runs(run_dirs)
    result["created_at_unix_s"] = round(time.time(), 3)
    write_eval_outputs(result, output_dir)
    print(f"Wrote perception eval: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
