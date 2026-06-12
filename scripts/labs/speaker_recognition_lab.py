#!/usr/bin/env python3
"""Standalone speaker-enrollment and recognition lab.

cd ~/argos-agent
source setup_shell.sh

# Enroll 3 clips into the temp lab DB
poetry run python -m scripts.labs.speaker_recognition_lab enroll --person-id person_me --clips 3

# Recognize one live clip against that temp lab DB
poetry run python -m scripts.labs.speaker_recognition_lab recognize --clips 1

# If you want to test final face/audio arbitration too
poetry run python -m scripts.labs.speaker_recognition_lab recognize \
  --clips 1 \
  --primary-face-person-id person_me \
  --visible-face-person-id person_me

# Use prerecorded WAVs instead of the mic
poetry run python -m scripts.labs.speaker_recognition_lab enroll \
  --person-id person_me \
  --audio-file /path/to/clip1.wav \
  --audio-file /path/to/clip2.wav

# Inspect saved refs in the temp session DB
poetry run python -m scripts.labs.speaker_recognition_lab list

# Reset the temp lab session
poetry run python -m scripts.labs.speaker_recognition_lab reset -y


This helper uses the real Argos speaker policy, VAD-based voiced trimming,
embedding backend, and speaker-reference matching, but it avoids the main
runtime and uses a separate lab session directory by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.labs.speaker_lab_common import (
    add_common_args,
    add_policy_override_args,
    build_lab_config,
    build_speaker_service,
    build_vad,
    capture_microphone_utterance,
    configure_logging,
    diagnose_enrollment_attempt,
    diagnose_recognition_attempt,
    ensure_session_dirs,
    inspect_vad_frames,
    json_print,
    load_audio_file_as_agent_pcm16,
    reset_session_dir,
    save_report,
    session_summary_payload,
    summarize_attempt_diagnostics,
    timestamp_label,
    write_pcm16_wav,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test Argos speaker enrollment and recognition without starting the full agent."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    enroll = subparsers.add_parser(
        "enroll",
        help="Capture or load enrollment clips and save/update one speaker reference.",
    )
    add_common_args(enroll)
    add_policy_override_args(enroll)
    enroll.add_argument(
        "--person-id",
        required=True,
        help="Speaker reference id to save. Use your face person_id if you want the ids to line up.",
    )
    enroll.add_argument(
        "--clips",
        type=int,
        default=1,
        help="How many microphone clips to capture. Ignored when --audio-file is used. Default: 1.",
    )
    enroll.add_argument(
        "--audio-file",
        action="append",
        default=[],
        help="Optional WAV file to enroll instead of microphone capture. Repeatable.",
    )

    recognize = subparsers.add_parser(
        "recognize",
        help="Capture or load a query clip and inspect speaker matching.",
    )
    add_common_args(recognize)
    add_policy_override_args(recognize)
    recognize.add_argument(
        "--clips",
        type=int,
        default=1,
        help="How many microphone clips to capture. Ignored when --audio-file is used. Default: 1.",
    )
    recognize.add_argument(
        "--audio-file",
        action="append",
        default=[],
        help="Optional WAV file to recognize instead of microphone capture. Repeatable.",
    )
    recognize.add_argument(
        "--primary-face-person-id",
        default="",
        help="Optional mocked primary visible face id for final arbitration diagnostics.",
    )
    recognize.add_argument(
        "--visible-face-person-id",
        action="append",
        default=[],
        help="Optional mocked visible face id. Repeat to simulate several visible people.",
    )
    recognize.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many scored matches to print. Default: 5.",
    )

    list_cmd = subparsers.add_parser(
        "list",
        help="List saved speaker references in the lab session DB.",
    )
    add_common_args(list_cmd)
    add_policy_override_args(list_cmd)

    reset = subparsers.add_parser(
        "reset",
        help="Delete the lab session directory and its temporary speaker DB.",
    )
    add_common_args(reset)
    add_policy_override_args(reset)
    reset.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Delete without prompting.",
    )

    return parser


def _capture_or_load_audio(
    *,
    config,
    vad: object,
    source_path: str | None,
    clip_index: int,
    clip_count: int,
) -> dict[str, Any]:
    if source_path:
        try:
            audio_pcm16, source_meta = load_audio_file_as_agent_pcm16(source_path)
        except Exception as exc:
            return {
                "success": False,
                "failure_reason": "audio_file_load_failed",
                "message": f"Could not load WAV file '{source_path}': {exc}",
                "source": "file",
                "clip_index": clip_index,
                "clip_count": clip_count,
            }
        return {
            "success": True,
            "audio_pcm16": audio_pcm16,
            "source": "file",
            "source_meta": source_meta,
            "clip_index": clip_index,
            "clip_count": clip_count,
        }
    prompt = (
        f"[{clip_index}/{clip_count}] Press Enter, then say 2-3 sentences. "
        "Recording will stop after silence."
    )
    input(prompt + "\n")
    captured = capture_microphone_utterance(config, vad=vad)
    captured["source"] = "microphone"
    captured["clip_index"] = clip_index
    captured["clip_count"] = clip_count
    return captured


def _persist_attempt_artifacts(
    *,
    config,
    attempt_key: str,
    raw_audio_pcm16: bytes,
    trimmed_audio_pcm16: bytes,
    payload: dict[str, Any],
) -> dict[str, str]:
    raw_path = write_pcm16_wav(
        Path(config.clips_dir) / f"{attempt_key}_raw_16k.wav",
        raw_audio_pcm16,
    )
    trimmed_path = write_pcm16_wav(
        Path(config.clips_dir) / f"{attempt_key}_trimmed_16k.wav",
        trimmed_audio_pcm16,
    )
    report_path = save_report(
        Path(config.reports_dir) / f"{attempt_key}.json",
        payload,
    )
    return {
        "raw_wav_path": raw_path,
        "trimmed_wav_path": trimmed_path,
        "report_path": report_path,
    }


def _run_enroll(args: argparse.Namespace) -> int:
    configure_logging(args.verbose)
    config = build_lab_config(args)
    ensure_session_dirs(config)
    capture_vad, vad_impl = build_vad(config.vad_threshold)
    analysis_vad, analysis_vad_impl = build_vad(config.vad_threshold)
    json_print(
        {
            "mode": "enroll",
            "config": session_summary_payload(config, vad_impl=analysis_vad_impl),
            "person_id": str(args.person_id).strip(),
            "audio_files": list(args.audio_file or []),
            "clips": int(args.clips),
        }
    )
    service = build_speaker_service(config)
    try:
        source_files = [str(path) for path in (args.audio_file or []) if str(path).strip()]
        total_clips = len(source_files) if source_files else max(1, int(args.clips))
        for clip_index in range(1, total_clips + 1):
            source_path = source_files[clip_index - 1] if source_files else None
            capture = _capture_or_load_audio(
                config=config,
                vad=capture_vad,
                source_path=source_path,
                clip_index=clip_index,
                clip_count=total_clips,
            )
            if not capture.get("success", False):
                json_print(capture)
                continue
            raw_audio_pcm16 = bytes(capture.get("audio_pcm16") or b"")
            trimmed_audio_pcm16 = service.trim_turn_audio(raw_audio_pcm16, vad=analysis_vad)
            attempt_key = f"{timestamp_label()}_enroll_{clip_index:02d}"
            raw_vad_frames = inspect_vad_frames(raw_audio_pcm16, vad=analysis_vad)
            trimmed_vad_frames = inspect_vad_frames(trimmed_audio_pcm16, vad=analysis_vad)
            payload = {
                "mode": "enroll_attempt",
                "config": session_summary_payload(config, vad_impl=analysis_vad_impl),
                "person_id": str(args.person_id).strip(),
                "clip_index": int(clip_index),
                "clip_count": int(total_clips),
                "source": str(capture.get("source") or ""),
                "capture": {
                    key: value
                    for key, value in capture.items()
                    if key not in {"success", "audio_pcm16"}
                },
                "vad_frames_raw": raw_vad_frames,
                "vad_frames_trimmed": trimmed_vad_frames,
            }
            attempt_diagnostics = diagnose_enrollment_attempt(
                service,
                person_id=str(args.person_id).strip(),
                raw_audio_pcm16=raw_audio_pcm16,
                trimmed_audio_pcm16=trimmed_audio_pcm16,
            )
            payload.update(attempt_diagnostics)
            payload["diagnostics"] = summarize_attempt_diagnostics(
                policy=service.policy,
                vad_impl=analysis_vad_impl,
                raw_audio_pcm16=raw_audio_pcm16,
                trimmed_audio_pcm16=trimmed_audio_pcm16,
                raw_vad_frames=raw_vad_frames,
                trimmed_vad_frames=trimmed_vad_frames,
                capture_vad_positive_blocks=int(capture.get("vad_positive_blocks", 0) or 0),
                enrollment_rejection=str(
                    ((attempt_diagnostics.get("enrollment_gate") or {}).get("rejection_reason") or "")
                ),
            )
            payload["artifacts"] = _persist_attempt_artifacts(
                config=config,
                attempt_key=attempt_key,
                raw_audio_pcm16=raw_audio_pcm16,
                trimmed_audio_pcm16=trimmed_audio_pcm16,
                payload=payload,
            )
            json_print(payload)
        final_record = service.db.get_reference(str(args.person_id).strip())
        json_print(
            {
                "summary": "enroll_complete",
                "person_id": str(args.person_id).strip(),
                "reference_saved": bool(final_record),
                "metadata": dict((final_record or {}).get("metadata") or {}),
            }
        )
        return 0
    finally:
        service.shutdown()


def _run_recognize(args: argparse.Namespace) -> int:
    configure_logging(args.verbose)
    config = build_lab_config(args)
    ensure_session_dirs(config)
    capture_vad, vad_impl = build_vad(config.vad_threshold)
    analysis_vad, analysis_vad_impl = build_vad(config.vad_threshold)
    json_print(
        {
            "mode": "recognize",
            "config": session_summary_payload(config, vad_impl=analysis_vad_impl),
            "audio_files": list(args.audio_file or []),
            "clips": int(args.clips),
            "face_context": {
                "primary_face_person_id": str(args.primary_face_person_id or "").strip() or None,
                "visible_face_person_ids": [
                    str(item or "").strip()
                    for item in (args.visible_face_person_id or [])
                    if str(item or "").strip()
                ],
            },
        }
    )
    service = build_speaker_service(config)
    try:
        source_files = [str(path) for path in (args.audio_file or []) if str(path).strip()]
        total_clips = len(source_files) if source_files else max(1, int(args.clips))
        for clip_index in range(1, total_clips + 1):
            source_path = source_files[clip_index - 1] if source_files else None
            capture = _capture_or_load_audio(
                config=config,
                vad=capture_vad,
                source_path=source_path,
                clip_index=clip_index,
                clip_count=total_clips,
            )
            if not capture.get("success", False):
                json_print(capture)
                continue
            raw_audio_pcm16 = bytes(capture.get("audio_pcm16") or b"")
            trimmed_audio_pcm16 = service.trim_turn_audio(raw_audio_pcm16, vad=analysis_vad)
            attempt_key = f"{timestamp_label()}_recognize_{clip_index:02d}"
            raw_vad_frames = inspect_vad_frames(raw_audio_pcm16, vad=analysis_vad)
            trimmed_vad_frames = inspect_vad_frames(trimmed_audio_pcm16, vad=analysis_vad)
            payload = {
                "mode": "recognize_attempt",
                "config": session_summary_payload(config, vad_impl=analysis_vad_impl),
                "clip_index": int(clip_index),
                "clip_count": int(total_clips),
                "source": str(capture.get("source") or ""),
                "capture": {
                    key: value
                    for key, value in capture.items()
                    if key not in {"success", "audio_pcm16"}
                },
                "vad_frames_raw": raw_vad_frames,
                "vad_frames_trimmed": trimmed_vad_frames,
            }
            attempt_diagnostics = diagnose_recognition_attempt(
                service,
                raw_audio_pcm16=raw_audio_pcm16,
                trimmed_audio_pcm16=trimmed_audio_pcm16,
                primary_face_person_id=(
                    str(args.primary_face_person_id or "").strip() or None
                ),
                visible_face_person_ids=tuple(
                    str(item or "").strip()
                    for item in (args.visible_face_person_id or [])
                    if str(item or "").strip()
                ),
                top_k=max(1, int(args.top_k)),
            )
            payload.update(attempt_diagnostics)
            decision_inputs = dict(attempt_diagnostics.get("decision_inputs") or {})
            payload["diagnostics"] = summarize_attempt_diagnostics(
                policy=service.policy,
                vad_impl=analysis_vad_impl,
                raw_audio_pcm16=raw_audio_pcm16,
                trimmed_audio_pcm16=trimmed_audio_pcm16,
                raw_vad_frames=raw_vad_frames,
                trimmed_vad_frames=trimmed_vad_frames,
                capture_vad_positive_blocks=int(capture.get("vad_positive_blocks", 0) or 0),
                query_safe=bool((attempt_diagnostics.get("query_gate") or {}).get("accepted")),
                top_score=float(decision_inputs.get("top_score", 0.0) or 0.0),
                reference_count=int(decision_inputs.get("reference_count", 0) or 0),
            )
            payload["artifacts"] = _persist_attempt_artifacts(
                config=config,
                attempt_key=attempt_key,
                raw_audio_pcm16=raw_audio_pcm16,
                trimmed_audio_pcm16=trimmed_audio_pcm16,
                payload=payload,
            )
            json_print(payload)
        return 0
    finally:
        service.shutdown()


def _run_list(args: argparse.Namespace) -> int:
    configure_logging(args.verbose)
    config = build_lab_config(args)
    ensure_session_dirs(config)
    vad, vad_impl = build_vad(config.vad_threshold)
    json_print(
        {
            "mode": "list",
            "config": session_summary_payload(config, vad_impl=vad_impl),
        }
    )
    service = build_speaker_service(config)
    try:
        references = service.db.list_all_references()
        json_print(
            {
                "summary": "list_references",
                "reference_count": int(len(references)),
                "references": references,
            }
        )
        return 0
    finally:
        service.shutdown()


def _run_reset(args: argparse.Namespace) -> int:
    configure_logging(args.verbose)
    config = build_lab_config(args)
    if not args.yes:
        rendered = input(
            f"Delete lab session directory '{config.session_dir}'? [y/N] "
        ).strip().lower()
        if rendered not in {"y", "yes"}:
            print("Aborted.")
            return 1
    reset_session_dir(config)
    json_print(
        {
            "summary": "session_reset",
            "session_dir": config.session_dir,
            "deleted": True,
        }
    )
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "enroll":
        return _run_enroll(args)
    if args.command == "recognize":
        return _run_recognize(args)
    if args.command == "list":
        return _run_list(args)
    if args.command == "reset":
        return _run_reset(args)
    parser.error(f"Unsupported command: {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
