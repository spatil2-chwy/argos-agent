#!/usr/bin/env python3
"""Structured audio detection capture lab."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from argos_src.speaker_recognition.policy import trim_voice_activity
from scripts.labs.perception_lab_common import DEFAULT_LAB_ROOT, LabRunWriter
from scripts.labs.speaker_lab_common import (
    add_common_args,
    add_policy_override_args,
    build_lab_config,
    build_vad,
    capture_microphone_utterance,
    configure_logging,
    inspect_vad_frames,
    load_audio_file_as_agent_pcm16,
    render_frame_rms_payload,
    render_stats_payload,
    session_summary_payload,
    write_pcm16_wav,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture structured audio detection lab samples for eval."
    )
    add_common_args(parser)
    add_policy_override_args(parser)
    parser.add_argument("--clips", type=int, default=10)
    parser.add_argument(
        "--audio-file",
        action="append",
        default=[],
        help="Optional WAV file to analyze instead of microphone capture. Repeatable.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_LAB_ROOT),
        help="Root directory for lab runs. Default: var/labs.",
    )
    parser.add_argument("--run-id", default="")
    return parser


def _label_template(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "artifacts": sample.get("artifacts", {}),
        "prediction_summary": {
            "speech_detected": (
                sample.get("components", {})
                .get("audio_detection", {})
                .get("speech_detected")
            ),
            "raw_duration_s": (
                sample.get("components", {})
                .get("audio_detection", {})
                .get("raw_stats", {})
                .get("duration_s")
            ),
            "voiced_fraction": (
                sample.get("components", {})
                .get("audio_detection", {})
                .get("vad_frames_raw", {})
                .get("voiced_fraction")
            ),
        },
        "labels": {
            "contains_speech": None,
            "speech_quality": None,
        },
    }


def _analyze_audio(
    *,
    writer: LabRunWriter,
    sample_id: str,
    raw_audio_pcm16: bytes,
    source: str,
    capture: dict[str, Any],
    vad: object,
    vad_impl: str,
    vad_threshold: float,
) -> dict[str, Any]:
    raw_waveform = np.frombuffer(raw_audio_pcm16 or b"", dtype=np.int16).copy()
    trimmed_waveform = trim_voice_activity(raw_waveform, vad=vad)
    trimmed_audio_pcm16 = trimmed_waveform.astype(np.int16).tobytes()
    raw_path = write_pcm16_wav(
        writer.artifacts_dir / f"{sample_id}_raw.wav",
        raw_audio_pcm16,
    )
    trimmed_path = write_pcm16_wav(
        writer.artifacts_dir / f"{sample_id}_trimmed.wav",
        trimmed_audio_pcm16,
    )
    raw_vad_frames = inspect_vad_frames(raw_audio_pcm16, vad=vad)
    trimmed_vad_frames = inspect_vad_frames(trimmed_audio_pcm16, vad=vad)
    speech_detected = (
        int(raw_vad_frames.get("voiced_frames", 0) or 0) > 0
        or int(capture.get("vad_positive_blocks", 0) or 0) > 0
    )
    return {
        "sample_id": sample_id,
        "source": source,
        "capture": capture,
        "artifacts": {
            "raw_wav_path": raw_path,
            "trimmed_wav_path": trimmed_path,
        },
        "components": {
            "audio_detection": {
                "measured": True,
                "speech_detected": bool(speech_detected),
                "vad_impl": vad_impl,
                "thresholds": {
                    "vad_threshold": float(vad_threshold),
                },
                "vad_frames_raw": raw_vad_frames,
                "vad_frames_trimmed": trimmed_vad_frames,
                "raw_stats": render_stats_payload(raw_audio_pcm16),
                "trimmed_stats": render_stats_payload(trimmed_audio_pcm16),
                "raw_frame_rms": render_frame_rms_payload(raw_audio_pcm16),
                "trimmed_frame_rms": render_frame_rms_payload(trimmed_audio_pcm16),
                "stop_reason": str(capture.get("stop_reason") or ""),
            }
        },
    }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    config = build_lab_config(args)
    vad, vad_impl = build_vad(config.vad_threshold)
    writer = LabRunWriter(
        component="audio",
        mode="detection",
        root=args.output_root,
        run_id=args.run_id or None,
    )
    writer.write_manifest(
        {
            "profile": config.profile_name,
            "profile_arg": args.profile,
            "enabled_components": {
                "face_detection": False,
                "face_enrollment": False,
                "face_recognition": False,
                "depth_gate": False,
                "attention_gate": False,
                "audio_detection": True,
            },
            "audio": session_summary_payload(config, vad_impl=vad_impl),
            "thresholds": {
                "vad_threshold": config.vad_threshold,
                "speaker_policy": asdict(config.policy),
            },
            "audio_files": list(args.audio_file or []),
            "requested_clips": int(args.clips),
        }
    )

    source_files = [str(path) for path in (args.audio_file or []) if str(path).strip()]
    total = len(source_files) if source_files else max(1, int(args.clips))
    successes = 0
    for index in range(1, total + 1):
        sample_id = f"clip_{index:04d}"
        if source_files:
            source_path = source_files[index - 1]
            try:
                raw_audio_pcm16, source_meta = load_audio_file_as_agent_pcm16(source_path)
            except Exception as exc:
                sample = {
                    "sample_id": sample_id,
                    "source": "file",
                    "capture": {
                        "success": False,
                        "failure_reason": "audio_file_load_failed",
                        "message": str(exc),
                    },
                    "artifacts": {},
                    "components": {
                        "audio_detection": {
                            "measured": False,
                            "skipped_reason": "audio_file_load_failed",
                        }
                    },
                }
            else:
                sample = _analyze_audio(
                    writer=writer,
                    sample_id=sample_id,
                    raw_audio_pcm16=raw_audio_pcm16,
                    source="file",
                    capture={"success": True, "source_meta": source_meta},
                    vad=vad,
                    vad_impl=vad_impl,
                    vad_threshold=config.vad_threshold,
                )
                successes += 1
        else:
            input(f"[{index}/{total}] Press Enter, then make/suppress sound for this sample.\n")
            capture = capture_microphone_utterance(config, vad=vad)
            if not capture.get("success"):
                sample = {
                    "sample_id": sample_id,
                    "source": "microphone",
                    "capture": capture,
                    "artifacts": {},
                    "components": {
                        "audio_detection": {
                            "measured": False,
                            "skipped_reason": str(capture.get("failure_reason") or "capture_failed"),
                        }
                    },
                }
            else:
                sample = _analyze_audio(
                    writer=writer,
                    sample_id=sample_id,
                    raw_audio_pcm16=bytes(capture.get("audio_pcm16") or b""),
                    source="microphone",
                    capture={key: value for key, value in capture.items() if key != "audio_pcm16"},
                    vad=vad,
                    vad_impl=vad_impl,
                    vad_threshold=config.vad_threshold,
                )
                successes += 1
        writer.append_sample(sample, _label_template(sample))
        print(
            {
                "sample_id": sample_id,
                "measured": sample["components"]["audio_detection"].get("measured"),
                "speech_detected": sample["components"]["audio_detection"].get("speech_detected"),
            }
        )

    writer.write_quick_summary(
        [
            "# Audio Detection Lab",
            "",
            f"- run_dir: `{writer.run_dir}`",
            f"- analyzed_clips: {successes}/{total}",
            f"- labels: `{writer.labels_path}`",
            "",
            "Edit `labels.todo.jsonl`, then run:",
            "",
            f"```bash\npoetry run python -m scripts.eval.perception_eval --run-dir {writer.run_dir}\n```",
        ]
    )
    print(f"Wrote audio lab run: {writer.run_dir}")
    return 0 if successes > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
