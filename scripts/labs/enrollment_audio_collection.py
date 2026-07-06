#!/usr/bin/env python3
"""Collect raw per-person microphone clips for enrollment/model comparison."""

from __future__ import annotations

import argparse
import audioop
import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from argos_src.speaker_recognition.policy import SAMPLE_RATE
from scripts.labs.enrollment_collection_common import (
    add_person_collection_args,
    create_display_runtime_for_profile,
    json_ready,
    load_profile,
    resolve_collection_session,
    write_session_manifest,
)
from scripts.labs.perception_lab_common import append_jsonl, write_json
from scripts.labs.speaker_lab_common import (
    AUDIO_CHANNELS,
    AUDIO_DTYPE,
    SpeakerLabConfig,
    build_lab_config,
    build_vad,
    configure_logging,
    render_frame_rms_payload,
    render_stats_payload,
    session_summary_payload,
    write_pcm16_wav,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect raw microphone clips for one person's speaker dataset."
    )
    add_person_collection_args(parser)
    parser.add_argument("--clips", type=int, default=5)
    parser.add_argument("--input-device", default="")
    parser.add_argument("--input-sample-rate", type=int, default=None)
    parser.add_argument("--input-block-size", type=int, default=None)
    parser.add_argument("--vad-threshold", type=float, default=None)
    parser.add_argument("--silence-grace-period", type=float, default=None)
    parser.add_argument("--listen-timeout-sec", type=float, default=10.0)
    parser.add_argument("--max-record-sec", type=float, default=8.0)
    parser.add_argument(
        "--prompt",
        default="Please say your name and one short sentence.",
        help="Prompt shown before each clip.",
    )
    return parser


def _capture_microphone_utterance_raw(
    config: SpeakerLabConfig,
    *,
    vad: object,
    on_listening: Callable[[], None] | None = None,
    on_recording_start: Callable[[], None] | None = None,
    on_recording_stop: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - local env dependent.
        return {
            "success": False,
            "failure_reason": "missing_sounddevice",
            "message": f"sounddevice is required for microphone mode: {exc}",
        }

    resample_state = None
    listening_started_at = time.time()
    recording_started_at = 0.0
    last_voice_at = 0.0
    recording = False
    stop_reason = ""
    status_flags: list[str] = []
    source_chunks: list[bytes] = []
    agent_chunks: list[bytes] = []
    vad_positive_blocks = 0

    try:
        if callable(on_listening):
            on_listening()
        with sd.InputStream(
            samplerate=int(config.input_sample_rate),
            blocksize=int(config.input_block_size),
            channels=AUDIO_CHANNELS,
            dtype=AUDIO_DTYPE,
            device=config.input_device or None,
        ) as stream:
            print("Mic admission active. Listening...")
            while True:
                chunk, overflowed = stream.read(int(config.input_block_size))
                if overflowed:
                    status_flags.append("overflowed")
                source_chunk = np.asarray(chunk, dtype=np.int16).reshape(-1).tobytes()
                try:
                    resampled, resample_state = audioop.ratecv(
                        source_chunk,
                        np.dtype(np.int16).itemsize,
                        AUDIO_CHANNELS,
                        int(config.input_sample_rate),
                        SAMPLE_RATE,
                        resample_state,
                    )
                    audio_16k = np.frombuffer(resampled, dtype=np.int16)
                except Exception as exc:
                    return {
                        "success": False,
                        "failure_reason": "resample_failed",
                        "message": f"Could not resample microphone audio: {exc}",
                    }

                try:
                    voice_detected, _ = vad(audio_16k, {})
                except Exception:
                    voice_detected = False

                now = time.time()
                if not recording:
                    if voice_detected:
                        recording = True
                        recording_started_at = now
                        last_voice_at = now
                        source_chunks.append(source_chunk)
                        agent_chunks.append(resampled)
                        vad_positive_blocks = 1
                        print("Recording...")
                        if callable(on_recording_start):
                            on_recording_start()
                        continue
                    if (now - listening_started_at) >= float(config.listen_timeout_sec):
                        return {
                            "success": False,
                            "failure_reason": "listen_timeout",
                            "message": "No speech was detected before the listen timeout.",
                        }
                    continue

                source_chunks.append(source_chunk)
                agent_chunks.append(resampled)
                if voice_detected:
                    last_voice_at = now
                    vad_positive_blocks += 1
                elif (now - last_voice_at) >= float(config.silence_grace_period):
                    stop_reason = "silence"
                    break

                if (now - recording_started_at) >= float(config.max_record_sec):
                    stop_reason = "max_record_sec"
                    break
    except Exception as exc:  # pragma: no cover - local env dependent.
        return {
            "success": False,
            "failure_reason": "microphone_open_failed",
            "message": f"Could not open the microphone input stream: {exc}",
        }

    if callable(on_recording_stop):
        on_recording_stop(stop_reason or "completed")
    source_audio_pcm16 = b"".join(source_chunks)
    agent_audio_pcm16 = b"".join(agent_chunks)
    if not source_audio_pcm16 or not agent_audio_pcm16:
        return {
            "success": False,
            "failure_reason": "empty_capture",
            "message": "Speech was detected, but the captured clip is empty.",
        }
    return {
        "success": True,
        "source_audio_pcm16": source_audio_pcm16,
        "agent_audio_pcm16": agent_audio_pcm16,
        "source_sample_rate_hz": int(config.input_sample_rate),
        "agent_sample_rate_hz": SAMPLE_RATE,
        "duration_s": round(
            float(len(source_audio_pcm16)) / float(2 * int(config.input_sample_rate)),
            4,
        ),
        "listen_started_at_unix_s": round(listening_started_at, 3),
        "recording_started_at_unix_s": round(recording_started_at, 3),
        "stop_reason": stop_reason or "completed",
        "status_flags": status_flags,
        "vad_positive_blocks": int(vad_positive_blocks),
    }


def _sample_payload(
    *,
    sample_id: str,
    capture: dict[str, Any],
    audio_dir: Path,
) -> dict[str, Any]:
    if not capture.get("success"):
        return {
            "sample_id": sample_id,
            "modality": "audio",
            "capture": capture,
            "artifacts": {},
        }

    source_audio = bytes(capture.get("source_audio_pcm16") or b"")
    agent_audio = bytes(capture.get("agent_audio_pcm16") or b"")
    source_rate = int(capture.get("source_sample_rate_hz") or 0)
    source_path = write_pcm16_wav(
        audio_dir / f"{sample_id}_input_{source_rate}hz.wav",
        source_audio,
        sample_rate=source_rate,
    )
    agent_path = write_pcm16_wav(
        audio_dir / f"{sample_id}_agent_{SAMPLE_RATE}hz.wav",
        agent_audio,
        sample_rate=SAMPLE_RATE,
    )
    cleaned_capture = {
        key: value
        for key, value in capture.items()
        if key not in {"source_audio_pcm16", "agent_audio_pcm16"}
    }
    sample = {
        "sample_id": sample_id,
        "modality": "audio",
        "captured_at_unix_s": round(time.time(), 3),
        "capture": cleaned_capture,
        "audio": {
            "source_stats": {
                "duration_s": round(float(len(source_audio)) / float(2 * source_rate), 4)
                if source_rate > 0
                else 0.0,
                "sample_rate_hz": source_rate,
                "sample_count": int(len(source_audio) // 2),
            },
            "agent_stats": render_stats_payload(agent_audio),
            "agent_frame_rms": render_frame_rms_payload(agent_audio),
        },
        "artifacts": {
            "source_wav_path": source_path,
            "agent_16k_wav_path": agent_path,
        },
    }
    metadata_path = audio_dir / f"{sample_id}.json"
    write_json(metadata_path, sample)
    sample["artifacts"]["metadata_path"] = str(metadata_path)
    return sample


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(bool(args.verbose))

    session = resolve_collection_session(
        output_root=args.output_root,
        person_name=args.person_name,
        person_id=args.person_id,
        session_id=args.session_id,
    )
    session_dir = Path(session["session_dir"])
    audio_dir = session_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    samples_path = session_dir / "audio_samples.jsonl"
    samples_path.write_text("", encoding="utf-8")

    profile = load_profile(args.profile)
    setattr(args, "session_dir", str(session_dir / "_speaker_lab_session"))
    config = build_lab_config(args)
    vad, vad_impl = build_vad(config.vad_threshold)

    display = create_display_runtime_for_profile(
        profile,
        disabled=bool(args.no_display),
        provider_transport=args.provider_transport,
    )
    write_session_manifest(
        session_dir=session_dir,
        filename="audio_manifest.json",
        payload={
            "collection_kind": "audio",
            "person_name": session["person_name"],
            "person_slug": session["person_slug"],
            "session_id": session["session_id"],
            "profile": profile.name,
            "profile_arg": args.profile,
            "requested_clips": int(args.clips),
            "prompt": str(args.prompt or ""),
            "audio": session_summary_payload(config, vad_impl=vad_impl),
        },
    )

    successes = 0
    total = max(1, int(args.clips))
    try:
        if display is not None:
            display.show_message(f"Audio collection: {session['person_name']}")
        for index in range(1, total + 1):
            sample_id = f"audio_{index:04d}"
            print(f"[{index}/{total}] {args.prompt}")
            input("Press Enter to arm mic recording, then have the person speak.\n")
            if display is not None:
                display.show_message("Mic admission active")
                display.show_subtitle(f"Audio {index}/{total}")
            capture = _capture_microphone_utterance_raw(
                config,
                vad=vad,
                on_listening=(
                    (lambda: display.show_message("Mic admission active"))
                    if display is not None
                    else None
                ),
                on_recording_start=display.show_recording if display is not None else None,
                on_recording_stop=(
                    (lambda reason: display.show_message("Saving audio..."))
                    if display is not None
                    else None
                ),
            )
            sample = _sample_payload(sample_id=sample_id, capture=capture, audio_dir=audio_dir)
            sample["person_name"] = session["person_name"]
            sample["person_slug"] = session["person_slug"]
            sample["session_id"] = session["session_id"]
            append_jsonl(samples_path, json_ready(sample))
            if capture.get("success"):
                successes += 1
                if display is not None:
                    display.show_message(f"Saved audio {index}/{total}")
            else:
                if display is not None:
                    display.show_message("Audio not saved")
            print(
                {
                    "sample_id": sample_id,
                    "success": bool(capture.get("success")),
                    "duration_s": capture.get("duration_s"),
                    "stop_reason": capture.get("stop_reason"),
                    "failure_reason": capture.get("failure_reason"),
                }
            )
        if display is not None:
            display.show_message(f"Saved audio: {successes}/{total}")
    finally:
        if display is not None:
            display.shutdown()

    print(f"Wrote audio collection: {session_dir}")
    return 0 if successes > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
