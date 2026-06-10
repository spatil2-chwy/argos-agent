#!/usr/bin/env python3
"""Shared utilities for standalone Argos speaker diagnostic scripts."""

from __future__ import annotations

import argparse
import audioop
import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import shutil
import tempfile
import time
from typing import TYPE_CHECKING
import wave
from typing import Any

import numpy as np

from argos_src.profile_config import load_scenario_profile
from argos_src.speaker_recognition.models import SpeakerRecognitionPolicy
from argos_src.speaker_recognition.policy import (
    FRAME_MS,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    clip_stats,
    enrollment_rejection_reason,
    is_query_clip_safe,
)

if TYPE_CHECKING:
    from argos_src.speaker_recognition.service import SpeakerRecognitionService

AUDIO_CHANNELS = 1
AUDIO_DTYPE = "int16"
DEFAULT_LISTEN_TIMEOUT_SEC = 10.0
DEFAULT_MAX_RECORD_SEC = 8.0
DEFAULT_SESSION_DIR = Path(tempfile.gettempdir()) / "argos_speaker_lab" / "default"


@dataclass(frozen=True)
class SpeakerLabConfig:
    profile_name: str
    session_dir: str
    profile_speaker_db_path: str
    speaker_db_path: str
    clips_dir: str
    reports_dir: str
    input_device: str | None
    input_sample_rate: int
    input_block_size: int
    vad_threshold: float
    silence_grace_period: float
    policy: SpeakerRecognitionPolicy
    listen_timeout_sec: float
    max_record_sec: float


class _RmsVAD:
    """Small fallback VAD when Silero is unavailable locally."""

    def __init__(self, rms_threshold: float) -> None:
        self.rms_threshold = float(rms_threshold)

    def __call__(
        self,
        audio_data: np.ndarray,
        input_parameters: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        samples = np.asarray(audio_data, dtype=np.float32).reshape(-1)
        rms = (
            float(np.sqrt(np.mean(np.square(samples))))
            if samples.size > 0
            else 0.0
        )
        result = dict(input_parameters)
        result["rms_vad"] = {
            "rms": round(rms, 4),
            "threshold": round(self.rms_threshold, 4),
        }
        return rms >= self.rms_threshold, result


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        default="static_interaction",
        help="Argos profile name or YAML path. Default: static_interaction.",
    )
    parser.add_argument(
        "--session-dir",
        default=str(DEFAULT_SESSION_DIR),
        help=(
            "Isolated lab session directory. A temporary speaker DB plus saved "
            "clips/reports live here instead of the agent speaker DB. "
            "Default: /tmp/argos_speaker_lab/default."
        ),
    )
    parser.add_argument(
        "--input-device",
        default="",
        help="Override realtime.input_device for microphone capture.",
    )
    parser.add_argument(
        "--input-sample-rate",
        type=int,
        default=None,
        help="Override realtime.input_sample_rate for microphone capture.",
    )
    parser.add_argument(
        "--input-block-size",
        type=int,
        default=None,
        help="Override realtime.input_block_size for microphone capture.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="Override realtime.vad_threshold.",
    )
    parser.add_argument(
        "--silence-grace-period",
        type=float,
        default=None,
        help="Override realtime.silence_grace_period.",
    )
    parser.add_argument(
        "--listen-timeout-sec",
        type=float,
        default=DEFAULT_LISTEN_TIMEOUT_SEC,
        help="How long to wait for speech before giving up. Default: 10.0.",
    )
    parser.add_argument(
        "--max-record-sec",
        type=float,
        default=DEFAULT_MAX_RECORD_SEC,
        help="Hard cap for one captured utterance. Default: 8.0.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )


def add_policy_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query-min-voiced-sec", type=float, default=None)
    parser.add_argument("--query-match-threshold", type=float, default=None)
    parser.add_argument("--query-margin-threshold", type=float, default=None)
    parser.add_argument("--reference-update-threshold", type=float, default=None)
    parser.add_argument("--enroll-min-voiced-sec", type=float, default=None)
    parser.add_argument("--enroll-max-voiced-sec", type=float, default=None)
    parser.add_argument("--enroll-min-rms-level", type=float, default=None)
    parser.add_argument("--max-clipped-fraction", type=float, default=None)


def build_lab_config(args: argparse.Namespace) -> SpeakerLabConfig:
    profile = load_scenario_profile(args.profile)
    realtime = profile.realtime
    base_policy = profile.speaker_recognition.policy
    replacements: dict[str, Any] = {}
    for attr in (
        "query_min_voiced_sec",
        "query_match_threshold",
        "query_margin_threshold",
        "reference_update_threshold",
        "enroll_min_voiced_sec",
        "enroll_max_voiced_sec",
        "enroll_min_rms_level",
        "max_clipped_fraction",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            replacements[attr] = value
    session_dir = Path(str(args.session_dir or DEFAULT_SESSION_DIR)).expanduser().resolve()
    speaker_db_path = session_dir / "speaker_db"
    clips_dir = session_dir / "clips"
    reports_dir = session_dir / "reports"
    profile_speaker_db_path = str(base_policy.db_path)
    policy = replace(base_policy, db_path=str(speaker_db_path), **replacements)
    return SpeakerLabConfig(
        profile_name=profile.name,
        session_dir=str(session_dir),
        profile_speaker_db_path=profile_speaker_db_path,
        speaker_db_path=str(speaker_db_path),
        clips_dir=str(clips_dir),
        reports_dir=str(reports_dir),
        input_device=(str(args.input_device).strip() or realtime.input_device),
        input_sample_rate=(
            int(args.input_sample_rate)
            if args.input_sample_rate is not None
            else int(realtime.input_sample_rate)
        ),
        input_block_size=(
            int(args.input_block_size)
            if args.input_block_size is not None
            else int(realtime.input_block_size)
        ),
        vad_threshold=(
            float(args.vad_threshold)
            if args.vad_threshold is not None
            else float(realtime.vad_threshold)
        ),
        silence_grace_period=(
            float(args.silence_grace_period)
            if args.silence_grace_period is not None
            else float(realtime.silence_grace_period)
        ),
        policy=policy,
        listen_timeout_sec=float(args.listen_timeout_sec),
        max_record_sec=float(args.max_record_sec),
    )


def ensure_session_dirs(config: SpeakerLabConfig) -> None:
    Path(config.session_dir).mkdir(parents=True, exist_ok=True)
    Path(config.speaker_db_path).mkdir(parents=True, exist_ok=True)
    Path(config.clips_dir).mkdir(parents=True, exist_ok=True)
    Path(config.reports_dir).mkdir(parents=True, exist_ok=True)


def reset_session_dir(config: SpeakerLabConfig) -> None:
    session_dir = Path(config.session_dir)
    if session_dir.exists():
        shutil.rmtree(session_dir)


def build_speaker_service(config: SpeakerLabConfig) -> "SpeakerRecognitionService":
    ensure_session_dirs(config)
    from argos_src.speaker_recognition.service import SpeakerRecognitionService

    return SpeakerRecognitionService(policy=config.policy)


def build_vad(vad_threshold: float) -> tuple[object, str]:
    try:
        from argos_src.audio import SileroVAD
    except Exception:
        return _RmsVAD(rms_threshold=350.0), "rms_fallback"
    return SileroVAD(SAMPLE_RATE, float(vad_threshold)), "silero"


def load_audio_file_as_agent_pcm16(path: str | Path) -> tuple[bytes, dict[str, Any]]:
    audio_path = Path(path).expanduser().resolve()
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = int(wav_file.getnchannels())
        sample_width = int(wav_file.getsampwidth())
        sample_rate = int(wav_file.getframerate())
        frame_count = int(wav_file.getnframes())
        payload = wav_file.readframes(frame_count)
    if channels <= 0 or sample_width <= 0 or sample_rate <= 0:
        raise ValueError(f"Unsupported WAV metadata for {audio_path}")
    mono_payload = payload
    if channels != 1:
        mono_payload = audioop.tomono(mono_payload, sample_width, 0.5, 0.5)
    pcm16_payload = mono_payload
    if sample_width != np.dtype(np.int16).itemsize:
        pcm16_payload = audioop.lin2lin(
            mono_payload,
            sample_width,
            np.dtype(np.int16).itemsize,
        )
    if sample_rate != SAMPLE_RATE:
        pcm16_payload, _ = audioop.ratecv(
            pcm16_payload,
            np.dtype(np.int16).itemsize,
            AUDIO_CHANNELS,
            sample_rate,
            SAMPLE_RATE,
            None,
        )
    return pcm16_payload, {
        "path": str(audio_path),
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate_hz": sample_rate,
        "frame_count": frame_count,
        "converted_to_agent_pcm16": True,
    }


def write_pcm16_wav(path: str | Path, audio_pcm16: bytes, *, sample_rate: int = SAMPLE_RATE) -> str:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as wav_file:
        wav_file.setnchannels(AUDIO_CHANNELS)
        wav_file.setsampwidth(np.dtype(np.int16).itemsize)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(bytes(audio_pcm16 or b""))
    return str(target)


def save_report(path: str | Path, payload: dict[str, Any]) -> str:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return str(target)


def render_stats_payload(audio_pcm16: bytes) -> dict[str, Any]:
    stats = clip_stats(np.frombuffer(audio_pcm16 or b"", dtype=np.int16).copy())
    return {
        "duration_s": round(float(stats.duration_s), 4),
        "rms_level": round(float(stats.rms_level), 4),
        "clipped_fraction": round(float(stats.clipped_fraction), 6),
        "sample_rate_hz": SAMPLE_RATE,
        "sample_count": int(len(audio_pcm16 or b"") // 2),
    }


def render_frame_rms_payload(audio_pcm16: bytes) -> dict[str, Any]:
    waveform = np.frombuffer(audio_pcm16 or b"", dtype=np.int16).copy()
    if waveform.size <= 0:
        return {
            "frame_ms": FRAME_MS,
            "total_frames": 0,
            "rms_p50": 0.0,
            "rms_p75": 0.0,
            "rms_p90": 0.0,
            "frames_rms_gt_100": 0,
            "frames_rms_gt_200": 0,
            "frames_rms_gt_300": 0,
            "frames_rms_gt_400": 0,
        }
    frame_rms: list[float] = []
    for start in range(0, int(waveform.size), FRAME_SAMPLES):
        frame = waveform[start : start + FRAME_SAMPLES]
        if frame.size <= 0:
            continue
        samples = frame.astype(np.float32, copy=False)
        frame_rms.append(float(np.sqrt(np.mean(np.square(samples)))))
    if not frame_rms:
        return {
            "frame_ms": FRAME_MS,
            "total_frames": 0,
            "rms_p50": 0.0,
            "rms_p75": 0.0,
            "rms_p90": 0.0,
            "frames_rms_gt_100": 0,
            "frames_rms_gt_200": 0,
            "frames_rms_gt_300": 0,
            "frames_rms_gt_400": 0,
        }
    frame_rms_np = np.asarray(frame_rms, dtype=np.float32)
    return {
        "frame_ms": FRAME_MS,
        "total_frames": int(frame_rms_np.size),
        "rms_p50": round(float(np.percentile(frame_rms_np, 50)), 4),
        "rms_p75": round(float(np.percentile(frame_rms_np, 75)), 4),
        "rms_p90": round(float(np.percentile(frame_rms_np, 90)), 4),
        "frames_rms_gt_100": int(np.sum(frame_rms_np > 100.0)),
        "frames_rms_gt_200": int(np.sum(frame_rms_np > 200.0)),
        "frames_rms_gt_300": int(np.sum(frame_rms_np > 300.0)),
        "frames_rms_gt_400": int(np.sum(frame_rms_np > 400.0)),
    }


def inspect_vad_frames(audio_pcm16: bytes, *, vad: object | None) -> dict[str, Any]:
    waveform = np.frombuffer(audio_pcm16 or b"", dtype=np.int16).copy()
    frame_samples = int(getattr(vad, "window_size", 0) or 0)
    if frame_samples <= 0:
        frame_samples = FRAME_SAMPLES
    frame_ms = round((1000.0 * float(frame_samples)) / float(SAMPLE_RATE), 1)
    if waveform.size <= 0:
        return {
            "frame_ms": frame_ms,
            "frame_samples": frame_samples,
            "total_frames": 0,
            "voiced_frames": 0,
            "voiced_fraction": 0.0,
        }
    total_frames = 0
    voiced_frames = 0
    context: dict[str, Any] = {}
    for start in range(0, int(waveform.size), frame_samples):
        frame = waveform[start : start + frame_samples]
        if frame.size <= 0:
            continue
        total_frames += 1
        if vad is None:
            continue
        try:
            voiced, _ = vad(frame, context)
        except Exception:
            voiced = False
        if voiced:
            voiced_frames += 1
    voiced_fraction = (
        float(voiced_frames) / float(total_frames)
        if total_frames > 0
        else 0.0
    )
    return {
        "frame_ms": frame_ms,
        "frame_samples": frame_samples,
        "total_frames": int(total_frames),
        "voiced_frames": int(voiced_frames),
        "voiced_fraction": round(voiced_fraction, 4),
    }


def summarize_attempt_diagnostics(
    *,
    policy: SpeakerRecognitionPolicy,
    vad_impl: str,
    raw_audio_pcm16: bytes,
    trimmed_audio_pcm16: bytes,
    raw_vad_frames: dict[str, Any],
    trimmed_vad_frames: dict[str, Any],
    capture_vad_positive_blocks: int = 0,
    query_safe: bool | None = None,
    top_score: float | None = None,
    reference_count: int | None = None,
    enrollment_rejection: str = "",
) -> dict[str, Any]:
    raw_stats = render_stats_payload(raw_audio_pcm16)
    trimmed_stats = render_stats_payload(trimmed_audio_pcm16)
    raw_frame_rms = render_frame_rms_payload(raw_audio_pcm16)
    trimmed_frame_rms = render_frame_rms_payload(trimmed_audio_pcm16)
    kept_ratio = (
        float(len(trimmed_audio_pcm16 or b"")) / float(len(raw_audio_pcm16 or b""))
        if raw_audio_pcm16
        else 0.0
    )
    raw_voiced_frames = int(raw_vad_frames.get("voiced_frames", 0) or 0)
    trimmed_voiced_frames = int(trimmed_vad_frames.get("voiced_frames", 0) or 0)
    notes: list[str] = []
    if vad_impl == "rms_fallback":
        notes.append("using_rms_fallback_vad")
    if raw_audio_pcm16 and raw_voiced_frames <= 0 and abs(kept_ratio - 1.0) <= 1e-4:
        notes.append("trim_used_raw_audio_fallback_no_voiced_frames")
    if capture_vad_positive_blocks > 0 and raw_voiced_frames <= 0:
        notes.append("capture_vad_detected_speech_but_trim_vad_found_none")
    if trimmed_stats["rms_level"] < float(policy.enroll_min_rms_level):
        notes.append("clip_quieter_than_enrollment_min_rms")
    if query_safe and top_score is not None and top_score < (
        float(policy.query_match_threshold) + 0.05
    ):
        notes.append("query_match_is_borderline")
    if query_safe and int(reference_count or 0) <= 1:
        notes.append("single_reference_match_not_discriminative")
    if enrollment_rejection:
        notes.append(str(enrollment_rejection))
    return {
        "vad_impl": vad_impl,
        "kept_ratio": round(kept_ratio, 4),
        "raw_voiced_frames": raw_voiced_frames,
        "trimmed_voiced_frames": trimmed_voiced_frames,
        "raw_stats": raw_stats,
        "trimmed_stats": trimmed_stats,
        "raw_frame_rms": raw_frame_rms,
        "trimmed_frame_rms": trimmed_frame_rms,
        "notes": notes,
    }


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float | None:
    lhs = np.asarray(left, dtype=np.float32).reshape(-1)
    rhs = np.asarray(right, dtype=np.float32).reshape(-1)
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm <= 1e-8 or rhs_norm <= 1e-8:
        return None
    return float(np.dot(lhs / lhs_norm, rhs / rhs_norm))


def diagnose_enrollment_attempt(
    service: "SpeakerRecognitionService",
    *,
    person_id: str,
    raw_audio_pcm16: bytes,
    trimmed_audio_pcm16: bytes,
) -> dict[str, Any]:
    existing = service.db.get_reference(person_id)
    trimmed_waveform = np.frombuffer(trimmed_audio_pcm16 or b"", dtype=np.int16).copy()
    rejection_reason = enrollment_rejection_reason(
        service.policy,
        audio_pcm16=trimmed_waveform,
    )
    consistency_score = None
    if (
        existing is not None
        and existing.get("embedding") is not None
        and not rejection_reason
        and trimmed_waveform.size > 0
    ):
        query_embedding = service.backend.embed_query_clip(
            trimmed_waveform,
            sample_rate=SAMPLE_RATE,
        )
        consistency_score = cosine_similarity(
            np.asarray(existing["embedding"], dtype=np.float32),
            query_embedding,
        )
    result = service.try_store_reference(
        person_id=person_id,
        audio_pcm16=trimmed_audio_pcm16,
        attempt_kind="silent",
    )
    stored = service.db.get_reference(person_id)
    metadata = dict((stored or {}).get("metadata") or {})
    return {
        "person_id": person_id,
        "raw_stats": render_stats_payload(raw_audio_pcm16),
        "trimmed_stats": render_stats_payload(trimmed_audio_pcm16),
        "trimmed_kept_ratio": round(
            (
                float(len(trimmed_audio_pcm16 or b"")) / float(len(raw_audio_pcm16 or b""))
                if raw_audio_pcm16
                else 0.0
            ),
            4,
        ),
        "enrollment_gate": {
            "accepted": not bool(rejection_reason),
            "rejection_reason": rejection_reason or "",
            "existing_reference_before_attempt": bool(existing),
            "consistency_score_to_existing": (
                round(float(consistency_score), 4)
                if consistency_score is not None
                else None
            ),
            "reference_update_threshold": float(service.policy.reference_update_threshold),
        },
        "service_result": {
            "saved": bool(result.saved),
            "reason": str(result.reason),
            "attempt_kind": str(result.attempt_kind),
        },
        "stored_reference": {
            "present": bool(stored),
            "clip_count": metadata.get("clip_count"),
            "query_duration_s": metadata.get("query_duration_s"),
            "total_voiced_sec": metadata.get("total_voiced_sec"),
            "rms_level": metadata.get("rms_level"),
            "mean_rms_level": metadata.get("mean_rms_level"),
            "model_name": metadata.get("model_name"),
        },
    }


def diagnose_recognition_attempt(
    service: "SpeakerRecognitionService",
    *,
    raw_audio_pcm16: bytes,
    trimmed_audio_pcm16: bytes,
    primary_face_person_id: str | None,
    visible_face_person_ids: list[str] | tuple[str, ...],
    top_k: int,
) -> dict[str, Any]:
    trimmed_waveform = np.frombuffer(trimmed_audio_pcm16 or b"", dtype=np.int16).copy()
    references = service.db.get_reference_embeddings()
    query_safe = is_query_clip_safe(service.policy, audio_pcm16=trimmed_waveform)
    scores_payload: list[dict[str, Any]] = []
    top_score = 0.0
    runner_up_score = 0.0
    top_person_id = None
    corroborated_by_face = False
    if query_safe and trimmed_waveform.size > 0 and references:
        query_embedding = service.backend.embed_query_clip(
            trimmed_waveform,
            sample_rate=SAMPLE_RATE,
        )
        scored = service.backend.score_against_references(query_embedding, references)
        for person_id, score in scored[: max(1, int(top_k))]:
            scores_payload.append(
                {
                    "person_id": str(person_id),
                    "score": round(float(score), 4),
                }
            )
        if scored:
            top_person_id = str(scored[0][0])
            top_score = float(scored[0][1])
            runner_up_score = float(scored[1][1]) if len(scored) > 1 else 0.0
            corroborated_by_face = top_person_id == str(primary_face_person_id or "").strip() or (
                top_person_id in tuple(str(item or "").strip() for item in visible_face_person_ids)
            )
    resolution = service.resolve_turn_owner(
        audio_pcm16=trimmed_audio_pcm16,
        primary_face_person_id=primary_face_person_id,
        visible_face_person_ids=visible_face_person_ids,
    )
    margin = max(0.0, top_score - runner_up_score)
    return {
        "raw_stats": render_stats_payload(raw_audio_pcm16),
        "trimmed_stats": render_stats_payload(trimmed_audio_pcm16),
        "trimmed_kept_ratio": round(
            (
                float(len(trimmed_audio_pcm16 or b"")) / float(len(raw_audio_pcm16 or b""))
                if raw_audio_pcm16
                else 0.0
            ),
            4,
        ),
        "query_gate": {
            "accepted": bool(query_safe),
            "rejection_reason": "" if query_safe else "query_too_short",
            "query_min_voiced_sec": float(service.policy.query_min_voiced_sec),
        },
        "scored_matches": scores_payload,
        "decision_inputs": {
            "top_person_id": top_person_id,
            "top_score": round(float(top_score), 4),
            "runner_up_score": round(float(runner_up_score), 4),
            "margin": round(float(margin), 4),
            "query_match_threshold": float(service.policy.query_match_threshold),
            "query_margin_threshold": float(service.policy.query_margin_threshold),
            "corroborated_by_face": bool(corroborated_by_face),
            "primary_face_person_id": str(primary_face_person_id or "").strip() or None,
            "visible_face_person_ids": [
                str(item or "").strip()
                for item in visible_face_person_ids
                if str(item or "").strip()
            ],
            "reference_count": int(len(references)),
        },
        "resolution": {
            "audio_speaker_id": resolution.audio_speaker_id,
            "owner_id": resolution.owner_id,
            "owner_source": resolution.owner_source,
            "owner_confidence": round(float(resolution.owner_confidence), 4),
            "speaker_visible": bool(resolution.speaker_visible),
            "top_score": round(float(resolution.top_score), 4),
            "runner_up_score": round(float(resolution.runner_up_score), 4),
            "margin": round(float(resolution.margin), 4),
        },
    }


def capture_microphone_utterance(
    config: SpeakerLabConfig,
    *,
    vad: object,
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
    resampled_chunks: list[bytes] = []
    vad_positive_blocks = 0

    try:
        with sd.InputStream(
            samplerate=int(config.input_sample_rate),
            blocksize=int(config.input_block_size),
            channels=AUDIO_CHANNELS,
            dtype=AUDIO_DTYPE,
            device=config.input_device or None,
        ) as stream:
            print("Listening...")
            while True:
                chunk, overflowed = stream.read(int(config.input_block_size))
                if overflowed:
                    status_flags.append("overflowed")
                raw_chunk = np.asarray(chunk, dtype=np.int16).reshape(-1).tobytes()
                try:
                    resampled, resample_state = audioop.ratecv(
                        raw_chunk,
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
                        resampled_chunks.append(resampled)
                        vad_positive_blocks = 1
                        continue
                    if (now - listening_started_at) >= float(config.listen_timeout_sec):
                        return {
                            "success": False,
                            "failure_reason": "listen_timeout",
                            "message": "No speech was detected before the listen timeout.",
                        }
                    continue

                resampled_chunks.append(resampled)
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

    audio_pcm16 = b"".join(resampled_chunks)
    if not audio_pcm16:
        return {
            "success": False,
            "failure_reason": "empty_capture",
            "message": "Speech was detected, but the captured clip is empty.",
        }
    return {
        "success": True,
        "audio_pcm16": audio_pcm16,
        "sample_rate_hz": SAMPLE_RATE,
        "duration_s": round(float(len(audio_pcm16)) / float(2 * SAMPLE_RATE), 4),
        "listen_started_at_unix_s": round(listening_started_at, 3),
        "recording_started_at_unix_s": round(recording_started_at, 3),
        "stop_reason": stop_reason or "completed",
        "status_flags": status_flags,
        "vad_positive_blocks": int(vad_positive_blocks),
    }


def timestamp_label() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def session_summary_payload(config: SpeakerLabConfig, *, vad_impl: str) -> dict[str, Any]:
    return {
        "profile_name": config.profile_name,
        "session_dir": config.session_dir,
        "profile_speaker_db_path": config.profile_speaker_db_path,
        "speaker_db_path": config.speaker_db_path,
        "speaker_db_isolated_from_agent": (
            str(Path(config.speaker_db_path).expanduser().resolve())
            != str(Path(config.profile_speaker_db_path).expanduser().resolve())
        ),
        "clips_dir": config.clips_dir,
        "reports_dir": config.reports_dir,
        "input_device": config.input_device,
        "input_sample_rate": config.input_sample_rate,
        "input_block_size": config.input_block_size,
        "vad_threshold": config.vad_threshold,
        "silence_grace_period": config.silence_grace_period,
        "listen_timeout_sec": config.listen_timeout_sec,
        "max_record_sec": config.max_record_sec,
        "vad_impl": vad_impl,
        "policy": asdict(config.policy),
    }
