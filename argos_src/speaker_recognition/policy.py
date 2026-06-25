"""Pure helpers for speaker-clip quality checks and ownership resolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argos_src.speaker_recognition.models import (
    SpeakerRecognitionPolicy,
    SpeakerResolutionResult,
)


SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * (FRAME_MS / 1000.0))


@dataclass(frozen=True)
class AudioClipStats:
    """Derived statistics for one mono PCM16 audio clip."""

    duration_s: float
    rms_level: float
    clipped_fraction: float


def _vad_frame_samples(vad: object | None) -> int:
    window_size = int(getattr(vad, "window_size", 0) or 0)
    if window_size > 0:
        return window_size
    return FRAME_SAMPLES


def trim_voice_activity(
    audio_pcm16: np.ndarray,
    *,
    vad: object | None = None,
) -> np.ndarray:
    """Use the existing local VAD to keep speech regions and drop non-speech spans."""
    waveform = np.asarray(audio_pcm16, dtype=np.int16).reshape(-1)
    if waveform.size <= 0 or vad is None:
        return waveform.copy()

    voiced_frames: list[np.ndarray] = []
    context = {}
    frame_samples = _vad_frame_samples(vad)
    for start in range(0, int(waveform.size), frame_samples):
        frame = waveform[start : start + frame_samples]
        if frame.size <= 0:
            continue
        try:
            voiced, _ = vad(frame, context)
        except Exception:
            voiced = False
        if not voiced:
            continue
        voiced_frames.append(frame.copy())

    if not voiced_frames:
        return waveform.copy()
    return np.concatenate(voiced_frames).astype(np.int16, copy=False)


def clip_stats(audio_pcm16: np.ndarray) -> AudioClipStats:
    waveform = np.asarray(audio_pcm16, dtype=np.int16).reshape(-1)
    if waveform.size <= 0:
        return AudioClipStats(duration_s=0.0, rms_level=0.0, clipped_fraction=0.0)
    waveform_float = waveform.astype(np.float32)
    duration_s = float(waveform.size) / float(SAMPLE_RATE)
    rms_level = float(np.sqrt(np.mean(np.square(waveform_float)))) if waveform_float.size else 0.0
    clipped_fraction = float(np.mean(np.abs(waveform_float) >= 32760.0))
    return AudioClipStats(
        duration_s=duration_s,
        rms_level=rms_level,
        clipped_fraction=clipped_fraction,
    )


def is_query_clip_safe(
    policy: SpeakerRecognitionPolicy,
    *,
    audio_pcm16: np.ndarray,
) -> bool:
    stats = clip_stats(audio_pcm16)
    if stats.duration_s < policy.query_min_voiced_sec:
        return False
    return True


def enrollment_rejection_reason(
    policy: SpeakerRecognitionPolicy,
    *,
    audio_pcm16: np.ndarray,
) -> str:
    stats = clip_stats(audio_pcm16)
    if stats.duration_s <= 0.0:
        return "reject_empty"
    if stats.duration_s < policy.enroll_min_voiced_sec:
        return "reject_too_short"
    if policy.enroll_max_voiced_sec > 0.0 and stats.duration_s > policy.enroll_max_voiced_sec:
        return "reject_too_long"
    if stats.rms_level < policy.enroll_min_rms_level:
        return "reject_too_quiet"
    if stats.clipped_fraction > policy.max_clipped_fraction:
        return "reject_clipped"
    return ""


def resolve_owner_id(
    *,
    policy: SpeakerRecognitionPolicy,
    primary_face_person_id: str | None,
    audio_speaker_id: str | None,
    top_score: float,
    runner_up_score: float,
    visible_face_person_ids: tuple[str, ...] | list[str] | None = None,
) -> SpeakerResolutionResult:
    rendered_primary = str(primary_face_person_id or "").strip() or None
    rendered_face_owner = rendered_primary
    rendered_audio_candidate = str(audio_speaker_id or "").strip() or None
    visible_face_ids = tuple(
        rendered
        for rendered in (
            str(person_id or "").strip()
            for person_id in (visible_face_person_ids or ())
        )
        if rendered
    )
    margin = max(0.0, float(top_score) - float(runner_up_score))
    audio_is_confident = (
        rendered_audio_candidate is not None
        and float(top_score) >= policy.query_match_threshold
        and margin >= policy.query_margin_threshold
    )
    rendered_audio = rendered_audio_candidate if audio_is_confident else None
    audio_visible = rendered_audio is not None and (
        rendered_audio in visible_face_ids
        or (not visible_face_ids and rendered_audio == rendered_primary)
    )

    if rendered_audio is not None:
        return SpeakerResolutionResult(
            audio_speaker_id=rendered_audio,
            top_score=float(top_score),
            runner_up_score=float(runner_up_score),
            margin=margin,
            speaker_visible=audio_visible,
            owner_id=rendered_audio,
            owner_source="audio_face_agree" if rendered_audio == rendered_primary else "audio",
            owner_confidence=float(top_score),
        )
    face_owner_visible = rendered_face_owner is not None and (
        rendered_face_owner in visible_face_ids or not visible_face_ids
    )
    if rendered_face_owner is not None:
        return SpeakerResolutionResult(
            audio_speaker_id=None,
            top_score=float(top_score),
            runner_up_score=float(runner_up_score),
            margin=margin,
            speaker_visible=face_owner_visible,
            owner_id=rendered_face_owner,
            owner_source="face",
            owner_confidence=0.0,
        )
    return SpeakerResolutionResult(
        audio_speaker_id=None,
        top_score=float(top_score),
        runner_up_score=float(runner_up_score),
        margin=margin,
        speaker_visible=False,
        owner_id=None,
        owner_source="unknown",
        owner_confidence=0.0,
    )
