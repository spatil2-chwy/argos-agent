"""Shared models for speaker recognition, ownership resolution, and enrollment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OwnerSource = Literal["audio", "audio_face_agree", "face", "unknown"]
VoiceEnrollmentAttemptKind = Literal["silent", "fallback"]


@dataclass(frozen=True)
class SpeakerRecognitionPolicy:
    """Pure runtime settings for speaker matching and voice enrollment."""

    backend: str = "speechbrain_ecapa"
    db_path: str = ""
    query_min_voiced_sec: float = 0.0
    query_match_threshold: float = 0.40
    query_margin_threshold: float = 0.20
    reference_update_threshold: float = 0.55
    enroll_min_voiced_sec: float = 0.0
    enroll_max_voiced_sec: float = 0.0
    explicit_prompt_after_silent_failures: int = 2
    enroll_min_rms_level: float = 0.0
    max_clipped_fraction: float = 0.02

    def __post_init__(self) -> None:
        if self.query_min_voiced_sec < 0.0:
            raise ValueError("query_min_voiced_sec must be >= 0")
        if not 0.0 <= self.query_match_threshold <= 1.0:
            raise ValueError("query_match_threshold must be within [0, 1]")
        if not 0.0 <= self.query_margin_threshold <= 1.0:
            raise ValueError("query_margin_threshold must be within [0, 1]")
        if not 0.0 <= self.reference_update_threshold <= 1.0:
            raise ValueError("reference_update_threshold must be within [0, 1]")
        if self.enroll_min_voiced_sec < 0.0:
            raise ValueError("enroll_min_voiced_sec must be >= 0")
        if 0.0 < self.enroll_max_voiced_sec < self.enroll_min_voiced_sec:
            raise ValueError("enroll_max_voiced_sec must be >= enroll_min_voiced_sec")
        if self.explicit_prompt_after_silent_failures < 1:
            raise ValueError("explicit_prompt_after_silent_failures must be >= 1")
        if self.enroll_min_rms_level < 0.0:
            raise ValueError("enroll_min_rms_level must be >= 0")
        if not 0.0 <= self.max_clipped_fraction <= 1.0:
            raise ValueError("max_clipped_fraction must be within [0, 1]")


@dataclass(frozen=True)
class SpeakerResolutionResult:
    """Immutable output of per-turn audio speaker resolution."""

    audio_speaker_id: str | None
    top_score: float
    runner_up_score: float
    margin: float
    speaker_visible: bool
    owner_id: str | None
    owner_source: OwnerSource
    owner_confidence: float


@dataclass(frozen=True)
class VoiceEnrollmentResult:
    """Result of trying to store a reusable speaker reference embedding."""

    saved: bool
    reason: str
    person_id: str
    attempt_kind: VoiceEnrollmentAttemptKind


@dataclass
class PendingVoiceEnrollment:
    """Session-local state for post-registration opportunistic voice enrollment."""

    person_id: str
    silent_failures: int = 0
    explicit_prompt_armed: bool = False
    explicit_prompt_used: bool = False
