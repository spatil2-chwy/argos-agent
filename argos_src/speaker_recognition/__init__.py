"""Speaker-recognition exports for the Argos runtime."""

from __future__ import annotations

from .constants import DEFAULT_SPEAKER_DB_PATH
from .models import (
    OwnerSource,
    PendingVoiceEnrollment,
    SpeakerRecognitionPolicy,
    SpeakerResolutionResult,
    VoiceEnrollmentAttemptKind,
    VoiceEnrollmentResult,
)

__all__ = [
    "DEFAULT_SPEAKER_DB_PATH",
    "OwnerSource",
    "PendingVoiceEnrollment",
    "SpeakerRecognitionPolicy",
    "SpeakerResolutionResult",
    "VoiceEnrollmentAttemptKind",
    "VoiceEnrollmentResult",
]
