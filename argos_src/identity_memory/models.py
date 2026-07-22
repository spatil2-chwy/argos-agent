"""Argos-facing identity and memory result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PersonMemoryContext:
    directory_profile_lines: tuple[str, ...] = ()
    profile_lines: tuple[str, ...] = ()
    followup_lines: tuple[str, ...] = ()
    context_markdown: str = ""
    preferred_language: str = "English"


@dataclass(frozen=True)
class PersonProfile:
    person_id: str
    display_name: str
    email: str = ""
    consent_status: str = ""
    status: str = ""
    interaction_count: int = 0
    last_seen: str | None = None
    directory_profile_lines: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BiometricCandidate:
    person_id: str
    display_name: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BiometricSearchResult:
    candidates: tuple[BiometricCandidate, ...] = ()
    recognized: bool = False
    status: str = "rejected"
    reason: str = "no_match"
    threshold: float = 0.0
    margin_threshold: float = 0.0
    top_score: float = 0.0
    runner_up_score: float = 0.0
    margin: float = 0.0


@dataclass(frozen=True)
class BiometricEnrollmentResult:
    saved: bool
    status: str
    reason: str
    person_id: str
    reference_id: str = ""


@dataclass(frozen=True)
class BiometricUpdateResult:
    accepted: bool = False
    status: str = "rejected"
    reason: str = ""
    person_id: str = ""
    reference_id: str = ""
    modality: str = ""
    sample_count: int = 0
    target_sample_count: int = 0
    similarity: float = 0.0


@dataclass(frozen=True)
class OwnerResolution:
    audio_speaker_id: str | None
    top_score: float
    runner_up_score: float
    margin: float
    speaker_visible: bool
    owner_id: str | None
    owner_source: str
    owner_confidence: float
    unresolved_reason: str = ""
