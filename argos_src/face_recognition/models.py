"""Shared data models for face recognition runtime state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CACHE_EXPIRE_SEC = 5.0
INTERACTION_DEDUPE_SEC = 60.0


@dataclass
class PersonContext:
    """Cached context for one recognized person in view."""

    person_id: str
    name: str
    interaction_count: int
    confidence: float
    bbox_area: int
    timestamp: float
    depth_m: float | None = None
    bearing_rad: float | None = None
    face_center_x_px: float | None = None
    face_center_y_px: float | None = None
    center_distance: float = 0.0
    directory_profile_lines: tuple[str, ...] = ()
    memory_profile_lines: tuple[str, ...] = ()
    preferred_language: str = ""
    potential_followups: tuple[str, ...] = ()
    visible: bool = True


@dataclass(frozen=True)
class FaceTurnTarget:
    """Current face-bearing target for a recognized person."""

    person_id: str
    name: str
    bearing_rad: float
    timestamp: float
    confidence: float = 0.0
    depth_m: float | None = None


@dataclass
class AttentionTarget:
    """Current social-attention target derived from all visible faces."""

    kind: Literal["recognized", "unknown"]
    depth_m: float | None
    bbox_area: int
    center_distance: float
    person_id: str | None = None
    name: str | None = None


@dataclass
class SocialSceneContext:
    """Compact, prompt-visible summary of the current face scene."""

    has_unrecognized_people: bool
    closest_person_kind: Literal["recognized", "unknown", "none"]
    nearest_recognized_name: str | None


@dataclass
class FacePresenceSnapshot:
    """Canonical face presence state consumed by ASR gating and proactive events."""

    status: str  # one of: none | unknown | recognized
    faces_detected: int
    recognized_count: int
    unknown_count: int
    recognized_names: list[str]
    has_mixed_scene: bool
    primary_face_kind: str
    primary_face_name: str
    nearest_recognized_name: str
    social_scene: SocialSceneContext
    updated_at: float
    expires_at: float


def empty_presence_snapshot(
    now: float,
    *,
    expires_in: float = CACHE_EXPIRE_SEC,
) -> FacePresenceSnapshot:
    """Return the canonical empty face-presence state."""
    return FacePresenceSnapshot(
        status="none",
        faces_detected=0,
        recognized_count=0,
        unknown_count=0,
        recognized_names=[],
        has_mixed_scene=False,
        primary_face_kind="none",
        primary_face_name="",
        nearest_recognized_name="",
        social_scene=SocialSceneContext(
            has_unrecognized_people=False,
            closest_person_kind="none",
            nearest_recognized_name=None,
        ),
        updated_at=now,
        expires_at=now + expires_in,
    )
