"""Pure helpers for social-scene analysis and target selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from argos_src.face_recognition.models import AttentionTarget, SocialSceneContext


@dataclass(frozen=True)
class FaceSceneCandidate:
    """Normalized per-face data used for social-scene selection."""

    kind: Literal["recognized", "unknown"]
    bbox_area: int
    center_distance: float
    depth_m: float | None = None
    person_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class FaceSceneAnalysis:
    """Output of scene analysis for prompting and memory ownership."""

    attention_target: AttentionTarget | None
    social_scene: SocialSceneContext


def _candidate_sort_key(candidate: FaceSceneCandidate) -> tuple[float, float, float]:
    depth_rank = (
        candidate.depth_m
        if candidate.depth_m is not None
        else float("inf")
    )
    return (depth_rank, -float(candidate.bbox_area), float(candidate.center_distance))


def _to_attention_target(candidate: FaceSceneCandidate | None) -> AttentionTarget | None:
    if candidate is None:
        return None
    return AttentionTarget(
        kind=candidate.kind,
        depth_m=candidate.depth_m,
        bbox_area=candidate.bbox_area,
        center_distance=candidate.center_distance,
        person_id=candidate.person_id,
        name=candidate.name,
    )


def _select_primary_face_person(
    candidates: list[FaceSceneCandidate],
) -> FaceSceneCandidate | None:
    if len(candidates) != 1:
        return None
    candidate = candidates[0]
    if candidate.kind != "recognized" or not candidate.person_id:
        return None
    return candidate


def analyze_face_scene(candidates: list[FaceSceneCandidate]) -> FaceSceneAnalysis:
    """Select attention and speaker targets plus prompt-facing social context."""
    primary_person = _select_primary_face_person(candidates)

    recognized = [candidate for candidate in candidates if candidate.kind == "recognized"]
    unknown_count = sum(1 for candidate in candidates if candidate.kind == "unknown")
    closest = min(candidates, key=_candidate_sort_key) if candidates else None
    nearest_recognized = min(recognized, key=_candidate_sort_key) if recognized else None
    social_scene = SocialSceneContext(
        has_unrecognized_people=unknown_count > 0,
        closest_person_kind=closest.kind if closest is not None else "none",
        nearest_recognized_name=nearest_recognized.name if nearest_recognized is not None else None,
    )
    return FaceSceneAnalysis(
        attention_target=_to_attention_target(primary_person),
        social_scene=social_scene,
    )
