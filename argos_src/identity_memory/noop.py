"""Disabled identity-memory client."""

from __future__ import annotations

from typing import Any

from .models import (
    BiometricEnrollmentResult,
    BiometricSearchResult,
    BiometricUpdateResult,
    OwnerResolution,
    PersonMemoryContext,
    PersonProfile,
)


class NoopIdentityMemoryClient:
    site_code = ""
    retention_class = "standard"

    def close(self) -> None:
        return None

    def health(self) -> bool:
        return True

    def resolve_identity(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "success": False,
            "status": "directory_unavailable",
            "message": "Identity memory is disabled.",
            "data": None,
            "candidates": [],
        }

    def get_verified_profile(self, **kwargs: Any) -> dict[str, Any] | None:
        return None

    def person_profile(self, person_id: str) -> PersonProfile | None:
        return None

    def person_context(self, person_id: str, **kwargs: Any) -> PersonMemoryContext:
        return PersonMemoryContext(
            profile_lines=tuple(kwargs.get("fallback_profile_lines") or ()),
            followup_lines=tuple(kwargs.get("fallback_followup_lines") or ()),
        )

    def site_blocks(self, site_code: str, *, current_person_id: str | None = None) -> tuple[str, ...]:
        return ()

    def search_face(self, **kwargs: Any) -> BiometricSearchResult:
        return BiometricSearchResult(reason="identity_memory_disabled")

    def enroll_face_reference(self, **kwargs: Any) -> BiometricEnrollmentResult:
        return BiometricEnrollmentResult(False, "rejected", "identity_memory_disabled", str(kwargs.get("person_id") or ""))

    def observe_face_embedding(self, **kwargs: Any) -> BiometricUpdateResult:
        return BiometricUpdateResult(
            accepted=False,
            status="rejected",
            reason="identity_memory_disabled",
            person_id=str(kwargs.get("person_id") or ""),
            modality="face",
        )

    def search_voice(self, **kwargs: Any) -> BiometricSearchResult:
        return BiometricSearchResult(reason="identity_memory_disabled")

    def enroll_voice_reference(self, **kwargs: Any) -> BiometricEnrollmentResult:
        return BiometricEnrollmentResult(False, "rejected", "identity_memory_disabled", str(kwargs.get("person_id") or ""))

    def observe_voice_embedding(self, **kwargs: Any) -> BiometricUpdateResult:
        return BiometricUpdateResult(
            accepted=False,
            status="rejected",
            reason="identity_memory_disabled",
            person_id=str(kwargs.get("person_id") or ""),
            modality="voice",
        )

    def has_voice_reference(self, person_id: str) -> bool:
        return False

    def resolve_turn_owner(self, **kwargs: Any) -> OwnerResolution:
        return OwnerResolution(
            audio_speaker_id=None,
            top_score=0.0,
            runner_up_score=0.0,
            margin=0.0,
            speaker_visible=False,
            owner_id=None,
            owner_source="unknown",
            owner_confidence=0.0,
            unresolved_reason="identity_memory_disabled",
        )

    def extract_and_store_segment(self, segment: Any, reason: str = "") -> None:
        return None

    def finish_active_episode(self, *, reason: str = "") -> None:
        return None

    def search_semantic_memory(self, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
        return {"episodes": [], "memory_items": []}
