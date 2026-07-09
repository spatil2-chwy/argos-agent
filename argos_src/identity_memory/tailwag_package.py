"""Tailwag package-backed identity and memory client for Argos."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import logging
import re
import threading
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from argos_src.agent.preference_types import PreferenceSegment
from argos_src.observability.observability import LatencyLogger
from .models import (
    BiometricCandidate,
    BiometricEnrollmentResult,
    BiometricSearchResult,
    BiometricUpdateResult,
    OwnerResolution,
    PersonMemoryContext,
    PersonProfile,
)


logger = logging.getLogger(__name__)

DEFAULT_PLACE_BUILDING = "ARGOS"
DEFAULT_PLACE_ROOM = "realtime"
DEFAULT_RETENTION_CLASS = "standard"
DEFAULT_PREFERRED_LANGUAGE = "English"


class TailwagPackageIdentityMemoryClient:
    """Single Argos adapter for Tailwag-owned identity and memory."""

    def __init__(
        self,
        *,
        client_factory: Callable[[], Any] | None = None,
        site_code: str = "",
        place_room_id: str = DEFAULT_PLACE_ROOM,
        retention_class: str = DEFAULT_RETENTION_CLASS,
        extract_live_turn_memory: bool = True,
    ) -> None:
        self._client_factory = client_factory
        self._client_instance: Any | None = None
        self.site_code = str(site_code or "").strip()
        self.place_room_id = str(place_room_id or "").strip() or DEFAULT_PLACE_ROOM
        self.retention_class = str(retention_class or "").strip() or DEFAULT_RETENTION_CLASS
        self.extract_live_turn_memory = bool(extract_live_turn_memory)
        self._episode_lock = threading.Lock()
        self._active_episode_id = ""
        self._active_started_at = ""
        self._active_segment_text: dict[str, str] = {}
        self._active_person_ids: set[str] = set()
        self._latency = LatencyLogger("identity_memory")

    def close(self) -> None:
        client = self._client_instance
        self._client_instance = None
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Failed to close Tailwag identity-memory client")

    def health(self) -> bool:
        try:
            self._client()
        except Exception:
            logger.exception("Tailwag identity-memory health check failed")
            return False
        return True

    def resolve_identity(
        self,
        *,
        shared_first_name: str,
        shared_last_name: str,
        shared_name: str = "",
    ) -> dict[str, Any]:
        result = self._client().resolve_identity(
            shared_first_name=shared_first_name,
            shared_last_name=shared_last_name,
            shared_name=shared_name,
            site_code=self.site_code,
        )
        return _plain(result)

    def get_verified_profile(self, *, username: str, official_name: str) -> dict[str, Any] | None:
        result = self._client().get_verified_profile(
            username=username,
            official_name=official_name,
            site_code=self.site_code,
        )
        if result is None:
            return None
        return _plain(result)

    def person_profile(self, person_id: str) -> PersonProfile | None:
        result = self._client().person_profile(str(person_id or "").strip())
        if result is None:
            return None
        payload = _plain(result)
        return self._profile_from_payload(payload, str(person_id or "").strip())

    @staticmethod
    def _profile_from_payload(payload: dict[str, Any], person_id: str) -> PersonProfile:
        return PersonProfile(
            person_id=str(payload.get("person_id") or person_id),
            display_name=str(payload.get("display_name") or payload.get("name") or person_id),
            email=str(payload.get("email") or ""),
            interaction_count=_safe_int(payload.get("interaction_count")),
            last_seen=str(payload.get("last_seen")) if payload.get("last_seen") is not None else None,
            directory_profile_lines=tuple(payload.get("directory_profile_lines") or ()),
            metadata=dict(payload.get("metadata") or payload),
        )

    def person_context(
        self,
        person_id: str,
        *,
        fallback_profile_lines: tuple[str, ...] = (),
        fallback_followup_lines: tuple[str, ...] = (),
        current_text: str | None = None,
    ) -> PersonMemoryContext:
        rendered_person_id = str(person_id or "").strip()
        if not rendered_person_id:
            return PersonMemoryContext(
                profile_lines=fallback_profile_lines,
                followup_lines=fallback_followup_lines,
            )
        try:
            structured = self._client().person_context_structured(
                rendered_person_id,
                current_text=current_text,
            )
            payload = _plain(structured)
            profile_lines = tuple(payload.get("memory_profile_lines") or ())
            directory_lines = tuple(payload.get("directory_profile_lines") or ())
            followup_lines = tuple(payload.get("potential_followups") or ())
            return PersonMemoryContext(
                directory_profile_lines=directory_lines,
                profile_lines=profile_lines or fallback_profile_lines,
                followup_lines=followup_lines or fallback_followup_lines,
                preferred_language=str(payload.get("preferred_language") or DEFAULT_PREFERRED_LANGUAGE),
            )
        except Exception:
            logger.exception("Tailwag structured context unavailable for person_id=%s", rendered_person_id)
            return PersonMemoryContext(
                profile_lines=fallback_profile_lines,
                followup_lines=fallback_followup_lines,
            )

    def site_blocks(self, site_code: str, *, current_person_id: str | None = None) -> tuple[str, ...]:
        del site_code, current_person_id
        return ()

    def search_face(self, *, embedding: Any, model: str, limit: int = 2) -> BiometricSearchResult:
        try:
            result = self._client().search_face(
                embedding=_embedding_list(embedding),
                model=model,
                limit=limit,
                site_code=self.site_code or None,
            )
            return _search_result(result)
        except Exception:
            logger.exception("Tailwag face search failed")
            return BiometricSearchResult(reason="tailwag_unavailable")

    def enroll_face_reference(
        self,
        *,
        person_id: str,
        embedding: Any,
        model: str,
        metadata: dict[str, Any] | None = None,
        consent_status: str = "consented",
    ) -> BiometricEnrollmentResult:
        result = self._client().enroll_face_reference(
            person_id=person_id,
            embedding=_embedding_list(embedding),
            model=model,
            metadata=dict(metadata or {}),
            consent_status=consent_status,
        )
        return _enrollment_result(result)

    def observe_face_embedding(
        self,
        *,
        person_id: str,
        embedding: Any,
        model: str,
        evidence: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> BiometricUpdateResult:
        try:
            result = self._client().observe_face_embedding(
                person_id=str(person_id or "").strip(),
                embedding=_embedding_list(embedding),
                model=model,
                evidence=dict(evidence or {}),
                metadata=dict(metadata or {}),
            )
            return _update_result(result)
        except Exception:
            logger.exception("Tailwag face observation failed person_id=%s", person_id)
            return BiometricUpdateResult(
                accepted=False,
                status="rejected",
                reason="tailwag_unavailable",
                person_id=str(person_id or "").strip(),
                modality="face",
            )

    def search_voice(self, *, embedding: Any, model: str, limit: int = 2) -> BiometricSearchResult:
        try:
            result = self._client().search_voice(
                embedding=_embedding_list(embedding),
                model=model,
                limit=limit,
                site_code=self.site_code or None,
            )
            return _search_result(result)
        except Exception:
            logger.exception("Tailwag voice search failed")
            return BiometricSearchResult(reason="tailwag_unavailable")

    def enroll_voice_reference(
        self,
        *,
        person_id: str,
        embedding: Any,
        model: str,
        metadata: dict[str, Any] | None = None,
        consent_status: str = "consented",
    ) -> BiometricEnrollmentResult:
        result = self._client().enroll_voice_reference(
            person_id=person_id,
            embedding=_embedding_list(embedding),
            model=model,
            metadata=dict(metadata or {}),
            consent_status=consent_status,
        )
        return _enrollment_result(result)

    def observe_voice_embedding(
        self,
        *,
        person_id: str,
        embedding: Any,
        model: str,
        evidence: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> BiometricUpdateResult:
        try:
            result = self._client().observe_voice_embedding(
                person_id=str(person_id or "").strip(),
                embedding=_embedding_list(embedding),
                model=model,
                evidence=dict(evidence or {}),
                metadata=dict(metadata or {}),
            )
            return _update_result(result)
        except Exception:
            logger.exception("Tailwag voice observation failed person_id=%s", person_id)
            return BiometricUpdateResult(
                accepted=False,
                status="rejected",
                reason="tailwag_unavailable",
                person_id=str(person_id or "").strip(),
                modality="voice",
            )

    def has_voice_reference(self, person_id: str) -> bool:
        try:
            return bool(self._client().has_voice_reference(str(person_id or "").strip()))
        except Exception:
            logger.exception("Tailwag voice reference check failed person_id=%s", person_id)
            return False

    def resolve_turn_owner(
        self,
        *,
        primary_face_candidate: Any = None,
        visible_face_candidates: tuple[Any, ...] = (),
        voice_candidate: Any = None,
        policy_context: dict[str, Any] | None = None,
    ) -> OwnerResolution:
        try:
            result = self._client().resolve_turn_owner(
                primary_face_candidate=_candidate_payload(primary_face_candidate),
                visible_face_candidates=[
                    _candidate_payload(candidate)
                    for candidate in tuple(visible_face_candidates or ())
                    if _candidate_payload(candidate)
                ],
                voice_candidate=_candidate_payload(voice_candidate),
                policy_context=dict(policy_context or {}),
            )
            payload = _plain(result)
            return OwnerResolution(
                audio_speaker_id=_optional_str(payload.get("audio_speaker_id")),
                top_score=float(payload.get("top_score") or 0.0),
                runner_up_score=float(payload.get("runner_up_score") or 0.0),
                margin=float(payload.get("margin") or 0.0),
                speaker_visible=bool(payload.get("speaker_visible")),
                owner_id=_optional_str(payload.get("owner_id")),
                owner_source=str(payload.get("owner_source") or "unknown"),
                owner_confidence=float(payload.get("owner_confidence") or 0.0),
                unresolved_reason=str(payload.get("unresolved_reason") or ""),
            )
        except Exception:
            logger.exception("Tailwag owner resolution failed")
            return OwnerResolution(
                audio_speaker_id=None,
                top_score=0.0,
                runner_up_score=0.0,
                margin=0.0,
                speaker_visible=False,
                owner_id=None,
                owner_source="unknown",
                owner_confidence=0.0,
                unresolved_reason="tailwag_unavailable",
            )

    def extract_and_store_segment(self, segment: PreferenceSegment, reason: str = "") -> None:
        try:
            episode = self._episode_from_segment(segment)
        except Exception:
            logger.exception("Tailwag live-turn episode construction failed")
            self._emit_live_episode_event(
                event="tailwag_episode_failed",
                segment=segment,
                reason=reason,
                error="episode_construction_failed",
            )
            if _is_terminal_flush(reason):
                self.finish_active_episode(reason=reason)
            return
        if episode is None:
            self._emit_live_episode_event(
                event="tailwag_episode_skipped",
                segment=segment,
                reason=reason,
                error="empty_segment",
            )
            return
        terminal = _is_terminal_flush(reason)
        try:
            result = self._client().record_episode(
                episode,
                extract_memory=self.extract_live_turn_memory,
            )
            self._emit_live_episode_event(
                event="tailwag_episode_recorded",
                segment=segment,
                reason=reason,
                episode_id=str(getattr(result, "episode_id", "") or getattr(episode, "id", "") or ""),
                extract_memory=self.extract_live_turn_memory,
                result=result,
            )
        except Exception:
            logger.exception("Tailwag live-turn episode ingestion failed")
            self._emit_live_episode_event(
                event="tailwag_episode_failed",
                segment=segment,
                reason=reason,
                episode_id=str(getattr(episode, "id", "") or ""),
                extract_memory=self.extract_live_turn_memory,
                error="record_episode_failed",
            )
        finally:
            if terminal:
                self.finish_active_episode(reason=reason)

    def finish_active_episode(self, *, reason: str = "") -> None:
        del reason
        self._reset_active_episode()

    def record_episode(self, episode: Any, *, extract_memory: bool = True) -> Any:
        return self._client().record_episode(episode, extract_memory=extract_memory)

    def search_semantic_memory(
        self,
        *,
        text: str,
        person_id: str,
        building_code: str | None = None,
        limit: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        rendered_text = str(text or "").strip()
        rendered_person_id = str(person_id or "").strip()
        if not rendered_text or not rendered_person_id:
            return {"episodes": [], "memory_items": []}
        return self._client().search_semantic_memory(
            text=rendered_text,
            person_id=rendered_person_id,
            building_code=building_code,
            limit=limit,
        )

    def _client(self) -> Any:
        if self._client_instance is None:
            if self._client_factory is not None:
                self._client_instance = self._client_factory()
            else:
                from tailwag_memory import TailwagMemoryClient

                self._client_instance = TailwagMemoryClient.from_env()
        return self._client_instance

    def _episode_from_segment(self, segment: PreferenceSegment) -> Any | None:
        person_id = str(getattr(segment, "person_id", "") or "").strip()
        turns = tuple(getattr(segment, "turns", ()) or ())
        if not person_id or not turns:
            return None
        segment_text = _segment_transcript(segment)
        if not segment_text:
            return None
        segment_id = str(getattr(segment, "segment_id", "") or "").strip()
        now = _utc_now_iso()
        with self._episode_lock:
            if not self._active_episode_id:
                self._active_episode_id = f"argos:conversation:{uuid4().hex}"
                self._active_started_at = now
            self._active_segment_text[segment_id or f"segment-{len(self._active_segment_text)}"] = segment_text
            self._active_person_ids.add(person_id)
            episode_id = self._active_episode_id
            started_at = self._active_started_at or now
            transcript = "\n\n".join(self._active_segment_text.values())
            participants = tuple(sorted(self._active_person_ids))
        return self._episode_input(
            id=episode_id,
            episode_type="conversation",
            start_time=started_at,
            end_time=now,
            transcript=transcript,
            retention_class=self.retention_class,
            place=self._place_input(
                building_code=self.site_code or DEFAULT_PLACE_BUILDING,
                room_id=self.place_room_id,
            ),
            participants=[
                self._person_input(id=participant_id, role="speaker", source="live_chat")
                for participant_id in participants
            ],
        )

    def _reset_active_episode(self) -> None:
        with self._episode_lock:
            self._active_episode_id = ""
            self._active_started_at = ""
            self._active_segment_text = {}
            self._active_person_ids = set()

    def _emit_live_episode_event(
        self,
        *,
        event: str,
        segment: PreferenceSegment,
        reason: str,
        episode_id: str = "",
        extract_memory: bool | None = None,
        result: Any | None = None,
        error: str = "",
    ) -> None:
        turns = tuple(getattr(segment, "turns", ()) or ())
        last_turn_id = str(getattr(turns[-1], "turn_id", "") or "") if turns else ""
        memory_results = tuple(getattr(result, "memory_results", ()) or ())
        created_count = sum(
            len(tuple(getattr(item, "created_memory_ids", ()) or ()))
            for item in memory_results
        )
        addressed_count = sum(
            len(tuple(getattr(item, "addressed_memory_ids", ()) or ()))
            for item in memory_results
        )
        supported_count = sum(
            len(tuple(getattr(item, "supported_memory_ids", ()) or ()))
            for item in memory_results
        )
        error_count = sum(
            1
            for item in memory_results
            if str(getattr(item, "error", "") or "").strip()
        )
        self._latency.emit(
            event=event,
            req_id=last_turn_id or None,
            memory_segment_id=getattr(segment, "segment_id", None),
            memory_person_id=getattr(segment, "person_id", None),
            memory_turn_count=len(turns),
            memory_flush_reason=reason,
            tailwag_episode_id=episode_id or None,
            tailwag_episode_extract_memory=extract_memory,
            tailwag_memory_result_count=len(memory_results),
            tailwag_memory_created_count=created_count,
            tailwag_memory_addressed_count=addressed_count,
            tailwag_memory_supported_count=supported_count,
            tailwag_memory_error_count=error_count,
            tailwag_episode_error=error or None,
        )

    @staticmethod
    def _person_input(**kwargs: Any) -> Any:
        from tailwag_memory import PersonInput

        return PersonInput(**kwargs)

    @staticmethod
    def _place_input(**kwargs: Any) -> Any:
        from tailwag_memory import PlaceInput

        return PlaceInput(**kwargs)

    @staticmethod
    def _episode_input(**kwargs: Any) -> Any:
        from tailwag_memory import EpisodeInput

        return EpisodeInput(**kwargs)


def _embedding_list(value: Any) -> list[float]:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(item) for item in vector.tolist()]


def _search_result(value: Any) -> BiometricSearchResult:
    payload = _plain(value)
    return BiometricSearchResult(
        candidates=tuple(_candidate(item) for item in payload.get("candidates", ()) or ()),
        recognized=bool(payload.get("recognized")),
        status=str(payload.get("status") or "rejected"),
        reason=str(payload.get("reason") or "no_match"),
        threshold=float(payload.get("threshold") or 0.0),
        margin_threshold=float(payload.get("margin_threshold") or 0.0),
        top_score=float(payload.get("top_score") or 0.0),
        runner_up_score=float(payload.get("runner_up_score") or 0.0),
        margin=float(payload.get("margin") or 0.0),
    )


def _candidate(value: Any) -> BiometricCandidate:
    payload = _plain(value)
    return BiometricCandidate(
        person_id=str(payload.get("person_id") or ""),
        display_name=str(payload.get("display_name") or payload.get("name") or ""),
        score=float(payload.get("score") or payload.get("similarity") or 0.0),
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {},
    )


def _enrollment_result(value: Any) -> BiometricEnrollmentResult:
    payload = _plain(value)
    return BiometricEnrollmentResult(
        saved=bool(payload.get("saved")),
        status=str(payload.get("status") or ""),
        reason=str(payload.get("reason") or ""),
        person_id=str(payload.get("person_id") or ""),
        reference_id=str(payload.get("reference_id") or ""),
    )


def _update_result(value: Any) -> BiometricUpdateResult:
    payload = _plain(value)
    return BiometricUpdateResult(
        accepted=bool(payload.get("accepted")),
        status=str(payload.get("status") or "rejected"),
        reason=str(payload.get("reason") or ""),
        person_id=str(payload.get("person_id") or ""),
        reference_id=str(payload.get("reference_id") or ""),
        modality=str(payload.get("modality") or ""),
        sample_count=_safe_int(payload.get("sample_count")),
        target_sample_count=_safe_int(payload.get("target_sample_count")),
        similarity=float(payload.get("similarity") or 0.0),
    )


def _candidate_payload(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = _plain(value)
    person_id = str(payload.get("person_id") or "").strip()
    if not person_id:
        return None
    return {
        "person_id": person_id,
        "display_name": str(payload.get("display_name") or payload.get("name") or ""),
        "score": float(payload.get("score") or payload.get("similarity") or 0.0),
        "metadata": dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {},
    }


def _plain(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    payload = getattr(value, "__dict__", None)
    return dict(payload) if isinstance(payload, dict) else {}


def _segment_transcript(segment: PreferenceSegment) -> str:
    lines: list[str] = []
    for turn in tuple(getattr(segment, "turns", ()) or ()):
        user_text = str(getattr(turn, "user_text", "") or "").strip()
        assistant_text = str(getattr(turn, "assistant_text", "") or "").strip()
        if user_text:
            lines.append(f"User: {user_text}")
        if assistant_text:
            lines.append(f"Assistant: {assistant_text}")
    return "\n".join(lines)


def _is_terminal_flush(reason: str) -> bool:
    return str(reason or "").strip() in {"idle_timeout", "shutdown"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _optional_str(value: Any) -> str | None:
    rendered = str(value or "").strip()
    return rendered or None
