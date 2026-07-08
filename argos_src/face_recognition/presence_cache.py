"""Cache and canonical presence-state bookkeeping for face recognition."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Any

from argos_src.face_recognition.models import (
    AttentionTarget,
    CACHE_EXPIRE_SEC,
    FaceTurnTarget,
    FacePresenceSnapshot,
    INTERACTION_DEDUPE_SEC,
    PersonContext,
    SocialSceneContext,
    empty_presence_snapshot,
)


class FacePresenceCache:
    """Store currently visible people plus the exported face-presence snapshot."""

    def __init__(
        self,
        *,
        cache_expire_sec: float = CACHE_EXPIRE_SEC,
        interaction_dedupe_sec: float = INTERACTION_DEDUPE_SEC,
    ) -> None:
        self._cache_expire_sec = float(cache_expire_sec)
        self._interaction_dedupe_sec = float(interaction_dedupe_sec)
        self._cached_persons: list[PersonContext] = []
        self._cache_lock = threading.Lock()
        self._active_ids: set[str] = set()
        self._active_last_seen: dict[str, float] = {}
        self._last_interaction_at: dict[str, float] = {}
        self._last_face_time = 0.0
        self._attention_target: AttentionTarget | None = None
        now = time.time()
        self._presence = empty_presence_snapshot(now, expires_in=self._cache_expire_sec)

    def _clear_locked(self, now: float) -> None:
        self._cached_persons = []
        self._presence = empty_presence_snapshot(
            now,
            expires_in=self._cache_expire_sec,
        )
        self._attention_target = None
        self._active_ids.clear()
        self._active_last_seen.clear()
        self._last_face_time = 0.0

    def _clear_if_expired_locked(self, now: float) -> bool:
        if not self._last_face_time:
            return False
        if (now - self._last_face_time) <= self._cache_expire_sec:
            return False
        self._clear_locked(now)
        return True

    def clear_if_expired(self, now: float) -> bool:
        """Clear stale cached state when no face has been seen recently."""
        with self._cache_lock:
            return self._clear_if_expired_locked(now)

    def mark_faces_seen(self, now: float) -> None:
        with self._cache_lock:
            self._last_face_time = now

    def should_record_interaction(self, person_id: str, now: float) -> bool:
        """Return True when a recognized person is outside the interaction dedupe window."""
        with self._cache_lock:
            last_interaction_at = self._last_interaction_at.get(person_id)
        if last_interaction_at is None:
            return True
        return (now - last_interaction_at) > self._interaction_dedupe_sec

    def mark_interaction_recorded(self, person_id: str, now: float) -> None:
        with self._cache_lock:
            self._last_interaction_at[person_id] = now

    def mark_person_seen(self, person_id: str, now: float) -> None:
        with self._cache_lock:
            self._active_ids.add(person_id)
            self._active_last_seen[person_id] = now

    def expire_inactive(self, current_ids: set[str], now: float) -> None:
        """Expire person ids that have not been seen for the cache window."""
        with self._cache_lock:
            for person_id in list(self._active_last_seen):
                if person_id in current_ids:
                    continue
                if (now - self._active_last_seen[person_id]) <= self._cache_expire_sec:
                    continue
                self._active_ids.discard(person_id)
                del self._active_last_seen[person_id]

    def update(
        self,
        *,
        persons: list[PersonContext],
        faces_detected: int,
        unknown_count: int,
        attentive_unknown_count: int = 0,
        attention_target: AttentionTarget | None,
        primary_attention_target: AttentionTarget | None = None,
        social_scene: SocialSceneContext,
        face_match_evidence: dict[str, Any] | None = None,
        now: float,
    ) -> FacePresenceSnapshot:
        """Replace cached persons and recompute the exported presence snapshot."""
        ordered_persons = sorted(persons, key=lambda person: person.bbox_area, reverse=True)
        recognized_names = [person.name for person in ordered_persons]
        attentive_persons = [person for person in ordered_persons if bool(person.attentive)]
        attentive_recognized_names = [person.name for person in attentive_persons]
        recognized_count = len(ordered_persons)
        attentive_recognized_count = len(attentive_persons)
        attentive_unknown_count = max(0, int(attentive_unknown_count or 0))
        attention_count = attentive_recognized_count + attentive_unknown_count
        primary_face_name = attention_target.name if attention_target and attention_target.name else ""
        primary_attention_name = (
            primary_attention_target.name
            if primary_attention_target and primary_attention_target.name
            else ""
        )
        primary_attention_person_id = (
            primary_attention_target.person_id
            if primary_attention_target and primary_attention_target.person_id
            else ""
        )
        attention_confidence = 0.0
        if primary_attention_person_id:
            for person in attentive_persons:
                if person.person_id == primary_attention_person_id:
                    attention_confidence = float(person.attention_confidence)
                    break
        elif attention_count:
            attention_confidence = max(
                [float(person.attention_confidence) for person in attentive_persons] + [0.0]
            )
        face_evidence = dict(face_match_evidence or {})
        status = (
            "recognized"
            if recognized_count > 0
            else ("unknown" if unknown_count > 0 else "none")
        )
        attention_status = (
            "attentive"
            if attention_count > 0
            else ("inattentive" if faces_detected > 0 else "none")
        )
        snapshot = FacePresenceSnapshot(
            status=status,
            faces_detected=faces_detected,
            recognized_count=recognized_count,
            unknown_count=unknown_count,
            recognized_names=recognized_names,
            has_mixed_scene=recognized_count > 0 and unknown_count > 0,
            primary_face_kind=attention_target.kind if attention_target else "none",
            primary_face_name=primary_face_name,
            attention_status=attention_status,
            attention_count=attention_count,
            attentive_recognized_count=attentive_recognized_count,
            attentive_unknown_count=attentive_unknown_count,
            attentive_recognized_names=attentive_recognized_names,
            has_attentive_mixed_scene=(
                attentive_recognized_count > 0 and attentive_unknown_count > 0
            ),
            primary_attention_kind=(
                primary_attention_target.kind if primary_attention_target else "none"
            ),
            primary_attention_name=primary_attention_name,
            primary_attention_person_id=primary_attention_person_id,
            attention_confidence=attention_confidence,
            face_match_status=str(face_evidence.get("status") or ""),
            face_match_reason=str(face_evidence.get("reason") or ""),
            face_match_name=str(face_evidence.get("name") or ""),
            face_match_person_id=str(face_evidence.get("person_id") or ""),
            face_score=float(face_evidence.get("similarity", 0.0) or 0.0),
            face_score_threshold=float(face_evidence.get("threshold", 0.0) or 0.0),
            face_runner_up_score=float(
                face_evidence.get("runner_up_similarity", 0.0) or 0.0
            ),
            face_score_margin=float(face_evidence.get("margin", 0.0) or 0.0),
            face_margin_threshold=float(
                face_evidence.get("margin_threshold", 0.0) or 0.0
            ),
            nearest_recognized_name=social_scene.nearest_recognized_name or "",
            social_scene=social_scene,
            updated_at=now,
            expires_at=now + self._cache_expire_sec,
        )
        with self._cache_lock:
            self._cached_persons = ordered_persons
            self._presence = snapshot
            self._attention_target = attention_target
        return snapshot

    def get_cached_persons(self) -> list[PersonContext]:
        """Return a snapshot of the current cached persons list."""
        with self._cache_lock:
            self._clear_if_expired_locked(time.time())
            return list(self._cached_persons)

    def get_attention_target_person_id(self) -> str | None:
        """Return the current recognized attention target person id, if any."""
        with self._cache_lock:
            self._clear_if_expired_locked(time.time())
            if self._attention_target is None:
                return None
            return self._attention_target.person_id

    def get_primary_face_person_id(self) -> str | None:
        """Return the current recognized primary visible person id, if any."""
        return self.get_attention_target_person_id()

    def get_primary_attention_person_id(self) -> str | None:
        """Return the current recognized primary attentive person id, if any."""
        with self._cache_lock:
            self._clear_if_expired_locked(time.time())
            rendered = str(self._presence.primary_attention_person_id or "").strip()
            return rendered or None

    def get_face_turn_target(self, person_id: str | None = None) -> FaceTurnTarget | None:
        """Return the latest bearing target for one recognized visible person."""
        now = time.time()
        requested_id = str(person_id or "").strip()
        with self._cache_lock:
            self._clear_if_expired_locked(now)
            target_id = requested_id
            if not target_id:
                if self._attention_target is None:
                    return None
                target_id = str(self._attention_target.person_id or "").strip()
            if not target_id:
                return None

            for person in self._cached_persons:
                if str(person.person_id or "").strip() != target_id:
                    continue
                if not bool(getattr(person, "visible", True)):
                    return None
                bearing = getattr(person, "bearing_rad", None)
                if bearing is None:
                    return None
                return FaceTurnTarget(
                    person_id=person.person_id,
                    name=person.name,
                    bearing_rad=float(bearing),
                    timestamp=float(person.timestamp),
                    confidence=float(person.confidence),
                    depth_m=person.depth_m,
                )
        return None

    def get_presence_snapshot(self) -> dict[str, Any]:
        """Return the exported face-presence state as a dict."""
        now = time.time()
        with self._cache_lock:
            self._clear_if_expired_locked(now)
            return asdict(self._presence)
