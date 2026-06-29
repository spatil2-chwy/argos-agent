"""Tailwag-backed social memory adapter.

This module intentionally keeps Tailwag imports lazy so Argos can keep running
without memory context when Tailwag or one of its backing services is
unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import re
import threading
from typing import Any, Callable
from uuid import uuid4

from argos_src.agent.preference_types import PreferenceSegment


logger = logging.getLogger(__name__)

DEFAULT_PLACE_BUILDING = "ARGOS"
DEFAULT_PLACE_ROOM = "realtime"
DEFAULT_RETENTION_CLASS = "standard"
DEFAULT_PREFERRED_LANGUAGE = "English"


@dataclass(frozen=True)
class PersonMemoryContext:
    """Prompt-compatible person memory projection for Argos."""

    profile_lines: tuple[str, ...] = ()
    followup_lines: tuple[str, ...] = ()
    preferred_language: str = DEFAULT_PREFERRED_LANGUAGE


class TailwagMemoryProvider:
    """Adapter between Argos runtime hooks and tailwag-memory."""

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

    def close(self) -> None:
        client = self._client_instance
        self._client_instance = None
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Failed to close Tailwag memory client")

    def health(self) -> bool:
        try:
            self._client()
        except Exception:
            logger.exception("Tailwag memory health check failed")
            return False
        return True

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
            raw = self._client().person_context(
                rendered_person_id,
                current_text=current_text,
            )
        except Exception:
            logger.exception(
                "Tailwag person context unavailable for person_id=%s",
                rendered_person_id,
            )
            return PersonMemoryContext(
                profile_lines=fallback_profile_lines,
                followup_lines=fallback_followup_lines,
            )

        profile_lines, followup_lines = _parse_tailwag_context(raw)
        preferred_language = _preferred_language_from_lines(
            (*profile_lines, *followup_lines)
        )
        return PersonMemoryContext(
            profile_lines=profile_lines or fallback_profile_lines,
            followup_lines=followup_lines or fallback_followup_lines,
            preferred_language=preferred_language,
        )

    def site_blocks(
        self,
        site_code: str,
        *,
        current_person_id: str | None = None,
    ) -> tuple[str, ...]:
        del site_code, current_person_id
        # Site context is intentionally deferred until Tailwag exposes a
        # first-class contract for prompt-ready site context.
        return ()

    def extract_and_store_segment(self, segment: PreferenceSegment, reason: str = "") -> None:
        """Append a flushed realtime segment to the active Tailwag episode."""
        try:
            episode = self._episode_from_segment(segment)
        except Exception:
            logger.exception(
                "Tailwag live-turn episode construction failed segment=%s person=%s",
                getattr(segment, "segment_id", ""),
                getattr(segment, "person_id", ""),
            )
            if _is_terminal_flush(reason):
                self.finish_active_episode(reason=reason)
            return
        if episode is None:
            return
        terminal = _is_terminal_flush(reason)
        try:
            self._client().record_episode(
                episode,
                extract_memory=self.extract_live_turn_memory,
            )
        except Exception:
            logger.exception(
                "Tailwag live-turn episode ingestion failed segment=%s person=%s",
                getattr(segment, "segment_id", ""),
                getattr(segment, "person_id", ""),
            )
        finally:
            if terminal:
                self.finish_active_episode(reason=reason)

    def finish_active_episode(self, *, reason: str = "") -> None:
        """End the current realtime episode without adding a new segment."""
        del reason
        self._reset_active_episode()

    def record_encounter(
        self,
        *,
        person_id: str,
        name: str,
        site_code: str = "",
        metadata: dict[str, Any] | None = None,
        observed_at: str | None = None,
    ) -> bool:
        """Update Tailwag person profile/last_seen for a locally seen person."""
        del site_code, observed_at
        rendered_person_id = str(person_id or "").strip()
        if not rendered_person_id:
            return False
        meta = dict(metadata or {})
        try:
            self.upsert_person(
                person_id=rendered_person_id,
                display_name=str(name or meta.get("name") or rendered_person_id),
                email=_metadata_email(meta),
            )
        except Exception:
            logger.exception(
                "Tailwag encounter person upsert failed person_id=%s",
                rendered_person_id,
            )
            return False
        return True

    def upsert_person(
        self,
        *,
        person_id: str,
        display_name: str = "",
        email: str = "",
        consent_status: str | None = None,
    ) -> str:
        rendered_person_id = str(person_id or "").strip()
        if not rendered_person_id:
            raise ValueError("person_id is required")
        normalized_email = _normalize_email(email)
        client = self._client()
        if normalized_email:
            try:
                rekeyed = client.rekey_person_by_email(
                    normalized_email,
                    rendered_person_id,
                )
                if rekeyed is False:
                    logger.info(
                        "Tailwag person rekey unresolved; identity review needed "
                        "person_id=%s email=%s reason=%s",
                        rendered_person_id,
                        normalized_email,
                        "no_unique_email_match_or_safe_temp_person",
                    )
            except Exception:
                logger.exception(
                    "Tailwag person rekey by email failed email=%s person_id=%s",
                    normalized_email,
                    rendered_person_id,
                )
        return client.upsert_person(
            self._person_input(
                id=rendered_person_id,
                display_name=str(display_name or "").strip() or None,
                email=normalized_email or None,
                consent_status=consent_status,
                role="participant",
                source="argos",
            )
        )

    def archive_person(self, person_id: str) -> bool:
        rendered_person_id = str(person_id or "").strip()
        if not rendered_person_id:
            return False
        try:
            return bool(self._client().archive_person(rendered_person_id))
        except Exception:
            logger.exception(
                "Tailwag person archive failed person_id=%s",
                rendered_person_id,
            )
            return False

    def record_episode(self, episode: Any, *, extract_memory: bool = True) -> Any:
        return self._client().record_episode(
            episode,
            extract_memory=extract_memory,
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
            self._active_segment_text[segment_id or f"segment-{len(self._active_segment_text)}"] = (
                segment_text
            )
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
                self._person_input(
                    id=participant_id,
                    role="speaker",
                    source="live_chat",
                )
                for participant_id in participants
            ],
        )

    def _reset_active_episode(self) -> None:
        with self._episode_lock:
            self._active_episode_id = ""
            self._active_started_at = ""
            self._active_segment_text = {}
            self._active_person_ids = set()

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


def _parse_tailwag_context(value: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    profile: list[str] = []
    followups: list[str] = []
    section = ""
    for raw_line in str(value or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "[PERSON MEMORY]":
            continue
        if line.startswith("[") and line.endswith("]"):
            section = "profile"
            continue
        if line.endswith(":") and not line.startswith("-"):
            section = line[:-1].strip().casefold()
            continue
        text = line[1:].strip() if line.startswith("-") else line
        text = _clean_context_line(text)
        if not text:
            continue
        if section in {"potential follow-ups", "potential followups", "followups"}:
            followups.append(text)
        else:
            profile.append(text)
    return tuple(_dedupe(profile)), tuple(_dedupe(followups))


def _preferred_language_from_lines(lines: tuple[str, ...]) -> str:
    for line in lines:
        match = re.search(
            r"\bpreferred\s+language\s*[:\-]\s*([A-Za-z][A-Za-z \-]+)",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(" .")
    return DEFAULT_PREFERRED_LANGUAGE


def _clean_context_line(value: str) -> str:
    return " ".join(str(value or "").split()).lstrip("#-*[]>` ").strip()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _normalize_email(value: Any) -> str:
    rendered = str(value or "").strip().casefold()
    return rendered if "@" in rendered else ""


def _metadata_email(metadata: dict[str, Any]) -> str:
    for key in ("email", "work_email", "employee_email"):
        email = _normalize_email(metadata.get(key))
        if email:
            return email
    return ""
