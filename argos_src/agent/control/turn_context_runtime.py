"""Frozen person, face, and memory context for realtime turns."""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Optional

from argos_src.agent.realtime_turns import FrozenTurnContext
from argos_src.face_recognition.models import PersonContext
from argos_src.identity_memory.normalization import normalize_directory_profile_lines


class TurnContextRuntime:
    """Build prompt-facing human context snapshots for a turn."""

    def __init__(self, host: Any) -> None:
        self._host = host

    def enrich_person_context_with_memory(self, person: PersonContext) -> PersonContext:
        host = self._host
        compiler = getattr(host, "memory_context_compiler", None)
        if compiler is None:
            return person
        person_id = str(getattr(person, "person_id", "") or "").strip()
        if not person_id:
            return person
        try:
            context = compiler.person_context(
                person_id,
            )
        except Exception:
            host.logger.exception("Failed to compile memory context for %s", person_id)
            return person
        person.context_markdown = str(getattr(context, "context_markdown", "") or "").strip()
        if context.preferred_language:
            person.preferred_language = context.preferred_language
        return person

    def compile_memory_context_blocks(
        self,
        current_person_id: Optional[str],
    ) -> tuple[str, ...]:
        host = self._host
        compiler = getattr(host, "memory_context_compiler", None)
        if compiler is None:
            return ()
        site_code = str(getattr(host, "_current_office_location", "") or "").strip()
        if not site_code:
            return ()
        try:
            return tuple(
                compiler.site_blocks(
                    site_code,
                    current_person_id=str(current_person_id or "").strip() or None,
                )
            )
        except Exception:
            host.logger.exception("Failed to compile site memory context")
            return ()

    def identity_person(self, person_id: Optional[str]) -> PersonContext | None:
        host = self._host
        rendered = str(person_id or "").strip()
        identity_memory = getattr(host, "identity_memory_client", None)
        if not rendered or identity_memory is None:
            return None
        try:
            profile = identity_memory.person_profile(rendered)
            if profile is None:
                return None
            metadata = dict(getattr(profile, "metadata", {}) or {})
            directory_lines = normalize_directory_profile_lines(
                getattr(profile, "directory_profile_lines", ())
            )
            person = PersonContext(
                person_id=rendered,
                name=str(getattr(profile, "display_name", "") or metadata.get("name") or rendered),
                interaction_count=int(getattr(profile, "interaction_count", 0) or 0),
                confidence=1.0,
                bbox_area=0,
                timestamp=time.time(),
                directory_profile_lines=directory_lines,
                memory_profile_lines=(),
                preferred_language="",
                potential_followups=(),
                visible=False,
            )
            return self.enrich_person_context_with_memory(person)
        except Exception:
            host.logger.exception("Failed to build identity context for %s", rendered)
            return None

    def capture_turn_context(
        self,
        *,
        primary_face_person_id: Optional[str] = None,
        audio_speaker_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        owner_source: str = "unknown",
        owner_confidence: float = 0.0,
        speaker_visible: bool = False,
    ) -> FrozenTurnContext:
        host = self._host
        memory_context_blocks = self.compile_memory_context_blocks(owner_id)
        if host.face_service is None:
            person = self.identity_person(owner_id)
            return FrozenTurnContext(
                persons=[person] if person is not None else [],
                primary_face_person_id=primary_face_person_id,
                audio_speaker_id=audio_speaker_id,
                owner_id=owner_id,
                owner_source=owner_source,
                owner_confidence=owner_confidence,
                speaker_visible=speaker_visible,
                memory_context_blocks=memory_context_blocks,
            )
        try:
            persons = deepcopy(host.face_service.get_cached_persons())
            owner_context_id = str(owner_id or "").strip()
            if owner_context_id:
                persons = [
                    self.enrich_person_context_with_memory(person)
                    if str(getattr(person, "person_id", "") or "").strip()
                    == owner_context_id
                    else person
                    for person in persons
                ]
            face_snapshot = deepcopy(host.face_service.get_presence_snapshot())
            if owner_context_id and not any(
                str(getattr(person, "person_id", "") or "").strip()
                == owner_context_id
                for person in persons
            ):
                person = self.identity_person(owner_context_id)
                if person is not None:
                    persons.append(person)
            if primary_face_person_id is None:
                attention_getter = getattr(
                    host.face_service,
                    "get_primary_attention_person_id",
                    None,
                )
                if callable(attention_getter):
                    primary_face_person_id = attention_getter()
                if primary_face_person_id is None:
                    getter = getattr(host.face_service, "get_primary_face_person_id", None)
                    if callable(getter):
                        primary_face_person_id = getter()
                    else:
                        primary_face_person_id = (
                            host.face_service.get_attention_target_person_id()
                        )
            return FrozenTurnContext(
                persons=persons,
                face_snapshot=face_snapshot,
                primary_face_person_id=primary_face_person_id,
                audio_speaker_id=audio_speaker_id,
                owner_id=owner_id,
                owner_source=owner_source,
                owner_confidence=owner_confidence,
                speaker_visible=speaker_visible,
                memory_context_blocks=memory_context_blocks,
            )
        except Exception:
            host.logger.exception("Failed to capture turn context snapshot")
            return FrozenTurnContext(
                primary_face_person_id=primary_face_person_id,
                audio_speaker_id=audio_speaker_id,
                owner_id=owner_id,
                owner_source=owner_source,
                owner_confidence=owner_confidence,
                speaker_visible=speaker_visible,
                memory_context_blocks=memory_context_blocks,
            )
