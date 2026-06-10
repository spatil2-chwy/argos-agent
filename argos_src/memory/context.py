"""Prompt-context compiler for Argos source-aware memory."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any

from argos_src.memory.prompting import (
    format_followup_lines,
    format_person_profile_lines,
    format_recent_encounters_block,
    format_site_memory_block,
)
from argos_src.memory.store import MemoryStore


@dataclass(frozen=True)
class PersonMemoryContext:
    profile_lines: tuple[str, ...] = ()
    followup_lines: tuple[str, ...] = ()
    preferred_language: str = ""


RELATION_FIELDS = (
    ("manager_name", "same manager"),
    ("cost_center", "same cost center"),
    ("business_function", "same business function"),
    ("senior_leadership_team", "same leadership org"),
)


class MemoryContextCompiler:
    """Build small prompt-visible memory projections for the current turn."""

    def __init__(self, memory_store: MemoryStore, identity_store: Any | None = None) -> None:
        self.memory_store = memory_store
        self.identity_store = identity_store

    def person_context(
        self,
        person_id: str,
        *,
        fallback_profile_lines: tuple[str, ...] = (),
        fallback_followup_lines: tuple[str, ...] = (),
    ) -> PersonMemoryContext:
        items = self.memory_store.list_active_items(
            scope_type="person",
            scope_id=person_id,
            kinds=("preference", "boundary", "pet", "fact", "note", "followup"),
            limit=100,
        )
        profile_lines = format_person_profile_lines(items)
        followup_lines = format_followup_lines(items)
        preferred_language = self._preferred_language(items)
        return PersonMemoryContext(
            profile_lines=profile_lines or fallback_profile_lines,
            followup_lines=followup_lines or fallback_followup_lines,
            preferred_language=preferred_language,
        )

    @staticmethod
    def _preferred_language(items) -> str:
        for item in items:
            if item.kind != "preference" or item.key != "preferred_language":
                continue
            value = str(item.metadata.get("value") or "").strip()
            if value:
                return value
            prefix = "preferred language:"
            summary = str(item.summary or "").strip()
            if summary.casefold().startswith(prefix):
                return summary[len(prefix):].strip()
        return ""

    def site_blocks(
        self,
        site_code: str,
        *,
        current_person_id: str | None = None,
    ) -> tuple[str, ...]:
        blocks: list[str] = []
        site_items = self.memory_store.list_active_items(
            scope_type="site",
            scope_id=site_code,
            kinds=("office_event",),
            limit=10,
        )
        site_block = format_site_memory_block(site_items)
        if site_block:
            blocks.append(site_block)
        encounters = self.memory_store.list_recent_encounters(
            site_code=site_code,
            since=datetime.now(timezone.utc) - timedelta(hours=2),
            exclude_person_id=current_person_id,
            limit=10,
        )
        encounters = self._relation_relevant_encounters(
            encounters,
            current_person_id=current_person_id,
        )
        encounter_block = format_recent_encounters_block(encounters, now=datetime.now(timezone.utc))
        if encounter_block:
            blocks.append(encounter_block)
        return tuple(blocks)

    def _relation_relevant_encounters(
        self,
        encounters,
        *,
        current_person_id: str | None,
    ):
        current_metadata = self._identity_metadata(current_person_id)
        if not current_metadata:
            return []
        relevant = []
        for item in encounters:
            relation = self._relation_label(current_metadata, item.metadata)
            if not relation:
                continue
            metadata = dict(item.metadata)
            metadata["relation_label"] = relation
            relevant.append(replace(item, metadata=metadata))
            if len(relevant) >= 3:
                break
        return relevant

    def _identity_metadata(self, person_id: str | None) -> dict[str, Any]:
        identity_store = getattr(self, "identity_store", None)
        rendered = str(person_id or "").strip()
        if identity_store is None or not rendered:
            return {}
        record = identity_store.get_person(rendered)
        if record is None:
            return {}
        return dict(record.get("metadata") or {})

    @staticmethod
    def _relation_label(current: dict[str, Any], encountered: dict[str, Any]) -> str:
        for field, label in RELATION_FIELDS:
            left = str(current.get(field) or "").strip().casefold()
            right = str(encountered.get(field) or "").strip().casefold()
            if left and right and left == right:
                return label
        return ""
