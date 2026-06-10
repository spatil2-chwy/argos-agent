"""Typed models and normalization helpers for Argos memory items."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


MemoryScope = Literal["person", "site"]
MemoryKind = Literal[
    "preference",
    "boundary",
    "pet",
    "fact",
    "note",
    "followup",
    "encounter",
    "office_event",
]
MemorySource = Literal["live_chat", "robot", "slack"]
MemoryStatus = Literal["active", "archived", "superseded"]

VALID_SCOPES = {"person", "site"}
VALID_KINDS = {
    "preference",
    "boundary",
    "pet",
    "fact",
    "note",
    "followup",
    "encounter",
    "office_event",
}
VALID_SOURCES = {"live_chat", "robot", "slack"}
VALID_STATUSES = {"active", "archived", "superseded"}


@dataclass(frozen=True)
class MemoryItem:
    """One source-aware memory record."""

    memory_id: str
    scope_type: MemoryScope
    scope_id: str
    kind: MemoryKind
    key: str
    summary: str
    source: MemorySource
    source_ref: str = ""
    status: MemoryStatus = "active"
    created_at: str = ""
    observed_at: str = ""
    updated_at: str = ""
    due_at: str = ""
    expires_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def is_expired(value: Any, *, now: datetime | None = None) -> bool:
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return False
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return parsed <= ref


def normalize_key(value: Any) -> str:
    rendered = str(value or "").strip().casefold()
    normalized = "".join(char if char.isalnum() else "_" for char in rendered)
    return "_".join(part for part in normalized.split("_") if part)


def require_valid(value: str, allowed: set[str], field: str) -> str:
    rendered = str(value or "").strip()
    if rendered not in allowed:
        raise ValueError(f"invalid {field}: {value!r}")
    return rendered
