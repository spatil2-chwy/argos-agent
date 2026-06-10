"""Prompt formatting helpers for source-aware Argos memory."""

from __future__ import annotations

from datetime import datetime, timezone

from argos_src.memory.models import MemoryItem


def _profile_item_rank(item: MemoryItem) -> tuple[int, str]:
    if item.kind == "preference":
        return (0, item.observed_at)
    if item.kind == "boundary":
        return (1, item.observed_at)
    if item.kind == "pet":
        return (2, item.observed_at)
    if item.kind == "fact":
        return (3, item.observed_at)
    field = str(item.metadata.get("field") or "").strip()
    if field == "pets" or item.summary.casefold().startswith("pet:"):
        return (2, item.observed_at)
    return (3, item.observed_at)


def format_person_profile_lines(
    items: list[MemoryItem],
    *,
    structured_limit: int = 20,
    note_limit: int = 10,
) -> tuple[str, ...]:
    lines: list[str] = []
    structured_items = [
        item for item in items if item.kind in {"preference", "boundary", "pet", "fact"}
    ]
    structured_items.sort(key=_profile_item_rank)
    note_items = [item for item in items if item.kind == "note"]
    note_items.sort(key=lambda item: item.observed_at, reverse=True)
    capped_structured = structured_items[: max(0, int(structured_limit or 0))]
    capped_notes = note_items[: max(0, int(note_limit or 0))]
    for item in [*capped_structured, *capped_notes]:
        text = item.summary.strip()
        if text and text not in lines:
            lines.append(text)
    return tuple(lines)


def _is_due(due_at: str, *, now: datetime | None = None) -> bool:
    text = str(due_at or "").strip()
    if not text:
        return True
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        due = datetime.fromisoformat(text)
    except Exception:
        return True
    if due.tzinfo is None:
        due = due.replace(tzinfo=timezone.utc)
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return due <= ref


def format_followup_lines(
    items: list[MemoryItem],
    *,
    limit: int = 3,
    now: datetime | None = None,
) -> tuple[str, ...]:
    lines: list[str] = []
    for item in items:
        if item.kind != "followup":
            continue
        if not _is_due(item.due_at, now=now):
            continue
        text = item.summary.strip()
        if text and text not in lines:
            lines.append(text)
        if len(lines) >= limit:
            break
    return tuple(lines)


def format_site_memory_block(items: list[MemoryItem], *, limit: int = 5) -> str:
    lines = [
        item.summary.strip()
        for item in items
        if item.kind == "office_event" and item.summary.strip()
    ][:limit]
    if not lines:
        return ""
    return "\n".join(["[OFFICE CONTEXT]", *(f"- {line}" for line in lines)])


def _relative_age(observed_at: str, *, now: datetime | None = None) -> str:
    text = str(observed_at or "").strip()
    if not text:
        return ""
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        observed = datetime.fromisoformat(text)
    except Exception:
        return ""
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    seconds = max(0, int((ref - observed).total_seconds()))
    minutes = seconds // 60
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} minutes ago"
    hours = minutes // 60
    return f"{hours} hours ago"


def format_recent_encounters_block(
    items: list[MemoryItem],
    *,
    now: datetime | None = None,
    limit: int = 3,
) -> str:
    lines: list[str] = []
    for item in items:
        name = str(item.metadata.get("name") or "").strip()
        relation = str(item.metadata.get("relation_label") or "").strip()
        age = _relative_age(item.observed_at, now=now)
        if name:
            line = f"You met {name}"
            if age:
                line += f" {age}"
            if relation:
                line += f"; they are in the current person's org context ({relation})"
            line += "."
        else:
            line = item.summary.strip()
        if line and line not in lines:
            lines.append(line)
        if len(lines) >= limit:
            break
    if not lines:
        return ""
    return "\n".join(["[RECENT ENCOUNTERS]", *(f"- {line}" for line in lines)])
