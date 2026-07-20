"""Pure event coalescing rules for robot/internal event batches."""

from __future__ import annotations

from typing import Any

CoalescedEvent = tuple[str, dict[str, Any]]


def dedup_events(events: list[CoalescedEvent]) -> list[CoalescedEvent]:
    """Deduplicate events within a coalescer batch.

    Rules:
    - Multiple FACE_EVENTs for the same person keep only the latest one.
    - Multiple NAV_EVENT goal results keep only the latest goal result.
    - NAV waypoint chatter is dropped when any goal result is present.
    """
    has_human = any(not metadata.get("internal") for _, metadata in events)
    has_face = any(metadata.get("internal_event") == "face" for _, metadata in events)
    has_nav_result = any(
        metadata.get("internal_event") == "navigation"
        and metadata.get("event_type") == "goal_result"
        for _, metadata in events
    )

    latest_face: dict[str, int] = {}
    latest_nav_result_idx: int | None = None

    for index, (_text, metadata) in enumerate(events):
        event_kind = str(metadata.get("internal_event", "") or "")
        if event_kind == "face":
            key = str(metadata.get("person_name", "") or "") or "__unknown__"
            latest_face[key] = index
        if event_kind == "navigation" and metadata.get("event_type") == "goal_result":
            latest_nav_result_idx = index

    result: list[CoalescedEvent] = []
    for index, (text, metadata) in enumerate(events):
        event_kind = str(metadata.get("internal_event", "") or "")

        if event_kind == "face":
            key = str(metadata.get("person_name", "") or "") or "__unknown__"
            if latest_face.get(key) != index:
                continue

        if event_kind == "navigation":
            event_type = str(metadata.get("event_type", "") or "")
            if event_type == "goal_result" and latest_nav_result_idx != index:
                continue
            if event_type != "goal_result" and has_nav_result:
                continue

        result.append((text, metadata))

    return result


def render_coalesced_text(events: list[CoalescedEvent]) -> tuple[str, dict[str, Any]]:
    """Render deduplicated events into the model-visible internal text payload."""
    internal = [(text, meta) for text, meta in events if meta.get("internal")]
    human = [(text, meta) for text, meta in events if not meta.get("internal")]

    parts: list[str] = []
    if internal:
        parts.append("[INTERNAL EVENT]" if len(internal) == 1 and not human else "[PENDING EVENTS]")
        for text, _meta in internal:
            parts.append(f"- {text}")
    if human:
        parts.append("[HUMAN INPUT]")
        for text, _meta in human:
            parts.append(text)

    primary_meta = human[-1][1] if human else events[-1][1]
    return "\n".join(parts), primary_meta


def render_internal_audio_turn_events(
    events: list[CoalescedEvent],
    metadata: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    """Render pending internal events that should ride along with an audio turn."""
    internal = [(text, meta) for text, meta in events if meta.get("internal")]
    if not internal:
        return None, dict(metadata)

    parts = ["[INTERNAL EVENT]" if len(internal) == 1 else "[PENDING EVENTS]"]
    for text, _meta in internal:
        parts.append(f"- {text}")
    merged_meta = dict(internal[-1][1])
    merged_meta.update(dict(metadata))
    return "\n".join(parts), merged_meta
