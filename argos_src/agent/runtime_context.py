"""Turn-scoped prompt context helpers for the realtime Argos agent."""

from __future__ import annotations

import ast
import json
from datetime import datetime
from typing import Any, Optional

from argos_src.identity_memory.normalization import normalize_directory_profile_lines
from argos_src.nav_support.locations import LocationStore

MAX_TOOL_SUMMARY_LEN = 240


def format_people_context(
    persons: list,
    *,
    primary_face_person_id: Optional[str] = None,
    face_snapshot: Optional[dict[str, Any]] = None,
    audio_speaker_id: Optional[str] = None,
    owner_id: Optional[str] = None,
    owner_source: str = "unknown",
    speaker_visible: bool = False,
) -> str:
    """Build owner-scoped person context for turn instructions."""
    del primary_face_person_id
    resolved_owner_id = str(owner_id or "").strip()
    if not resolved_owner_id:
        return ""

    owner_person = next(
        (
            person
            for person in persons
            if str(getattr(person, "person_id", "") or "").strip() == resolved_owner_id
        ),
        None,
    )
    lines = ["[PERSON SPEAKING TO YOU]"]
    owner_name = (
        str(getattr(owner_person, "name", "") or "").strip()
        if owner_person is not None
        else ""
    ) or resolved_owner_id
    interaction_count = (
        int(getattr(owner_person, "interaction_count", 0) or 0)
        if owner_person is not None
        else 0
    )
    if interaction_count == 0:
        count_str = "first time"
    elif interaction_count == 1:
        count_str = "met once before"
    else:
        count_str = f"met {interaction_count} times"
    owner_visible = (
        bool(getattr(owner_person, "visible", True))
        if owner_person is not None
        else False
    )
    if owner_visible:
        lines.append(f"- {owner_name} ({count_str})")
    else:
        lines.append(f"- {owner_name} ({count_str}; not visible)")

    directory_items = (
        getattr(owner_person, "directory_profile_lines", ())
        if owner_person is not None
        else ()
    )
    directory_profile_lines = normalize_directory_profile_lines(directory_items)
    if directory_profile_lines:
        lines.append(f"  Directory: {'; '.join(directory_profile_lines)}")

    context_markdown = (
        str(getattr(owner_person, "context_markdown", "") or "").strip()
        if owner_person is not None
        else ""
    )
    if context_markdown:
        lines.append(context_markdown)

    snapshot = face_snapshot or {}
    unknown_count = int(snapshot.get("unknown_count", 0) or 0)
    snapshot_recognized_names = [
        str(name or "").strip()
        for name in (snapshot.get("recognized_names") or ())
        if str(name or "").strip()
    ]
    visible_other_names: list[str] = []
    for name in snapshot_recognized_names:
        if name and name != owner_name and name not in visible_other_names:
            visible_other_names.append(name)
    for person in persons:
        person_id = str(getattr(person, "person_id", "") or "").strip()
        name = str(getattr(person, "name", "") or "").strip()
        if person_id == resolved_owner_id or not name:
            continue
        if not bool(getattr(person, "visible", True)):
            continue
        if name not in visible_other_names:
            visible_other_names.append(name)

    if visible_other_names or unknown_count > 0:
        lines.append("")
        lines.append("[OTHER PEOPLE IN VIEW]")
    for name in visible_other_names:
        lines.append(f"- {name}")
    if unknown_count > 0:
        label = "unrecognized person" if unknown_count == 1 else "unrecognized people"
        lines.append(f"- {unknown_count} {label}")

    prioritized_language = (
        str(getattr(owner_person, "preferred_language", "") or "").strip()
        if owner_person is not None
        else ""
    )
    if prioritized_language:
        lines.append(
            f"- Prioritize talking in this language to this user: {prioritized_language}."
        )

    return "\n".join(lines)


def format_current_time_block(now: Optional[datetime] = None) -> str:
    """Build the current local date/time block for dynamic prompt context."""
    local_now = now or datetime.now().astimezone()
    return (
        "[CURRENT TIME] "
        + local_now.strftime("%A, %B %d, %Y at %I:%M %p %Z").replace(" 0", " ")
    )


def format_current_office_location_block(site_code: Optional[str]) -> str:
    """Build the current office-location block for turn-scoped instructions."""
    cleaned = str(site_code or "").strip()
    return f"[CURRENT OFFICE LOCATION] {cleaned}" if cleaned else ""


def format_saved_locations(store: LocationStore) -> str:
    """Build the [SAVED LOCATIONS] block for turn-scoped instructions."""
    names = store.names()
    if not names:
        return (
            "[SAVED LOCATIONS] (none yet - use get_current_location "
            "with save=True to save a spot)"
        )
    return "[SAVED LOCATIONS] " + ", ".join(names)


def parse_tool_output(content: object) -> Optional[dict[str, Any]]:
    """Parse tool output content as a dict when possible."""
    if isinstance(content, dict):
        return content
    text = str(content).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return None


def summarize_tool_payload(
    tool_name: Optional[str],
    content: object,
) -> tuple[Optional[str], Optional[str]]:
    """Return (posture, compact summary) extracted from a tool result payload."""
    posture: Optional[str] = None
    summary: Optional[str] = None
    payload = parse_tool_output(content)
    if payload is not None:
        payload_posture = payload.get("robot_state_after")
        if isinstance(payload_posture, str) and payload_posture.strip():
            posture = payload_posture.strip()
        error_text = str(payload.get("error", "") or "").strip()
        message_text = str(payload.get("message", "") or "").strip()
        status_text = str(payload.get("status", "") or "").strip()
        verification_text = str(payload.get("charging_verification", "") or "").strip()
        success_value = payload.get("success")
        if error_text:
            summary = f"error={error_text}"
        elif success_value is True and status_text:
            summary = f"success ({status_text})"
        elif success_value is True:
            summary = "success"
        elif success_value is False and message_text and status_text:
            summary = f"{status_text}: {message_text}"
        elif success_value is False and message_text:
            summary = message_text
        elif success_value is False and status_text:
            summary = f"failed ({status_text})"
        elif message_text:
            summary = message_text
        elif status_text:
            summary = status_text
        elif verification_text:
            summary = f"charging_verification={verification_text}"
        else:
            summary = str(payload).strip()
    else:
        summary = str(content or "").strip()

    if summary:
        summary = summary[:MAX_TOOL_SUMMARY_LEN]
    summary = summary or None
    if tool_name and summary:
        return posture, f"{tool_name}: {summary}"
    return posture, summary


def format_robot_state_block(
    posture: str,
    last_tool_name: Optional[str],
    last_tool_summary: Optional[str],
    *,
    stand_tool_name: str,
    supports_navigation: bool,
) -> str:
    """Build the [ROBOT STATE] block for turn-scoped instructions."""
    posture_text = posture if posture else "unknown"
    lines = [f"[ROBOT STATE] posture={posture_text}"]
    if last_tool_name:
        lines.append(f"Last tool call: {last_tool_name}.")
    if last_tool_summary:
        lines.append(f"Last tool result: {last_tool_summary}")
    if supports_navigation:
        lines.append(
            f"If posture is not standing, call {stand_tool_name} before navigation or movement tools."
        )
    else:
        lines.append(
            f"If posture is not standing, call {stand_tool_name} before movement tools."
        )
    return " ".join(lines)
