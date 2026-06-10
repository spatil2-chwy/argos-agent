"""Turn-scoped prompt context helpers for the realtime Argos agent."""

from __future__ import annotations

import ast
import json
from datetime import datetime
from typing import Any, Optional

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
    """Build the [PEOPLE IN VIEW] block for turn-scoped instructions."""
    del primary_face_person_id
    lines = ["[PEOPLE IN VIEW]"]
    resolved_owner_id = owner_id
    for person in persons:
        is_owner = bool(resolved_owner_id and person.person_id == resolved_owner_id)
        is_visible = bool(getattr(person, "visible", True))
        if person.interaction_count == 0:
            count_str = "first time"
        elif person.interaction_count == 1:
            count_str = "met once before"
        else:
            count_str = f"met {person.interaction_count} times"
        speaker_tag = " [talking to you]" if is_owner else ""
        if is_visible:
            lines.append(f"- {person.name} ({count_str}){speaker_tag}")
        else:
            lines.append(
                f"- Recognized speaker identity: {person.name} ({count_str}; not visible){speaker_tag}"
            )
        directory_profile_lines = tuple(
            str(item).strip()
            for item in (getattr(person, "directory_profile_lines", ()) or ())
            if str(item).strip()
        )
        if directory_profile_lines:
            lines.append(f"  Directory: {'; '.join(directory_profile_lines)}")
        if not is_owner:
            continue
        memory_profile_lines = tuple(
            str(item).strip()
            for item in (getattr(person, "memory_profile_lines", ()) or ())
            if str(item).strip()
        )
        if memory_profile_lines:
            lines.append(f"  About: {'; '.join(memory_profile_lines)}")
        else:
            lines.append(
                "  About: No durable social memory stored yet. Use this conversation to learn one useful social detail."
            )
        followups = tuple(getattr(person, "potential_followups", ()) or ())
        if followups:
            rendered = " ".join(
                str(item).strip() for item in followups if str(item).strip()
            )
            if rendered:
                lines.append(f"  Potential Followups: {rendered}")

    snapshot = face_snapshot or {}
    recognized_count = int(snapshot.get("recognized_count", 0) or 0)
    unknown_count = int(snapshot.get("unknown_count", 0) or 0)
    nearest_recognized = str(snapshot.get("nearest_recognized_name", "") or "").strip()
    snapshot_recognized_names = [
        str(name or "").strip()
        for name in (snapshot.get("recognized_names") or ())
        if str(name or "").strip()
    ]
    recognized_names = snapshot_recognized_names or [
        str(getattr(person, "name", "") or "").strip()
        for person in persons
        if bool(getattr(person, "visible", True))
        and str(getattr(person, "name", "") or "").strip()
    ]
    audio_speaker_name = next(
        (person.name for person in persons if person.person_id == audio_speaker_id),
        "",
    ).strip()
    owner_name = next(
        (person.name for person in persons if person.person_id == resolved_owner_id),
        "",
    ).strip()

    if not persons and unknown_count > 0:
        lines.append("- No recognized people in view yet.")

    if recognized_count > 0 and unknown_count > 0:
        if recognized_names:
            lines.append(
                "- Social scene: recognized people in view: "
                + ", ".join(recognized_names)
                + ". There is also at least one unrecognized person nearby."
            )
        elif nearest_recognized:
            lines.append(
                f"- Social scene: there is at least one unrecognized person nearby. The nearest recognized person is {nearest_recognized}."
            )
        else:
            lines.append(
                "- Social scene: there is at least one recognized person in view and at least one unrecognized person nearby."
            )
    elif unknown_count > 0:
        lines.append("- Social scene: there is at least one unrecognized person in view.")
    elif recognized_names:
        lines.append(
            "- Social scene: recognized people in view: " + ", ".join(recognized_names) + "."
        )
    elif nearest_recognized:
        lines.append(f"- Social scene: the nearest recognized person in view is {nearest_recognized}.")

    if audio_speaker_name and not speaker_visible:
        lines.append(
            f"- Current speaker voice matches {audio_speaker_name}, but {audio_speaker_name} is not visible right now."
        )
    elif audio_speaker_name:
        lines.append(f"- Current speaker voice matches {audio_speaker_name}.")
    elif owner_source == "face" and owner_name:
        lines.append(
            f"- Current speaker appears to be {owner_name} from the visible face captured for this turn."
        )
    elif recognized_count > 0 or unknown_count > 0:
        lines.append("- Current speaker is not safely identified.")

    prioritized_language = ""
    if resolved_owner_id:
        prioritized_language = next(
            (
                str(getattr(person, "preferred_language", "") or "").strip()
                for person in persons
                if person.person_id == resolved_owner_id
            ),
            "",
        )
    if not prioritized_language and len(persons) == 1:
        prioritized_language = str(
            getattr(persons[0], "preferred_language", "") or ""
        ).strip()
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
