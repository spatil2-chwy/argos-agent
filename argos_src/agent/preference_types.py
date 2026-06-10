"""Shared types for buffered live-chat memory extraction segments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreferenceSegmentTurn:
    """One completed attributed conversational turn."""

    turn_id: str
    person_id: str
    user_text: str = ""
    assistant_text: str = ""


@dataclass(frozen=True)
class PreferenceSegment:
    """Buffered consecutive turns owned by one recognized speaker."""

    segment_id: str
    person_id: str
    turns: tuple[PreferenceSegmentTurn, ...] = ()
