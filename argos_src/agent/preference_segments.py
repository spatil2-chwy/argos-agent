"""Helpers for buffering speaker-owned preference extraction segments."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Optional

from argos_src.agent.preference_types import (
    PreferenceSegment,
    PreferenceSegmentTurn,
)


@dataclass
class _BufferedPreferenceSegment:
    """Mutable buffer of consecutive turns attributed to one speaker."""

    segment_id: str
    person_id: str
    turns: list[PreferenceSegmentTurn] = field(default_factory=list)

    def to_segment(self) -> PreferenceSegment:
        return PreferenceSegment(
            segment_id=self.segment_id,
            person_id=self.person_id,
            turns=tuple(self.turns),
        )


class _PreferenceSegmentCoordinator:
    """Thread-safe owner of the active preference-extraction segment."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_segment: Optional[_BufferedPreferenceSegment] = None

    def add_completed_turn(self, turn: PreferenceSegmentTurn) -> Optional[PreferenceSegment]:
        """Append one completed attributed turn, flushing on speaker handoff."""
        with self._lock:
            if self._active_segment is None:
                self._active_segment = _BufferedPreferenceSegment(
                    segment_id=turn.turn_id,
                    person_id=turn.person_id,
                    turns=[turn],
                )
                return None

            if self._active_segment.person_id == turn.person_id:
                self._active_segment.turns.append(turn)
                return None

            completed = self._active_segment.to_segment()
            self._active_segment = _BufferedPreferenceSegment(
                segment_id=turn.turn_id,
                person_id=turn.person_id,
                turns=[turn],
            )
            return completed

    def flush_active(self) -> Optional[PreferenceSegment]:
        """Return and clear the active segment, if any."""
        with self._lock:
            if self._active_segment is None or not self._active_segment.turns:
                self._active_segment = None
                return None
            completed = self._active_segment.to_segment()
            self._active_segment = None
            return completed
