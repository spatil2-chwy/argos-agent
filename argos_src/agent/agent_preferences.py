"""Preference-extraction helpers for the Argos agent runtime."""

from __future__ import annotations

from typing import Any

from argos_src.agent.preference_types import PreferenceSegmentTurn


class RealtimeAgentPreferenceMixin:
    def _maybe_note_preference_turn(self, turn) -> None:
        if self._preference_segments is None or turn.source_is_internal:
            return
        flush_unattributed = False
        with self._turn_lock:
            if getattr(turn, "preference_noted", False):
                return
            user_transcript = str(turn.user_transcript or "").strip()
            assistant_transcript = str(turn.assistant_transcript or "").strip()
            owner_id = str(turn.owner_id or "").strip()
            if turn.kind == "audio" and not user_transcript:
                self.logger.info(
                    "Preference extraction waiting req_id=%s "
                    "missing=input_transcript owner_id=%s user_item_id=%s phase=%s",
                    turn.req_id,
                    turn.owner_id,
                    getattr(turn, "user_item_id", ""),
                    getattr(turn, "phase", ""),
                )
                return
            if not assistant_transcript:
                self.logger.info(
                    "Preference extraction waiting req_id=%s "
                    "missing=assistant_transcript owner_id=%s user_item_id=%s phase=%s",
                    turn.req_id,
                    turn.owner_id,
                    getattr(turn, "user_item_id", ""),
                    getattr(turn, "phase", ""),
                )
                return
            if not owner_id:
                if getattr(turn, "preference_unattributed_flushed", False):
                    return
                self.logger.info(
                    "Preference extraction skipped unattributed turn req_id=%s "
                    "reason=missing_owner_id action=flush_active_segment user_item_id=%s "
                    "phase=%s user_transcript_len=%s",
                    turn.req_id,
                    getattr(turn, "user_item_id", ""),
                    getattr(turn, "phase", ""),
                    len(user_transcript),
                )
                turn.preference_unattributed_flushed = True
                flush_unattributed = True
            if not flush_unattributed:
                turn.preference_noted = True

        if flush_unattributed:
            # This turn cannot be stored yet, but it still marks a break after
            # any previous attributed memory segment.
            self.flush_preference_segments(reason="speaker_unattributed")
            return

        self._cancel_preference_idle_flush()
        completed_segment = self._preference_segments.add_completed_turn(
            PreferenceSegmentTurn(
                turn_id=turn.req_id,
                person_id=owner_id,
                user_text=user_transcript,
                assistant_text=assistant_transcript,
            )
        )
        if completed_segment is not None:
            self._schedule_preference_segment_extraction(
                completed_segment,
                reason="speaker_handoff",
            )

    def _schedule_preference_segment_extraction(self, segment: Any, *, reason: str) -> None:
        if not self.preference_extraction_enabled or self.preference_extractor is None:
            return
        if not getattr(segment, "turns", None):
            return

        with self._pending_lock:
            if segment.segment_id in self._pending_preference_segment_ids:
                return
            self._pending_preference_segment_ids.add(segment.segment_id)

        def run_then_clear() -> None:
            try:
                try:
                    self.preference_extractor.extract_and_store_segment(segment, reason=reason)
                except TypeError:
                    self.preference_extractor.extract_and_store_segment(segment)
            finally:
                with self._pending_lock:
                    self._pending_preference_segment_ids.discard(segment.segment_id)

        try:
            self._preference_executor.submit(run_then_clear)
        except RuntimeError:
            with self._pending_lock:
                self._pending_preference_segment_ids.discard(segment.segment_id)
            self.logger.exception(
                "Failed to schedule preference extraction segment=%s person=%s reason=%s",
                segment.segment_id,
                segment.person_id,
                reason,
            )
            return
        self.logger.info(
            "Scheduled preference extraction segment=%s person=%s reason=%s",
            segment.segment_id,
            segment.person_id,
            reason,
        )

    def _retry_ready_preference_turns(self) -> None:
        if self._preference_segments is None:
            return
        finalized_phase = "finalized"
        with self._turn_lock:
            turns = [
                turn
                for turn in self._turns_by_req_id.values()
                if getattr(turn, "phase", "") == finalized_phase
                and not getattr(turn, "preference_noted", False)
                and not getattr(turn, "source_is_internal", False)
                and str(getattr(turn, "user_transcript", "") or "").strip()
                and str(getattr(turn, "assistant_transcript", "") or "").strip()
                and str(getattr(turn, "owner_id", "") or "").strip()
            ]
        for turn in turns:
            self._maybe_note_preference_turn(turn)
