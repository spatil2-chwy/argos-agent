"""Speaker-owned preference extraction controller."""

from __future__ import annotations

import threading
from typing import Any

from argos_src.agent.preference_types import PreferenceSegmentTurn


class PreferenceRuntime:
    """Buffer finalized turns and schedule preference extraction work."""

    def __init__(self, host: Any) -> None:
        self._host = host

    def maybe_note_turn(self, turn: Any) -> None:
        host = self._host
        if host._preference_segments is None or turn.source_is_internal:
            return
        flush_unattributed = False
        with host._turn_lock:
            if getattr(turn, "preference_noted", False):
                return
            user_transcript = str(turn.user_transcript or "").strip()
            assistant_transcript = str(turn.assistant_transcript or "").strip()
            owner_id = str(turn.owner_id or "").strip()
            if turn.kind == "audio" and not user_transcript:
                host.logger.info(
                    "Preference extraction waiting req_id=%s "
                    "missing=input_transcript owner_id=%s user_item_id=%s phase=%s",
                    turn.req_id,
                    turn.owner_id,
                    getattr(turn, "user_item_id", ""),
                    getattr(turn, "phase", ""),
                )
                return
            if not assistant_transcript:
                host.logger.info(
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
                host.logger.info(
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
            host.flush_preference_segments(reason="speaker_unattributed")
            return

        self.cancel_idle_flush()
        completed_segment = host._preference_segments.add_completed_turn(
            PreferenceSegmentTurn(
                turn_id=turn.req_id,
                person_id=owner_id,
                user_text=user_transcript,
                assistant_text=assistant_transcript,
            )
        )
        if completed_segment is not None:
            self._emit_memory_segment_flushed(completed_segment, reason="speaker_handoff")
            self.schedule_segment_extraction(
                completed_segment,
                reason="speaker_handoff",
            )

    def flush_segments(self, reason: str = "idle") -> None:
        host = self._host
        if host._preference_segments is None:
            return
        if reason == "idle":
            self.schedule_idle_flush()
            return
        self.cancel_idle_flush()
        self.retry_ready_turns()
        completed_segment = host._preference_segments.flush_active()
        if completed_segment is None:
            if reason in {"idle_timeout", "shutdown"}:
                finish_episode = getattr(
                    host.preference_extractor,
                    "finish_active_episode",
                    None,
                )
                if callable(finish_episode):
                    finish_episode(reason=reason)
            return
        self._emit_memory_segment_flushed(completed_segment, reason=reason)
        self.schedule_segment_extraction(completed_segment, reason=reason)

    def _emit_memory_segment_flushed(self, segment: Any, *, reason: str) -> None:
        host = self._host
        latency = getattr(host, "_latency", None)
        emit = getattr(latency, "emit", None)
        if not callable(emit):
            return
        turns = tuple(getattr(segment, "turns", ()) or ())
        last_turn_id = str(getattr(turns[-1], "turn_id", "") or "") if turns else ""
        emit(
            event="memory_segment_flushed",
            req_id=last_turn_id or None,
            memory_segment_id=getattr(segment, "segment_id", None),
            memory_person_id=getattr(segment, "person_id", None),
            memory_turn_count=len(turns),
            memory_flush_reason=reason,
            memory_extraction_enabled=bool(getattr(host, "preference_extraction_enabled", False)),
            memory_extraction_scheduled=bool(
                getattr(host, "preference_extraction_enabled", False)
                and getattr(host, "preference_extractor", None) is not None
            ),
        )

    def schedule_idle_flush(self) -> None:
        host = self._host
        if host._preference_segments is None:
            return

        def run_flush() -> None:
            with host._preference_idle_flush_lock:
                host._preference_idle_flush_timer = None
            host.flush_preference_segments(reason="idle_timeout")

        with host._preference_idle_flush_lock:
            if host._preference_idle_flush_timer is not None:
                host._preference_idle_flush_timer.cancel()
            timer = threading.Timer(host._preference_idle_flush_delay_sec, run_flush)
            timer.daemon = True
            host._preference_idle_flush_timer = timer
            timer.start()

    def cancel_idle_flush(self) -> None:
        host = self._host
        with host._preference_idle_flush_lock:
            if host._preference_idle_flush_timer is not None:
                host._preference_idle_flush_timer.cancel()
                host._preference_idle_flush_timer = None

    def schedule_segment_extraction(self, segment: Any, *, reason: str) -> None:
        host = self._host
        if not host.preference_extraction_enabled or host.preference_extractor is None:
            return
        if not getattr(segment, "turns", None):
            return

        with host._pending_lock:
            if segment.segment_id in host._pending_preference_segment_ids:
                return
            host._pending_preference_segment_ids.add(segment.segment_id)

        def run_then_clear() -> None:
            try:
                try:
                    host.preference_extractor.extract_and_store_segment(
                        segment,
                        reason=reason,
                    )
                except TypeError:
                    host.preference_extractor.extract_and_store_segment(segment)
            finally:
                with host._pending_lock:
                    host._pending_preference_segment_ids.discard(segment.segment_id)

        try:
            host._preference_executor.submit(run_then_clear)
        except RuntimeError:
            with host._pending_lock:
                host._pending_preference_segment_ids.discard(segment.segment_id)
            host.logger.exception(
                "Failed to schedule preference extraction segment=%s person=%s reason=%s",
                segment.segment_id,
                segment.person_id,
                reason,
            )
            return
        host.logger.info(
            "Scheduled preference extraction segment=%s person=%s reason=%s",
            segment.segment_id,
            segment.person_id,
            reason,
        )

    def retry_ready_turns(self) -> None:
        host = self._host
        if host._preference_segments is None:
            return
        finalized_phase = "finalized"
        with host._turn_lock:
            turns = [
                turn
                for turn in host._turns_by_req_id.values()
                if getattr(turn, "phase", "") == finalized_phase
                and not getattr(turn, "preference_noted", False)
                and not getattr(turn, "source_is_internal", False)
                and str(getattr(turn, "user_transcript", "") or "").strip()
                and str(getattr(turn, "assistant_transcript", "") or "").strip()
                and str(getattr(turn, "owner_id", "") or "").strip()
            ]
        for turn in turns:
            self.maybe_note_turn(turn)
