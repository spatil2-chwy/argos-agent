"""Timer-backed coalescer for internal robot events."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from argos_src.agent.control.observers import safe_ignored
from argos_src.agent.control.reducers.coalescing import (
    dedup_events,
    render_coalesced_text,
    render_internal_audio_turn_events,
)
from argos_src.agent.control.types import StateAxis

logger = logging.getLogger(__name__)


class EventCoalescer:
    """Buffers rapid events and flushes them as one combined message.

    Internal events (face, nav, battery) start/extend a debounce timer.
    Human input flushes immediately, bundling any pending internal events into
    the same batch. A max-wait cap prevents indefinite buffering when internal
    events keep arriving.
    """

    def __init__(
        self,
        agent: Any,
        engagement: Any,
        debounce_sec: float = 0.4,
        max_wait_sec: float = 2.0,
        state_observer: Any = None,
    ):
        self._agent = agent
        self._engagement = engagement
        self._debounce_sec = debounce_sec
        self._max_wait_sec = max_wait_sec
        self._state_observer = state_observer
        self._buffer: list[tuple[str, dict]] = []
        self._lock = threading.RLock()
        self._timer: Optional[threading.Timer] = None
        self._first_event_time: Optional[float] = None

    def submit(self, text: str, metadata: Optional[dict] = None) -> None:
        """Submit an event (human or internal) for coalescing."""
        meta = dict(metadata or {})
        is_human = not meta.get("internal", False)

        with self._lock:
            self._buffer.append((text, meta))
            if self._first_event_time is None:
                self._first_event_time = time.time()

            if is_human:
                self._cancel_timer_locked()
                self._flush_locked()
            else:
                elapsed = time.time() - self._first_event_time
                if elapsed >= self._max_wait_sec:
                    if self._should_defer_internal_flush_locked():
                        self._restart_timer_locked()
                    else:
                        self._cancel_timer_locked()
                        self._flush_locked()
                else:
                    self._restart_timer_locked()

        # Notify engagement outside the coalescer lock to avoid ABBA deadlocks
        # with the engagement watchdog, which may call force_flush.
        if is_human:
            self._engagement.on_human_input(meta.get("req_id"))

    def force_flush(self) -> None:
        """Flush all buffered events immediately."""
        with self._lock:
            self._flush_locked()

    def drain_internal_events_for_audio_turn(
        self,
        metadata: Optional[dict] = None,
    ) -> tuple[Optional[str], dict]:
        """Return any pending internal-event text for a live audio turn."""
        with self._lock:
            if not self._buffer:
                return None, dict(metadata or {})

            events = list(self._buffer)
            self._buffer.clear()
            self._first_event_time = None
            self._cancel_timer_locked()

        events = dedup_events(events)
        return render_internal_audio_turn_events(events, dict(metadata or {}))

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        events = list(self._buffer)
        self._buffer.clear()
        self._first_event_time = None
        self._cancel_timer_locked()

        events = dedup_events(events)
        if not events:
            return
        combined, primary_meta = render_coalesced_text(events)
        self._agent.enqueue_internal_event(combined, metadata=primary_meta)

    def _dedup(self, events: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
        """Compatibility wrapper for older tests and labs."""
        return dedup_events(events)

    def _restart_timer_locked(self) -> None:
        self._cancel_timer_locked()
        self._timer = threading.Timer(self._debounce_sec, self._timer_flush)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _timer_flush(self) -> None:
        with self._lock:
            if self._should_defer_internal_flush_locked():
                safe_ignored(
                    self._state_observer,
                    axis=StateAxis.COALESCER,
                    trigger="timer_flush",
                    reason="recording_active",
                )
                self._restart_timer_locked()
                return
            self._flush_locked()

    def _should_defer_internal_flush_locked(self) -> bool:
        """Hold internal-only events while a human utterance is being recorded."""
        if not self._buffer:
            return False
        if any(not meta.get("internal") for _, meta in self._buffer):
            return False
        is_recording_active = getattr(self._engagement, "is_recording_active", None)
        if not callable(is_recording_active):
            return False
        try:
            return bool(is_recording_active())
        except Exception:
            logger.exception("Failed to check recording state before flushing events")
            return False
