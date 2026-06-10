"""Event coalescing and engagement state machine for the Argos realtime agent.

EventCoalescer: buffers rapid events and flushes them as one combined
turn payload to the realtime agent queue (debounce for internal,
immediate for human).

EngagementStateMachine: IDLE / ALERT / ENGAGED / SPEAKING / COOLDOWN —
formalises interaction flow, controls patrol suppression, playback-aware
state transitions, and proactive battery-low prompts.
"""

from dataclasses import dataclass
import enum
import logging
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event Coalescer
# ---------------------------------------------------------------------------


class EventCoalescer:
    """Buffers rapid events and flushes them as one combined message.

    Internal events (face, nav, patrol, battery) start/extend a debounce
    timer.  Human input flushes immediately, bundling any pending internal
    events into the same batch.  A max-wait cap prevents indefinite
    buffering when internal events keep arriving.
    """

    def __init__(
        self,
        agent: Any,
        engagement: "EngagementStateMachine",
        debounce_sec: float = 0.4,
        max_wait_sec: float = 2.0,
    ):
        self._agent = agent
        self._engagement = engagement
        self._debounce_sec = debounce_sec
        self._max_wait_sec = max_wait_sec
        self._buffer: list[tuple[str, dict]] = []
        self._lock = threading.RLock()
        self._timer: Optional[threading.Timer] = None
        self._first_event_time: Optional[float] = None

    # -- public API ----------------------------------------------------------

    def submit(self, text: str, metadata: Optional[dict] = None) -> None:
        """Submit an event (human or internal) for coalescing."""
        meta = dict(metadata or {})
        is_human = not meta.get("internal", False)
        is_patrol = meta.get("internal_event") == "patrol_continue"

        if is_patrol and self._engagement.should_suppress_patrol():
            return

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

        # Notify engagement *outside* coalescer lock to avoid ABBA deadlock with the engagement watchdog (which may call force_flush).
        if is_human:
            self._engagement.on_human_input(meta.get("req_id"))

    def force_flush(self) -> None:
        """Flush all buffered events immediately (e.g. on ALERT timeout)."""
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

        events = self._dedup(events)
        internal = [(text, meta) for text, meta in events if meta.get("internal")]
        if not internal:
            return None, dict(metadata or {})

        parts = ["[INTERNAL EVENT]" if len(internal) == 1 else "[PENDING EVENTS]"]
        for text, _meta in internal:
            parts.append(f"- {text}")
        merged_meta = dict(internal[-1][1])
        merged_meta.update(dict(metadata or {}))
        return "\n".join(parts), merged_meta

    # -- internals -----------------------------------------------------------

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        events = list(self._buffer)
        self._buffer.clear()
        self._first_event_time = None
        self._cancel_timer_locked()

        events = self._dedup(events)
        if not events:
            return

        internal = [(t, m) for t, m in events if m.get("internal")]
        human = [(t, m) for t, m in events if not m.get("internal")]

        parts: list[str] = []
        if internal:
            parts.append("[INTERNAL EVENT]" if len(internal) == 1 and not human else "[PENDING EVENTS]")
            for text, _ in internal:
                parts.append(f"- {text}")
        if human:
            parts.append("[HUMAN INPUT]")
            for text, _ in human:
                parts.append(text)

        combined = "\n".join(parts)
        primary_meta = human[-1][1] if human else events[-1][1]
        self._agent.enqueue_internal_event(combined, metadata=primary_meta)

    def _dedup(self, events: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
        """Deduplicate events within a batch.

        Rules:
        - Multiple FACE_EVENTs for the same person -> keep latest only.
        - Multiple NAV_EVENTs -> keep only the final goal_result; drop
          intermediate waypoint events when a goal_result is present.
        - PATROL_EVENT suppressed if face or human input in the same batch.
        """
        has_human = any(not m.get("internal") for _, m in events)
        has_face = any(m.get("internal_event") == "face" for _, m in events)
        has_nav_result = any(
            m.get("internal_event") == "navigation"
            and m.get("event_type") == "goal_result"
            for _, m in events
        )

        latest_face: dict[str, int] = {}
        latest_nav_result_idx: Optional[int] = None

        for i, (_text, meta) in enumerate(events):
            evt = meta.get("internal_event", "")
            if evt == "face":
                key = meta.get("person_name", "") or "__unknown__"
                latest_face[key] = i
            if evt == "navigation" and meta.get("event_type") == "goal_result":
                latest_nav_result_idx = i

        result: list[tuple[str, dict]] = []
        for i, (text, meta) in enumerate(events):
            evt = meta.get("internal_event", "")

            if evt == "patrol_continue" and (has_face or has_human):
                continue

            if evt == "face":
                key = meta.get("person_name", "") or "__unknown__"
                if latest_face.get(key) != i:
                    continue

            if evt == "navigation":
                event_type = meta.get("event_type", "")
                if event_type == "goal_result" and latest_nav_result_idx != i:
                    continue
                if event_type != "goal_result" and has_nav_result:
                    continue

            result.append((text, meta))

        return result

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


# ---------------------------------------------------------------------------
# Engagement State Machine
# ---------------------------------------------------------------------------


class EngagementState(enum.Enum):
    IDLE = "idle"
    ALERT = "alert"
    ENGAGED = "engaged"
    SPEAKING = "speaking"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class InteractionSnapshot:
    state: str
    req_id: str
    entered_at: float
    expires_at: Optional[float]
    nav_active: bool
    nav_source: str
    nav_interruptible: bool
    nav_passive_listen_allowed: bool


class EngagementStateMachine:
    """Tracks playback-aware interaction phases for the Argos realtime agent."""

    def __init__(
        self,
        *,
        voice_cmd_publisher: Optional[Callable[[str], None]] = None,
        alert_timeout_sec: float = 15.0,
        cooldown_sec: float = 7.0,
        speaking_timeout_sec: float = 30.0,
        on_idle_entered: Optional[Callable[[], None]] = None,
        nav_state: Any = None,
        battery_cache: Any = None,
        self_charge_available: bool = True,
    ):
        self._state = EngagementState.IDLE
        self._lock = threading.RLock()
        self._last_transition = time.time()
        self._voice_cmd_pub = voice_cmd_publisher
        self._alert_timeout = alert_timeout_sec
        self._cooldown_timeout = cooldown_sec
        self._speaking_timeout = speaking_timeout_sec
        self._on_idle_entered = on_idle_entered
        self._nav_state = nav_state
        self._battery_cache = battery_cache
        self._self_charge_available = bool(self_charge_available)
        self._current_req_id = ""
        self._awaiting_playback_req_id = ""
        self._awaiting_playback_stream_id = ""
        self._latest_playback_stream_id = ""
        self._awaiting_playback_terminal = False

        # Set by factory after construction (circular refs)
        self._coalescer: Optional[EventCoalescer] = None
        self._battery_low_submit: Optional[Callable[[str, dict], None]] = None
        self._recording_state_provider: Optional[Callable[[], bool]] = None
        self._battery_low_latch = False

        self._watchdog_stop = threading.Event()
        self._watchdog = threading.Thread(
            target=self._run_timeout_watchdog, daemon=True
        )
        self._watchdog.start()

    # -- public API ----------------------------------------------------------

    @property
    def state(self) -> EngagementState:
        with self._lock:
            return self._state

    @property
    def state_name(self) -> str:
        return self.state.value

    def snapshot(self) -> InteractionSnapshot:
        with self._lock:
            state = self._state
            entered_at = self._last_transition
            snapshot = InteractionSnapshot(
                state=state.value,
                req_id=self._current_req_id,
                entered_at=entered_at,
                expires_at=self._expires_at_for(state, entered_at),
                nav_active=False,
                nav_source="",
                nav_interruptible=True,
                nav_passive_listen_allowed=True,
            )
        if self._nav_state is None:
            return snapshot
        nav_context = self._nav_state.build_interaction_context()
        return InteractionSnapshot(
            state=snapshot.state,
            req_id=snapshot.req_id,
            entered_at=snapshot.entered_at,
            expires_at=snapshot.expires_at,
            nav_active=bool(nav_context.get("nav_active", False)),
            nav_source=str(nav_context.get("nav_source", "") or ""),
            nav_interruptible=bool(nav_context.get("nav_interruptible", True)),
            nav_passive_listen_allowed=bool(
                nav_context.get("nav_passive_listen_allowed", True)
            ),
        )

    def should_suppress_patrol(self) -> bool:
        return self.state != EngagementState.IDLE

    def attach_coalescer(self, coalescer: Optional[EventCoalescer]) -> None:
        """Attach the runtime coalescer after construction."""
        with self._lock:
            self._coalescer = coalescer

    def attach_battery_low_submitter(
        self,
        submitter: Optional[Callable[[str, dict], None]],
    ) -> None:
        """Attach the battery-low event submitter after construction."""
        with self._lock:
            self._battery_low_submit = submitter

    def attach_recording_state_provider(
        self,
        provider: Optional[Callable[[], bool]],
    ) -> None:
        """Attach a callback that reports whether local speech capture is active."""
        with self._lock:
            self._recording_state_provider = provider

    def is_recording_active(self) -> bool:
        """Return True while the local runtime is actively buffering a human utterance."""
        with self._lock:
            provider = self._recording_state_provider
        if provider is None:
            return False
        try:
            return bool(provider())
        except Exception:
            logger.exception("Recording state provider failed")
            return False

    def _clear_playback_tracking_locked(self) -> None:
        self._awaiting_playback_req_id = ""
        self._awaiting_playback_stream_id = ""
        self._latest_playback_stream_id = ""
        self._awaiting_playback_terminal = False

    def on_face_or_wake(self) -> None:
        """Called when a proactive face event claims the robot."""
        should_act = False
        with self._lock:
            if self._state == EngagementState.IDLE:
                self._current_req_id = ""
                self._clear_playback_tracking_locked()
                self._set_state_locked(
                    EngagementState.ALERT,
                    reason="face_detected",
                    req_id="",
                )
                should_act = True
        if should_act:
            self._publish_voice_cmd("stop")
            self._cancel_active_navigation()

    def on_human_input(self, req_id: Optional[str] = None) -> None:
        """Called after the coalescer flushes a human message."""
        should_cancel = False
        req = str(req_id or "").strip()
        with self._lock:
            if self._state in (
                EngagementState.IDLE,
                EngagementState.ALERT,
                EngagementState.COOLDOWN,
            ):
                should_cancel = self._state in (
                    EngagementState.IDLE,
                    EngagementState.COOLDOWN,
                )
                self._current_req_id = req
                self._clear_playback_tracking_locked()
                self._set_state_locked(
                    EngagementState.ENGAGED,
                    reason="human_input",
                    req_id=req,
                )
        if should_cancel:
            self._publish_voice_cmd("stop")
            self._cancel_active_navigation()

    def on_agent_output_started(
        self,
        req_id: Optional[str],
        *,
        stream_id: Optional[str] = None,
    ) -> None:
        """Called when the current turn first emits audible model output."""
        req = str(req_id or "").strip()
        stream = str(stream_id or "").strip()
        if not req:
            return
        with self._lock:
            if self._state not in (
                EngagementState.ALERT,
                EngagementState.ENGAGED,
                EngagementState.SPEAKING,
            ):
                return
            self._current_req_id = req
            if stream:
                self._latest_playback_stream_id = stream
            if self._state in (
                EngagementState.ALERT,
                EngagementState.ENGAGED,
            ):
                self._set_state_locked(
                    EngagementState.SPEAKING,
                    reason="agent_output_started",
                    req_id=req,
                )
            return

    def on_agent_done(self, *, has_reply: bool, req_id: Optional[str]) -> None:
        """Called when the agent finishes processing a message."""
        req = str(req_id or "").strip()
        with self._lock:
            if has_reply:
                if self._state not in (
                    EngagementState.ALERT,
                    EngagementState.ENGAGED,
                    EngagementState.SPEAKING,
                ):
                    return
                chosen_req = req or self._current_req_id
                chosen_stream = self._latest_playback_stream_id
                self._current_req_id = chosen_req
                self._awaiting_playback_req_id = chosen_req
                self._awaiting_playback_stream_id = chosen_stream
                self._awaiting_playback_terminal = True
                if self._state in (
                    EngagementState.ALERT,
                    EngagementState.ENGAGED,
                ):
                    self._set_state_locked(
                        EngagementState.SPEAKING,
                        reason="agent_done_with_reply",
                        req_id=chosen_req,
                    )
                return
            if self._state in (
                EngagementState.ALERT,
                EngagementState.ENGAGED,
            ):
                self._current_req_id = req
                self._clear_playback_tracking_locked()
                self._set_state_locked(
                    EngagementState.COOLDOWN,
                    reason="agent_done",
                    req_id=req,
                )

    def on_playback_event(
        self,
        event: str,
        req_id: Optional[str],
        *,
        stream_id: Optional[str] = None,
    ) -> None:
        """Called when a reply playback lifecycle changes."""
        req = str(req_id or "").strip()
        stream = str(stream_id or "").strip()
        if not req and not stream:
            logger.warning(
                "Ignoring playback event without req_id or stream_id: event=%s",
                event,
            )
            return
        with self._lock:
            if event == "playback_started":
                if stream:
                    self._latest_playback_stream_id = stream
                if self._state in (
                    EngagementState.ALERT,
                    EngagementState.ENGAGED,
                ):
                    self._current_req_id = req
                    self._set_state_locked(
                        EngagementState.SPEAKING,
                        reason="playback_started",
                        req_id=req,
                )
                return
            if event not in ("playback_completed", "playback_stopped"):
                return
            if not self._awaiting_playback_terminal:
                if self._state == EngagementState.SPEAKING:
                    logger.warning(
                        "Ignoring terminal playback event while not awaiting one: event=%s req_id=%s stream_id=%s state=%s",
                        event,
                        req,
                        stream,
                        self._state.value,
                    )
                return
            req_matches = bool(req) and req == self._awaiting_playback_req_id
            stream_matches = bool(stream) and stream == self._awaiting_playback_stream_id
            if self._awaiting_playback_stream_id:
                if not stream_matches:
                    logger.warning(
                        "Ignoring terminal playback event due to stream mismatch: event=%s req_id=%s stream_id=%s awaiting_req_id=%s awaiting_stream_id=%s",
                        event,
                        req,
                        stream,
                        self._awaiting_playback_req_id,
                        self._awaiting_playback_stream_id,
                    )
                    return
            elif self._awaiting_playback_req_id:
                if not req_matches:
                    logger.warning(
                        "Ignoring terminal playback event due to req_id mismatch: event=%s req_id=%s stream_id=%s awaiting_req_id=%s",
                        event,
                        req,
                        stream,
                        self._awaiting_playback_req_id,
                    )
                    return
            else:
                logger.warning(
                    "Ignoring terminal playback event with no awaited req_id or stream_id: event=%s req_id=%s stream_id=%s",
                    event,
                    req,
                    stream,
                )
                return
            if self._state not in (
                EngagementState.SPEAKING,
                EngagementState.ENGAGED,
            ):
                logger.warning(
                    "Ignoring terminal playback event because state is %s instead of an awaited playback state: event=%s req_id=%s stream_id=%s",
                    self._state.value,
                    event,
                    req,
                    stream,
                )
                return
            if req:
                self._current_req_id = req
            self._clear_playback_tracking_locked()
            logger.info(
                "Accepted terminal playback event event=%s req_id=%s stream_id=%s",
                event,
                req,
                stream,
            )
            self._set_state_locked(
                EngagementState.COOLDOWN,
                reason=event,
                req_id=req,
            )

    def shutdown(self) -> None:
        self._watchdog_stop.set()
        self._watchdog.join(timeout=2.0)

    # -- internals -----------------------------------------------------------

    def _expires_at_for(self, state: EngagementState, entered_at: float) -> Optional[float]:
        if state == EngagementState.ALERT:
            return entered_at + self._alert_timeout
        if state == EngagementState.COOLDOWN:
            return entered_at + self._cooldown_timeout
        return None

    def _set_state_locked(
        self,
        new_state: EngagementState,
        *,
        reason: str,
        req_id: str,
    ) -> None:
        old = self._state
        self._state = new_state
        self._last_transition = time.time()
        if new_state == EngagementState.IDLE:
            self._current_req_id = ""
            self._clear_playback_tracking_locked()
        logger.info(
            "Engagement: %s -> %s reason=%s req_id=%s",
            old.value,
            new_state.value,
            reason,
            req_id,
        )

    def _publish_voice_cmd(self, cmd: str) -> None:
        if self._voice_cmd_pub is not None:
            try:
                self._voice_cmd_pub(cmd)
            except Exception:
                pass

    def _cancel_active_navigation(self) -> None:
        if self._nav_state is None:
            return
        if not self._nav_state.active_goal_allows_auto_interrupt():
            return
        handle = self._nav_state.take_last_goal_handle()
        if handle is not None:
            try:
                handle.cancel_goal_async()
            except Exception:
                pass

    def _run_timeout_watchdog(self) -> None:
        while not self._watchdog_stop.wait(1.0):
            action: Optional[str] = None
            with self._lock:
                elapsed = time.time() - self._last_transition
                if (
                    self._state == EngagementState.ALERT
                    and elapsed >= self._alert_timeout
                ):
                    self._set_state_locked(
                        EngagementState.IDLE,
                        reason="timeout",
                        req_id="",
                    )
                    action = "alert_timeout"
                elif (
                    self._state == EngagementState.COOLDOWN
                    and elapsed >= self._cooldown_timeout
                ):
                    self._set_state_locked(
                        EngagementState.IDLE,
                        reason="timeout",
                        req_id="",
                    )
                    action = "cooldown_timeout"
                elif (
                    (
                        (
                            self._awaiting_playback_terminal
                            and self._state
                            in (
                                EngagementState.ENGAGED,
                                EngagementState.SPEAKING,
                            )
                        )
                        or self._state == EngagementState.SPEAKING
                    )
                    and elapsed >= self._speaking_timeout
                ):
                    req = self._awaiting_playback_req_id or self._current_req_id
                    logger.warning(
                        "Playback watchdog fallback fired after %.2fs awaiting req_id=%s stream_id=%s current_req_id=%s",
                        elapsed,
                        self._awaiting_playback_req_id,
                        self._awaiting_playback_stream_id,
                        self._current_req_id,
                    )
                    self._clear_playback_tracking_locked()
                    self._set_state_locked(
                        EngagementState.COOLDOWN,
                        reason="fallback",
                        req_id=req,
                    )

            if action == "alert_timeout":
                if self._coalescer:
                    self._coalescer.force_flush()
                if self._on_idle_entered:
                    threading.Thread(
                        target=self._on_idle_entered, daemon=True
                    ).start()
            elif action == "cooldown_timeout":
                if self._on_idle_entered:
                    threading.Thread(
                        target=self._on_idle_entered, daemon=True
                    ).start()

            self._check_low_battery()

    def _check_low_battery(self) -> None:
        """Inject BATTERY_LOW_EVENT when IDLE + battery is below threshold + no nav."""
        if self._battery_cache is None or self._battery_low_submit is None:
            return
        if self.state != EngagementState.IDLE:
            self._battery_low_latch = False
            return
        if (
            self._nav_state is not None
            and self._nav_state.get_active_goal() is not None
        ):
            return
        if not self._battery_cache.should_block_general_navigation():
            self._battery_low_latch = False
            return
        if self._battery_low_latch:
            return
        self._battery_low_latch = True
        snap = self._battery_cache.snapshot()
        pct = getattr(snap, "percentage", 0) if snap else 0
        low_pct = getattr(self._battery_cache, "low_battery_pct", 0)
        can_self_charge = self._self_charge_available
        if self._battery_cache is not None and hasattr(
            self._battery_cache, "can_self_charge"
        ):
            can_self_charge = bool(self._battery_cache.can_self_charge())
        if can_self_charge:
            text = (
                f"BATTERY_LOW_EVENT: Battery is at {pct:.0f}%. You need to charge. "
                "Call charging_dock now."
            )
        else:
            text = (
                f"BATTERY_LOW_EVENT: Battery is at {pct:.0f}%. "
                f"Tell the user your battery is below {low_pct:.0f}% and ask to be "
                "charged soon for continued functionality."
            )
        self._battery_low_submit(
            text,
            {
                "internal": True,
                "internal_event": "battery_low",
                "source": "battery_state",
            },
        )
