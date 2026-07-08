"""Playback-aware engagement runtime for the realtime agent."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Callable, Optional

from argos_src.agent.control.coalescer import EventCoalescer
from argos_src.agent.control.observers import safe_transition
from argos_src.agent.control.reducers.engagement import (
    EngagementTrigger,
    decision_has_action,
    reduce_engagement,
)
from argos_src.agent.control.types import EngagementMode, StateAxis, StateTransition

logger = logging.getLogger(__name__)


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
        state_observer: Any = None,
    ):
        self._state = EngagementMode.IDLE
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
        self._state_observer = state_observer
        self._current_req_id = ""
        self._awaiting_playback_req_id = ""
        self._awaiting_playback_stream_id = ""
        self._latest_playback_stream_id = ""
        self._awaiting_playback_terminal = False

        self._coalescer: Optional[EventCoalescer] = None
        self._battery_low_submit: Optional[Callable[[str, dict], None]] = None
        self._recording_state_provider: Optional[Callable[[], bool]] = None
        self._battery_low_latch = False

        self._watchdog_stop = threading.Event()
        self._watchdog = threading.Thread(
            target=self._run_timeout_watchdog, daemon=True
        )
        self._watchdog.start()

    @property
    def state(self) -> EngagementMode:
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
        return self.state != EngagementMode.IDLE

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

    def _apply_decision_locked(
        self,
        decision: Any,
        *,
        req_id: str,
    ) -> None:
        if not decision.changed:
            return
        self._set_state_locked(
            EngagementMode(decision.new_state),
            reason=decision.reason,
            req_id=req_id,
        )

    def on_face_or_wake(self) -> None:
        """Called when a proactive face event claims the robot."""
        should_act = False
        with self._lock:
            decision = reduce_engagement(
                self._state.value,
                EngagementTrigger.FACE_OR_WAKE,
            )
            if decision.changed:
                self._current_req_id = ""
                self._clear_playback_tracking_locked()
                self._apply_decision_locked(decision, req_id="")
                should_act = decision_has_action(decision, "cancel_active_navigation")
        if should_act:
            self._publish_voice_cmd("stop")
            self._cancel_active_navigation()

    def on_human_input(self, req_id: Optional[str] = None) -> None:
        """Called after the coalescer flushes a human message."""
        should_cancel = False
        req = str(req_id or "").strip()
        with self._lock:
            decision = reduce_engagement(
                self._state.value,
                EngagementTrigger.HUMAN_INPUT,
            )
            if decision.changed:
                should_cancel = decision_has_action(decision, "cancel_active_navigation")
                self._current_req_id = req
                self._clear_playback_tracking_locked()
                self._apply_decision_locked(decision, req_id=req)
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
                EngagementMode.ALERT,
                EngagementMode.ENGAGED,
                EngagementMode.SPEAKING,
            ):
                return
            self._current_req_id = req
            if stream:
                self._latest_playback_stream_id = stream
            decision = reduce_engagement(
                self._state.value,
                EngagementTrigger.AGENT_OUTPUT_STARTED,
            )
            if decision.changed:
                self._apply_decision_locked(decision, req_id=req)
            return

    def on_agent_done(self, *, has_reply: bool, req_id: Optional[str]) -> None:
        """Called when the agent finishes processing a message."""
        req = str(req_id or "").strip()
        with self._lock:
            decision = reduce_engagement(
                self._state.value,
                EngagementTrigger.AGENT_DONE,
                has_reply=has_reply,
            )
            if has_reply:
                if self._state not in (
                    EngagementMode.ALERT,
                    EngagementMode.ENGAGED,
                    EngagementMode.SPEAKING,
                ):
                    return
                chosen_req = req or self._current_req_id
                chosen_stream = self._latest_playback_stream_id
                self._current_req_id = chosen_req
                self._awaiting_playback_req_id = chosen_req
                self._awaiting_playback_stream_id = chosen_stream
                self._awaiting_playback_terminal = True
                if decision.changed:
                    self._apply_decision_locked(decision, req_id=chosen_req)
                return
            if decision.changed:
                self._current_req_id = req
                self._clear_playback_tracking_locked()
                self._apply_decision_locked(decision, req_id=req)

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
                    EngagementMode.ALERT,
                    EngagementMode.ENGAGED,
                ):
                    self._current_req_id = req
                    self._set_state_locked(
                        EngagementMode.SPEAKING,
                        reason="playback_started",
                        req_id=req,
                    )
                return
            if event not in ("playback_completed", "playback_stopped"):
                return
            if not self._awaiting_playback_terminal:
                if self._state == EngagementMode.SPEAKING:
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
                EngagementMode.SPEAKING,
                EngagementMode.ENGAGED,
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
            decision = reduce_engagement(
                self._state.value,
                EngagementTrigger.PLAYBACK_TERMINAL,
            )
            self._apply_decision_locked(decision, req_id=req)

    def shutdown(self) -> None:
        self._watchdog_stop.set()
        self._watchdog.join(timeout=2.0)

    def _expires_at_for(self, state: EngagementMode, entered_at: float) -> Optional[float]:
        if state == EngagementMode.ALERT:
            return entered_at + self._alert_timeout
        if state == EngagementMode.COOLDOWN:
            return entered_at + self._cooldown_timeout
        return None

    def _set_state_locked(
        self,
        new_state: EngagementMode,
        *,
        reason: str,
        req_id: str,
    ) -> None:
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        self._last_transition = time.time()
        if new_state == EngagementMode.IDLE:
            self._current_req_id = ""
            self._clear_playback_tracking_locked()
        logger.info(
            "Engagement: %s -> %s reason=%s req_id=%s",
            old.value,
            new_state.value,
            reason,
            req_id,
        )
        safe_transition(
            self._state_observer,
            StateTransition(
                axis=StateAxis.ENGAGEMENT,
                old_state=old.value,
                new_state=new_state.value,
                trigger=reason,
                req_id=req_id,
                reason=reason,
            ),
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
                    self._state == EngagementMode.ALERT
                    and elapsed >= self._alert_timeout
                ):
                    decision = reduce_engagement(
                        self._state.value,
                        EngagementTrigger.ALERT_TIMEOUT,
                    )
                    self._apply_decision_locked(decision, req_id="")
                    action = "alert_timeout"
                elif (
                    self._state == EngagementMode.COOLDOWN
                    and elapsed >= self._cooldown_timeout
                ):
                    decision = reduce_engagement(
                        self._state.value,
                        EngagementTrigger.COOLDOWN_TIMEOUT,
                    )
                    self._apply_decision_locked(decision, req_id="")
                    action = "cooldown_timeout"
                elif (
                    (
                        (
                            self._awaiting_playback_terminal
                            and self._state
                            in (
                                EngagementMode.ENGAGED,
                                EngagementMode.SPEAKING,
                            )
                        )
                        or self._state == EngagementMode.SPEAKING
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
                    decision = reduce_engagement(
                        self._state.value,
                        EngagementTrigger.PLAYBACK_FALLBACK,
                    )
                    self._apply_decision_locked(decision, req_id=req)

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
        if self.state != EngagementMode.IDLE:
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
