"""Face-event and patrol bridges for the Argos realtime agent.

FaceEventBridge: polls face recognition, publishes presence snapshots,
emits FACE_EVENTs through the coalescer with per-person greeting cooldowns.

PatrolLoopBridge: schedules the next patrol hop after a successful nav
event (with a configurable delay).  Patrol-resume after idle is handled
by the engagement state machine's COOLDOWN→IDLE callback, not here.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Optional

from argos_src.nav_support.locations import NavigationState

from .orchestrator import EngagementState

if TYPE_CHECKING:
    from argos_src.face_recognition.face_recognition_service import FaceRecognitionService

FACE_EVENT_POLL_SEC = 0.5
RECOGNIZED_GREET_COOLDOWN_SEC = 45.0
UNKNOWN_GREET_COOLDOWN_SEC = 30.0
PATROL_NEXT_HOP_DELAY_SEC = 5.0


class FaceEventBridge:
    """Publishes face presence snapshots and emits proactive face events.

    Per-person greeting cooldowns (_recognized_last_greet_s at 45 s,
    _last_unknown_greet_s at 30 s) remain here — they decide *whether* a
    sighting becomes an event.  Proactive FACE_EVENTs are only enqueued
    when engagement is IDLE (not ALERT / ENGAGED / COOLDOWN), so vision
    does not start a second greet right after a turn. Higher-level flow
    like patrol suppression and playback interruption is handled by the
    engagement state machine.
    """

    def __init__(
        self,
        *,
        face_service: FaceRecognitionService,
        robot_client,
        coalescer,
        engagement,
        nav_state: NavigationState,
        presence_callback=None,
        recognized_greet_enabled: bool = True,
        unknown_greet_enabled: bool = True,
        require_attention: bool = False,
        recognized_greet_cooldown_sec: float = RECOGNIZED_GREET_COOLDOWN_SEC,
        unknown_greet_cooldown_sec: float = UNKNOWN_GREET_COOLDOWN_SEC,
    ):
        self._face_service = face_service
        self._robot_client = robot_client
        self._coalescer = coalescer
        self._engagement = engagement
        self._nav_state = nav_state
        self._presence_callback = presence_callback
        self._recognized_greet_enabled = recognized_greet_enabled
        self._unknown_greet_enabled = unknown_greet_enabled
        self._require_attention = bool(require_attention)
        self._recognized_greet_cooldown_sec = float(recognized_greet_cooldown_sec)
        self._unknown_greet_cooldown_sec = float(unknown_greet_cooldown_sec)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_unknown_greet_s = 0.0
        self._recognized_last_greet_s: dict[str, float] = {}
        self._previous_status = "none"
        self._previous_ids: set[str] = set()
        self._previous_unknown_count = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- internals -----------------------------------------------------------

    def _publish_presence(self, snapshot: dict) -> None:
        callback = getattr(self, "_presence_callback", None)
        if callable(callback):
            try:
                callback(snapshot)
            except Exception:
                pass
        try:
            publisher = getattr(self._robot_client, "publish_face_presence", None)
            if callable(publisher):
                publisher(snapshot)
        except Exception:
            return

    def _enqueue_face_event(
        self, text: str, face_status: str, name: str = ""
    ) -> None:
        nav_active = self._nav_state.get_active_goal() is not None
        self._coalescer.submit(
            text=text,
            metadata={
                "internal": True,
                "internal_event": "face",
                "source": "face_recognition",
                "face_status": face_status,
                "person_name": name,
                "nav_active": nav_active,
            },
        )
        self._engagement.on_face_or_wake()

    def _maybe_enqueue_face_events(
        self,
        *,
        snapshot: dict,
        persons: list,
        now: float,
    ) -> None:
        require_attention = bool(getattr(self, "_require_attention", False))
        if require_attention:
            unknown_count = int(snapshot.get("attentive_unknown_count", 0) or 0)
            has_mixed_scene = bool(snapshot.get("has_attentive_mixed_scene", False))
            nearest_recognized_name = str(
                snapshot.get("primary_attention_name")
                or snapshot.get("nearest_recognized_name", "")
                or ""
            ).strip()
            event_persons = [
                p for p in persons if bool(getattr(p, "attentive", False))
            ]
        else:
            unknown_count = int(snapshot.get("unknown_count", 0) or 0)
            has_mixed_scene = bool(snapshot.get("has_mixed_scene", False))
            nearest_recognized_name = str(snapshot.get("nearest_recognized_name", "") or "").strip()
            event_persons = list(persons)
        ids_now = {p.person_id for p in event_persons}
        new_ids = ids_now - self._previous_ids
        is_recording_active = getattr(self._engagement, "is_recording_active", None)
        recording_active = False
        if callable(is_recording_active):
            try:
                recording_active = bool(is_recording_active())
            except Exception:
                recording_active = False
        allow_face_attention = (
            self._engagement.state == EngagementState.IDLE
            and self._nav_state.allows_proactive_face_attention()
            and not recording_active
        )

        emitted_mixed = False
        if (
            has_mixed_scene
            and allow_face_attention
            and self._recognized_greet_enabled
            and self._unknown_greet_enabled
            and (unknown_count > 0)
            and ((self._previous_unknown_count == 0) or bool(new_ids))
            and (
                now - self._last_unknown_greet_s
                >= self._unknown_greet_cooldown_sec
            )
        ):
            recognized_ids_for_mixed = new_ids or ids_now
            recognized_ready = True
            if recognized_ids_for_mixed:
                recognized_ready = any(
                    (now - self._recognized_last_greet_s.get(pid, 0.0))
                    >= self._recognized_greet_cooldown_sec
                    for pid in recognized_ids_for_mixed
                )
            if recognized_ready:
                self._last_unknown_greet_s = now
                for pid in recognized_ids_for_mixed:
                    self._recognized_last_greet_s[pid] = now
                if nearest_recognized_name:
                    self._enqueue_face_event(
                        f"FACE_EVENT: recognized person '{nearest_recognized_name}' is in front of you, and there is also an unrecognized person nearby.",
                        face_status="mixed",
                        name=nearest_recognized_name,
                    )
                else:
                    self._enqueue_face_event(
                        "FACE_EVENT: there is a recognized person in front of you, and there is also an unrecognized person nearby.",
                        face_status="mixed",
                    )
                emitted_mixed = True

        if (
            not emitted_mixed
            and self._unknown_greet_enabled
            and unknown_count > 0
            and self._previous_unknown_count == 0
            and allow_face_attention
        ):
            if (
                now - self._last_unknown_greet_s
                >= self._unknown_greet_cooldown_sec
            ):
                self._last_unknown_greet_s = now
                self._enqueue_face_event(
                    "FACE_EVENT: an unrecognized person is in front of you.",
                    face_status="unknown",
                )

        if self._recognized_greet_enabled and not emitted_mixed:
            allow_recognized_greet = bool(new_ids) and allow_face_attention
            for p in event_persons:
                if p.person_id not in new_ids:
                    continue
                if not allow_recognized_greet:
                    continue
                last = self._recognized_last_greet_s.get(p.person_id, 0.0)
                if now - last < self._recognized_greet_cooldown_sec:
                    continue
                self._recognized_last_greet_s[p.person_id] = now
                self._enqueue_face_event(
                    f"FACE_EVENT: recognized person '{p.name}' appeared "
                    "in front of you.",
                    face_status="recognized",
                    name=p.name,
                )

        self._previous_ids = ids_now
        self._previous_unknown_count = unknown_count

    def _run(self) -> None:
        while not self._stop.wait(FACE_EVENT_POLL_SEC):
            snapshot = self._face_service.get_presence_snapshot()
            self._publish_presence(snapshot)

            now = time.time()
            persons = self._face_service.get_cached_persons()
            self._maybe_enqueue_face_events(
                snapshot=snapshot,
                persons=persons,
                now=now,
            )


class PatrolLoopBridge:
    """Schedules the next patrol hop after a successful navigation event.

    The idle-resume watchdog that existed in the old _PatrolLoopBridge is
    gone.  Patrol resume after interaction is now handled by the
    engagement state machine's COOLDOWN→IDLE ``on_idle_entered`` callback.
    """

    def __init__(
        self,
        *,
        coalescer,
        nav_state: NavigationState,
        next_hop_delay_sec: float = PATROL_NEXT_HOP_DELAY_SEC,
    ):
        self._coalescer = coalescer
        self._nav_state = nav_state
        self._next_hop_delay_sec = float(next_hop_delay_sec)

    def on_nav_event(self, event: dict) -> None:
        if event.get("event_type") != "goal_result":
            return
        if event.get("outcome") != "succeeded":
            return

        patrol = self._nav_state.get_patrol()
        if not patrol.get("enabled", False):
            return

        arrived_target = str(event.get("target_label", "")).strip()
        if not arrived_target:
            return

        next_target = self._nav_state.patrol_mark_arrived_and_get_next(
            arrived_target
        )
        if not next_target:
            return

        def _emit_next_patrol_hop_after_delay() -> None:
            time.sleep(self._next_hop_delay_sec)
            patrol_state = self._nav_state.get_patrol()
            if not patrol_state.get("enabled", False):
                return
            self._coalescer.submit(
                text=(
                    "PATROL_EVENT: patrol loop is active. "
                    "Continue patrol by calling "
                    f"navigate_to_location(location_name='{next_target}')."
                ),
                metadata={
                    "internal": True,
                    "internal_event": "patrol_continue",
                    "source": "navigation",
                    "target_label": next_target,
                },
            )

        threading.Thread(
            target=_emit_next_patrol_hop_after_delay,
            daemon=True,
        ).start()
