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

from .control.robot_arbitration import decide_proactive_face_attention
from .control.types import EngagementMode

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
        self._previous_unknown_greet_ready = False
        self._unknown_greet_stability_frames = (
            self._resolve_unknown_greet_stability_frames(face_service)
        )

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
        self,
        text: str,
        face_status: str,
        name: str = "",
        person_id: str = "",
    ) -> None:
        nav_active = self._nav_state.get_active_goal() is not None
        self._coalescer.submit(
            text=text,
            metadata={
                "internal": True,
                "internal_event": "face",
                "source": "face_recognition",
                "face_status": face_status,
                "person_id": person_id,
                "person_name": name,
                "nav_active": nav_active,
            },
        )
        self._engagement.on_face_or_wake()

    @staticmethod
    def _resolve_unknown_greet_stability_frames(
        face_service: FaceRecognitionService,
    ) -> int:
        stability = getattr(face_service, "_recognition_stability", None)
        settings = getattr(stability, "settings", None)
        try:
            return max(1, int(getattr(settings, "window_frames", 1) or 1))
        except (TypeError, ValueError):
            return 1

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
            unknown_stability_frames = int(
                snapshot.get("attentive_unknown_stability_frames", 0) or 0
            )
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
            unknown_stability_frames = int(
                snapshot.get("unknown_stability_frames", 0) or 0
            )
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
        face_attention_decision = decide_proactive_face_attention(
            engagement_state=self._engagement.state,
            nav_state=self._nav_state,
            recording_active=recording_active,
        )
        self._last_face_arbitration_decision = face_attention_decision
        allow_face_attention = face_attention_decision.allowed
        required_unknown_stability_frames = max(
            1,
            int(getattr(self, "_unknown_greet_stability_frames", 1) or 1),
        )
        if unknown_count > 0 and unknown_stability_frames <= 0:
            unknown_stability_frames = required_unknown_stability_frames
        unknown_greet_ready = (
            unknown_count > 0
            and unknown_stability_frames >= required_unknown_stability_frames
        )
        previous_unknown_greet_ready = bool(
            getattr(self, "_previous_unknown_greet_ready", False)
        )

        emitted_mixed = False
        if (
            has_mixed_scene
            and allow_face_attention
            and self._recognized_greet_enabled
            and self._unknown_greet_enabled
            and unknown_greet_ready
            and ((not previous_unknown_greet_ready) or bool(new_ids))
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
            and unknown_greet_ready
            and not previous_unknown_greet_ready
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
                    person_id=p.person_id,
                )

        self._previous_ids = ids_now
        self._previous_unknown_count = unknown_count
        self._previous_unknown_greet_ready = unknown_greet_ready

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
        robot_client=None,
        nav_state: NavigationState,
        battery_cache=None,
        next_hop_delay_sec: float = PATROL_NEXT_HOP_DELAY_SEC,
    ):
        self._robot_client = robot_client
        self._nav_state = nav_state
        self._battery_cache = battery_cache
        self._next_hop_delay_sec = float(next_hop_delay_sec)

    def _start_patrol_navigation(self, target: str) -> bool:
        rendered_target = str(target or "").strip()
        if not rendered_target or self._robot_client is None:
            return False
        try:
            from argos_src.tools.unitree_go2.navigation.toolset import (
                start_navigation_to_saved_location,
            )

            result = start_navigation_to_saved_location(
                robot_client=self._robot_client,
                state=self._nav_state,
                location_name=rendered_target,
                battery=self._battery_cache,
                tool_name="patrol_navigation",
            )
        except Exception:
            return False
        return bool(result.get("success", False))

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
            if self._nav_state.get_active_goal() is not None:
                return
            if str(patrol_state.get("awaiting_target", "")).strip() != next_target:
                return
            self._start_patrol_navigation(next_target)

        threading.Thread(
            target=_emit_next_patrol_hop_after_delay,
            daemon=True,
        ).start()
