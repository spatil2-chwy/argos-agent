"""Engagement state machine and internal-event coalescer tests."""

import time

from argos_src.agent.control.coalescer import EventCoalescer
from argos_src.agent.control.engagement_runtime import EngagementState, EngagementStateMachine


class _FakeGoalHandle:
    def __init__(self):
        self.cancel_calls = 0

    def cancel_goal_async(self):
        self.cancel_calls += 1


class _FakeNavState:
    def __init__(self, *, interruptible: bool = True, passive_listen_allowed: bool = True):
        self._interruptible = interruptible
        self._passive_listen_allowed = passive_listen_allowed
        self.handle = _FakeGoalHandle()
        self.take_calls = 0
        self._active_goal = None

    def active_goal_allows_auto_interrupt(self) -> bool:
        return self._interruptible

    def take_last_goal_handle(self):
        self.take_calls += 1
        return self.handle

    def get_active_goal(self):
        return self._active_goal

    def build_interaction_context(self):
        if self._active_goal is None:
            return {
                "nav_active": False,
                "nav_source": "",
                "nav_interruptible": True,
                "nav_passive_listen_allowed": True,
            }
        return {
            "nav_active": True,
            "nav_source": "human_task",
            "nav_interruptible": self._interruptible,
            "nav_passive_listen_allowed": self._passive_listen_allowed,
        }


def _make_machine(**kwargs):
    idle_calls = []
    machine = EngagementStateMachine(
        on_idle_entered=lambda: idle_calls.append("idle"),
        **kwargs,
    )
    return machine, idle_calls


def test_playback_events_drive_speaking_to_cooldown():
    machine, _ = _make_machine(
        alert_timeout_sec=1.0,
        cooldown_sec=1.0,
        speaking_timeout_sec=5.0,
    )
    try:
        machine.on_human_input("rt-1")
        machine.on_agent_output_started("rt-1", stream_id="resp-1")
        machine.on_agent_done(has_reply=True, req_id="rt-1")
        machine.on_playback_event("playback_completed", "rt-1", stream_id="resp-1")

        assert machine.state == EngagementState.COOLDOWN
    finally:
        machine.shutdown()


def test_matching_stream_id_completes_playback_without_req_id():
    machine, _ = _make_machine(
        alert_timeout_sec=1.0,
        cooldown_sec=1.0,
        speaking_timeout_sec=5.0,
    )
    try:
        machine.on_human_input("rt-1")
        machine.on_agent_output_started("rt-1", stream_id="resp-1")
        machine.on_agent_done(has_reply=True, req_id="rt-1")
        machine.on_playback_event("playback_completed", "", stream_id="resp-1")

        assert machine.state == EngagementState.COOLDOWN
    finally:
        machine.shutdown()


def test_snapshot_exposes_live_nav_context():
    nav_state = _FakeNavState(interruptible=False, passive_listen_allowed=False)
    nav_state._active_goal = {"goal_id": "nav-1"}
    machine, _ = _make_machine(nav_state=nav_state)
    try:
        machine.on_human_input("rt-1")
        snapshot = machine.snapshot()

        assert snapshot.state == "engaged"
        assert snapshot.req_id == "rt-1"
        assert snapshot.nav_active is True
        assert snapshot.nav_interruptible is False
        assert snapshot.nav_passive_listen_allowed is False
    finally:
        machine.shutdown()


def test_speaking_watchdog_falls_back_to_cooldown():
    machine, _ = _make_machine(
        alert_timeout_sec=1.0,
        cooldown_sec=1.0,
        speaking_timeout_sec=0.2,
    )
    try:
        machine.on_human_input("rt-1")
        machine.on_agent_output_started("rt-1", stream_id="resp-1")
        machine.on_agent_done(has_reply=True, req_id="rt-1")

        time.sleep(1.3)

        assert machine.state == EngagementState.COOLDOWN
    finally:
        machine.shutdown()


def test_recording_state_provider_reports_capture_activity():
    recording_active = False
    machine, _ = _make_machine()
    machine.attach_recording_state_provider(lambda: recording_active)
    try:
        assert machine.is_recording_active() is False
        recording_active = True
        assert machine.is_recording_active() is True
    finally:
        machine.shutdown()


def test_internal_event_flush_waits_until_recording_stops():
    machine, _ = _make_machine()
    recording_active = True
    machine.attach_recording_state_provider(lambda: recording_active)
    enqueued = []
    coalescer = EventCoalescer(
        agent=type(
            "_FakeAgent",
            (),
            {"enqueue_internal_event": lambda self, text, metadata: enqueued.append((text, metadata))},
        )(),
        engagement=machine,
        debounce_sec=10.0,
        max_wait_sec=10.0,
    )
    try:
        coalescer.submit(
            "BATTERY_EVENT: Battery is low.",
            {"internal": True, "internal_event": "battery_low"},
        )

        coalescer._timer_flush()
        assert enqueued == []

        recording_active = False
        coalescer._timer_flush()
        assert len(enqueued) == 1
        assert "BATTERY_EVENT" in enqueued[0][0]
    finally:
        with coalescer._lock:
            coalescer._cancel_timer_locked()
        machine.shutdown()


def test_coalescer_dedups_face_and_navigation_events_for_audio_turn():
    machine, _ = _make_machine()
    agent = type(
        "_FakeAgent",
        (),
        {"enqueue_internal_event": lambda self, text, metadata: None},
    )()
    coalescer = EventCoalescer(
        agent=agent,
        engagement=machine,
        debounce_sec=60.0,
        max_wait_sec=60.0,
    )
    try:
        coalescer.submit(
            "FACE_EVENT: Sam appeared.",
            {"internal": True, "internal_event": "face", "person_name": "Sam"},
        )
        coalescer.submit(
            "FACE_EVENT: Sam is attentive.",
            {"internal": True, "internal_event": "face", "person_name": "Sam"},
        )
        coalescer.submit(
            "NAV_EVENT: waypoint reached.",
            {"internal": True, "internal_event": "navigation", "event_type": "waypoint"},
        )
        coalescer.submit(
            "NAV_EVENT: goal reached.",
            {
                "internal": True,
                "internal_event": "navigation",
                "event_type": "goal_result",
            },
        )

        text, metadata = coalescer.drain_internal_events_for_audio_turn({"req_id": "rt-1"})

        assert text is not None
        assert "Sam is attentive" in text
        assert "Sam appeared" not in text
        assert "goal reached" in text
        assert "waypoint reached" not in text
        assert metadata["req_id"] == "rt-1"
    finally:
        with coalescer._lock:
            coalescer._cancel_timer_locked()
        machine.shutdown()


def test_coalescer_drops_patrol_when_face_is_in_same_batch():
    machine, _ = _make_machine()
    agent = type(
        "_FakeAgent",
        (),
        {"enqueue_internal_event": lambda self, text, metadata: None},
    )()
    coalescer = EventCoalescer(
        agent=agent,
        engagement=machine,
        debounce_sec=60.0,
        max_wait_sec=60.0,
    )
    try:
        coalescer.submit(
            "PATROL_EVENT: resume patrol.",
            {"internal": True, "internal_event": "patrol_continue"},
        )
        coalescer.submit(
            "FACE_EVENT: Sam appeared.",
            {"internal": True, "internal_event": "face", "person_name": "Sam"},
        )

        text, _metadata = coalescer.drain_internal_events_for_audio_turn({"req_id": "rt-1"})

        assert text is not None
        assert "FACE_EVENT" in text
        assert "PATROL_EVENT" not in text
    finally:
        with coalescer._lock:
            coalescer._cancel_timer_locked()
        machine.shutdown()
