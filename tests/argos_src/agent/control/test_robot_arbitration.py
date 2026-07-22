from __future__ import annotations

from types import SimpleNamespace

from argos_src.agent.control.robot_arbitration import (
    decide_idle_patrol_resume,
    decide_proactive_face_attention,
)


class _NavState:
    def __init__(self) -> None:
        self.patrol = {"enabled": True, "awaiting_target": "kitchen"}
        self.active_goal = None
        self.dock_alignment_active = False
        self.face_attention_allowed = True

    def get_patrol(self):
        return dict(self.patrol)

    def get_active_goal(self):
        return self.active_goal

    def has_active_dock_alignment(self):
        return self.dock_alignment_active

    def allows_proactive_face_attention(self):
        return self.face_attention_allowed


def test_idle_patrol_resume_policy_prioritizes_battery_block() -> None:
    decision = decide_idle_patrol_resume(
        nav_state=_NavState(),
        coalescer=object(),
        battery_cache=SimpleNamespace(should_block_general_navigation=lambda: True),
    )

    assert decision.allowed is False
    assert decision.state == "battery_low_blocking"
    assert decision.reason == "battery_blocks_navigation"


def test_idle_patrol_resume_policy_allows_targeted_patrol() -> None:
    decision = decide_idle_patrol_resume(
        nav_state=_NavState(),
        coalescer=object(),
        battery_cache=SimpleNamespace(should_block_general_navigation=lambda: False),
    )

    assert decision.allowed is True
    assert decision.state == "patrol_allowed"
    assert decision.fields == {"target_label": "kitchen"}


def test_idle_patrol_resume_policy_blocks_active_dock_alignment() -> None:
    nav_state = _NavState()
    nav_state.dock_alignment_active = True

    decision = decide_idle_patrol_resume(
        nav_state=nav_state,
        coalescer=object(),
        battery_cache=SimpleNamespace(should_block_general_navigation=lambda: False),
    )

    assert decision.allowed is False
    assert decision.state == "patrol_suppressed"
    assert decision.reason == "active_dock_alignment"


def test_proactive_face_attention_policy_blocks_recording_before_greeting() -> None:
    decision = decide_proactive_face_attention(
        engagement_state="idle",
        nav_state=_NavState(),
        recording_active=True,
    )

    assert decision.allowed is False
    assert decision.state == "face_attention_suppressed"
    assert decision.reason == "recording_active"


def test_proactive_face_attention_policy_allows_idle_interruptible_scene() -> None:
    decision = decide_proactive_face_attention(
        engagement_state="idle",
        nav_state=_NavState(),
        recording_active=False,
    )

    assert decision.allowed is True
    assert decision.state == "face_attention_allowed"


def test_proactive_face_attention_policy_blocks_active_human_turn() -> None:
    decision = decide_proactive_face_attention(
        engagement_state="idle",
        nav_state=_NavState(),
        recording_active=False,
        human_turn_active=True,
    )

    assert decision.allowed is False
    assert decision.reason == "human_turn_active"
