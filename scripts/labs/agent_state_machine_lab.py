"""Synthetic admission, engagement, and coalescer lab for the realtime agent."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

from argos_src.agent.control.coalescer import EventCoalescer
from argos_src.agent.control.engagement_runtime import EngagementStateMachine
from argos_src.profile_config import load_scenario_profile
from argos_src.runtime.audio_admission import resolve_record_admission


@dataclass(frozen=True)
class AdmissionCase:
    name: str
    face_present: bool
    attention_present: bool
    interaction_state: str
    wake_detected: bool
    wake_window_until_s: float = 0.0
    nav_active: bool = False
    nav_interruptible: bool = True
    nav_passive_listen_allowed: bool = True


def admission_cases() -> tuple[AdmissionCase, ...]:
    return (
        AdmissionCase(
            name="idle_blocked",
            face_present=False,
            attention_present=False,
            interaction_state="idle",
            wake_detected=False,
        ),
        AdmissionCase(
            name="wake_from_idle",
            face_present=False,
            attention_present=False,
            interaction_state="idle",
            wake_detected=True,
        ),
        AdmissionCase(
            name="alert_followup",
            face_present=False,
            attention_present=False,
            interaction_state="alert",
            wake_detected=False,
        ),
        AdmissionCase(
            name="cooldown_followup",
            face_present=False,
            attention_present=False,
            interaction_state="cooldown",
            wake_detected=False,
        ),
        AdmissionCase(
            name="attention_present",
            face_present=True,
            attention_present=True,
            interaction_state="idle",
            wake_detected=False,
        ),
        AdmissionCase(
            name="speaking_blocks",
            face_present=True,
            attention_present=True,
            interaction_state="speaking",
            wake_detected=False,
        ),
        AdmissionCase(
            name="focused_nav_blocks_passive",
            face_present=True,
            attention_present=True,
            interaction_state="idle",
            wake_detected=False,
            nav_active=True,
            nav_interruptible=False,
            nav_passive_listen_allowed=False,
        ),
        AdmissionCase(
            name="focused_nav_allows_wake",
            face_present=False,
            attention_present=False,
            interaction_state="idle",
            wake_detected=True,
            nav_active=True,
            nav_interruptible=False,
            nav_passive_listen_allowed=False,
        ),
    )


def run_admission_sweep(profile_name: str) -> list[dict[str, Any]]:
    profile = load_scenario_profile(profile_name)
    rows: list[dict[str, Any]] = []
    now_s = 100.0
    for case in admission_cases():
        allowed, reason, wake_until = resolve_record_admission(
            face_present=case.face_present,
            attention_present=case.attention_present,
            interaction_state=case.interaction_state,
            now_s=now_s,
            wake_window_until_s=case.wake_window_until_s,
            wake_detected=case.wake_detected,
            wake_window_sec=profile.realtime.wake_window_sec,
            block_during_speaking=profile.realtime.admission.block_during_speaking,
            block_during_engaged=profile.realtime.admission.block_during_engaged,
            open_on_face_presence=profile.realtime.admission.open_on_face_presence,
            open_on_attention_presence=profile.realtime.admission.open_on_attention_presence,
            open_on_interaction_states=profile.realtime.admission.open_on_interaction_states,
            open_on_wake_window=profile.realtime.admission.open_on_wake_window,
            nav_active=case.nav_active,
            nav_interruptible=case.nav_interruptible,
            nav_passive_listen_allowed=case.nav_passive_listen_allowed,
        )
        rows.append(
            {
                **asdict(case),
                "allowed": bool(allowed),
                "reason": reason,
                "wake_window_until_after_s": round(float(wake_until), 3),
            }
        )
    return rows


def run_engagement_sequence() -> dict[str, Any]:
    voice_cmds: list[str] = []
    machine = EngagementStateMachine(
        voice_cmd_publisher=voice_cmds.append,
        alert_timeout_sec=15.0,
        cooldown_sec=7.0,
        speaking_timeout_sec=30.0,
    )
    states = [{"step": "initial", "state": machine.state_name}]
    try:
        machine.on_face_or_wake()
        states.append({"step": "face_or_wake", "state": machine.state_name})
        machine.on_human_input("lab-1")
        states.append({"step": "human_input", "state": machine.state_name})
        machine.on_agent_output_started("lab-1", stream_id="stream-1")
        states.append({"step": "agent_output_started", "state": machine.state_name})
        machine.on_agent_done(has_reply=True, req_id="lab-1")
        states.append({"step": "agent_done_with_reply", "state": machine.state_name})
        machine.on_playback_event(
            "playback_completed",
            "lab-1",
            stream_id="stream-1",
        )
        states.append({"step": "playback_completed", "state": machine.state_name})
        return {"states": states, "voice_cmds": voice_cmds}
    finally:
        machine.shutdown()


class _FakeAgent:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, dict[str, Any]]] = []

    def enqueue_internal_event(self, text: str, metadata: dict[str, Any]) -> None:
        self.enqueued.append((text, metadata))


def run_coalescer_sequence() -> dict[str, Any]:
    machine = EngagementStateMachine()
    agent = _FakeAgent()
    coalescer = EventCoalescer(
        agent=agent,
        engagement=machine,
        debounce_sec=60.0,
        max_wait_sec=60.0,
    )
    try:
        coalescer.submit(
            "FACE_EVENT: Sam is visible.",
            {"internal": True, "internal_event": "face", "person_name": "Sam"},
        )
        coalescer.submit(
            "FACE_EVENT: Sam is attentive.",
            {"internal": True, "internal_event": "face", "person_name": "Sam"},
        )
        coalescer.submit(
            "NAV_EVENT: Reached waypoint.",
            {"internal": True, "internal_event": "navigation", "event_type": "waypoint"},
        )
        coalescer.submit(
            "NAV_EVENT: Goal reached.",
            {
                "internal": True,
                "internal_event": "navigation",
                "event_type": "goal_result",
            },
        )
        text, metadata = coalescer.drain_internal_events_for_audio_turn(
            {"req_id": "lab-audio"}
        )
        return {
            "drained_text": text or "",
            "metadata": metadata,
            "enqueued_count": len(agent.enqueued),
        }
    finally:
        with coalescer._lock:
            coalescer._cancel_timer_locked()
        machine.shutdown()


def build_report(profile_name: str) -> dict[str, Any]:
    profile = load_scenario_profile(profile_name)
    return {
        "profile": profile.name,
        "source_path": str(profile.source_path),
        "admission_profile": {
            "block_during_speaking": profile.realtime.admission.block_during_speaking,
            "block_during_engaged": profile.realtime.admission.block_during_engaged,
            "open_on_face_presence": profile.realtime.admission.open_on_face_presence,
            "open_on_attention_presence": profile.realtime.admission.open_on_attention_presence,
            "open_on_interaction_states": list(
                profile.realtime.admission.open_on_interaction_states
            ),
        },
        "admission": run_admission_sweep(profile_name),
        "engagement": run_engagement_sequence(),
        "coalescer": run_coalescer_sequence(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="static_interaction")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report(args.profile)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
