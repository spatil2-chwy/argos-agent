from __future__ import annotations

import time

from argos_src.runtime.audio_admission import FacePresenceGate, resolve_record_admission


def test_face_presence_gate_tracks_attention_status():
    gate = FacePresenceGate(stale_after_sec=5.0)

    gate.update_from_snapshot(
        {
            "status": "recognized",
            "attention_status": "attentive",
            "expires_at": time.time() + 5.0,
        },
        now_s=100.0,
    )

    assert gate.is_face_present() is True
    assert gate.is_attention_present() is True


def test_record_admission_can_require_attention_instead_of_face_presence():
    allowed, reason, _ = resolve_record_admission(
        face_present=True,
        attention_present=False,
        interaction_state="idle",
        now_s=100.0,
        wake_window_until_s=0.0,
        wake_detected=False,
        wake_window_sec=5.0,
        open_on_face_presence=False,
        open_on_attention_presence=True,
    )

    assert allowed is False
    assert reason == "blocked"

    allowed, reason, _ = resolve_record_admission(
        face_present=True,
        attention_present=True,
        interaction_state="idle",
        now_s=100.0,
        wake_window_until_s=0.0,
        wake_detected=False,
        wake_window_sec=5.0,
        open_on_face_presence=False,
        open_on_attention_presence=True,
    )

    assert allowed is True
    assert reason == "attention_present"


def test_record_admission_can_block_during_engaged_even_with_attention():
    allowed, reason, _ = resolve_record_admission(
        face_present=False,
        attention_present=True,
        interaction_state="engaged",
        now_s=100.0,
        wake_window_until_s=0.0,
        wake_detected=True,
        wake_window_sec=5.0,
        block_during_engaged=True,
        open_on_attention_presence=True,
    )

    assert allowed is False
    assert reason == "engaged"


def test_record_admission_alert_opens_by_interaction_state():
    allowed, reason, _ = resolve_record_admission(
        face_present=False,
        attention_present=False,
        interaction_state="alert",
        now_s=100.0,
        wake_window_until_s=0.0,
        wake_detected=False,
        wake_window_sec=5.0,
        open_on_face_presence=False,
        open_on_attention_presence=False,
        open_on_interaction_states=("alert",),
        open_on_wake_window=False,
    )

    assert allowed is True
    assert reason == "alert"


def test_record_admission_cooldown_needs_attention_or_wake_without_state_opener():
    allowed, reason, _ = resolve_record_admission(
        face_present=False,
        attention_present=False,
        interaction_state="cooldown",
        now_s=100.0,
        wake_window_until_s=0.0,
        wake_detected=False,
        wake_window_sec=5.0,
        open_on_face_presence=False,
        open_on_attention_presence=True,
        open_on_interaction_states=("alert",),
        open_on_wake_window=False,
    )

    assert allowed is False
    assert reason == "blocked"

    allowed, reason, _ = resolve_record_admission(
        face_present=False,
        attention_present=True,
        interaction_state="cooldown",
        now_s=100.0,
        wake_window_until_s=0.0,
        wake_detected=False,
        wake_window_sec=5.0,
        open_on_face_presence=False,
        open_on_attention_presence=True,
        open_on_interaction_states=("alert",),
        open_on_wake_window=False,
    )

    assert allowed is True
    assert reason == "attention_present"

    allowed, reason, wake_until = resolve_record_admission(
        face_present=False,
        attention_present=False,
        interaction_state="cooldown",
        now_s=100.0,
        wake_window_until_s=0.0,
        wake_detected=True,
        wake_window_sec=5.0,
        open_on_face_presence=False,
        open_on_attention_presence=True,
        open_on_interaction_states=("alert",),
        open_on_wake_window=True,
    )

    assert allowed is True
    assert reason == "wake_word"
    assert wake_until == 105.0
