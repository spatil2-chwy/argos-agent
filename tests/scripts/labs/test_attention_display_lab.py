from __future__ import annotations

from scripts.labs.attention_display_lab import (
    RecognitionNameWindow,
    _attention_display_state,
)


def test_attention_display_state_formats_attentive_named_person() -> None:
    state = _attention_display_state(
        {
            "attention_status": "attentive",
            "faces_detected": 1,
            "attention_count": 1,
            "primary_attention_name": "Sam",
            "recognized_names": ["Sam"],
            "attention_confidence": 0.876,
        }
    )

    assert state.text == "Detected | Attentive\nrecognized: Sam"
    assert state.signature == ("Detected | Attentive", ("Sam",))


def test_attention_display_state_formats_inattentive_recognized_faces() -> None:
    state = _attention_display_state(
        {
            "attention_status": "inattentive",
            "faces_detected": 2,
            "attention_count": 0,
            "recognized_names": ["Sam", "Alex"],
        }
    )

    assert state.text == "Detected | Non-Attentive\nrecognized: Sam, Alex"


def test_attention_display_state_formats_no_face() -> None:
    state = _attention_display_state(
        {
            "attention_status": "none",
            "faces_detected": 0,
            "attention_count": 0,
        }
    )

    assert state.text == "Not Detected"


def test_recognition_name_window_promotes_after_min_matches() -> None:
    window = RecognitionNameWindow(window_frames=5, min_matches=2)

    first = window.names_for_snapshot(
        {
            "updated_at": 1.0,
            "faces_detected": 1,
            "primary_face_name": "Sam",
        }
    )
    second = window.names_for_snapshot(
        {
            "updated_at": 2.0,
            "faces_detected": 1,
            "recognized_names": [],
        }
    )
    third = window.names_for_snapshot(
        {
            "updated_at": 3.0,
            "faces_detected": 1,
            "recognized_names": ["Sam"],
        }
    )
    fourth = window.names_for_snapshot(
        {
            "updated_at": 4.0,
            "faces_detected": 1,
            "recognized_names": [],
        }
    )

    assert first == []
    assert second == []
    assert third == ["Sam"]
    assert fourth == ["Sam"]


def test_recognition_name_window_decays_when_matches_leave_window() -> None:
    window = RecognitionNameWindow(window_frames=5, min_matches=2)
    snapshots = [
        {"updated_at": 1.0, "faces_detected": 1, "recognized_names": ["Sam"]},
        {"updated_at": 2.0, "faces_detected": 1, "recognized_names": ["Sam"]},
        {"updated_at": 3.0, "faces_detected": 1, "recognized_names": []},
        {"updated_at": 4.0, "faces_detected": 1, "recognized_names": []},
        {"updated_at": 5.0, "faces_detected": 1, "recognized_names": []},
        {"updated_at": 6.0, "faces_detected": 1, "recognized_names": []},
    ]

    results = [window.names_for_snapshot(snapshot) for snapshot in snapshots]

    assert results[1] == ["Sam"]
    assert results[-1] == []


def test_recognition_name_window_does_not_double_count_same_snapshot() -> None:
    window = RecognitionNameWindow(window_frames=5, min_matches=2)
    snapshot = {
        "updated_at": 1.0,
        "faces_detected": 1,
        "recognized_names": ["Sam"],
    }

    assert window.names_for_snapshot(snapshot) == []
    assert window.names_for_snapshot(snapshot) == []
