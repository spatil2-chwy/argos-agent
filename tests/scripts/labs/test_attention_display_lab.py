from __future__ import annotations

from scripts.labs.attention_display_lab import (
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
