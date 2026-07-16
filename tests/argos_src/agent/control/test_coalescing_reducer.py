from argos_src.agent.control.reducers.coalescing import (
    dedup_events,
    render_coalesced_text,
    render_internal_audio_turn_events,
)


def test_dedup_events_keeps_latest_face_and_goal_result() -> None:
    events = [
        ("FACE_EVENT: Sam appeared.", {"internal": True, "internal_event": "face", "person_name": "Sam"}),
        ("FACE_EVENT: Sam attentive.", {"internal": True, "internal_event": "face", "person_name": "Sam"}),
        ("NAV_EVENT: waypoint.", {"internal": True, "internal_event": "navigation", "event_type": "waypoint"}),
        ("NAV_EVENT: goal.", {"internal": True, "internal_event": "navigation", "event_type": "goal_result"}),
    ]

    result = dedup_events(events)

    assert [text for text, _meta in result] == [
        "FACE_EVENT: Sam attentive.",
        "NAV_EVENT: goal.",
    ]


def test_render_coalesced_text_preserves_model_visible_headers() -> None:
    text, metadata = render_coalesced_text(
        [
            ("BATTERY_EVENT: low.", {"internal": True, "internal_event": "battery"}),
            ("hello", {"req_id": "rt-1"}),
        ]
    )

    assert text == "[PENDING EVENTS]\n- BATTERY_EVENT: low.\n[HUMAN INPUT]\nhello"
    assert metadata == {"req_id": "rt-1"}


def test_render_internal_audio_turn_events_merges_latest_internal_metadata() -> None:
    text, metadata = render_internal_audio_turn_events(
        [
            ("BATTERY_EVENT: low.", {"internal": True, "internal_event": "battery"}),
            (
                "NAV_EVENT: goal.",
                {"internal": True, "internal_event": "navigation", "goal_id": "nav-1"},
            ),
        ],
        {"req_id": "rt-1"},
    )

    assert text == "[PENDING EVENTS]\n- BATTERY_EVENT: low.\n- NAV_EVENT: goal."
    assert metadata["internal_event"] == "navigation"
    assert metadata["goal_id"] == "nav-1"
    assert metadata["req_id"] == "rt-1"
