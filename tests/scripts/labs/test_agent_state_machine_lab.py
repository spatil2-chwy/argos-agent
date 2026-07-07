from scripts.labs.agent_state_machine_lab import build_report


def test_state_machine_lab_uses_static_profile_admission_policy() -> None:
    report = build_report("static_interaction")

    cases = {row["name"]: row for row in report["admission"]}

    assert cases["idle_blocked"]["allowed"] is False
    assert cases["wake_from_idle"]["allowed"] is True
    assert cases["wake_from_idle"]["reason"] == "wake_word"
    assert cases["alert_followup"]["allowed"] is True
    assert cases["cooldown_followup"]["allowed"] is False
    assert cases["cooldown_followup"]["reason"] == "blocked"
    assert cases["focused_nav_blocks_passive"]["allowed"] is False
    assert cases["focused_nav_blocks_passive"]["reason"] == "focused_navigation"
    assert cases["focused_nav_allows_wake"]["allowed"] is True


def test_state_machine_lab_reports_engagement_and_coalescer_sequences() -> None:
    report = build_report("static_interaction")

    engagement_states = [
        item["state"] for item in report["engagement"]["states"]
    ]
    assert engagement_states == [
        "idle",
        "alert",
        "engaged",
        "speaking",
        "speaking",
        "cooldown",
    ]

    drained = report["coalescer"]["drained_text"]
    assert "Sam is attentive" in drained
    assert "Sam is visible" not in drained
    assert "Goal reached" in drained
    assert "Reached waypoint" not in drained
    assert report["coalescer"]["metadata"]["req_id"] == "lab-audio"
