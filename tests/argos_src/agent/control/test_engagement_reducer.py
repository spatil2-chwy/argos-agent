from argos_src.agent.control.reducers.engagement import (
    EngagementTrigger,
    decision_has_action,
    reduce_engagement,
)


def test_face_or_wake_claims_idle_attention_and_requests_nav_stop() -> None:
    decision = reduce_engagement("idle", EngagementTrigger.FACE_OR_WAKE)

    assert decision.new_state == "alert"
    assert decision.reason == "face_detected"
    assert decision_has_action(decision, "publish_voice_command")
    assert decision_has_action(decision, "cancel_active_navigation")


def test_human_input_from_alert_does_not_publish_redundant_stop() -> None:
    decision = reduce_engagement("alert", EngagementTrigger.HUMAN_INPUT)

    assert decision.new_state == "engaged"
    assert decision.actions == ()


def test_human_input_from_cooldown_requests_stop_and_navigation_cancel() -> None:
    decision = reduce_engagement("cooldown", EngagementTrigger.HUMAN_INPUT)

    assert decision.new_state == "engaged"
    assert decision_has_action(decision, "publish_voice_command")
    assert decision_has_action(decision, "cancel_active_navigation")


def test_agent_done_with_reply_arms_playback_without_leaving_speaking() -> None:
    decision = reduce_engagement(
        "speaking",
        EngagementTrigger.AGENT_DONE,
        has_reply=True,
    )

    assert decision.new_state == "speaking"
    assert decision_has_action(decision, "await_playback_terminal")


def test_alert_timeout_returns_idle_and_requests_flush_and_idle_callback() -> None:
    decision = reduce_engagement("alert", EngagementTrigger.ALERT_TIMEOUT)

    assert decision.new_state == "idle"
    assert decision_has_action(decision, "force_flush_coalescer")
    assert decision_has_action(decision, "notify_idle_entered")
