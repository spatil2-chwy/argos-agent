from __future__ import annotations

from argos_src.agent.control.observers import NullStateObserver, safe_ignored, safe_transition
from argos_src.agent.control.types import StateAxis, StateTransition


class _FailingObserver:
    def transition(self, transition: StateTransition) -> None:
        raise RuntimeError("boom")

    def ignored(self, *, axis: StateAxis | str, trigger: str, reason: str, **fields):
        raise RuntimeError("boom")


def test_safe_transition_ignores_missing_or_failing_observers() -> None:
    transition = StateTransition(
        axis=StateAxis.TURN,
        old_state="committed",
        new_state="response_requested",
        trigger="test",
    )

    safe_transition(None, transition)
    safe_transition(NullStateObserver(), transition)
    safe_transition(_FailingObserver(), transition)


def test_safe_ignored_ignores_missing_or_failing_observers() -> None:
    safe_ignored(None, axis=StateAxis.COALESCER, trigger="test", reason="not_ready")
    safe_ignored(
        NullStateObserver(),
        axis=StateAxis.COALESCER,
        trigger="test",
        reason="not_ready",
    )
    safe_ignored(
        _FailingObserver(),
        axis=StateAxis.COALESCER,
        trigger="test",
        reason="not_ready",
    )
