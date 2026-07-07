"""State observer implementations backed by Argos structured logs."""

from __future__ import annotations

from typing import Any

from argos_src.agent.control.types import StateAxis, StateTransition
from argos_src.observability.observability import LatencyLogger


class StructuredStateObserver:
    """Emit state transitions into the same structured log stream as latency events."""

    def __init__(self, *, component: str = "state") -> None:
        self._logger = LatencyLogger(component)

    def transition(self, transition: StateTransition) -> None:
        self._logger.emit(
            event="transition",
            axis=_axis_value(transition.axis),
            old_state=transition.old_state,
            new_state=transition.new_state,
            trigger=transition.trigger,
            req_id=transition.req_id or None,
            stream_id=transition.stream_id or None,
            reason=transition.reason or None,
            **transition.fields,
        )

    def ignored(self, *, axis: StateAxis | str, trigger: str, reason: str, **fields: Any) -> None:
        self._logger.emit(
            event="ignored",
            axis=_axis_value(axis),
            trigger=trigger,
            ignored_reason=reason,
            **fields,
        )


def _axis_value(axis: StateAxis | str) -> str:
    return axis.value if isinstance(axis, StateAxis) else str(axis)
