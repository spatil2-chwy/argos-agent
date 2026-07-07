"""State-observer protocol and safe wrappers for the realtime control plane."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from argos_src.agent.control.types import StateAxis, StateTransition

logger = logging.getLogger(__name__)


class StateObserver(Protocol):
    def transition(self, transition: StateTransition) -> None:
        """Observe a state transition."""

    def ignored(self, *, axis: StateAxis | str, trigger: str, reason: str, **fields: Any) -> None:
        """Observe an ignored event."""


class NullStateObserver:
    def transition(self, transition: StateTransition) -> None:
        return None

    def ignored(self, *, axis: StateAxis | str, trigger: str, reason: str, **fields: Any) -> None:
        return None


def safe_transition(observer: Any, transition: StateTransition) -> None:
    """Best-effort transition observation that never affects runtime behavior."""
    if observer is None:
        return
    transition_fn = getattr(observer, "transition", None)
    if not callable(transition_fn):
        return
    try:
        transition_fn(transition)
    except Exception:
        logger.debug("State transition observer failed", exc_info=True)


def safe_ignored(
    observer: Any,
    *,
    axis: StateAxis | str,
    trigger: str,
    reason: str,
    **fields: Any,
) -> None:
    if observer is None:
        return
    ignored_fn = getattr(observer, "ignored", None)
    if not callable(ignored_fn):
        return
    try:
        ignored_fn(axis=axis, trigger=trigger, reason=reason, **fields)
    except Exception:
        logger.debug("State ignored observer failed", exc_info=True)
