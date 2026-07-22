"""Runtime controls available to the currently executing model tool."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator


TimeoutSetter = Callable[[float], None]
SideEffectGuard = Callable[[], AbstractContextManager[bool]]

_TIMEOUT_SETTER: ContextVar[TimeoutSetter | None] = ContextVar(
    "argos_tool_timeout_setter",
    default=None,
)
_SIDE_EFFECT_GUARD: ContextVar[SideEffectGuard | None] = ContextVar(
    "argos_tool_side_effect_guard",
    default=None,
)


@contextmanager
def tool_execution_context(
    *,
    set_timeout: TimeoutSetter,
    side_effect_guard: SideEffectGuard,
) -> Iterator[None]:
    timeout_token = _TIMEOUT_SETTER.set(set_timeout)
    guard_token = _SIDE_EFFECT_GUARD.set(side_effect_guard)
    try:
        yield
    finally:
        _SIDE_EFFECT_GUARD.reset(guard_token)
        _TIMEOUT_SETTER.reset(timeout_token)


def set_tool_execution_timeout(timeout_sec: float) -> None:
    setter = _TIMEOUT_SETTER.get()
    if setter is not None:
        setter(float(timeout_sec))


@contextmanager
def guard_tool_side_effect_start() -> Iterator[bool]:
    """Atomically reject a side effect if its owning turn is already terminal."""
    guard = _SIDE_EFFECT_GUARD.get()
    if guard is None:
        yield True
        return
    with guard() as allowed:
        yield bool(allowed)


__all__ = [
    "guard_tool_side_effect_start",
    "set_tool_execution_timeout",
    "tool_execution_context",
]
