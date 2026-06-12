"""Shared process-local locks for motion command channels."""

from __future__ import annotations

import threading

_MOTION_LOCKS_GUARD = threading.Lock()
_MOTION_LOCKS: dict[str, threading.Lock] = {}


def motion_lock_for_channel(channel: str) -> threading.Lock:
    """Return a process-local lock for a movement command channel."""
    rendered = str(channel or "").strip()
    with _MOTION_LOCKS_GUARD:
        lock = _MOTION_LOCKS.get(rendered)
        if lock is None:
            lock = threading.Lock()
            _MOTION_LOCKS[rendered] = lock
        return lock


__all__ = ["motion_lock_for_channel"]
