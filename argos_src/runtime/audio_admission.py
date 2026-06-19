"""Pure policy helpers for local speech admission."""

from __future__ import annotations

import json
import time
from threading import Lock
from typing import Tuple


class FacePresenceGate:
    """Tracks visual presence snapshots published by the robot perception path."""

    def __init__(self, stale_after_sec: float = 6.0):
        self._stale_after_sec = stale_after_sec
        self._lock = Lock()
        self._status = "none"
        self._attention_status = "none"
        self._expires_at = 0.0

    def update_from_json(self, data: str) -> None:
        now = time.time()
        try:
            payload = json.loads(data)
        except Exception:
            payload = {"status": "none", "expires_at": now + self._stale_after_sec}
        self.update_from_snapshot(payload, now_s=now)

    def update_from_snapshot(self, payload: dict, *, now_s: float | None = None) -> None:
        now = time.time() if now_s is None else float(now_s)
        try:
            status = str(payload.get("status", "none"))
            attention_status = str(payload.get("attention_status", "none"))
            expires_at = float(payload.get("expires_at", now + self._stale_after_sec))
        except Exception:
            status = "none"
            attention_status = "none"
            expires_at = now + self._stale_after_sec
        with self._lock:
            self._status = status
            self._attention_status = attention_status
            self._expires_at = expires_at

    def status(self) -> str:
        now = time.time()
        with self._lock:
            if now > self._expires_at:
                return "none"
            return self._status

    def attention_status(self) -> str:
        now = time.time()
        with self._lock:
            if now > self._expires_at:
                return "none"
            return self._attention_status

    def is_face_present(self) -> bool:
        return self.status() in ("unknown", "recognized")

    def is_attention_present(self) -> bool:
        return self.attention_status() == "attentive"


def resolve_record_admission(
    *,
    face_present: bool,
    interaction_state: str,
    now_s: float,
    wake_window_until_s: float,
    wake_detected: bool,
    wake_window_sec: float,
    attention_present: bool = False,
    block_during_speaking: bool = True,
    block_during_engaged: bool = False,
    open_on_face_presence: bool = True,
    open_on_attention_presence: bool = False,
    open_on_interaction_states: tuple[str, ...] = ("alert", "cooldown"),
    open_on_wake_window: bool = True,
    nav_active: bool = False,
    nav_interruptible: bool = True,
    nav_passive_listen_allowed: bool = True,
) -> Tuple[bool, str, float]:
    """Resolve whether speech should start recording for this sample."""
    if block_during_speaking and interaction_state == "speaking":
        return False, interaction_state, wake_window_until_s
    if block_during_engaged and interaction_state == "engaged":
        return False, interaction_state, wake_window_until_s
    if nav_active and (not nav_interruptible) and (not nav_passive_listen_allowed):
        if wake_detected:
            next_wake_until = wake_window_until_s
            if open_on_wake_window:
                next_wake_until = now_s + max(0.1, float(wake_window_sec))
            return True, "wake_word", next_wake_until
        return False, "focused_navigation", wake_window_until_s
    if open_on_attention_presence and attention_present:
        return True, "attention_present", wake_window_until_s
    if open_on_face_presence and face_present:
        return True, "face_present", wake_window_until_s
    if interaction_state in open_on_interaction_states:
        return True, interaction_state, wake_window_until_s
    if open_on_wake_window and now_s < wake_window_until_s:
        return True, "wake_window", wake_window_until_s
    if wake_detected:
        next_wake_until = wake_window_until_s
        if open_on_wake_window:
            next_wake_until = now_s + max(0.1, float(wake_window_sec))
        return True, "wake_word", next_wake_until
    return False, "blocked", wake_window_until_s
