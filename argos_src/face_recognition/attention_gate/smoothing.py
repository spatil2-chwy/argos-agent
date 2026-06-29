"""Temporal smoothing for face-attention observations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionSmoothingSettings:
    """Runtime knobs for attention hysteresis."""

    window_sec: float = 1.0
    min_observations: int = 2
    hold_sec: float = 0.8

    def __post_init__(self) -> None:
        if self.window_sec <= 0.0:
            raise ValueError("window_sec must be > 0")
        if self.min_observations < 1:
            raise ValueError("min_observations must be >= 1")
        if self.hold_sec < 0.0:
            raise ValueError("hold_sec must be >= 0")


class AttentionSmoother:
    """Smooth per-face attention decisions over a short rolling window."""

    def __init__(self, settings: AttentionSmoothingSettings | None = None) -> None:
        self.settings = settings or AttentionSmoothingSettings()
        self._history: dict[str, deque[tuple[float, bool, float]]] = {}
        self._last_attentive_at: dict[str, float] = {}

    def update(
        self,
        *,
        track_id: str,
        now: float,
        attentive: bool,
        confidence: float,
    ) -> tuple[bool, float]:
        rendered = str(track_id or "").strip()
        if not rendered:
            return bool(attentive), float(confidence)

        window_sec = float(self.settings.window_sec)
        cutoff = float(now) - window_sec
        history = self._history.setdefault(rendered, deque())
        history.append((float(now), bool(attentive), float(confidence)))
        while history and history[0][0] < cutoff:
            history.popleft()

        attentive_items = [item for item in history if item[1]]
        enough = len(attentive_items) >= int(self.settings.min_observations)
        if enough:
            self._last_attentive_at[rendered] = float(now)
            self._prune(now)
            return True, 1.0

        last_attentive = self._last_attentive_at.get(rendered)
        if (
            last_attentive is not None
            and (float(now) - last_attentive) <= float(self.settings.hold_sec)
        ):
            self._prune(now)
            return True, 1.0

        self._prune(now)
        return False, 0.0

    def _prune(self, now: float) -> None:
        stale_before = float(now) - max(
            float(self.settings.window_sec),
            float(self.settings.hold_sec),
            1.0,
        ) - 1.0
        for track_id in list(self._history):
            history = self._history[track_id]
            while history and history[0][0] < stale_before:
                history.popleft()
            if not history:
                del self._history[track_id]
        for track_id, last_seen in list(self._last_attentive_at.items()):
            if last_seen < stale_before:
                del self._last_attentive_at[track_id]
