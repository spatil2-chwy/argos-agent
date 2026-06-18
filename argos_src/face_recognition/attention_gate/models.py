"""Data models for face-attention estimation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaceAttentionObservation:
    """Attention estimate for one visible face."""

    attentive: bool
    confidence: float
    reason: str = ""
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    raw_attentive: bool = False
    raw_confidence: float = 0.0
