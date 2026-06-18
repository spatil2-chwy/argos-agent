"""Attention-gating helpers for face recognition."""

from __future__ import annotations

from .gate import AttentionGateSettings, FaceAttentionGate
from .models import FaceAttentionObservation, HeadPoseObservation
from .overlay import draw_attention_overlay
from .smoothing import AttentionSmoother, AttentionSmoothingSettings

__all__ = [
    "AttentionGateSettings",
    "AttentionSmoother",
    "AttentionSmoothingSettings",
    "FaceAttentionGate",
    "FaceAttentionObservation",
    "HeadPoseObservation",
    "draw_attention_overlay",
]
