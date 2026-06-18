"""Attention-gating helpers for face recognition."""

from __future__ import annotations

from .gate import AttentionGateSettings, FaceAttentionGate
from .head_pose import HeadPoseObservation, estimate_head_pose
from .models import FaceAttentionObservation
from .smoothing import AttentionSmoother, AttentionSmoothingSettings

__all__ = [
    "AttentionGateSettings",
    "AttentionSmoother",
    "AttentionSmoothingSettings",
    "FaceAttentionGate",
    "FaceAttentionObservation",
    "HeadPoseObservation",
    "estimate_head_pose",
]
