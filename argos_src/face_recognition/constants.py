"""Shared constants for the Argos face-recognition stack."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACE_DB_PATH = str((REPO_ROOT / "var" / "face_recognition").resolve())
MIN_FACE_DETECTION_CONFIDENCE = 0.9
