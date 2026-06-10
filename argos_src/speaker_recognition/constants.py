"""Shared constants for the Argos speaker-recognition stack."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEAKER_DB_PATH = str(
    (REPO_ROOT / "argos_src" / "speaker_recognition" / "db").resolve()
)
