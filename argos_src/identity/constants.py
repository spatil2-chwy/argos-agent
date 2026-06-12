"""Shared constants for Argos identity storage."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IDENTITY_DB_PATH = str(
    (REPO_ROOT / "var" / "identity" / "identity.sqlite3").resolve()
)
