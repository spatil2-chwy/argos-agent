"""Shared constants for Argos source-aware memory storage."""

from __future__ import annotations

from pathlib import Path


DEFAULT_MEMORY_DB_PATH = (
    Path(__file__).resolve().parents[2] / "var" / "memory" / "memory.sqlite3"
)
