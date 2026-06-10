"""Shared constants for Argos source-aware memory storage."""

from __future__ import annotations

from pathlib import Path


DEFAULT_MEMORY_DB_PATH = (
    Path(__file__).resolve().parents[1] / "memory" / "db" / "memory.sqlite3"
)
