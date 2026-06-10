"""Shared BaseTool import with a lightweight fallback for test-only environments."""

from __future__ import annotations

try:
    from langchain_core.tools import BaseTool
except ImportError:  # pragma: no cover - fallback for local/unit-test environments.
    from pydantic import BaseModel

    class BaseTool(BaseModel):
        """Minimal fallback that preserves Pydantic field behavior used by tool tests."""

        class Config:
            arbitrary_types_allowed = True


__all__ = ["BaseTool"]
