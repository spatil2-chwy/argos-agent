"""Compatibility normalization for Argos-facing identity metadata."""

from __future__ import annotations

import ast
from typing import Any


def normalize_directory_profile_lines(value: Any) -> tuple[str, ...]:
    """Return directory profile metadata as complete, cleaned lines.

    Tailwag normally returns a list, but older or intermediate payloads may
    contain a plain string or the string representation of a list. Strings
    must always remain whole rather than being treated as iterables of
    characters.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return (text,)
            if isinstance(parsed, (list, tuple)):
                return _clean_directory_profile_lines(parsed)
        return (text,)
    if isinstance(value, (list, tuple)):
        return _clean_directory_profile_lines(value)
    return ()


def _clean_directory_profile_lines(value: list[Any] | tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(
        line
        for item in value
        if isinstance(item, str) and (line := item.strip())
    )
