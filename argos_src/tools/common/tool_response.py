"""Shared helpers for deterministic tool response envelopes."""

from __future__ import annotations

import json
from typing import Any, Optional


def build_tool_response(
    *,
    success: bool,
    status: str,
    message: str,
    robot_state_after: Optional[str] = None,
    eventual: Optional[bool] = None,
    result_source: Optional[str] = None,
    data: Optional[dict[str, Any]] = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a normalized tool response payload.

    Not every tool needs every field, but all tool responses should at least provide:
    - success
    - status
    - message
    """
    payload: dict[str, Any] = {
        "success": bool(success),
        "status": str(status or "").strip() or ("completed" if success else "error"),
        "message": str(message or "").strip(),
    }
    if robot_state_after is not None:
        payload["robot_state_after"] = str(robot_state_after)
    if eventual is not None:
        payload["eventual"] = bool(eventual)
    if result_source is not None:
        payload["result_source"] = str(result_source)
    if data:
        payload["data"] = data
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    return payload


def tool_response_json(
    *,
    success: bool,
    status: str,
    message: str,
    robot_state_after: Optional[str] = None,
    eventual: Optional[bool] = None,
    result_source: Optional[str] = None,
    data: Optional[dict[str, Any]] = None,
    **extra: Any,
) -> str:
    """Return a normalized tool response envelope as JSON."""
    return json.dumps(
        build_tool_response(
            success=success,
            status=status,
            message=message,
            robot_state_after=robot_state_after,
            eventual=eventual,
            result_source=result_source,
            data=data,
            **extra,
        )
    )
