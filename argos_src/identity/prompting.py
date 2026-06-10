"""Prompt helpers for identity/directory metadata."""

from __future__ import annotations

from typing import Any


def _clean(value: Any) -> str:
    return str(value or "").strip()


def format_identity_profile_lines(metadata: dict[str, Any] | None) -> tuple[str, ...]:
    """Return compact prompt-safe directory lines from identity metadata."""
    meta = dict(metadata or {})
    lines: list[str] = []

    title = _clean(meta.get("business_title"))
    job_level = _clean(meta.get("job_level"))
    if title and job_level:
        lines.append(f"title: {title} ({job_level})")
    elif title:
        lines.append(f"title: {title}")
    elif job_level:
        lines.append(f"job level: {job_level}")

    manager = _clean(meta.get("manager_name"))
    if manager:
        lines.append(f"manager: {manager}")

    business_function = _clean(meta.get("business_function"))
    cost_center = _clean(meta.get("cost_center"))
    if business_function and cost_center:
        lines.append(f"org: {business_function} / {cost_center}")
    elif business_function:
        lines.append(f"org: {business_function}")
    elif cost_center:
        lines.append(f"cost center: {cost_center}")

    tenure = _clean(meta.get("tenure"))
    if tenure:
        lines.append(f"tenure: {tenure}")

    return tuple(lines)
