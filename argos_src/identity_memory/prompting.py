"""Prompt and encounter helpers for Tailwag-backed identity memory."""

from __future__ import annotations

from typing import Any


ENCOUNTER_PROFILE_FIELDS = (
    "official_name",
    "employee_name",
    "username",
    "email",
    "work_email",
    "employee_email",
    "business_title",
    "job_family",
    "job_family_group",
    "job_level",
    "c_level",
    "manager_name",
    "team",
    "cost_center",
    "business_function",
    "senior_leadership_team",
    "tenure",
)


def build_encounter_metadata(
    *,
    name: str,
    site_code: str,
    identity_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return compact non-biometric metadata for a recognized-person encounter."""
    metadata: dict[str, Any] = {
        "name": str(name or "").strip(),
        "site_code": str(site_code or "").strip(),
    }
    for field in ENCOUNTER_PROFILE_FIELDS:
        value = str(identity_metadata.get(field) or "").strip()
        if value:
            metadata[field] = value
    return metadata


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


def _clean(value: Any) -> str:
    return str(value or "").strip()
