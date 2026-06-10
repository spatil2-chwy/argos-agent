"""Encounter normalization helpers for robot-observed memory."""

from __future__ import annotations

from typing import Any


ENCOUNTER_PROFILE_FIELDS = (
    "official_name",
    "employee_name",
    "username",
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
    """Return a compact metadata payload for a recognized-person encounter."""
    metadata: dict[str, Any] = {
        "name": str(name or "").strip(),
        "site_code": str(site_code or "").strip(),
    }
    for field in ENCOUNTER_PROFILE_FIELDS:
        value = str(identity_metadata.get(field) or "").strip()
        if value:
            metadata[field] = value
    return metadata
