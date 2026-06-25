#!/usr/bin/env python3
"""Shared helpers for person-centered enrollment data collection scripts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from argos_src.display import DisplayRuntime
from argos_src.profile_config import ScenarioProfile, load_scenario_profile
from argos_src.provider_api.factory import create_provider_client
from scripts.labs.perception_lab_common import current_git_commit, write_json

DEFAULT_COLLECTION_ROOT = _REPO_ROOT / "data_collection"


def safe_path_part(value: str) -> str:
    rendered = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(value or "").strip()
    ).strip("_")
    return rendered or "unknown"


def make_session_id(prefix: str = "") -> str:
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    rendered_prefix = safe_path_part(prefix) if prefix else ""
    return f"{rendered_prefix}_{suffix}" if rendered_prefix else suffix


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    return value


def add_person_collection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("person_name", help="Name/label for the person being collected.")
    parser.add_argument(
        "--person-id",
        default="",
        help="Optional stable person id. Defaults to a slug derived from person_name.",
    )
    parser.add_argument(
        "--profile",
        default="static_interaction",
        help="Argos profile name or YAML path. Default: static_interaction.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_COLLECTION_ROOT),
        help="Root directory for collected enrollment data. Default: data_collection.",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="Optional session id. Defaults to <person_slug>_<timestamp>.",
    )
    parser.add_argument(
        "--provider-transport",
        default="",
        help="Override provider transport for capture. Use 'fake' for smoke tests.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not update the configured interaction display.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose capture logging.",
    )


def resolve_collection_session(
    *,
    output_root: str | Path,
    person_name: str,
    person_id: str = "",
    session_id: str = "",
) -> dict[str, Any]:
    person_slug = safe_path_part(person_id or person_name)
    rendered_session_id = safe_path_part(session_id) if session_id else make_session_id(person_slug)
    session_dir = Path(output_root).expanduser().resolve() / person_slug / rendered_session_id
    return {
        "person_name": str(person_name or "").strip(),
        "person_slug": person_slug,
        "session_id": rendered_session_id,
        "session_dir": session_dir,
    }


def write_session_manifest(
    *,
    session_dir: str | Path,
    payload: dict[str, Any],
    filename: str = "session_manifest.json",
) -> Path:
    target = Path(session_dir) / (Path(str(filename or "session_manifest.json")).name)
    manifest = {
        "created_at_unix_s": round(datetime.now().timestamp(), 3),
        "git_commit": current_git_commit(),
        "command": list(sys.argv),
        **dict(payload),
    }
    write_json(target, json_ready(manifest))
    return target


def parse_camera_specs(
    values: list[str] | tuple[str, ...] | None,
    *,
    default_resource_id: str = "",
    default_alias: str = "face_camera",
) -> list[dict[str, str]]:
    raw_values = list(values or [])
    if not raw_values:
        rendered_resource = str(default_resource_id or "").strip()
        if not rendered_resource:
            raise ValueError(
                "No --camera was provided and the selected profile has no resources.face_camera."
            )
        raw_values = [f"{safe_path_part(default_alias)}={rendered_resource}"]
    specs: list[dict[str, str]] = []
    seen_aliases: set[str] = set()
    for raw in raw_values:
        rendered = str(raw or "").strip()
        if not rendered:
            continue
        if "=" in rendered:
            alias_raw, resource_raw = rendered.split("=", 1)
            alias = safe_path_part(alias_raw)
            resource_id = resource_raw.strip()
        else:
            resource_id = rendered
            alias = safe_path_part(rendered)
        if not resource_id:
            raise ValueError(f"Camera spec {rendered!r} is missing a resource id.")
        if alias in seen_aliases:
            raise ValueError(f"Duplicate camera alias: {alias}")
        seen_aliases.add(alias)
        specs.append({"alias": alias, "resource_id": resource_id})
    if not specs:
        raise ValueError("At least one camera resource is required.")
    return specs


def load_profile(selection: str) -> ScenarioProfile:
    return load_scenario_profile(selection)


def create_provider_for_resource(
    profile: ScenarioProfile,
    *,
    resource_id: str,
    provider_transport: str = "",
):
    manifest = profile.manifest
    resource = manifest.resource_by_id(resource_id) if manifest is not None else None
    provider = manifest.provider_by_id(resource.provider) if manifest is not None and resource else None
    if provider is None and manifest is not None:
        primary = manifest.resource_by_id(profile.resources.primary_robot)
        provider = manifest.provider_by_id(primary.provider) if primary is not None else None

    transport = str(provider_transport or "").strip() or (
        str(getattr(provider, "transport", "") or "").strip()
        or str(profile.robot.bridge.transport or "").strip()
    )
    key_prefix = (
        str(getattr(provider, "key_prefix", "") or "").strip()
        or str(profile.robot.bridge.key_prefix or "").strip()
    )
    connect_endpoints = (
        tuple(getattr(provider, "connect_endpoints", ()) or ())
        or tuple(profile.robot.bridge.connect_endpoints or ())
    )
    return create_provider_client(
        transport=transport,
        key_prefix=key_prefix,
        connect_endpoints=connect_endpoints,
        resource_id=resource_id,
        manifest=manifest,
    )


def create_display_runtime_for_profile(
    profile: ScenarioProfile,
    *,
    disabled: bool = False,
    provider_transport: str = "",
) -> DisplayRuntime | None:
    if disabled or not bool(getattr(profile.display, "enabled", True)):
        return None
    resource_id = str(getattr(profile.resources, "interaction_display", "") or "").strip()
    manifest = profile.manifest
    if not resource_id or manifest is None:
        return None
    resource = manifest.resource_by_id(resource_id)
    if resource is None or not resource.has_capability("display.command"):
        return None
    provider = manifest.provider_by_id(resource.provider)
    if provider is None:
        return None
    client = create_provider_client(
        transport=(str(provider_transport or "").strip() or provider.transport),
        key_prefix=provider.key_prefix,
        connect_endpoints=provider.connect_endpoints,
        resource_id=resource.id,
        manifest=manifest,
    )
    runtime = DisplayRuntime(client=client, resource_id=resource.id)
    runtime.start()
    return runtime
