"""Static provider manifest models and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFESTS_DIR = REPO_ROOT / "config" / "manifests"

ALLOWED_CAPABILITY_IDS = frozenset(
    {
        "motion.velocity",
        "posture.command",
        "embodiment.action",
        "camera.rgb",
        "camera.rgbd",
        "camera.intrinsics",
        "transform.lookup",
        "battery.state",
        "navigation.goal",
        "dock.charging",
        "display.command",
        "display.interaction",
        "presence.face.publish",
        "voice_command.publish",
        "lidar.scan",
        "arm.pose",
        "gripper.command",
        "manipulation.pick_place",
    }
)


class ManifestValidationError(ValueError):
    """Raised when a provider manifest is missing or invalid."""


@dataclass(frozen=True)
class ProviderRoute:
    id: str
    transport: str
    key_prefix: str
    connect_endpoints: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderResource:
    id: str
    kind: str
    provider: str
    capabilities: tuple[str, ...]
    family: str = ""
    hardware: str = ""

    def has_capability(self, capability_id: str) -> bool:
        return str(capability_id or "").strip() in self.capabilities


@dataclass(frozen=True)
class ProviderManifest:
    id: str
    display_name: str
    providers: tuple[ProviderRoute, ...]
    resources: tuple[ProviderResource, ...]

    def provider_by_id(self, provider_id: str) -> ProviderRoute | None:
        wanted = str(provider_id or "").strip()
        return next((item for item in self.providers if item.id == wanted), None)

    def resource_by_id(self, resource_id: str) -> ProviderResource | None:
        wanted = str(resource_id or "").strip()
        return next((item for item in self.resources if item.id == wanted), None)


def resolve_manifest_path(name_or_path: str) -> Path:
    """Resolve a manifest name or path to an on-disk YAML file."""
    raw = str(name_or_path or "").strip()
    if not raw:
        raise ManifestValidationError("Manifest selection is required.")

    candidate = Path(raw)
    looks_like_path = (
        candidate.is_absolute()
        or candidate.parent != Path(".")
        or raw.startswith(".")
        or "/" in raw
    )
    if looks_like_path:
        path = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
        resolved = path.resolve()
        if not resolved.exists():
            raise ManifestValidationError(f"Manifest file not found: {resolved}")
        return resolved

    filename = raw if raw.endswith((".yaml", ".yml")) else f"{raw}.yaml"
    resolved = (MANIFESTS_DIR / filename).resolve()
    if not resolved.exists():
        raise ManifestValidationError(
            f"Manifest '{raw}' not found under {MANIFESTS_DIR}."
        )
    return resolved


def load_provider_manifest(name_or_path: str) -> ProviderManifest:
    """Load and validate one static provider manifest."""
    path = resolve_manifest_path(name_or_path)
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ManifestValidationError(f"Manifest {path} must contain a mapping.")
    return parse_provider_manifest(payload, source_path=path)


def parse_provider_manifest(
    payload: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> ProviderManifest:
    """Parse a manifest mapping into validated dataclasses."""
    context = str(source_path or "manifest")
    data = dict(payload)
    manifest_id = _required_str(data, "id", context)
    display_name = _optional_str(data, "display_name", default=manifest_id)
    providers = tuple(
        _parse_provider(item, index=i, context=context)
        for i, item in enumerate(_required_list(data, "providers", context))
    )
    resources = tuple(
        _parse_resource(item, index=i, context=context)
        for i, item in enumerate(_required_list(data, "resources", context))
    )
    _reject_unknown(data, context)

    provider_ids = {item.id for item in providers}
    if len(provider_ids) != len(providers):
        raise ManifestValidationError(f"{context}.providers contains duplicate ids.")
    resource_ids = {item.id for item in resources}
    if len(resource_ids) != len(resources):
        raise ManifestValidationError(f"{context}.resources contains duplicate ids.")
    for resource in resources:
        if resource.provider not in provider_ids:
            raise ManifestValidationError(
                f"{context}.resources[{resource.id}].provider "
                f"references unknown provider '{resource.provider}'."
            )

    return ProviderManifest(
        id=manifest_id,
        display_name=display_name,
        providers=providers,
        resources=resources,
    )


def _parse_provider(item: Any, *, index: int, context: str) -> ProviderRoute:
    item_context = f"{context}.providers[{index}]"
    if not isinstance(item, dict):
        raise ManifestValidationError(f"{item_context} must be a mapping.")
    data = dict(item)
    provider_id = _required_str(data, "id", item_context)
    transport = _optional_str(data, "transport", default="zenoh")
    key_prefix = _optional_str(
        data,
        "key_prefix",
        default=f"argos/providers/{provider_id}",
    )
    connect_endpoints = tuple(
        _coerce_string_list(
            data.pop("connect_endpoints", []),
            context=f"{item_context}.connect_endpoints",
        )
    )
    _reject_unknown(data, item_context)
    return ProviderRoute(
        id=provider_id,
        transport=transport,
        key_prefix=key_prefix,
        connect_endpoints=connect_endpoints,
    )


def _parse_resource(item: Any, *, index: int, context: str) -> ProviderResource:
    item_context = f"{context}.resources[{index}]"
    if not isinstance(item, dict):
        raise ManifestValidationError(f"{item_context} must be a mapping.")
    data = dict(item)
    resource_id = _required_str(data, "id", item_context)
    kind = _required_str(data, "kind", item_context)
    provider = _required_str(data, "provider", item_context)
    capabilities = tuple(
        _coerce_string_list(
            _required_list(data, "capabilities", item_context),
            context=f"{item_context}.capabilities",
        )
    )
    for capability_id in capabilities:
        if capability_id not in ALLOWED_CAPABILITY_IDS:
            raise ManifestValidationError(
                f"{item_context}.capabilities contains unsupported capability "
                f"'{capability_id}'."
            )
    family = _optional_str(data, "family", default="")
    hardware = _optional_str(data, "hardware", default="")
    _reject_unknown(data, item_context)
    return ProviderResource(
        id=resource_id,
        kind=kind,
        provider=provider,
        capabilities=capabilities,
        family=family,
        hardware=hardware,
    )


def _required_str(data: dict[str, Any], key: str, context: str) -> str:
    value = data.pop(key, None)
    rendered = str(value or "").strip()
    if not rendered:
        raise ManifestValidationError(f"{context}.{key} is required.")
    return rendered


def _optional_str(data: dict[str, Any], key: str, *, default: str) -> str:
    value = data.pop(key, None)
    if value is None:
        return default
    return str(value or "").strip()


def _required_list(data: dict[str, Any], key: str, context: str) -> list[Any]:
    value = data.pop(key, None)
    if not isinstance(value, list):
        raise ManifestValidationError(f"{context}.{key} must be a list.")
    return list(value)


def _coerce_string_list(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, list):
        raise ManifestValidationError(f"{context} must be a list.")
    result: list[str] = []
    for item in value:
        rendered = str(item or "").strip()
        if not rendered:
            raise ManifestValidationError(f"{context} contains an empty item.")
        result.append(rendered)
    return result


def _reject_unknown(data: dict[str, Any], context: str) -> None:
    if data:
        keys = ", ".join(sorted(data))
        raise ManifestValidationError(f"Unknown key(s) in {context}: {keys}")


__all__ = [
    "ALLOWED_CAPABILITY_IDS",
    "MANIFESTS_DIR",
    "ManifestValidationError",
    "ProviderManifest",
    "ProviderResource",
    "ProviderRoute",
    "load_provider_manifest",
    "parse_provider_manifest",
    "resolve_manifest_path",
]
