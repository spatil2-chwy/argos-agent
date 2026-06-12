"""Provider/resource contracts for Argos capability integrations."""

from argos_src.provider_api.client import ProviderClient
from argos_src.provider_api.factory import DEFAULT_PROVIDER_TRANSPORT, create_provider_client
from argos_src.provider_api.fake import FakeProviderClient
from argos_src.provider_api.manifest import (
    ALLOWED_CAPABILITY_IDS,
    MANIFESTS_DIR,
    ManifestValidationError,
    ProviderManifest,
    ProviderResource,
    ProviderRoute,
    load_provider_manifest,
    resolve_manifest_path,
)
from argos_src.provider_api.models import (
    BatterySnapshot,
    CameraIntrinsics,
    ImageFrame,
    RGBDFrame,
    RobotTransform,
    VelocityCommand,
)
from argos_src.provider_api.errors import ProviderError, ProviderTimeout, is_provider_error

__all__ = [
    "ALLOWED_CAPABILITY_IDS",
    "BatterySnapshot",
    "CameraIntrinsics",
    "DEFAULT_PROVIDER_TRANSPORT",
    "FakeProviderClient",
    "ImageFrame",
    "MANIFESTS_DIR",
    "ManifestValidationError",
    "ProviderError",
    "ProviderClient",
    "ProviderManifest",
    "ProviderResource",
    "ProviderRoute",
    "ProviderTimeout",
    "RGBDFrame",
    "RobotTransform",
    "VelocityCommand",
    "create_provider_client",
    "is_provider_error",
    "load_provider_manifest",
    "resolve_manifest_path",
]
