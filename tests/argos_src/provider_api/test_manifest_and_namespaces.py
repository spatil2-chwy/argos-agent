from pathlib import Path

import pytest

from argos_src.provider_api.manifest import (
    ManifestValidationError,
    load_provider_manifest,
    parse_provider_manifest,
)
from argos_src.provider_api.namespaces import (
    provider_event_key,
    provider_manifest_key,
    provider_request_key,
    provider_response_key,
    provider_state_key,
)


def test_puffle_manifest_loads_resources_and_capabilities():
    manifest = load_provider_manifest("puffle")

    assert manifest.id == "puffle"
    assert manifest.display_name == "Puffle"
    assert manifest.provider_by_id("puffle-local").key_prefix == (
        "argos/providers/puffle-local"
    )
    base = manifest.resource_by_id("base")
    assert base is not None
    assert base.family == "unitree_go2"
    assert base.has_capability("motion.velocity")
    assert manifest.resource_by_id("head_realsense").has_capability("camera.rgbd")


def test_manifest_rejects_unknown_capability():
    with pytest.raises(ManifestValidationError, match="unsupported capability"):
        parse_provider_manifest(
            {
                "id": "bad",
                "providers": [{"id": "provider"}],
                "resources": [
                    {
                        "id": "thing",
                        "kind": "camera",
                        "provider": "provider",
                        "capabilities": ["camera.telepathy"],
                    }
                ],
            },
            source_path=Path("/tmp/bad.yaml"),
        )


def test_provider_resource_namespace_helpers():
    prefix = "argos/providers/puffle-local"

    assert provider_manifest_key(prefix) == "argos/providers/puffle-local/manifest"
    assert provider_request_key(prefix, "base", "req1") == (
        "argos/providers/puffle-local/resources/base/request/req1"
    )
    assert provider_response_key(prefix, "base", "req1") == (
        "argos/providers/puffle-local/resources/base/response/req1"
    )
    assert provider_event_key(prefix, "base", "battery.event") == (
        "argos/providers/puffle-local/resources/base/event/battery.event"
    )
    assert provider_state_key(prefix, "base", "battery") == (
        "argos/providers/puffle-local/resources/base/state/battery"
    )
