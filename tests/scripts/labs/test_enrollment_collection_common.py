from pathlib import Path
from types import SimpleNamespace

import pytest

from argos_src.provider_api.manifest import (
    ProviderAuth,
    ProviderManifest,
    ProviderResource,
    ProviderRoute,
)
from scripts.labs.enrollment_collection_common import (
    create_identity_memory_client_for_profile,
    parse_camera_specs,
    resolve_collection_session,
    safe_path_part,
)


def test_safe_path_part_keeps_collection_names_filesystem_friendly():
    assert safe_path_part("Jane Doe / BOS3") == "Jane_Doe___BOS3"
    assert safe_path_part("  ") == "unknown"


def test_parse_camera_specs_uses_profile_face_camera_by_default():
    specs = parse_camera_specs([], default_resource_id="arducam_001")

    assert specs == [{"alias": "face_camera", "resource_id": "arducam_001"}]


def test_parse_camera_specs_requires_default_resource_when_no_camera_given():
    with pytest.raises(ValueError, match="resources.face_camera"):
        parse_camera_specs([])


def test_parse_camera_specs_supports_alias_resource_pairs():
    specs = parse_camera_specs(["front=cam_1", "side_cam"])

    assert specs == [
        {"alias": "front", "resource_id": "cam_1"},
        {"alias": "side_cam", "resource_id": "side_cam"},
    ]


def test_parse_camera_specs_rejects_duplicate_aliases():
    with pytest.raises(ValueError, match="Duplicate camera alias"):
        parse_camera_specs(["front=cam_1", "front=cam_2"])


def test_resolve_collection_session_uses_person_slug_and_session(tmp_path: Path):
    session = resolve_collection_session(
        output_root=tmp_path,
        person_name="Jane Doe",
        person_id="person_jane",
        session_id="trial_1",
    )

    assert session["person_slug"] == "person_jane"
    assert session["session_id"] == "trial_1"
    assert session["session_dir"] == tmp_path / "person_jane" / "trial_1"


def _identity_memory_profile(
    *,
    manifest: ProviderManifest | None,
    resource_id: str = "memory",
    site_code: str = "BOS3",
    backend: str = "tailwag_http",
):
    return SimpleNamespace(
        manifest=manifest,
        resources=SimpleNamespace(identity_memory=resource_id),
        identity_memory=SimpleNamespace(
            enabled=True,
            backend=backend,
            site_code=site_code,
            place_room_id="robotics-lab",
            retention_class="priority",
            extract_live_turn_memory=True,
        ),
    )


def _memory_manifest(*, capability: str = "memory.identity") -> ProviderManifest:
    return ProviderManifest(
        id="lab",
        display_name="Lab",
        providers=(
            ProviderRoute(
                id="memory-provider",
                transport="http",
                key_prefix="argos/providers/memory",
                connect_endpoints=("http://localhost:8000",),
                auth=ProviderAuth(type="bearer", token_env="TAILWAG_API_BEARER_TOKEN"),
            ),
        ),
        resources=(
            ProviderResource(
                id="memory",
                kind="memory",
                provider="memory-provider",
                capabilities=(capability,),
            ),
        ),
    )


def test_create_identity_memory_client_for_profile_uses_manifest_resource(monkeypatch):
    calls = []

    class FakeProviderClient:
        def shutdown(self):
            return None

    def fake_create_provider_client(**kwargs):
        calls.append(kwargs)
        return FakeProviderClient()

    monkeypatch.setattr(
        "scripts.labs.enrollment_collection_common.create_provider_client",
        fake_create_provider_client,
    )
    profile = _identity_memory_profile(manifest=_memory_manifest())

    client = create_identity_memory_client_for_profile(profile, site_code="NYC1")

    assert client.site_code == "NYC1"
    assert client.place_room_id == "robotics-lab"
    assert client.retention_class == "priority"
    assert client.extract_live_turn_memory is True
    assert getattr(client, "_resource_id") == "memory"
    assert calls == [
        {
            "transport": "http",
            "key_prefix": "argos/providers/memory",
            "connect_endpoints": ("http://localhost:8000",),
            "resource_id": "memory",
            "manifest": profile.manifest,
            "auth_token_env": "TAILWAG_API_BEARER_TOKEN",
        }
    ]


def test_create_identity_memory_client_for_profile_rejects_non_memory_resource():
    profile = _identity_memory_profile(manifest=_memory_manifest(capability="camera.rgb"))

    with pytest.raises(ValueError, match="memory.identity"):
        create_identity_memory_client_for_profile(profile)


def test_create_identity_memory_client_for_profile_rejects_noop_backend():
    profile = _identity_memory_profile(manifest=_memory_manifest(), backend="noop")

    with pytest.raises(ValueError, match="tailwag_http"):
        create_identity_memory_client_for_profile(profile)
