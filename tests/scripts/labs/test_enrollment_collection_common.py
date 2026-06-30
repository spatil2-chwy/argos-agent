from pathlib import Path

import pytest

from scripts.labs.enrollment_collection_common import (
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
