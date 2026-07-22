import json
from pathlib import Path

import numpy as np
import pytest

from scripts.labs import biometric_enrollment_bundle as bundle


def _finalized_bundle(tmp_path: Path):
    created = bundle.create_bundle(tmp_path, person_name="Jane Doe", person_id="person_jane")
    raw_dir = created.path / "raw"
    raw_dir.mkdir()
    (raw_dir / "face.jpg").write_bytes(b"raw unencrypted face")
    (raw_dir / "voice.wav").write_bytes(b"raw unencrypted voice")
    return bundle.finalize_bundle(
        created,
        face_embeddings=[np.array([3.0, 0.0]) for _ in range(5)],
        voice_embeddings=[np.array([0.0, 4.0]) for _ in range(5)],
    )


def test_normalize_mean_embedding_uses_every_normalized_sample_and_float32() -> None:
    result = bundle.normalize_mean_embedding(
        [np.array([10.0, 0.0]), np.array([0.0, 2.0]), np.array([1.0, 1.0])]
    )
    expected = np.array([1.0 + 1.0 / np.sqrt(2), 1.0 + 1.0 / np.sqrt(2)])
    expected = (expected / np.linalg.norm(expected)).astype(np.float32)

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected)
    with pytest.raises(bundle.BundleError, match="same shape"):
        bundle.normalize_mean_embedding([np.ones(2), np.ones(3)])
    with pytest.raises(bundle.BundleError, match="zero norm"):
        bundle.normalize_mean_embedding([np.zeros(2)])


def test_same_name_creates_distinct_opaque_uuid_bundles(tmp_path: Path) -> None:
    first = bundle.create_bundle(tmp_path, person_name="Jane Doe")
    second = bundle.create_bundle(tmp_path, person_name="Jane Doe")

    assert first.bundle_id != second.bundle_id
    assert first.path.parent == tmp_path / bundle.BUNDLE_STORE_DIRECTORY
    assert first.path.name == first.bundle_id
    assert "Jane" not in first.path.as_posix()


def test_finalize_round_trip_load_and_discovery(tmp_path: Path) -> None:
    completed = _finalized_bundle(tmp_path)
    collecting = bundle.create_bundle(tmp_path, person_name="Still Collecting")

    loaded = bundle.load_bundle(completed.path, output_root=tmp_path)
    aggregates = bundle.load_aggregate_embeddings(loaded)
    discovered = bundle.discover_completed_bundles(tmp_path)
    all_bundles = bundle.discover_bundles(tmp_path)

    assert loaded.bundle_id == completed.bundle_id
    assert loaded.completed is True
    assert aggregates["face"].dtype == np.float32
    assert aggregates["voice"].dtype == np.float32
    assert set(loaded.manifest["artifacts"][0]) == {"path", "size_bytes", "sha256"}
    assert [item.bundle_id for item in discovered] == [completed.bundle_id]
    assert {item.bundle_id for item in all_bundles} == {
        completed.bundle_id,
        collecting.bundle_id,
    }
    assert collecting.bundle_id not in {item.bundle_id for item in discovered}
    assert loaded.ready_for_upload is True
    assert bundle.is_bundle_ready_for_upload(loaded) is True


def test_artifact_inventory_detects_tampering_and_discovery_excludes_it(tmp_path: Path) -> None:
    completed = _finalized_bundle(tmp_path)
    (completed.path / "raw" / "face.jpg").write_bytes(b"tampered")

    assert bundle.verify_artifact_inventory(completed.path) is False
    with pytest.raises(bundle.BundleError, match="verification failed"):
        bundle.load_bundle(completed.path, output_root=tmp_path)
    assert bundle.discover_completed_bundles(tmp_path) == []
    assert bundle.discover_invalid_bundle_paths(tmp_path) == [completed.path]


def test_upload_state_transitions_are_atomic(tmp_path: Path, monkeypatch) -> None:
    completed = _finalized_bundle(tmp_path)
    assert completed.upload_state["modalities"]["face"]["details"] is None
    updated = bundle.update_upload_state(
        completed,
        "face",
        "uploaded",
        details={"reference_id": "face-ref", "sample_count": np.int64(5)},
    )

    assert updated.upload_state["modalities"]["face"]["state"] == "uploaded"
    assert updated.upload_state["modalities"]["face"]["details"] == {
        "reference_id": "face-ref",
        "sample_count": 5,
    }
    state_path = completed.path / bundle.UPLOAD_STATE_FILENAME
    before = state_path.read_bytes()

    def fail_replace(source, target):
        raise OSError("simulated interrupted replace")

    monkeypatch.setattr(bundle.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        bundle.update_upload_state(completed, "voice", "failed", error="offline")

    assert state_path.read_bytes() == before
    leftovers = list(completed.path.glob(f".{bundle.UPLOAD_STATE_FILENAME}.*.tmp"))
    assert leftovers == []


def test_cleanup_refuses_untrusted_incomplete_or_not_uploaded_bundle(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(bundle.BundleError, match="outside configured"):
        bundle.cleanup_bundle(tmp_path, outside)

    collecting = bundle.create_bundle(tmp_path, person_name="Collecting")
    with pytest.raises(bundle.BundleError, match="incomplete"):
        bundle.cleanup_bundle(tmp_path, collecting.path)

    completed = _finalized_bundle(tmp_path)
    with pytest.raises(bundle.BundleError, match="uploads"):
        bundle.cleanup_bundle(tmp_path, completed.path)
    assert completed.path.exists()


def test_cleanup_succeeds_only_after_verified_terminal_states(tmp_path: Path) -> None:
    completed = _finalized_bundle(tmp_path)
    completed = bundle.update_upload_state(completed, "face", "uploaded")
    completed = bundle.update_upload_state(completed, "voice", "skipped")

    assert bundle.upload_complete(completed) is True
    assert bundle.is_cleanup_eligible(completed) is True
    bundle.cleanup_bundle(tmp_path, completed.path)

    assert not completed.path.exists()


def test_failed_state_records_error_and_is_not_complete(tmp_path: Path) -> None:
    completed = _finalized_bundle(tmp_path)
    completed = bundle.update_upload_state(completed, "face", "failed", error="service down")

    state_payload = json.loads(
        (completed.path / bundle.UPLOAD_STATE_FILENAME).read_text(encoding="utf-8")
    )
    assert state_payload["modalities"]["face"]["error"] == "service down"
    assert bundle.upload_complete(completed) is False
    with pytest.raises(bundle.BundleError, match="mapping or null"):
        bundle.update_upload_state(completed, "voice", "failed", details=["invalid"])
    assert bundle.is_bundle_ready_for_upload(completed) is True
    assert completed.bundle_id in {
        item.bundle_id for item in bundle.discover_bundles(tmp_path)
    }


def test_finalize_rejects_fewer_than_five_samples(tmp_path: Path) -> None:
    created = bundle.create_bundle(tmp_path, person_name="Jane Doe")
    with pytest.raises(bundle.BundleError, match="at least five"):
        bundle.finalize_bundle(
            created,
            face_embeddings=[np.ones(2) for _ in range(4)],
            voice_embeddings=[np.ones(2) for _ in range(5)],
        )
    assert created.manifest["status"] == "collecting"


def test_bundle_records_sample_counts_and_binds_one_person(tmp_path: Path) -> None:
    completed = _finalized_bundle(tmp_path)

    assert completed.manifest["metadata"]["sample_counts"] == {"face": 5, "voice": 5}
    bound = bundle.bind_bundle_person(completed, "person_jane")
    assert bound.upload_state["person_id"] == "person_jane"
    assert bundle.bind_bundle_person(bound, "person_jane").upload_state["person_id"] == (
        "person_jane"
    )
    with pytest.raises(bundle.BundleError, match="already bound"):
        bundle.bind_bundle_person(bound, "person_other")


def test_create_bundle_rejects_symlinked_store(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    store = tmp_path / bundle.BUNDLE_STORE_DIRECTORY
    store.symlink_to(outside, target_is_directory=True)

    with pytest.raises(bundle.BundleError, match="symbolic link"):
        bundle.create_bundle(tmp_path, person_name="Jane Doe")
    assert list(outside.iterdir()) == []


def test_successful_upload_states_are_terminal(tmp_path: Path) -> None:
    completed = _finalized_bundle(tmp_path)
    completed = bundle.update_upload_state(completed, "face", "uploaded")

    with pytest.raises(bundle.BundleError, match="terminal face state"):
        bundle.update_upload_state(completed, "face", "pending")
