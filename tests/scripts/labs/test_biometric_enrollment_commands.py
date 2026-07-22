from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.labs import biometric_enrollment_commands as commands
from scripts.labs.biometric_enrollment_bundle import (
    create_bundle,
    bind_bundle_person,
    finalize_bundle,
    load_bundle,
    update_upload_state,
)


def _ready_bundle(tmp_path, *, person_id: str = ""):
    bundle = create_bundle(
        tmp_path,
        person_name="Jane Doe",
        person_id=person_id,
        metadata={
            "username": "jdoe",
            "site_code": "BOS3",
            "profile_arg": "static_interaction",
            "embedding_models": {"face": "facenet", "voice": "ecapa"},
        },
    )
    (bundle.path / "photo.png").write_bytes(b"photo")
    (bundle.path / "voice.wav").write_bytes(b"voice")
    return finalize_bundle(
        bundle,
        face_embeddings=[
            np.asarray([1.0, 0.0]), np.asarray([0.9, 0.1]),
            np.asarray([1.0, 0.05]), np.asarray([0.95, 0.05]), np.asarray([1.0, 0.1]),
        ],
        voice_embeddings=[
            np.asarray([0.0, 1.0]), np.asarray([0.1, 0.9]),
            np.asarray([0.05, 1.0]), np.asarray([0.05, 0.95]), np.asarray([0.1, 1.0]),
        ],
    )


class FakeIdentityMemory:
    def __init__(self, *, face_exists: bool, voice_exists: bool) -> None:
        self.exists = {"face": face_exists, "voice": voice_exists}
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def biometric_reference_exists(self, *, modality: str, person_id: str) -> bool:
        self.calls.append(("exists", {"modality": modality, "person_id": person_id}))
        return self.exists[modality]

    def search_face(self, **kwargs):
        self.calls.append(("search_face", kwargs))
        return {
            "recognized": False,
            "reason": "below_threshold",
            "threshold": 0.6,
            "candidates": [],
        }

    def search_voice(self, **kwargs):
        self.calls.append(("search_voice", kwargs))
        return {
            "recognized": False,
            "reason": "below_threshold",
            "threshold": 0.5,
            "candidates": [],
        }

    def enroll_face_reference(self, **kwargs):
        self.calls.append(("enroll_face", kwargs))
        return {
            "saved": True,
            "status": "saved",
            "reason": "saved",
            "person_id": kwargs["person_id"],
            "reference_id": "face-ref",
        }

    def enroll_voice_reference(self, **kwargs):
        self.calls.append(("enroll_voice", kwargs))
        return {
            "saved": True,
            "status": "saved",
            "reason": "saved",
            "person_id": kwargs["person_id"],
            "reference_id": "voice-ref",
        }

    def close(self):
        self.closed = True


def _args(tmp_path):
    return SimpleNamespace(
        output_root=str(tmp_path),
        profile="static_interaction",
        site_code="",
    )


def _install_push_fakes(monkeypatch, fake):
    monkeypatch.setattr(
        commands,
        "load_profile",
        lambda _selection: SimpleNamespace(
            identity_memory=SimpleNamespace(site_code="BOS3")
        ),
    )
    monkeypatch.setattr(
        commands,
        "create_identity_memory_client_for_profile",
        lambda _profile, *, site_code: fake,
    )
    monkeypatch.setattr(
        commands,
        "_resolve_bundle_identity",
        lambda identity_memory, *, bundle, site_code: {
            "person_id": "person_jdoe",
            "display_name": "Jane Doe",
            "official_name": "Jane Doe",
            "username": "jdoe",
            "site_code": site_code,
        },
    )


def test_normalize_cli_argv_preserves_legacy_capture_and_maps_commands():
    assert commands.normalize_cli_argv(["capture", "Jane Doe"]) == ["Jane Doe"]
    assert commands.normalize_cli_argv(["list"]) == ["--list"]
    assert commands.normalize_cli_argv(["push", "--site-code", "BOS3"]) == [
        "--push",
        "--site-code",
        "BOS3",
    ]
    assert commands.normalize_cli_argv(["Jane Doe"]) == ["Jane Doe"]


def test_push_uploads_only_missing_modality_after_typed_approval(monkeypatch, tmp_path):
    bundle = _ready_bundle(tmp_path)
    fake = FakeIdentityMemory(face_exists=False, voice_exists=True)
    _install_push_fakes(monkeypatch, fake)
    answers = iter(["1", "Jane Doe"])

    assert commands.push_local_bundle(
        _args(tmp_path), input_fn=lambda _prompt: next(answers)
    ) == 0

    loaded = load_bundle(bundle.path, output_root=tmp_path)
    assert loaded.modality_states == {"face": "uploaded", "voice": "skipped"}
    assert [name for name, _payload in fake.calls] == [
        "exists",
        "exists",
        "search_face",
        "enroll_face",
    ]
    assert fake.calls[2][1]["global_scope"] is True
    assert fake.calls[2][1]["strict_response"] is True
    enrollment = fake.calls[-1][1]
    assert enrollment["person_id"] == "person_jdoe"
    assert enrollment["consent_status"] == "consented"
    assert enrollment["metadata"]["bundle_id"] == bundle.bundle_id
    assert "sample_count" not in enrollment["metadata"]
    assert fake.closed is True


def test_push_cancellation_sends_no_embedding(monkeypatch, tmp_path):
    bundle = _ready_bundle(tmp_path)
    fake = FakeIdentityMemory(face_exists=False, voice_exists=False)
    _install_push_fakes(monkeypatch, fake)
    answers = iter(["1", "not the name"])

    assert commands.push_local_bundle(
        _args(tmp_path), input_fn=lambda _prompt: next(answers)
    ) == 0

    loaded = load_bundle(bundle.path, output_root=tmp_path)
    assert loaded.modality_states == {"face": "pending", "voice": "pending"}
    assert [name for name, _payload in fake.calls] == ["exists", "exists"]
    assert fake.closed is True


def test_push_conflict_fails_before_enrollment_and_journals_failure(
    monkeypatch, tmp_path
):
    bundle = _ready_bundle(tmp_path)
    fake = FakeIdentityMemory(face_exists=False, voice_exists=True)
    _install_push_fakes(monkeypatch, fake)
    fake.search_face = lambda **kwargs: {
        "recognized": True,
        "reason": "matched",
        "threshold": 0.6,
        "candidates": [{"person_id": "person_other", "score": 0.95}],
    }
    answers = iter(["1", "Jane Doe"])

    with pytest.raises(RuntimeError, match="person_other"):
        commands.push_local_bundle(
            _args(tmp_path), input_fn=lambda _prompt: next(answers)
        )

    loaded = load_bundle(bundle.path, output_root=tmp_path)
    assert loaded.modality_states["face"] == "failed"
    assert loaded.modality_states["voice"] == "pending"
    assert fake.closed is True


def test_push_preflights_all_conflicts_before_first_enrollment(monkeypatch, tmp_path):
    bundle = _ready_bundle(tmp_path)
    fake = FakeIdentityMemory(face_exists=False, voice_exists=False)
    _install_push_fakes(monkeypatch, fake)

    def conflicting_voice(**kwargs):
        fake.calls.append(("search_voice", kwargs))
        return {
            "recognized": True,
            "reason": "matched",
            "threshold": 0.5,
            "candidates": [{"person_id": "person_other", "score": 0.95}],
        }

    fake.search_voice = conflicting_voice
    answers = iter(["1", "Jane Doe"])

    with pytest.raises(RuntimeError, match="person_other"):
        commands.push_local_bundle(
            _args(tmp_path), input_fn=lambda _prompt: next(answers)
        )

    loaded = load_bundle(bundle.path, output_root=tmp_path)
    assert loaded.modality_states == {"face": "pending", "voice": "failed"}
    assert [name for name, _payload in fake.calls] == [
        "exists",
        "exists",
        "search_face",
        "search_voice",
    ]
    assert fake.closed is True


def test_push_existence_failure_sends_no_embedding(monkeypatch, tmp_path):
    bundle = _ready_bundle(tmp_path)
    fake = FakeIdentityMemory(face_exists=False, voice_exists=False)
    _install_push_fakes(monkeypatch, fake)

    def failed_exists(*, modality, person_id):
        fake.calls.append(("exists", {"modality": modality, "person_id": person_id}))
        raise RuntimeError("malformed existence response")

    fake.biometric_reference_exists = failed_exists

    with pytest.raises(RuntimeError, match="malformed existence response"):
        commands.push_local_bundle(_args(tmp_path), input_fn=lambda _prompt: "1")

    loaded = load_bundle(bundle.path, output_root=tmp_path)
    assert loaded.modality_states == {"face": "pending", "voice": "pending"}
    assert [name for name, _payload in fake.calls] == ["exists"]
    assert fake.closed is True


def test_push_rejects_person_change_after_partial_upload(monkeypatch, tmp_path):
    bundle = _ready_bundle(tmp_path)
    bundle = update_upload_state(
        bundle,
        "face",
        "uploaded",
        details={"person_id": "person_original"},
    )
    bundle = bind_bundle_person(bundle, "person_original")
    fake = FakeIdentityMemory(face_exists=True, voice_exists=False)
    _install_push_fakes(monkeypatch, fake)

    with pytest.raises(RuntimeError, match="bound to person_original"):
        commands.push_local_bundle(_args(tmp_path), input_fn=lambda _prompt: "1")

    assert fake.calls == []
    assert fake.closed is True


def test_claimed_archived_person_is_not_enrollable(tmp_path):
    bundle = _ready_bundle(tmp_path, person_id="person_archived")

    class ArchivedIdentity:
        def person_profile(self, person_id):
            return {
                "person_id": person_id,
                "display_name": "Jane Doe",
                "status": "archived",
            }

    with pytest.raises(RuntimeError, match="not active"):
        commands._resolve_bundle_identity(
            ArchivedIdentity(), bundle=bundle, site_code="BOS3"
        )


def test_retry_after_remote_save_rechecks_existence_without_duplicate(
    monkeypatch, tmp_path
):
    bundle = _ready_bundle(tmp_path)
    fake = FakeIdentityMemory(face_exists=False, voice_exists=True)
    _install_push_fakes(monkeypatch, fake)
    original_enroll = fake.enroll_face_reference

    def save_remotely(**kwargs):
        result = original_enroll(**kwargs)
        fake.exists["face"] = True
        return result

    fake.enroll_face_reference = save_remotely
    real_update = commands.update_upload_state
    failed_once = False

    def fail_first_uploaded(bundle_value, modality, state, **kwargs):
        nonlocal failed_once
        if modality == "face" and state == "uploaded" and not failed_once:
            failed_once = True
            raise OSError("local journal interrupted")
        return real_update(bundle_value, modality, state, **kwargs)

    monkeypatch.setattr(commands, "update_upload_state", fail_first_uploaded)
    answers = iter(["1", "Jane Doe"])
    with pytest.raises(OSError, match="local journal interrupted"):
        commands.push_local_bundle(
            _args(tmp_path), input_fn=lambda _prompt: next(answers)
        )

    monkeypatch.setattr(commands, "update_upload_state", real_update)
    assert commands.push_local_bundle(_args(tmp_path), input_fn=lambda _prompt: "1") == 0

    loaded = load_bundle(bundle.path, output_root=tmp_path)
    assert loaded.modality_states == {"face": "skipped", "voice": "skipped"}
    assert [name for name, _payload in fake.calls].count("enroll_face") == 1


def test_verified_directory_person_can_be_first_enrollment(tmp_path):
    bundle = _ready_bundle(tmp_path)

    class DirectoryIdentity:
        def resolve_identity(self, **kwargs):
            return {
                "success": True,
                "data": {
                    "candidate": {
                        "username": "jdoe",
                        "official_name": "Jane Doe",
                    }
                },
            }

        def get_verified_profile(self, **kwargs):
            return {
                "person_id": "person_jdoe",
                "official_name": "Jane Doe",
                "display_name": "Jane Doe",
                "username": "jdoe",
            }

        def person_profile(self, person_id):
            return None

    resolved = commands._resolve_bundle_identity(
        DirectoryIdentity(), bundle=bundle, site_code="BOS3"
    )

    assert resolved["person_id"] == "person_jdoe"
    assert resolved["site_code"] == "BOS3"


@pytest.mark.parametrize(
    "search_result, message",
    [
        (
            {
                "recognized": False,
                "reason": "margin_too_small",
                "threshold": 0.6,
                "candidates": [
                    {"person_id": "person_other", "score": 0.95},
                    {"person_id": "person_jdoe", "score": 0.94},
                ],
            },
            "strongly matches person_other",
        ),
        (
            {"recognized": False, "reason": "no_match", "candidates": []},
            "conflict search was malformed",
        ),
    ],
)
def test_push_aborts_on_ambiguous_or_malformed_conflict_search(
    monkeypatch, tmp_path, search_result, message
):
    bundle = _ready_bundle(tmp_path)
    fake = FakeIdentityMemory(face_exists=False, voice_exists=True)
    _install_push_fakes(monkeypatch, fake)
    fake.search_face = lambda **kwargs: search_result
    answers = iter(["1", "Jane Doe"])

    with pytest.raises(RuntimeError, match=message):
        commands.push_local_bundle(
            _args(tmp_path), input_fn=lambda _prompt: next(answers)
        )

    loaded = load_bundle(bundle.path, output_root=tmp_path)
    assert loaded.modality_states["face"] == "failed"
    assert not any(name.startswith("enroll") for name, _payload in fake.calls)


def test_push_rejects_mismatched_enrollment_person(monkeypatch, tmp_path):
    bundle = _ready_bundle(tmp_path)
    fake = FakeIdentityMemory(face_exists=False, voice_exists=True)
    _install_push_fakes(monkeypatch, fake)

    def mismatched_enroll(**kwargs):
        fake.calls.append(("enroll_face", kwargs))
        return {
            "saved": True,
            "status": "saved",
            "reason": "saved",
            "person_id": "person_other",
            "reference_id": "face-ref",
        }

    fake.enroll_face_reference = mismatched_enroll
    answers = iter(["1", "Jane Doe"])

    with pytest.raises(RuntimeError, match="person_other"):
        commands.push_local_bundle(
            _args(tmp_path), input_fn=lambda _prompt: next(answers)
        )

    loaded = load_bundle(bundle.path, output_root=tmp_path)
    assert loaded.modality_states["face"] == "failed"


def test_bundle_label_distinguishes_capture_state_and_uuid(tmp_path):
    collecting = create_bundle(tmp_path, person_name="Collecting Person")
    label = commands.bundle_label(collecting)

    assert "capture=collecting" in label
    assert f"id={collecting.bundle_id}" in label


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"consent_status": "consented"}, "not active"),
        ({"status": "active", "consent_status": "opted_out"}, "consent status"),
    ],
)
def test_existing_profile_eligibility_is_fail_closed(payload, message):
    with pytest.raises(RuntimeError, match=message):
        commands._validate_enrollable_profile(payload, person_id="person_jdoe")

    commands._validate_enrollable_profile(
        {"status": "active", "consent_status": ""},
        person_id="person_jdoe",
    )


def test_cleanup_deletes_only_upload_complete_selected_bundle(tmp_path):
    bundle = _ready_bundle(tmp_path)
    bundle = update_upload_state(bundle, "face", "uploaded")
    bundle = update_upload_state(bundle, "voice", "skipped")
    answers = iter(["1", "DELETE Jane Doe"])

    assert commands.cleanup_local_bundle(
        _args(tmp_path), input_fn=lambda _prompt: next(answers)
    ) == 0
    assert not bundle.path.exists()
