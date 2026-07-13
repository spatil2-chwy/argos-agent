from __future__ import annotations

import numpy as np

from scripts.labs import biometric_enrollment_lab as lab


class FakeIdentityMemory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

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

    def observe_face_embedding(self, **kwargs):
        self.calls.append(("observe_face", kwargs))
        return {
            "accepted": True,
            "status": "updated",
            "reason": "updated",
            "person_id": kwargs["person_id"],
            "reference_id": "face-ref",
            "modality": "face",
            "sample_count": len([call for call in self.calls if call[0] == "observe_face"]) + 1,
            "target_sample_count": 5,
        }

    def observe_voice_embedding(self, **kwargs):
        self.calls.append(("observe_voice", kwargs))
        return {
            "accepted": True,
            "status": "updated",
            "reason": "updated",
            "person_id": kwargs["person_id"],
            "reference_id": "voice-ref",
            "modality": "voice",
            "sample_count": len([call for call in self.calls if call[0] == "observe_voice"]) + 1,
            "target_sample_count": 5,
        }


def test_commit_modality_enrolls_once_then_observes_remaining_face_samples() -> None:
    fake = FakeIdentityMemory()
    embeddings = [np.asarray([1.0, float(index)], dtype=np.float32) for index in range(5)]

    result = lab._commit_modality(
        identity_memory=fake,
        modality="face",
        person_id="person_jane",
        embeddings=embeddings,
        metadata={"display_name": "Jane Doe"},
    )

    assert [name for name, _ in fake.calls] == [
        "enroll_face",
        "observe_face",
        "observe_face",
        "observe_face",
        "observe_face",
    ]
    assert result["enrollment"]["saved"] is True
    assert len(result["updates"]) == 4
    assert fake.calls[1][1]["evidence"]["owner_source"] == "audio_face_agree"
    assert fake.calls[1][1]["metadata"]["enrollment_mode"] == "operator_controlled_live"


def test_commit_modality_enrolls_once_then_observes_remaining_voice_samples() -> None:
    fake = FakeIdentityMemory()
    embeddings = [np.asarray([1.0, float(index)], dtype=np.float32) for index in range(5)]

    result = lab._commit_modality(
        identity_memory=fake,
        modality="voice",
        person_id="person_jane",
        embeddings=embeddings,
        metadata={"display_name": "Jane Doe"},
    )

    assert [name for name, _ in fake.calls] == [
        "enroll_voice",
        "observe_voice",
        "observe_voice",
        "observe_voice",
        "observe_voice",
    ]
    assert result["enrollment"]["saved"] is True
    assert len(result["updates"]) == 4
    assert fake.calls[1][1]["evidence"]["audio_speaker_id"] == "person_jane"


def test_slug_person_id_prefers_username_then_email_then_name() -> None:
    assert lab._slug_person_id("Jane Doe", {"username": "jdoe"}) == "person_jdoe"
    assert lab._slug_person_id("Jane Doe", {"employee_email": "jane@example.com"}) == "person_jane"
    assert lab._slug_person_id("Jane A. Doe", {}) == "person_jane_a_doe"
