from __future__ import annotations

import pytest

from argos_src.identity_memory.tailwag_package import TailwagPackageIdentityMemoryClient


class _StrictTailwagClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def search_face(self, *, embedding, limit=2, site_code=None):
        self.calls.append(
            ("search_face", {"embedding": embedding, "limit": limit, "site_code": site_code})
        )
        return {
            "candidates": [{"person_id": "person-1", "display_name": "Alex", "score": 0.91}],
            "recognized": True,
            "status": "accepted",
            "reason": "matched",
            "top_score": 0.91,
        }

    def search_voice(self, *, embedding, limit=2, site_code=None):
        self.calls.append(
            ("search_voice", {"embedding": embedding, "limit": limit, "site_code": site_code})
        )
        return {
            "candidates": [{"person_id": "person-1", "display_name": "Alex", "score": 0.87}],
            "recognized": True,
            "status": "accepted",
            "reason": "matched",
            "top_score": 0.87,
        }

    def enroll_face_reference(
        self,
        *,
        person_id,
        embedding,
        metadata=None,
        consent_status="consented",
    ):
        self.calls.append(
            (
                "enroll_face_reference",
                {
                    "person_id": person_id,
                    "embedding": embedding,
                    "metadata": dict(metadata or {}),
                    "consent_status": consent_status,
                },
            )
        )
        return {
            "saved": True,
            "status": "saved",
            "reason": "saved",
            "person_id": person_id,
            "reference_id": "face-ref-1",
        }

    def enroll_voice_reference(
        self,
        *,
        person_id,
        embedding,
        metadata=None,
        consent_status="consented",
    ):
        self.calls.append(
            (
                "enroll_voice_reference",
                {
                    "person_id": person_id,
                    "embedding": embedding,
                    "metadata": dict(metadata or {}),
                    "consent_status": consent_status,
                },
            )
        )
        return {
            "saved": True,
            "status": "saved",
            "reason": "saved",
            "person_id": person_id,
            "reference_id": "voice-ref-1",
        }

    def observe_face_embedding(self, *, person_id, embedding, evidence, metadata=None):
        self.calls.append(
            (
                "observe_face_embedding",
                {
                    "person_id": person_id,
                    "embedding": embedding,
                    "evidence": dict(evidence or {}),
                    "metadata": dict(metadata or {}),
                },
            )
        )
        return {
            "accepted": True,
            "status": "accepted",
            "reason": "accepted",
            "person_id": person_id,
            "modality": "face",
        }

    def observe_voice_embedding(self, *, person_id, embedding, evidence, metadata=None):
        self.calls.append(
            (
                "observe_voice_embedding",
                {
                    "person_id": person_id,
                    "embedding": embedding,
                    "evidence": dict(evidence or {}),
                    "metadata": dict(metadata or {}),
                },
            )
        )
        return {
            "accepted": True,
            "status": "accepted",
            "reason": "accepted",
            "person_id": person_id,
            "modality": "voice",
        }


def test_tailwag_search_calls_match_package_contract():
    raw_client = _StrictTailwagClient()
    client = TailwagPackageIdentityMemoryClient(
        client_factory=lambda: raw_client,
        site_code="BOS3",
    )

    face = client.search_face(embedding=(0.1, 0.2), limit=3)
    voice = client.search_voice(embedding=(0.3, 0.4), limit=4)

    assert face.recognized is True
    assert face.top_score == 0.91
    assert voice.recognized is True
    assert voice.top_score == 0.87
    assert [name for name, _payload in raw_client.calls] == ["search_face", "search_voice"]
    face_call = raw_client.calls[0][1]
    voice_call = raw_client.calls[1][1]
    assert face_call["embedding"] == pytest.approx([0.1, 0.2])
    assert face_call["limit"] == 3
    assert face_call["site_code"] == "BOS3"
    assert voice_call["embedding"] == pytest.approx([0.3, 0.4])
    assert voice_call["limit"] == 4
    assert voice_call["site_code"] == "BOS3"


def test_tailwag_biometric_write_calls_match_package_contract():
    raw_client = _StrictTailwagClient()
    client = TailwagPackageIdentityMemoryClient(client_factory=lambda: raw_client)

    face = client.enroll_face_reference(
        person_id="person-1",
        embedding=(0.1, 0.2),
        metadata={"source": "test"},
    )
    voice = client.enroll_voice_reference(
        person_id="person-1",
        embedding=(0.3, 0.4),
        metadata={"source": "test"},
    )
    face_update = client.observe_face_embedding(
        person_id="person-1",
        embedding=(0.5, 0.6),
        evidence={"owner_source": "audio_face_agree"},
    )
    voice_update = client.observe_voice_embedding(
        person_id="person-1",
        embedding=(0.7, 0.8),
        evidence={"owner_source": "audio_face_agree"},
    )

    assert face.saved is True
    assert voice.saved is True
    assert face_update.accepted is True
    assert voice_update.accepted is True
    assert [name for name, _payload in raw_client.calls] == [
        "enroll_face_reference",
        "enroll_voice_reference",
        "observe_face_embedding",
        "observe_voice_embedding",
    ]
