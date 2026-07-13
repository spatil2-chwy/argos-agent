from __future__ import annotations

import pytest

from argos_src.identity_memory.tailwag_http import TailwagHttpIdentityMemoryClient


class _StrictProviderClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def request(self, *, resource_id, operation, args=None, timeout_ms=None):
        del timeout_ms
        self.calls.append((operation, {"resource_id": resource_id, **dict(args or {})}))
        if operation == "memory.biometrics_face_search":
            return {
                "candidates": [
                    {"person_id": "person-1", "display_name": "Alex", "score": 0.91}
                ],
                "recognized": True,
                "status": "accepted",
                "reason": "matched",
                "top_score": 0.91,
            }
        if operation == "memory.biometrics_voice_search":
            return {
                "candidates": [
                    {"person_id": "person-1", "display_name": "Alex", "score": 0.87}
                ],
                "recognized": True,
                "status": "accepted",
                "reason": "matched",
                "top_score": 0.87,
            }
        if operation in {
            "memory.biometrics_face_references",
            "memory.biometrics_voice_references",
        }:
            modality = "face" if "face" in operation else "voice"
            return {
                "saved": True,
                "status": "saved",
                "reason": "saved",
                "person_id": (args or {}).get("person_id", ""),
                "reference_id": f"{modality}-ref-1",
            }
        if operation in {
            "memory.biometrics_face_observations",
            "memory.biometrics_voice_observations",
        }:
            modality = "face" if "face" in operation else "voice"
            return {
                "accepted": True,
                "status": "accepted",
                "reason": "accepted",
                "person_id": (args or {}).get("person_id", ""),
                "modality": modality,
            }
        if operation in {"memory.identity_verified_profile", "memory.people_profile"}:
            return None
        raise AssertionError(f"Unexpected operation: {operation}")

    def shutdown(self):
        return None


def test_tailwag_search_calls_match_http_provider_contract():
    provider = _StrictProviderClient()
    client = TailwagHttpIdentityMemoryClient(
        provider_client=provider,
        resource_id="memory",
        site_code="BOS3",
    )

    face = client.search_face(embedding=(0.1, 0.2), limit=3)
    voice = client.search_voice(embedding=(0.3, 0.4), limit=4)

    assert face.recognized is True
    assert face.top_score == 0.91
    assert voice.recognized is True
    assert voice.top_score == 0.87
    assert [name for name, _payload in provider.calls] == [
        "memory.biometrics_face_search",
        "memory.biometrics_voice_search",
    ]
    face_call = provider.calls[0][1]
    voice_call = provider.calls[1][1]
    assert face_call["resource_id"] == "memory"
    assert face_call["embedding"] == pytest.approx([0.1, 0.2])
    assert face_call["limit"] == 3
    assert face_call["site_code"] == "BOS3"
    assert voice_call["embedding"] == pytest.approx([0.3, 0.4])
    assert voice_call["limit"] == 4
    assert voice_call["site_code"] == "BOS3"


def test_tailwag_biometric_write_calls_match_http_provider_contract():
    provider = _StrictProviderClient()
    client = TailwagHttpIdentityMemoryClient(
        provider_client=provider,
        resource_id="memory",
    )

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
    assert [name for name, _payload in provider.calls] == [
        "memory.biometrics_face_references",
        "memory.biometrics_voice_references",
        "memory.biometrics_face_observations",
        "memory.biometrics_voice_observations",
    ]


def test_tailwag_optional_profiles_preserve_http_null_results():
    provider = _StrictProviderClient()
    client = TailwagHttpIdentityMemoryClient(
        provider_client=provider,
        resource_id="memory",
    )

    assert client.get_verified_profile(username="missing", official_name="Missing") is None
    assert client.person_profile("person-missing") is None
    assert [name for name, _payload in provider.calls] == [
        "memory.identity_verified_profile",
        "memory.people_profile",
    ]
