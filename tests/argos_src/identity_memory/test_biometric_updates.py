from __future__ import annotations

from types import SimpleNamespace

from argos_src.identity_memory.biometric_updates import (
    AdaptiveBiometricObservation,
    AdaptiveBiometricUpdateCoordinator,
)


class _FakeIdentityMemory:
    def __init__(self, results):
        self.results = list(results)
        self.face_calls = []
        self.voice_calls = []

    def observe_face_embedding(self, **kwargs):
        self.face_calls.append(kwargs)
        return self.results.pop(0)

    def observe_voice_embedding(self, **kwargs):
        self.voice_calls.append(kwargs)
        return self.results.pop(0)


def test_coordinator_suppresses_after_tailwag_reports_complete():
    memory = _FakeIdentityMemory(
        [
            SimpleNamespace(
                accepted=False,
                status="complete",
                reason="sample_target_reached",
                sample_count=5,
                target_sample_count=5,
                similarity=1.0,
            )
        ]
    )
    coordinator = AdaptiveBiometricUpdateCoordinator(memory, cooldown_sec=0.0)
    observation = AdaptiveBiometricObservation(
        modality="face",
        person_id="person_alice",
        embedding=[1.0, 0.0],
        model="facenet-vggface2",
        evidence={"owner_id": "person_alice"},
    )

    coordinator.submit_sync(observation)
    coordinator.submit(observation)
    coordinator.close()

    assert len(memory.face_calls) == 1


def test_coordinator_routes_voice_observation():
    memory = _FakeIdentityMemory(
        [
            SimpleNamespace(
                accepted=True,
                status="updated",
                reason="updated",
                sample_count=2,
                target_sample_count=5,
                similarity=0.91,
            )
        ]
    )
    coordinator = AdaptiveBiometricUpdateCoordinator(memory, cooldown_sec=0.0)

    result = coordinator.submit_sync(
        AdaptiveBiometricObservation(
            modality="voice",
            person_id="person_alice",
            embedding=[0.1, 0.9],
            model="ecapa",
            evidence={"owner_id": "person_alice"},
            metadata={"source": "turn_audio"},
        )
    )
    coordinator.close()

    assert result is not None
    assert result.accepted is True
    assert memory.voice_calls[0]["person_id"] == "person_alice"
    assert memory.voice_calls[0]["metadata"] == {"source": "turn_audio"}
