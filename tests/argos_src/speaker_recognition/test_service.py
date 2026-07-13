from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from argos_src.speaker_recognition.models import SpeakerRecognitionPolicy
from argos_src.speaker_recognition.service import SpeakerRecognitionService


def _pcm16_with_amplitude(amplitude: int, *, duration_s: float) -> bytes:
    samples = max(1, int(16000 * duration_s))
    return np.full(samples, amplitude, dtype=np.int16).tobytes()


class _FakeBackend:
    model_name = "fake-ecapa"

    def __init__(self, *, embeddings=None):
        self.embeddings = list(
            embeddings or [np.asarray([1.0, 0.0, 0.0], dtype=np.float32)]
        )

    def embed_query_clip(self, audio_pcm16, *, sample_rate):
        del audio_pcm16, sample_rate
        if len(self.embeddings) > 1:
            return self.embeddings.pop(0)
        return self.embeddings[0]


class _FakeIdentityMemory:
    def __init__(self, *, search=None):
        self.search = search or SimpleNamespace(
            candidates=(),
            recognized=False,
            status="rejected",
            reason="below_threshold",
            top_score=0.0,
            runner_up_score=0.0,
            margin=0.0,
        )
        self.voice_searches = []
        self.owner_requests = []
        self.enrollments = []
        self.has_voice = set()

    def has_voice_reference(self, person_id):
        return str(person_id or "").strip() in self.has_voice

    def search_voice(self, **kwargs):
        self.voice_searches.append(kwargs)
        return self.search

    def resolve_turn_owner(self, **kwargs):
        self.owner_requests.append(kwargs)
        voice_candidate = kwargs.get("voice_candidate")
        primary_face = kwargs.get("primary_face_candidate") or {}
        policy = kwargs.get("policy_context") or {}
        if voice_candidate is not None:
            person_id = voice_candidate.person_id
            face_id = str(primary_face.get("person_id") or "").strip()
            return SimpleNamespace(
                audio_speaker_id=person_id,
                top_score=policy.get("voice_top_score", 0.0),
                runner_up_score=policy.get("voice_runner_up_score", 0.0),
                margin=policy.get("voice_margin", 0.0),
                speaker_visible=bool(face_id and face_id == person_id),
                owner_id=person_id,
                owner_source="audio_face_agree" if face_id == person_id else "audio",
                owner_confidence=policy.get("voice_top_score", 0.0),
            )
        face_id = str(primary_face.get("person_id") or "").strip() or None
        return SimpleNamespace(
            audio_speaker_id=None,
            top_score=policy.get("voice_top_score", 0.0),
            runner_up_score=policy.get("voice_runner_up_score", 0.0),
            margin=policy.get("voice_margin", 0.0),
            speaker_visible=bool(face_id),
            owner_id=face_id,
            owner_source="face" if face_id else "unknown",
            owner_confidence=0.0,
        )

    def enroll_voice_reference(self, **kwargs):
        self.enrollments.append(kwargs)
        return SimpleNamespace(saved=True, reason="saved", person_id=kwargs["person_id"])


class _FakeAdaptiveCoordinator:
    def __init__(self):
        self.observations = []

    def submit(self, observation):
        self.observations.append(observation)


def test_resolve_turn_owner_uses_tailwag_search_and_face_fallback():
    memory = _FakeIdentityMemory(
        search=SimpleNamespace(
            candidates=(),
            recognized=False,
            status="rejected",
            reason="below_threshold",
            top_score=0.515,
            runner_up_score=0.0,
            margin=0.515,
        )
    )
    service = SpeakerRecognitionService(
        policy=SpeakerRecognitionPolicy(query_match_threshold=0.60),
        backend=_FakeBackend(),
        identity_memory_client=memory,
    )

    result = service.resolve_turn_owner(
        audio_pcm16=_pcm16_with_amplitude(1200, duration_s=1.0),
        primary_face_person_id="alice",
        visible_face_person_ids=("alice",),
    )

    assert "model" not in memory.voice_searches[0]
    assert memory.owner_requests[0]["voice_candidate"] is None
    assert result.audio_speaker_id is None
    assert result.owner_id == "alice"
    assert result.owner_source == "face"
    assert result.top_score == 0.515


def test_resolve_turn_owner_accepts_tailwag_voice_candidate():
    candidate = SimpleNamespace(person_id="alice", display_name="Alice", score=0.82)
    memory = _FakeIdentityMemory(
        search=SimpleNamespace(
            candidates=(candidate,),
            recognized=True,
            status="accepted",
            reason="matched",
            top_score=0.82,
            runner_up_score=0.2,
            margin=0.62,
        )
    )
    service = SpeakerRecognitionService(
        policy=SpeakerRecognitionPolicy(),
        backend=_FakeBackend(),
        identity_memory_client=memory,
    )

    result = service.resolve_turn_owner(
        audio_pcm16=_pcm16_with_amplitude(1200, duration_s=1.0),
        primary_face_person_id=None,
        visible_face_person_ids=(),
    )

    assert memory.owner_requests[0]["voice_candidate"] is candidate
    assert result.audio_speaker_id == "alice"
    assert result.owner_id == "alice"
    assert result.owner_source == "audio"
    assert result.owner_confidence == 0.82


def test_voice_adaptive_update_skips_voice_only_owner():
    candidate = SimpleNamespace(person_id="alice", display_name="Alice", score=0.82)
    memory = _FakeIdentityMemory(
        search=SimpleNamespace(
            candidates=(candidate,),
            recognized=True,
            status="accepted",
            reason="matched",
            top_score=0.82,
            runner_up_score=0.2,
            margin=0.62,
        )
    )
    memory.has_voice.add("alice")
    coordinator = _FakeAdaptiveCoordinator()
    service = SpeakerRecognitionService(
        policy=SpeakerRecognitionPolicy(),
        backend=_FakeBackend(),
        identity_memory_client=memory,
        adaptive_update_coordinator=coordinator,
    )

    service.resolve_turn_owner(
        audio_pcm16=_pcm16_with_amplitude(1200, duration_s=1.0),
        primary_face_person_id=None,
        visible_face_person_ids=(),
    )

    assert coordinator.observations == []


def test_voice_adaptive_update_skips_face_only_owner():
    memory = _FakeIdentityMemory(
        search=SimpleNamespace(
            candidates=(),
            recognized=False,
            status="rejected",
            reason="below_threshold",
            top_score=0.515,
            runner_up_score=0.0,
            margin=0.515,
        )
    )
    memory.has_voice.add("alice")
    coordinator = _FakeAdaptiveCoordinator()
    service = SpeakerRecognitionService(
        policy=SpeakerRecognitionPolicy(),
        backend=_FakeBackend(embeddings=[np.asarray([0.2, 0.8, 0.0], dtype=np.float32)]),
        identity_memory_client=memory,
        adaptive_update_coordinator=coordinator,
    )

    service.resolve_turn_owner(
        audio_pcm16=_pcm16_with_amplitude(1200, duration_s=1.0),
        primary_face_person_id="alice",
        visible_face_person_ids=("alice",),
        face_evidence={
            "face_score_margin": 0.31,
            "recognized_count": 1,
            "unknown_count": 0,
        },
    )

    assert coordinator.observations == []


def test_voice_adaptive_update_uses_audio_face_agreement():
    candidate = SimpleNamespace(person_id="alice", display_name="Alice", score=0.82)
    memory = _FakeIdentityMemory(
        search=SimpleNamespace(
            candidates=(candidate,),
            recognized=True,
            status="accepted",
            reason="matched",
            top_score=0.82,
            runner_up_score=0.2,
            margin=0.62,
        )
    )
    memory.has_voice.add("alice")
    coordinator = _FakeAdaptiveCoordinator()
    service = SpeakerRecognitionService(
        policy=SpeakerRecognitionPolicy(),
        backend=_FakeBackend(embeddings=[np.asarray([0.2, 0.8, 0.0], dtype=np.float32)]),
        identity_memory_client=memory,
        adaptive_update_coordinator=coordinator,
    )

    service.resolve_turn_owner(
        audio_pcm16=_pcm16_with_amplitude(1200, duration_s=1.0),
        primary_face_person_id="alice",
        visible_face_person_ids=("alice",),
        face_evidence={
            "face_score_margin": 0.31,
            "recognized_count": 1,
            "unknown_count": 0,
        },
    )

    assert len(coordinator.observations) == 1
    observation = coordinator.observations[0]
    assert observation.modality == "voice"
    assert observation.person_id == "alice"
    assert observation.evidence["owner_source"] == "audio_face_agree"
    assert observation.evidence["primary_face_person_id"] == "alice"
    assert observation.evidence["audio_speaker_id"] == "alice"
    assert observation.evidence["recognized_count"] == 1
    assert observation.evidence["unknown_count"] == 0
    assert observation.metadata["source"] == "turn_audio"


def test_voice_adaptive_update_skips_when_initial_reference_missing():
    memory = _FakeIdentityMemory(
        search=SimpleNamespace(
            candidates=(),
            recognized=False,
            status="rejected",
            reason="below_threshold",
            top_score=0.515,
            runner_up_score=0.0,
            margin=0.515,
        )
    )
    coordinator = _FakeAdaptiveCoordinator()
    service = SpeakerRecognitionService(
        policy=SpeakerRecognitionPolicy(),
        backend=_FakeBackend(),
        identity_memory_client=memory,
        adaptive_update_coordinator=coordinator,
    )

    service.resolve_turn_owner(
        audio_pcm16=_pcm16_with_amplitude(1200, duration_s=1.0),
        primary_face_person_id="alice",
        visible_face_person_ids=("alice",),
        face_evidence={"face_score_margin": 0.31, "recognized_count": 1, "unknown_count": 0},
    )

    assert coordinator.observations == []


def test_try_store_reference_enrolls_voice_reference_in_tailwag():
    memory = _FakeIdentityMemory()
    service = SpeakerRecognitionService(
        policy=SpeakerRecognitionPolicy(),
        backend=_FakeBackend(embeddings=[np.asarray([1.0, 0.0, 0.0], dtype=np.float32)]),
        identity_memory_client=memory,
    )

    result = service.try_store_reference(
        person_id="alice",
        audio_pcm16=_pcm16_with_amplitude(1200, duration_s=2.5),
        attempt_kind="silent",
    )

    assert result.saved is True
    assert memory.enrollments[0]["person_id"] == "alice"
    assert "model" not in memory.enrollments[0]
    metadata = memory.enrollments[0]["metadata"]
    assert metadata["query_duration_s"] == 2.5
    assert metadata["rms_level"] == 1200.0
    assert metadata["clipped_fraction"] == 0.0
    assert metadata["attempt_kind"] == "silent"
    assert "clip_count" not in metadata
    assert "total_voiced_sec" not in metadata
    assert "mean_rms_level" not in metadata
