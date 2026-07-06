from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

from argos_src.speaker_recognition.models import SpeakerRecognitionPolicy


def _load_service_module(monkeypatch):
    store_mod = types.ModuleType("argos_src.identity.embeddings.speaker_store")
    store_mod.SpeakerEmbeddingStore = object
    monkeypatch.setitem(sys.modules, "argos_src.identity.embeddings.speaker_store", store_mod)

    module_name = "test_argos_speaker_recognition_service_module"
    module_path = (
        Path(__file__).resolve().parents[3]
        / "argos_src/speaker_recognition/service.py"
    )
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _pcm16_with_amplitude(amplitude: int, *, duration_s: float) -> bytes:
    samples = max(1, int(16000 * duration_s))
    return np.full(samples, amplitude, dtype=np.int16).tobytes()


class _FakeBackend:
    model_name = "fake"

    def __init__(self, *, embeddings=None, scores=None):
        self.embeddings = list(embeddings or [np.asarray([1.0, 0.0, 0.0], dtype=np.float32)])
        self.scores = list(scores or [])

    def embed_query_clip(self, audio_pcm16, *, sample_rate):
        del audio_pcm16, sample_rate
        if len(self.embeddings) > 1:
            return self.embeddings.pop(0)
        return self.embeddings[0]

    def score_against_references(self, query_embedding, references):
        del query_embedding, references
        return list(self.scores)


class _FakeSpeakerDb:
    def __init__(self, *, references=None, stored=None):
        self.references = dict(references or {})
        self.stored = dict(stored or {})
        self.upserts = []

    def has_reference(self, person_id):
        return str(person_id or "").strip() in self.stored

    def get_reference_embeddings(self):
        return dict(self.references)

    def get_reference(self, person_id):
        return self.stored.get(str(person_id or "").strip())

    def upsert_reference(self, **kwargs):
        self.upserts.append(kwargs)
        self.stored[kwargs["person_id"]] = {
            "embedding": np.asarray(kwargs["embedding"], dtype=np.float32),
            "metadata": {
                "clip_count": kwargs.get("clip_count", 1),
                "total_voiced_sec": kwargs.get("total_voiced_sec", kwargs["query_duration_s"]),
                "mean_rms_level": kwargs.get("mean_rms_level", kwargs["rms_level"]),
                "query_duration_s": kwargs["query_duration_s"],
                "rms_level": kwargs["rms_level"],
            },
        }


def test_resolve_turn_owner_only_sets_audio_speaker_id_above_threshold(monkeypatch):
    module = _load_service_module(monkeypatch)
    service = module.SpeakerRecognitionService(
        policy=SpeakerRecognitionPolicy(query_match_threshold=0.60),
        backend=_FakeBackend(scores=[("alice", 0.515)]),
        speaker_db=_FakeSpeakerDb(references={"alice": np.asarray([1.0, 0.0, 0.0])}),
    )
    try:
        result = service.resolve_turn_owner(
            audio_pcm16=_pcm16_with_amplitude(1200, duration_s=1.0),
            primary_face_person_id="alice",
            visible_face_person_ids=("alice",),
        )
    finally:
        service.shutdown()

    assert result.audio_speaker_id is None
    assert result.owner_id == "alice"
    assert result.owner_source == "face"
    assert result.top_score == 0.515


def test_resolve_turn_owner_leaves_unresolved_without_face_or_audio_match(monkeypatch):
    module = _load_service_module(monkeypatch)
    service = module.SpeakerRecognitionService(
        policy=SpeakerRecognitionPolicy(query_match_threshold=0.60),
        backend=_FakeBackend(scores=[("alice", 0.515)]),
        speaker_db=_FakeSpeakerDb(references={"alice": np.asarray([1.0, 0.0, 0.0])}),
    )
    try:
        result = service.resolve_turn_owner(
            audio_pcm16=_pcm16_with_amplitude(1200, duration_s=1.0),
            primary_face_person_id=None,
            visible_face_person_ids=(),
        )
    finally:
        service.shutdown()

    assert result.audio_speaker_id is None
    assert result.owner_id is None
    assert result.owner_source == "unknown"


def test_progressive_voice_reference_saves_first_good_clip_immediately(monkeypatch):
    module = _load_service_module(monkeypatch)
    db = _FakeSpeakerDb()
    service = module.SpeakerRecognitionService(
        policy=SpeakerRecognitionPolicy(),
        backend=_FakeBackend(embeddings=[np.asarray([1.0, 0.0, 0.0], dtype=np.float32)]),
        speaker_db=db,
    )
    try:
        result = service.try_store_reference(
            person_id="alice",
            audio_pcm16=_pcm16_with_amplitude(1200, duration_s=2.5),
            attempt_kind="silent",
        )
    finally:
        service.shutdown()

    assert result.saved is True
    assert db.upserts[0]["clip_count"] == 1
    assert db.upserts[0]["total_voiced_sec"] == 2.5


def test_progressive_voice_reference_averages_consistent_clip(monkeypatch):
    module = _load_service_module(monkeypatch)
    db = _FakeSpeakerDb(
        stored={
            "alice": {
                "embedding": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                "metadata": {
                    "clip_count": 1,
                    "total_voiced_sec": 2.5,
                    "mean_rms_level": 1200.0,
                },
            }
        }
    )
    service = module.SpeakerRecognitionService(
        policy=SpeakerRecognitionPolicy(
            reference_update_threshold=0.55,
        ),
        backend=_FakeBackend(embeddings=[np.asarray([0.8, 0.6, 0.0], dtype=np.float32)]),
        speaker_db=db,
    )
    try:
        result = service.try_store_reference(
            person_id="alice",
            audio_pcm16=_pcm16_with_amplitude(1000, duration_s=2.0),
            attempt_kind="silent",
        )
    finally:
        service.shutdown()

    assert result.saved is True
    assert db.upserts[0]["clip_count"] == 2
    assert db.upserts[0]["total_voiced_sec"] == 4.5
    assert abs(float(np.linalg.norm(db.upserts[0]["embedding"])) - 1.0) < 1e-6


def test_progressive_voice_reference_rejects_inconsistent_clip_without_update(monkeypatch):
    module = _load_service_module(monkeypatch)
    db = _FakeSpeakerDb(
        stored={
            "alice": {
                "embedding": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                "metadata": {"clip_count": 1},
            }
        }
    )
    service = module.SpeakerRecognitionService(
        policy=SpeakerRecognitionPolicy(
            reference_update_threshold=0.55,
        ),
        backend=_FakeBackend(embeddings=[np.asarray([0.0, 1.0, 0.0], dtype=np.float32)]),
        speaker_db=db,
    )
    try:
        result = service.try_store_reference(
            person_id="alice",
            audio_pcm16=_pcm16_with_amplitude(1000, duration_s=2.0),
            attempt_kind="silent",
        )
    finally:
        service.shutdown()

    assert result.saved is False
    assert result.reason == "reject_inconsistent"
    assert db.upserts == []
