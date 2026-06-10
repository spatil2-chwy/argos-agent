from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest


HAS_CHROMADB = importlib.util.find_spec("chromadb") is not None


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb is not installed")
def test_unified_delete_removes_identity_face_and_speaker(tmp_path):
    from argos_src.face_recognition.store import FaceRecognitionStore
    from argos_src.identity import IdentityStore
    from argos_src.identity.manage_identity import delete_identity
    from argos_src.embedding_stores.speaker_store import SpeakerEmbeddingStore

    identity_path = tmp_path / "identity.sqlite3"
    face_path = tmp_path / "face_db"
    speaker_path = tmp_path / "speaker_db"

    identity_store = IdentityStore(identity_path)
    face_db = FaceRecognitionStore(db_path=face_path, identity_store=identity_store)
    speaker_db = SpeakerEmbeddingStore(db_path=speaker_path)
    embedding = np.random.default_rng(seed=22).random(FaceRecognitionStore.EMBEDDING_DIM)
    embedding = embedding / np.linalg.norm(embedding)
    person_id = face_db.add_person(name="Sakshee Patil", face_embedding=embedding)
    speaker_db.upsert_reference(
        person_id=person_id,
        embedding=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        model_name="test-model",
        query_duration_s=2.5,
        rms_level=500.0,
    )

    exit_code = delete_identity(
        "Sakshee Patil",
        identity_db_path=str(identity_path),
        face_db_path=str(face_path),
        speaker_db_path=str(speaker_path),
        yes=True,
    )

    assert exit_code == 0
    assert identity_store.get_person(person_id) is None
    assert face_db.get_person(person_id) is None
    assert speaker_db.has_reference(person_id) is False


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb is not installed")
def test_unified_delete_can_remove_orphan_face_embedding_by_person_id(tmp_path):
    from argos_src.embedding_stores.face_store import FaceEmbeddingStore
    from argos_src.face_recognition.store import FaceRecognitionStore
    from argos_src.identity.manage_identity import delete_identity

    identity_path = tmp_path / "identity.sqlite3"
    face_path = tmp_path / "face_db"
    speaker_path = tmp_path / "speaker_db"

    face_store = FaceEmbeddingStore(db_path=face_path)
    person_id = "person_orphan_face_20260507_120000"
    embedding = np.random.default_rng(seed=24).random(FaceRecognitionStore.EMBEDDING_DIM)
    embedding = embedding / np.linalg.norm(embedding)
    face_store.add_embedding(person_id=person_id, embedding=embedding)

    exit_code = delete_identity(
        person_id,
        identity_db_path=str(identity_path),
        face_db_path=str(face_path),
        speaker_db_path=str(speaker_path),
        yes=True,
    )

    assert exit_code == 0
    assert face_store.get_embedding(person_id) is None


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb is not installed")
def test_show_identity_prints_identity_and_modality_metadata(tmp_path, capsys):
    from argos_src.face_recognition.store import FaceRecognitionStore
    from argos_src.identity import IdentityStore
    from argos_src.identity.manage_identity import show_identity

    identity_path = tmp_path / "identity.sqlite3"
    face_path = tmp_path / "face_db"
    speaker_path = tmp_path / "speaker_db"

    identity_store = IdentityStore(identity_path)
    face_db = FaceRecognitionStore(db_path=face_path, identity_store=identity_store)
    embedding = np.random.default_rng(seed=25).random(FaceRecognitionStore.EMBEDDING_DIM)
    embedding = embedding / np.linalg.norm(embedding)
    face_db.add_person(
        name="Sakshee Patil",
        face_embedding=embedding,
        metadata={"username": "spatil2", "business_title": "AI Technologist II"},
    )

    exit_code = show_identity(
        "Sakshee Patil",
        identity_db_path=str(identity_path),
        face_db_path=str(face_path),
        speaker_db_path=str(speaker_path),
    )

    rendered = capsys.readouterr().out
    assert exit_code == 0
    assert "Identity: Sakshee Patil" in rendered
    assert "face embedding:" in rendered
    assert "Identity Metadata:" in rendered
    assert "username: spatil2" in rendered


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb is not installed")
def test_show_identity_json_excludes_memory_payload(tmp_path, capsys):
    from argos_src.face_recognition.store import FaceRecognitionStore
    from argos_src.identity import IdentityStore
    from argos_src.identity.manage_identity import show_identity

    identity_path = tmp_path / "identity.sqlite3"
    face_path = tmp_path / "face_db"
    speaker_path = tmp_path / "speaker_db"

    identity_store = IdentityStore(identity_path)
    face_db = FaceRecognitionStore(db_path=face_path, identity_store=identity_store)
    embedding = np.random.default_rng(seed=26).random(FaceRecognitionStore.EMBEDDING_DIM)
    embedding = embedding / np.linalg.norm(embedding)
    person_id = face_db.add_person(name="Sakshee Patil", face_embedding=embedding)

    exit_code = show_identity(
        person_id,
        identity_db_path=str(identity_path),
        face_db_path=str(face_path),
        speaker_db_path=str(speaker_path),
        as_json=True,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["person_id"] == person_id
    assert payload["modalities"]["face"]["present"] is True
    assert "memory" not in payload
