from __future__ import annotations

import importlib.util

import numpy as np
import pytest


HAS_CHROMADB = importlib.util.find_spec("chromadb") is not None


def _normalized_embedding() -> np.ndarray:
    embedding = np.random.default_rng(seed=11).random(512)
    return embedding / np.linalg.norm(embedding)


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb is not installed")
def test_face_recognition_store_adds_identity_and_face_embedding(tmp_path):
    from argos_src.face_recognition.store import FaceRecognitionStore

    db = FaceRecognitionStore(db_path=tmp_path / "faces_db")
    embedding = _normalized_embedding()
    person_id = db.add_person(
        name="Test Person",
        face_embedding=embedding,
        metadata={"official_name": "Test Person", "username": "tperson"},
    )

    identity = db.identity_store.get_person(person_id)
    face = db.embedding_store.get_embedding(person_id)

    assert identity is not None
    assert identity["name"] == "Test Person"
    assert identity["metadata"]["username"] == "tperson"
    assert face is not None


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb is not installed")
def test_face_recognition_store_recognize_face_uses_identity_name(tmp_path):
    from argos_src.face_recognition.store import FaceRecognitionStore

    db = FaceRecognitionStore(db_path=tmp_path / "faces_db")
    embedding = _normalized_embedding()
    person_id = db.add_person(name="Test Person", face_embedding=embedding)

    matches = db.recognize_face(embedding, threshold=0.5)

    assert matches
    assert matches[0]["person_id"] == person_id
    assert matches[0]["name"] == "Test Person"


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb is not installed")
def test_face_recognition_store_recognize_face_returns_empty_when_database_is_empty(tmp_path):
    from argos_src.face_recognition.store import FaceRecognitionStore

    db = FaceRecognitionStore(db_path=tmp_path / "faces_db")
    matches = db.recognize_face(_normalized_embedding(), threshold=0.5)

    assert matches == []
