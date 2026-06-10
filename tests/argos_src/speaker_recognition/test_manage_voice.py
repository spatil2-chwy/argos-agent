from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

pytest.importorskip("chromadb")

from argos_src.embedding_stores.speaker_store import SpeakerEmbeddingStore
from argos_src.speaker_recognition.manage_voice import (
    list_voice_references,
    show_voice_reference,
)


def test_speaker_embedding_store_list_get_delete_reference(tmp_path):
    db = SpeakerEmbeddingStore(db_path=tmp_path / "speaker_db")
    db.upsert_reference(
        person_id="person-1",
        embedding=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        model_name="test-model",
        query_duration_s=3.2,
        rms_level=500.0,
    )

    listed = db.list_all_references()
    record = db.get_reference("person-1")

    assert len(listed) == 1
    assert listed[0]["person_id"] == "person-1"
    assert record is not None
    assert record["person_id"] == "person-1"
    assert record["metadata"]["model_name"] == "test-model"
    assert record["embedding"].shape == (3,)
    assert db.delete_reference("person-1") is True
    assert db.has_reference("person-1") is False


def test_list_voice_references_humanizes_person_id_without_face_name(tmp_path):
    speaker_db = SpeakerEmbeddingStore(db_path=tmp_path / "speaker_db")
    speaker_db.upsert_reference(
        person_id="person_sakshee_patil_20260504_152002",
        embedding=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        model_name="test-model",
        query_duration_s=7.4,
        rms_level=2980.2,
    )

    captured = io.StringIO()
    with redirect_stdout(captured):
        list_voice_references(
            speaker_db_path=str(tmp_path / "speaker_db"),
            identity_db_path=str(tmp_path / "missing_identity.sqlite3"),
        )

    rendered = captured.getvalue()
    assert "Name: Sakshee Patil" in rendered
    assert "ID:           person_sakshee_patil_20260504_152002" in rendered


def test_show_voice_reference_resolves_humanized_name_without_face_db(tmp_path):
    speaker_db = SpeakerEmbeddingStore(db_path=tmp_path / "speaker_db")
    speaker_db.upsert_reference(
        person_id="person_sakshee_patil_20260504_152002",
        embedding=np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
        model_name="test-model",
        query_duration_s=4.5,
        rms_level=700.0,
    )

    captured = io.StringIO()
    with redirect_stdout(captured):
        exit_code = show_voice_reference(
            "Sakshee Patil",
            speaker_db_path=str(tmp_path / "speaker_db"),
            identity_db_path=str(tmp_path / "missing_identity.sqlite3"),
        )

    rendered = captured.getvalue()
    assert exit_code == 0
    assert "Voice Reference: Sakshee Patil" in rendered
