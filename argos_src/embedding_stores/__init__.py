"""Embedding stores keyed by identity person_id."""

from argos_src.embedding_stores.face_store import FaceEmbeddingStore
from argos_src.embedding_stores.speaker_store import SpeakerEmbeddingStore

__all__ = ["FaceEmbeddingStore", "SpeakerEmbeddingStore"]
