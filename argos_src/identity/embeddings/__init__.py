"""Identity-owned embedding stores for recognition modalities."""

from argos_src.identity.embeddings.face_store import FaceEmbeddingStore
from argos_src.identity.embeddings.speaker_store import SpeakerEmbeddingStore

__all__ = ["FaceEmbeddingStore", "SpeakerEmbeddingStore"]
