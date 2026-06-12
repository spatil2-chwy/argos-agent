"""Chroma-backed face embedding store keyed by identity person_id."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
import numpy as np

from argos_src.face_recognition.constants import DEFAULT_FACE_DB_PATH


class FaceEmbeddingStore:
    """Persistent face-vector store.

    Identity lives in :class:`argos_src.identity.store.IdentityStore`; social/context
    memory lives in :class:`argos_src.memory.store.MemoryStore`.
    This class stores only face embeddings and modality-specific metadata.
    """

    EMBEDDING_DIM = 512
    COLLECTION_NAME = "face_embeddings"

    def __init__(self, db_path: str | Path = DEFAULT_FACE_DB_PATH) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Face recognition embeddings keyed by Argos person_id"},
        )

    def count(self) -> int:
        return int(self.collection.count())

    def add_embedding(
        self,
        *,
        person_id: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        rendered_person_id = str(person_id or "").strip()
        if not rendered_person_id:
            raise ValueError("person_id is required")
        normalized = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if normalized.shape[0] != self.EMBEDDING_DIM:
            raise ValueError(
                f"Expected embedding dimension {self.EMBEDDING_DIM}, got {normalized.shape[0]}"
            )
        self.collection.add(
            ids=[rendered_person_id],
            embeddings=[normalized.tolist()],
            metadatas=[dict(metadata or {"person_id": rendered_person_id})],
        )

    def query(
        self,
        *,
        embedding: np.ndarray,
        top_k: int,
    ) -> dict[str, Any]:
        normalized = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if normalized.shape[0] != self.EMBEDDING_DIM:
            raise ValueError(
                f"Expected embedding dimension {self.EMBEDDING_DIM}, got {normalized.shape[0]}"
            )
        available = self.count()
        if available <= 0 or top_k <= 0:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        return self.collection.query(
            query_embeddings=[normalized.tolist()],
            n_results=min(top_k, available),
        )

    def get_embedding(self, person_id: str) -> dict[str, Any] | None:
        rendered = str(person_id or "").strip()
        if not rendered:
            return None
        result = self.collection.get(ids=[rendered], include=["metadatas", "embeddings"])
        ids = list(result.get("ids", []) or [])
        if not ids:
            return None
        embeddings = list(result.get("embeddings", []) or [])
        metadatas = list(result.get("metadatas", []) or [])
        embedding = embeddings[0] if embeddings else None
        return {
            "person_id": ids[0],
            "metadata": dict(metadatas[0]) if metadatas and isinstance(metadatas[0], dict) else {},
            "embedding": (
                np.asarray(embedding, dtype=np.float32)
                if embedding is not None
                else None
            ),
        }

    def list_embeddings(self) -> list[dict[str, Any]]:
        result = self.collection.get(include=["metadatas"])
        ids = list(result.get("ids", []) or [])
        metadatas = list(result.get("metadatas", []) or [])
        return [
            {
                "person_id": str(person_id),
                "metadata": dict(metadata) if isinstance(metadata, dict) else {},
            }
            for person_id, metadata in zip(ids, metadatas)
        ]

    def delete_embedding(self, person_id: str) -> bool:
        rendered = str(person_id or "").strip()
        if not rendered:
            return False
        try:
            self.collection.delete(ids=[rendered])
            return True
        except Exception:
            return False

    def reset(self) -> None:
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Face recognition embeddings keyed by Argos person_id"},
        )
