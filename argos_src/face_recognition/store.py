"""Identity-aware face recognition store."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from argos_src.embedding_stores.face_store import FaceEmbeddingStore
from argos_src.face_recognition.constants import DEFAULT_FACE_DB_PATH
from argos_src.identity.constants import DEFAULT_IDENTITY_DB_PATH
from argos_src.identity.store import IdentityStore

logger = logging.getLogger(__name__)


def _default_identity_path_for_face_db(face_db_path: str | Path) -> str:
    resolved_face_path = Path(face_db_path).expanduser().resolve()
    if resolved_face_path == Path(DEFAULT_FACE_DB_PATH).expanduser().resolve():
        return DEFAULT_IDENTITY_DB_PATH
    return str((resolved_face_path.parent / "identity.sqlite3").resolve())


class FaceRecognitionStore:
    """Coordinates face embeddings with canonical Argos identities."""

    EMBEDDING_DIM = FaceEmbeddingStore.EMBEDDING_DIM

    def __init__(
        self,
        db_path: str | Path = DEFAULT_FACE_DB_PATH,
        *,
        identity_store: IdentityStore | None = None,
        identity_db_path: str | Path | None = None,
        embedding_store: FaceEmbeddingStore | None = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.embedding_store = embedding_store or FaceEmbeddingStore(db_path=self.db_path)
        self.identity_store = identity_store or IdentityStore(
            identity_db_path or _default_identity_path_for_face_db(self.db_path)
        )
        logger.info("Initialized face embedding store at %s", self.db_path)
        logger.info("Initialized identity store at %s", self.identity_store.db_path)

    @property
    def collection(self):
        """Expose the underlying Chroma collection for operational counts."""
        return self.embedding_store.collection

    def add_person(
        self,
        name: str,
        face_embedding: np.ndarray,
        person_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        normalized = np.asarray(face_embedding, dtype=np.float32).reshape(-1)
        if normalized.shape[0] != self.EMBEDDING_DIM:
            raise ValueError(
                f"Expected embedding dimension {self.EMBEDDING_DIM}, got {normalized.shape[0]}"
            )
        rendered_person_id = self.identity_store.create_person(
            name=name,
            person_id=person_id,
            metadata=metadata,
        )
        try:
            self.embedding_store.add_embedding(
                person_id=rendered_person_id,
                embedding=normalized,
                metadata={"person_id": rendered_person_id},
            )
        except Exception:
            self.identity_store.delete_person(rendered_person_id)
            raise
        logger.info("Added person '%s' with ID %s", name, rendered_person_id)
        return rendered_person_id

    def recognize_face(
        self,
        face_embedding: np.ndarray,
        threshold: float = 0.6,
        top_k: int = 1,
    ) -> list[dict[str, Any]]:
        results = self.embedding_store.query(embedding=face_embedding, top_k=top_k)
        matches: list[dict[str, Any]] = []
        if results["ids"] and results["ids"][0]:
            for i, person_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity = 1.0 - (distance * distance / 2.0)
                if similarity < threshold:
                    continue
                record = self.identity_store.get_person(person_id)
                if record is None:
                    logger.warning(
                        "Face embedding exists without identity row person_id=%s",
                        person_id,
                    )
                    continue
                metadata = dict(record.get("metadata") or {})
                matches.append(
                    {
                        "person_id": person_id,
                        "name": str(record.get("name") or metadata.get("name") or person_id),
                        "similarity": similarity,
                        "metadata": metadata,
                    }
                )
        return matches

    def update_interaction(self, person_id: str) -> dict[str, Any] | None:
        metadata = self.identity_store.update_interaction(person_id)
        if metadata is None:
            logger.warning("Person ID %s not found in identity store", person_id)
            return None
        logger.info(
            "Updated interaction for %s (count: %s)",
            metadata.get("name", person_id),
            metadata.get("interaction_count", 0),
        )
        return dict(metadata)

    def get_person(self, person_id: str) -> dict[str, Any] | None:
        record = self.embedding_store.get_embedding(person_id)
        identity = self.identity_store.get_person(person_id)
        if record is None or identity is None:
            return None
        return {
            "person_id": person_id,
            "metadata": dict(identity.get("metadata") or {}),
            "embedding": record.get("embedding"),
        }

    def list_all_people(self) -> list[dict[str, Any]]:
        people: list[dict[str, Any]] = []
        for record in self.embedding_store.list_embeddings():
            person_id = str(record.get("person_id") or "").strip()
            if not person_id:
                continue
            identity = self.identity_store.get_person(person_id)
            if identity is None:
                continue
            metadata = dict(identity.get("metadata") or {})
            people.append(
                {
                    "person_id": person_id,
                    "name": str(identity.get("name") or metadata.get("name") or person_id),
                    "metadata": metadata,
                }
            )
        people.sort(key=lambda item: (str(item["name"]).casefold(), item["person_id"]))
        return people

    def delete_person(self, person_id: str) -> bool:
        face_deleted = self.embedding_store.delete_embedding(person_id)
        identity_deleted = self.identity_store.delete_person(person_id)
        return face_deleted or identity_deleted

    def reset(self) -> None:
        person_ids = [
            str(record.get("person_id") or "").strip()
            for record in self.embedding_store.list_embeddings()
            if str(record.get("person_id") or "").strip()
        ]
        self.embedding_store.reset()
        for person_id in person_ids:
            self.identity_store.delete_person(person_id)
