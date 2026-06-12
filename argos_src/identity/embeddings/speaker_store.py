"""Chroma-backed speaker embedding store keyed by identity person_id."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
import numpy as np

from argos_src.speaker_recognition.constants import DEFAULT_SPEAKER_DB_PATH


class SpeakerEmbeddingStore:
    """Store one normalized speaker-reference embedding per person_id."""

    COLLECTION_NAME = "speaker_reference_embeddings"

    def __init__(self, db_path: str | Path = DEFAULT_SPEAKER_DB_PATH):
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
            metadata={"description": "Speaker reference embeddings keyed by Argos person_id"},
        )

    def has_reference(self, person_id: str) -> bool:
        rendered = str(person_id or "").strip()
        if not rendered:
            return False
        result = self.collection.get(ids=[rendered], include=[])
        return bool(result.get("ids"))

    def upsert_reference(
        self,
        *,
        person_id: str,
        embedding: np.ndarray,
        model_name: str,
        query_duration_s: float,
        rms_level: float,
        clip_count: int = 1,
        total_voiced_sec: float | None = None,
        mean_rms_level: float | None = None,
    ) -> None:
        rendered_person_id = str(person_id or "").strip()
        if not rendered_person_id:
            raise ValueError("person_id is required")
        normalized = np.asarray(embedding, dtype=np.float32).reshape(-1)
        existing = self.get_reference_metadata(rendered_person_id) or {}
        created_at = str(existing.get("created_at") or datetime.now(timezone.utc).isoformat())
        updated_at = datetime.now(timezone.utc).isoformat()
        rendered_clip_count = max(1, int(clip_count or 1))
        rendered_total_voiced_sec = (
            float(total_voiced_sec)
            if total_voiced_sec is not None
            else float(query_duration_s)
        )
        rendered_mean_rms = (
            float(mean_rms_level)
            if mean_rms_level is not None
            else float(rms_level)
        )
        metadata = {
            "person_id": rendered_person_id,
            "model_name": str(model_name or "").strip(),
            "created_at": created_at,
            "last_updated_at": updated_at,
            "query_duration_s": float(query_duration_s),
            "total_voiced_sec": rendered_total_voiced_sec,
            "clip_count": rendered_clip_count,
            "rms_level": float(rms_level),
            "mean_rms_level": rendered_mean_rms,
        }
        self.collection.upsert(
            ids=[rendered_person_id],
            embeddings=[normalized.tolist()],
            metadatas=[metadata],
        )

    def get_reference_embeddings(
        self,
        person_ids: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, np.ndarray]:
        if person_ids:
            cleaned_ids = [
                str(person_id or "").strip()
                for person_id in person_ids
                if str(person_id or "").strip()
            ]
            if not cleaned_ids:
                return {}
            result = self.collection.get(ids=cleaned_ids, include=["embeddings"])
        else:
            result = self.collection.get(include=["embeddings"])
        ids = list(result.get("ids", []) or [])
        embeddings = list(result.get("embeddings", []) or [])
        payload: dict[str, np.ndarray] = {}
        for person_id, embedding in zip(ids, embeddings):
            if embedding is None:
                continue
            payload[str(person_id)] = np.asarray(embedding, dtype=np.float32)
        return payload

    def get_reference_metadata(self, person_id: str) -> dict[str, Any] | None:
        rendered = str(person_id or "").strip()
        if not rendered:
            return None
        result = self.collection.get(ids=[rendered], include=["metadatas"])
        metadatas = list(result.get("metadatas", []) or [])
        if not metadatas:
            return None
        metadata = metadatas[0]
        return dict(metadata) if isinstance(metadata, dict) else None

    def get_reference(self, person_id: str) -> dict[str, Any] | None:
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
        metadata = metadatas[0] if metadatas else None
        return {
            "person_id": ids[0],
            "metadata": dict(metadata) if isinstance(metadata, dict) else {},
            "embedding": (
                np.asarray(embedding, dtype=np.float32)
                if embedding is not None
                else None
            ),
        }

    def list_all_references(self) -> list[dict[str, Any]]:
        result = self.collection.get(include=["metadatas"])
        ids = list(result.get("ids", []) or [])
        metadatas = list(result.get("metadatas", []) or [])
        payload: list[dict[str, Any]] = []
        for person_id, metadata in zip(ids, metadatas):
            payload.append(
                {
                    "person_id": str(person_id),
                    "metadata": dict(metadata) if isinstance(metadata, dict) else {},
                }
            )
        return payload

    def delete_reference(self, person_id: str) -> bool:
        rendered = str(person_id or "").strip()
        if not rendered:
            return False
        try:
            self.collection.delete(ids=[rendered])
            return True
        except Exception:
            return False
