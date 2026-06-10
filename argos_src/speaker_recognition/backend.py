"""Speaker embedding backend interfaces and SpeechBrain ECAPA implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class SpeakerEmbeddingBackend(Protocol):
    """Minimal backend contract used by the speaker-recognition service."""

    model_name: str

    def embed_query_clip(
        self,
        audio_pcm16: np.ndarray,
        *,
        sample_rate: int,
    ) -> np.ndarray:
        """Return one normalized speaker embedding for the provided audio clip."""

    def score_against_references(
        self,
        query_embedding: np.ndarray,
        references: dict[str, np.ndarray],
    ) -> list[tuple[str, float]]:
        """Return cosine-like similarity scores sorted descending."""


def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
    rendered = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(rendered))
    if norm <= 1e-8:
        raise ValueError("speaker embedding norm must be > 0")
    return rendered / norm


@dataclass
class SpeechBrainEcapaBackend:
    """SpeechBrain ECAPA-TDNN speaker embedding backend."""

    source: str = "speechbrain/spkrec-ecapa-voxceleb"
    savedir: str | None = None
    run_opts: dict | None = None

    def __post_init__(self) -> None:
        self.model_name = "speechbrain_ecapa"
        self._classifier = None

    def _get_classifier(self):
        if self._classifier is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                import torch
            except Exception as exc:  # pragma: no cover - exercised only in live environments
                raise RuntimeError(
                    "SpeechBrain is required for speaker recognition. "
                    "Install the argos runtime dependencies with speechbrain available."
                ) from exc
            run_opts = dict(self.run_opts or {})
            run_opts.setdefault(
                "device",
                "cuda:0" if torch.cuda.is_available() else "cpu",
            )
            self._classifier = EncoderClassifier.from_hparams(
                source=self.source,
                savedir=self.savedir,
                run_opts=run_opts,
            )
        return self._classifier

    def prewarm(self) -> None:
        """Load the ECAPA model early so first live use avoids HF/model startup cost."""
        self._get_classifier()

    def embed_query_clip(
        self,
        audio_pcm16: np.ndarray,
        *,
        sample_rate: int,
    ) -> np.ndarray:
        waveform = np.asarray(audio_pcm16, dtype=np.float32).reshape(-1)
        if waveform.size <= 0:
            raise ValueError("speaker clip must contain at least one sample")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        waveform = waveform / 32768.0
        classifier = self._get_classifier()
        try:
            import torch
        except Exception as exc:  # pragma: no cover - exercised only in live environments
            raise RuntimeError("Torch is required for the SpeechBrain backend.") from exc
        tensor = torch.from_numpy(waveform).unsqueeze(0)
        embedding = classifier.encode_batch(tensor).detach().cpu().numpy().reshape(-1)
        return _normalize_embedding(embedding)

    def score_against_references(
        self,
        query_embedding: np.ndarray,
        references: dict[str, np.ndarray],
    ) -> list[tuple[str, float]]:
        normalized_query = _normalize_embedding(query_embedding)
        scores: list[tuple[str, float]] = []
        for person_id, reference in references.items():
            normalized_reference = _normalize_embedding(reference)
            score = float(np.dot(normalized_query, normalized_reference))
            scores.append((person_id, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores
