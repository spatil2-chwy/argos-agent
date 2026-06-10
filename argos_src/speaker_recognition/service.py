"""Speaker recognition orchestration for Argos turn ownership and enrollment."""

from __future__ import annotations

import logging

import numpy as np

from argos_src.speaker_recognition.backend import (
    SpeakerEmbeddingBackend,
    SpeechBrainEcapaBackend,
)
from argos_src.speaker_recognition.constants import DEFAULT_SPEAKER_DB_PATH
from argos_src.embedding_stores.speaker_store import SpeakerEmbeddingStore
from argos_src.speaker_recognition.models import (
    SpeakerRecognitionPolicy,
    SpeakerResolutionResult,
    VoiceEnrollmentResult,
)
from argos_src.speaker_recognition.policy import (
    clip_stats,
    enrollment_rejection_reason,
    is_query_clip_safe,
    resolve_owner_id,
    trim_voice_activity,
)


logger = logging.getLogger(__name__)


class SpeakerRecognitionService:
    """Owns speaker embedding lookup, ownership resolution, and voice enrollment."""

    def __init__(
        self,
        *,
        policy: SpeakerRecognitionPolicy,
        backend: SpeakerEmbeddingBackend | None = None,
        speaker_db: SpeakerEmbeddingStore | None = None,
    ) -> None:
        self.policy = policy
        self.backend = backend or SpeechBrainEcapaBackend()
        self.db = speaker_db or SpeakerEmbeddingStore(
            db_path=policy.db_path or DEFAULT_SPEAKER_DB_PATH
        )
        self._logged_no_references_notice = False
        logger.info(
            "Speaker recognition initialized backend=%s db_path=%s "
            "match_threshold=%.3f reference_update_threshold=%.3f",
            self.policy.backend,
            self.policy.db_path or DEFAULT_SPEAKER_DB_PATH,
            self.policy.query_match_threshold,
            self.policy.reference_update_threshold,
        )

    def shutdown(self) -> None:
        return None

    def prewarm(self) -> None:
        """Load backend assets eagerly so the first live turn does not pay setup cost."""
        prewarm_fn = getattr(self.backend, "prewarm", None)
        if callable(prewarm_fn):
            logger.info("Prewarming speaker backend=%s", self.policy.backend)
            prewarm_fn()
            logger.info("Speaker backend prewarm complete backend=%s", self.policy.backend)
            return
        logger.info(
            "Speaker backend=%s does not expose a prewarm hook; skipping eager warmup.",
            self.policy.backend,
        )

    def has_reference(self, person_id: str) -> bool:
        return self.db.has_reference(person_id)

    def resolve_turn_owner(
        self,
        *,
        audio_pcm16: bytes,
        primary_face_person_id: str | None,
        visible_face_person_ids: tuple[str, ...] | list[str] | None = None,
    ) -> SpeakerResolutionResult:
        waveform = np.frombuffer(audio_pcm16 or b"", dtype=np.int16).copy()
        if waveform.size <= 0:
            return resolve_owner_id(
                policy=self.policy,
                primary_face_person_id=primary_face_person_id,
                audio_speaker_id=None,
                top_score=0.0,
                runner_up_score=0.0,
                visible_face_person_ids=visible_face_person_ids,
            )
        if not is_query_clip_safe(self.policy, audio_pcm16=waveform):
            return resolve_owner_id(
                policy=self.policy,
                primary_face_person_id=primary_face_person_id,
                audio_speaker_id=None,
                top_score=0.0,
                runner_up_score=0.0,
                visible_face_person_ids=visible_face_person_ids,
            )
        references = self.db.get_reference_embeddings()
        if not references:
            if not self._logged_no_references_notice:
                logger.info(
                    "Speaker recognition has no enrolled voice references yet; using strict face ownership until a voice reference is saved."
                )
                self._logged_no_references_notice = True
            return resolve_owner_id(
                policy=self.policy,
                primary_face_person_id=primary_face_person_id,
                audio_speaker_id=None,
                top_score=0.0,
                runner_up_score=0.0,
                visible_face_person_ids=visible_face_person_ids,
            )
        query_embedding = self.backend.embed_query_clip(
            waveform,
            sample_rate=16000,
        )
        scored = self.backend.score_against_references(query_embedding, references)
        audio_speaker_id = scored[0][0] if scored else None
        top_score = scored[0][1] if scored else 0.0
        runner_up_score = scored[1][1] if len(scored) > 1 else 0.0
        return resolve_owner_id(
            policy=self.policy,
            primary_face_person_id=primary_face_person_id,
            audio_speaker_id=audio_speaker_id,
            top_score=top_score,
            runner_up_score=runner_up_score,
            visible_face_person_ids=visible_face_person_ids,
        )

    def trim_turn_audio(
        self,
        audio_pcm16: bytes,
        *,
        vad: object | None = None,
    ) -> bytes:
        waveform = np.frombuffer(audio_pcm16 or b"", dtype=np.int16).copy()
        trimmed = trim_voice_activity(waveform, vad=vad)
        return trimmed.astype(np.int16).tobytes()

    def try_store_reference(
        self,
        *,
        person_id: str,
        audio_pcm16: bytes,
        attempt_kind: str,
    ) -> VoiceEnrollmentResult:
        waveform = np.frombuffer(audio_pcm16 or b"", dtype=np.int16).copy()
        rejection = enrollment_rejection_reason(
            self.policy,
            audio_pcm16=waveform,
        )
        if rejection:
            return VoiceEnrollmentResult(
                saved=False,
                reason=rejection,
                person_id=str(person_id or "").strip(),
                attempt_kind=attempt_kind,  # type: ignore[arg-type]
            )
        stats = clip_stats(waveform)
        embedding = self.backend.embed_query_clip(
            waveform,
            sample_rate=16000,
        )
        existing = self.db.get_reference(str(person_id or "").strip())
        if existing is not None and existing.get("embedding") is not None:
            existing_embedding = np.asarray(existing["embedding"], dtype=np.float32).reshape(-1)
            current_embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
            existing_norm = float(np.linalg.norm(existing_embedding))
            current_norm = float(np.linalg.norm(current_embedding))
            if existing_norm <= 1e-8 or current_norm <= 1e-8:
                return VoiceEnrollmentResult(
                    saved=False,
                    reason="reject_inconsistent",
                    person_id=str(person_id or "").strip(),
                    attempt_kind=attempt_kind,  # type: ignore[arg-type]
                )
            consistency = float(
                np.dot(existing_embedding / existing_norm, current_embedding / current_norm)
            )
            if consistency < self.policy.reference_update_threshold:
                return VoiceEnrollmentResult(
                    saved=False,
                    reason="reject_inconsistent",
                    person_id=str(person_id or "").strip(),
                    attempt_kind=attempt_kind,  # type: ignore[arg-type]
                )
            metadata = dict(existing.get("metadata") or {})
            clip_count = max(1, int(metadata.get("clip_count", 1) or 1))
            total_voiced_sec = float(
                metadata.get("total_voiced_sec", metadata.get("query_duration_s", 0.0)) or 0.0
            )
            previous_mean_rms = float(
                metadata.get("mean_rms_level", metadata.get("rms_level", 0.0)) or 0.0
            )
            embedding = ((existing_embedding * clip_count) + current_embedding) / float(
                clip_count + 1
            )
            embedding_norm = float(np.linalg.norm(embedding))
            if embedding_norm > 1e-8:
                embedding = embedding / embedding_norm
            updated_clip_count = clip_count + 1
            updated_total_voiced_sec = total_voiced_sec + stats.duration_s
            updated_mean_rms = (
                (previous_mean_rms * clip_count) + stats.rms_level
            ) / float(updated_clip_count)
        else:
            updated_clip_count = 1
            updated_total_voiced_sec = stats.duration_s
            updated_mean_rms = stats.rms_level
        self.db.upsert_reference(
            person_id=str(person_id or "").strip(),
            embedding=embedding,
            model_name=str(getattr(self.backend, "model_name", "unknown") or "unknown"),
            query_duration_s=stats.duration_s,
            rms_level=stats.rms_level,
            clip_count=updated_clip_count,
            total_voiced_sec=updated_total_voiced_sec,
            mean_rms_level=updated_mean_rms,
        )
        self._logged_no_references_notice = False
        return VoiceEnrollmentResult(
            saved=True,
            reason="saved",
            person_id=str(person_id or "").strip(),
            attempt_kind=attempt_kind,  # type: ignore[arg-type]
        )
