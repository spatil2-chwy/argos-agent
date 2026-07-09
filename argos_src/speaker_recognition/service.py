"""Speaker recognition orchestration for Argos turn ownership and enrollment."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from argos_src.identity_memory.biometric_updates import AdaptiveBiometricObservation
from argos_src.speaker_recognition.backend import (
    SpeakerEmbeddingBackend,
    SpeechBrainEcapaBackend,
)
from argos_src.speaker_recognition.models import (
    SpeakerRecognitionPolicy,
    SpeakerResolutionResult,
    VoiceEnrollmentResult,
)
from argos_src.speaker_recognition.policy import (
    clip_stats,
    enrollment_rejection_reason,
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
        identity_memory_client: Any | None = None,
        adaptive_update_coordinator: Any | None = None,
    ) -> None:
        self.policy = policy
        self.backend = backend or SpeechBrainEcapaBackend()
        self.identity_memory_client = identity_memory_client
        self.adaptive_update_coordinator = adaptive_update_coordinator
        self._logged_no_references_notice = False
        logger.info(
            "Speaker recognition initialized backend=%s identity_memory=%s "
            "fallback_match_threshold=%.3f",
            self.policy.backend,
            type(identity_memory_client).__name__ if identity_memory_client is not None else "disabled",
            self.policy.query_match_threshold,
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
        identity_memory = getattr(self, "identity_memory_client", None)
        if identity_memory is None:
            return False
        try:
            return bool(identity_memory.has_voice_reference(str(person_id or "").strip()))
        except Exception:
            logger.exception("Voice reference check failed person_id=%s", person_id)
            return False

    def resolve_turn_owner(
        self,
        *,
        audio_pcm16: bytes,
        primary_face_person_id: str | None,
        visible_face_person_ids: tuple[str, ...] | list[str] | None = None,
        face_evidence: dict[str, Any] | None = None,
        log_fields: dict[str, Any] | None = None,
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
        identity_memory = getattr(self, "identity_memory_client", None)
        if identity_memory is None:
            if not self._logged_no_references_notice:
                logger.info(
                    "Speaker recognition has no identity-memory client; using strict face ownership."
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
        query_embedding = self.backend.embed_query_clip(waveform, sample_rate=16000)
        try:
            search = identity_memory.search_voice(
                embedding=query_embedding,
                model=str(getattr(self.backend, "model_name", self.policy.backend) or self.policy.backend),
                limit=2,
            )
            candidates = tuple(getattr(search, "candidates", ()) or ())
            voice_candidate = candidates[0] if bool(getattr(search, "recognized", False)) and candidates else None
            top_score = float(getattr(search, "top_score", 0.0) or 0.0)
            runner_up_score = float(getattr(search, "runner_up_score", 0.0) or 0.0)
            margin = float(getattr(search, "margin", max(0.0, top_score - runner_up_score)) or 0.0)
            visible_candidates = tuple(
                {"person_id": rendered}
                for rendered in (
                    str(person_id or "").strip()
                    for person_id in (visible_face_person_ids or ())
                )
                if rendered
            )
            owner = identity_memory.resolve_turn_owner(
                primary_face_candidate=(
                    {"person_id": str(primary_face_person_id).strip()}
                    if str(primary_face_person_id or "").strip()
                    else None
                ),
                visible_face_candidates=visible_candidates,
                voice_candidate=voice_candidate,
                policy_context={
                    "voice_top_score": top_score,
                    "voice_runner_up_score": runner_up_score,
                    "voice_margin": margin,
                    "voice_status": str(getattr(search, "status", "") or ""),
                    "voice_reason": str(getattr(search, "reason", "") or ""),
                },
            )
            resolution = SpeakerResolutionResult(
                audio_speaker_id=getattr(owner, "audio_speaker_id", None),
                top_score=float(getattr(owner, "top_score", top_score) or 0.0),
                runner_up_score=float(getattr(owner, "runner_up_score", runner_up_score) or 0.0),
                margin=float(getattr(owner, "margin", margin) or 0.0),
                speaker_visible=bool(getattr(owner, "speaker_visible", False)),
                owner_id=getattr(owner, "owner_id", None),
                owner_source=str(getattr(owner, "owner_source", "unknown") or "unknown"),  # type: ignore[arg-type]
                owner_confidence=float(getattr(owner, "owner_confidence", 0.0) or 0.0),
            )
            self._maybe_submit_adaptive_voice_observation(
                resolution=resolution,
                query_embedding=query_embedding,
                waveform=waveform,
                primary_face_person_id=primary_face_person_id,
                visible_face_person_ids=visible_face_person_ids,
                face_evidence=face_evidence,
                log_fields=log_fields,
            )
            return resolution
        except Exception:
            logger.exception("Tailwag speaker owner resolution failed; falling back to face-only ownership.")
            return resolve_owner_id(
                policy=self.policy,
                primary_face_person_id=primary_face_person_id,
                audio_speaker_id=None,
                top_score=0.0,
                runner_up_score=0.0,
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
        identity_memory = getattr(self, "identity_memory_client", None)
        if identity_memory is None:
            return VoiceEnrollmentResult(
                saved=False,
                reason="identity_memory_unavailable",
                person_id=str(person_id or "").strip(),
                attempt_kind=attempt_kind,  # type: ignore[arg-type]
            )
        result = identity_memory.enroll_voice_reference(
            person_id=str(person_id or "").strip(),
            embedding=embedding,
            model=str(getattr(self.backend, "model_name", "unknown") or "unknown"),
            metadata={
                "query_duration_s": stats.duration_s,
                "rms_level": stats.rms_level,
                "clipped_fraction": stats.clipped_fraction,
                "attempt_kind": attempt_kind,
            },
            consent_status="consented",
        )
        self._logged_no_references_notice = False
        return VoiceEnrollmentResult(
            saved=bool(getattr(result, "saved", False)),
            reason=str(getattr(result, "reason", "") or ("saved" if getattr(result, "saved", False) else "save_failed")),
            person_id=str(person_id or "").strip(),
            attempt_kind=attempt_kind,  # type: ignore[arg-type]
        )

    def _maybe_submit_adaptive_voice_observation(
        self,
        *,
        resolution: SpeakerResolutionResult,
        query_embedding: Any,
        waveform: np.ndarray,
        primary_face_person_id: str | None,
        visible_face_person_ids: tuple[str, ...] | list[str] | None,
        face_evidence: dict[str, Any] | None,
        log_fields: dict[str, Any] | None,
    ) -> None:
        coordinator = getattr(self, "adaptive_update_coordinator", None)
        owner_id = str(getattr(resolution, "owner_id", "") or "").strip()
        owner_source = str(getattr(resolution, "owner_source", "") or "").strip()
        if coordinator is None or not owner_id or owner_source != "audio_face_agree":
            return
        if enrollment_rejection_reason(self.policy, audio_pcm16=waveform):
            return
        if not self.has_reference(owner_id):
            return
        stats = clip_stats(waveform)
        face_payload = dict(face_evidence or {})
        visible_ids = tuple(
            rendered
            for rendered in (
                str(person_id or "").strip()
                for person_id in (visible_face_person_ids or ())
            )
            if rendered
        )
        primary_face_id = str(primary_face_person_id or "").strip()
        evidence = {
            **face_payload,
            "owner_id": owner_id,
            "owner_source": owner_source,
            "primary_face_person_id": primary_face_id,
            "audio_speaker_id": str(getattr(resolution, "audio_speaker_id", "") or "").strip(),
            "voice_score": float(getattr(resolution, "top_score", 0.0) or 0.0),
            "voice_margin": float(getattr(resolution, "margin", 0.0) or 0.0),
            "audio_score_margin": float(getattr(resolution, "margin", 0.0) or 0.0),
            "visible_face_count": len(visible_ids),
            "recognized_count": _safe_int(
                face_payload.get("recognized_count"),
                default=len(set(visible_ids)),
            ),
            "unknown_count": _safe_int(face_payload.get("unknown_count")),
        }
        coordinator.submit(
            AdaptiveBiometricObservation(
                modality="voice",
                person_id=owner_id,
                embedding=query_embedding,
                model=str(getattr(self.backend, "model_name", self.policy.backend) or self.policy.backend),
                evidence=evidence,
                metadata={
                    "query_duration_s": stats.duration_s,
                    "rms_level": stats.rms_level,
                    "clipped_fraction": stats.clipped_fraction,
                    "source": "turn_audio",
                },
                log_fields=dict(log_fields or {}),
            )
        )


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)
