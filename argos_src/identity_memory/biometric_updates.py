"""Session-local adaptive biometric update coordination."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import threading
from typing import Any

from .models import BiometricUpdateResult
from argos_src.observability.observability import LatencyLogger


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptiveBiometricObservation:
    modality: str
    person_id: str
    embedding: Any
    model: str
    evidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    log_fields: dict[str, Any] = field(default_factory=dict)


class AdaptiveBiometricUpdateCoordinator:
    """Best-effort background updater without durable memory ownership."""

    def __init__(
        self,
        identity_memory_client: Any,
        *,
        logger_: logging.Logger | None = None,
        max_workers: int = 1,
    ) -> None:
        self.identity_memory_client = identity_memory_client
        self.logger = logger_ or logger
        self._lock = threading.Lock()
        self._completed: set[tuple[str, str]] = set()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="adaptive-biometric-update",
        )
        self._latency = LatencyLogger("identity_memory")

    def submit(self, observation: AdaptiveBiometricObservation) -> None:
        """Queue one observation if this session has not completed/cooled it down."""
        if self._should_skip(observation):
            return
        try:
            self._executor.submit(self.submit_sync, observation)
        except RuntimeError:
            self.logger.debug(
                "Adaptive biometric update executor unavailable modality=%s person_id=%s",
                observation.modality,
                observation.person_id,
            )

    def submit_sync(self, observation: AdaptiveBiometricObservation) -> BiometricUpdateResult | None:
        """Submit synchronously; useful for tests and for the executor worker."""
        modality = str(observation.modality or "").strip()
        person_id = str(observation.person_id or "").strip()
        if modality not in {"face", "voice"} or not person_id:
            return None
        try:
            if modality == "face":
                result = self.identity_memory_client.observe_face_embedding(
                    person_id=person_id,
                    embedding=observation.embedding,
                    model=observation.model,
                    evidence=dict(observation.evidence or {}),
                    metadata=dict(observation.metadata or {}),
                )
            else:
                result = self.identity_memory_client.observe_voice_embedding(
                    person_id=person_id,
                    embedding=observation.embedding,
                    model=observation.model,
                    evidence=dict(observation.evidence or {}),
                    metadata=dict(observation.metadata or {}),
                )
        except Exception:
            self.logger.exception(
                "Adaptive biometric update failed modality=%s person_id=%s",
                modality,
                person_id,
            )
            return None

        if self._is_complete(result):
            with self._lock:
                self._completed.add((person_id, modality))
        self._latency.emit(
            event="adaptive_biometric_update",
            **dict(observation.log_fields or {}),
            biometric_update_modality=modality,
            biometric_update_person_id=person_id,
            biometric_update_accepted=bool(getattr(result, "accepted", False)),
            biometric_update_status=str(getattr(result, "status", "") or ""),
            biometric_update_reason=str(getattr(result, "reason", "") or ""),
            biometric_update_reference_id=str(getattr(result, "reference_id", "") or ""),
            biometric_update_sample_count=int(getattr(result, "sample_count", 0) or 0),
            biometric_update_target_sample_count=int(
                getattr(result, "target_sample_count", 0) or 0
            ),
            biometric_update_similarity=float(getattr(result, "similarity", 0.0) or 0.0),
        )
        self.logger.info(
            "Adaptive biometric update modality=%s person_id=%s accepted=%s status=%s "
            "reason=%s sample_count=%s target_sample_count=%s similarity=%.3f",
            modality,
            person_id,
            bool(getattr(result, "accepted", False)),
            str(getattr(result, "status", "") or ""),
            str(getattr(result, "reason", "") or ""),
            int(getattr(result, "sample_count", 0) or 0),
            int(getattr(result, "target_sample_count", 0) or 0),
            float(getattr(result, "similarity", 0.0) or 0.0),
        )
        return result

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _should_skip(self, observation: AdaptiveBiometricObservation) -> bool:
        modality = str(observation.modality or "").strip()
        person_id = str(observation.person_id or "").strip()
        if modality not in {"face", "voice"} or not person_id:
            return True
        key = (person_id, modality)
        with self._lock:
            if key in self._completed:
                return True
        return False

    @staticmethod
    def _is_complete(result: Any) -> bool:
        sample_count = int(getattr(result, "sample_count", 0) or 0)
        target_sample_count = int(getattr(result, "target_sample_count", 0) or 0)
        return (
            str(getattr(result, "status", "") or "") == "complete"
            or str(getattr(result, "reason", "") or "") == "sample_target_reached"
            or (target_sample_count > 0 and sample_count >= target_sample_count)
        )
