"""Face recognition service orchestration for Go2."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional
from uuid import uuid4

import cv2
import numpy as np
import torch

from argos_src.face_recognition.bearing import estimate_robot_yaw_error_rad, face_center_px
from argos_src.face_recognition.constants import (
    MIN_FACE_DETECTION_CONFIDENCE,
)
from argos_src.face_recognition.attention_gate import (
    AttentionGateSettings,
    FaceAttentionGate,
    draw_attention_overlay,
)
from argos_src.face_recognition.depth_gate import DepthGateSettings, filter_detections_by_depth
from argos_src.face_recognition.models import (
    AttentionTarget,
    CACHE_EXPIRE_SEC,
    PersonContext,
    SocialSceneContext,
)
from argos_src.face_recognition.pipeline import FaceEmbeddingPipeline, FacePipelineCudaUnavailable
from argos_src.face_recognition.presence_cache import FacePresenceCache
from argos_src.face_recognition.scene_analysis import FaceSceneCandidate, analyze_face_scene
from argos_src.media.image_encoding import preprocess_image
from argos_src.observability.observability import LatencyLogger, perf_now
from argos_src.provider_api.errors import is_provider_error
from argos_src.provider_api.models import CameraIntrinsics


logger = logging.getLogger(__name__)


ENROLLMENT_BURST_FRAMES = 5
ENROLLMENT_REQUIRED_STABLE_FRAMES = 3
ENROLLMENT_BURST_SLEEP_SEC = 0.1
LOOP_HEARTBEAT_LOG_SEC = 5.0


@dataclass(frozen=True)
class FaceEnrollmentPolicy:
    min_face_area: int = 1300
    min_brightness: float = 35.0
    max_brightness: float = 220.0
    min_contrast: float = 15.5
    min_embedding_similarity: float = 0.60


DEFAULT_FACE_ENROLLMENT_POLICY = FaceEnrollmentPolicy()


@dataclass(frozen=True)
class FaceRecognitionStabilitySettings:
    window_frames: int = 5
    min_hits: int = 2

    def __post_init__(self) -> None:
        if int(self.window_frames) < 1:
            raise ValueError("window_frames must be >= 1")
        if int(self.min_hits) < 1:
            raise ValueError("min_hits must be >= 1")
        if int(self.min_hits) > int(self.window_frames):
            raise ValueError("min_hits must be <= window_frames")


class RecognitionStabilityWindow:
    """Promote recognized identities only after repeated recent frame hits."""

    def __init__(self, settings: FaceRecognitionStabilitySettings | None = None) -> None:
        self.settings = settings or FaceRecognitionStabilitySettings()
        self._samples: deque[set[str]] = deque(maxlen=int(self.settings.window_frames))
        self._latest_persons: dict[str, PersonContext] = {}

    def reset(self) -> None:
        self._samples.clear()
        self._latest_persons.clear()

    def update(self, persons: list[PersonContext]) -> tuple[list[PersonContext], set[str]]:
        current_ids: set[str] = set()
        latest_by_id: dict[str, PersonContext] = {}
        for person in persons:
            person_id = str(getattr(person, "person_id", "") or "").strip()
            if not person_id:
                continue
            current_ids.add(person_id)
            previous = latest_by_id.get(person_id)
            if previous is None or int(person.bbox_area) > int(previous.bbox_area):
                latest_by_id[person_id] = person

        self._samples.append(current_ids)
        for person_id, person in latest_by_id.items():
            self._latest_persons[person_id] = person

        counts: dict[str, int] = {}
        for sample in self._samples:
            for person_id in sample:
                counts[person_id] = counts.get(person_id, 0) + 1
        stable_ids = {
            person_id
            for person_id, count in counts.items()
            if count >= int(self.settings.min_hits)
        }

        live_window_ids = set(counts)
        for person_id in list(self._latest_persons):
            if person_id not in live_window_ids:
                del self._latest_persons[person_id]

        stable_persons = [
            person
            for person_id, person in self._latest_persons.items()
            if person_id in stable_ids
        ]
        stable_persons.sort(key=lambda person: int(person.bbox_area), reverse=True)
        return stable_persons, stable_ids


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "na"
    try:
        return f"{float(value):.1f}"
    except Exception:
        return "na"


@dataclass(frozen=True)
class FaceEnrollmentQuality:
    accepted: bool
    reason: str = ""
    guidance: str = ""
    metric: float = 0.0


@dataclass(frozen=True)
class FaceEnrollmentQualityMetrics:
    bbox_area: int = 0
    clipped: bool = False
    crop_valid: bool = False
    brightness: float = 0.0
    contrast: float = 0.0


@dataclass(frozen=True)
class FacePreparationResult:
    faces: list[dict[str, Any]]
    reason: str = ""
    detected_count: int = 0
    rejected_count: int = 0
    rejection_details: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FaceEnrollmentCandidate:
    cleaned_name: str
    verified_durable: dict[str, str]
    averaged_embedding: np.ndarray
    reference_face: dict[str, Any]
    image_shape: tuple[int, ...]
    preview_image: Any


class FaceRecognitionService:
    """
    Face recognition service providing detection and recognition via robot API.

    Pipeline:
    - MTCNN: face detection + alignment
    - InceptionResnetV1 (FaceNet, vggface2): 512-d face embeddings
    - Tailwag-owned biometric search and enrollment
    """

    def __init__(
        self,
        robot_client: Any | None = None,
        depth_gate_settings: Optional[DepthGateSettings] = None,
        attention_gate_settings: AttentionGateSettings | None = None,
        enrollment_policy: FaceEnrollmentPolicy | None = None,
        identity_memory_client: Any | None = None,
        memory_store: Any | None = None,
        site_code: str = "",
        camera_resource_id: str = "",
        camera_yaw_offset_rad: float = 0.0,
        display_runtime: Any | None = None,
        live_image_title: str = "Camera",
        live_image_ttl_ms: int = 1000,
        live_image_enabled: bool = True,
        recognition_stability_settings: FaceRecognitionStabilitySettings | None = None,
    ):
        self.device = FaceEmbeddingPipeline.resolve_device()
        self._pipeline: Optional[FaceEmbeddingPipeline] = None
        self.mtcnn = None
        self.resnet = None

        self.robot_client = robot_client
        self.identity_memory_client = identity_memory_client
        self.memory_store = memory_store
        self.site_code = str(site_code or "").strip()
        self._depth_gate_settings = depth_gate_settings
        self._attention_gate = FaceAttentionGate(attention_gate_settings)
        self._camera_resource_id = str(camera_resource_id or "").strip()
        self._camera_yaw_offset_rad = float(camera_yaw_offset_rad)
        self._camera_intrinsics: CameraIntrinsics | None = None
        self._display_runtime = display_runtime
        self._live_image_title = str(live_image_title or "Camera").strip() or "Camera"
        self._live_image_ttl_ms = max(1, int(live_image_ttl_ms))
        self._live_image_enabled = bool(live_image_enabled)
        self._enrollment_policy = enrollment_policy or DEFAULT_FACE_ENROLLMENT_POLICY
        self._presence_cache = FacePresenceCache(cache_expire_sec=CACHE_EXPIRE_SEC)
        self._recognition_stability = RecognitionStabilityWindow(
            recognition_stability_settings
        )
        self._unknown_stability_frames = 0
        self._attentive_unknown_stability_frames = 0
        self._presence_subscribers: list[Callable[[dict[str, Any]], None]] = []
        self._presence_subscribers_lock = threading.Lock()
        self._recent_face_observations: dict[str, dict[str, Any]] = {}
        self._recent_face_observations_lock = threading.Lock()
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_stop = threading.Event()
        self._latest_loop_frame_lock = threading.Lock()
        self._latest_loop_frame = None
        self._latest_loop_frame_resource_id: str | None = None
        self._latest_loop_frame_at = 0.0
        self._loop_log_heartbeat_at: dict[str, float] = {}
        self._loop_metric_heartbeat_at: dict[str, float] = {}
        self._loop_latency = LatencyLogger("face_loop")

        gpu_name = (
            f", gpu={torch.cuda.get_device_name(0)}"
            if self.device.type == "cuda"
            else ""
        )
        logger.info(
            "Face Recognition Service initialized (device=%s%s, identity_memory=%s)",
            self.device,
            gpu_name,
            type(identity_memory_client).__name__ if identity_memory_client is not None else "disabled",
        )

    def subscribe_presence(
        self,
        callback: Callable[[dict[str, Any]], None],
        *,
        replay_latest: bool = True,
    ) -> Callable[[], None]:
        """Subscribe to immediate face-presence updates from the recognition loop."""
        if not callable(callback):
            raise TypeError("callback must be callable")
        if not hasattr(self, "_presence_subscribers_lock"):
            self._presence_subscribers_lock = threading.Lock()
        if not hasattr(self, "_presence_subscribers"):
            self._presence_subscribers = []
        with self._presence_subscribers_lock:
            self._presence_subscribers.append(callback)
        if replay_latest:
            try:
                callback(self.get_presence_snapshot())
            except Exception:
                logger.exception("Face presence subscriber failed during replay")

        def unsubscribe() -> None:
            with self._presence_subscribers_lock:
                try:
                    self._presence_subscribers.remove(callback)
                except ValueError:
                    pass

        return unsubscribe

    def _notify_presence_subscribers(self, snapshot: dict[str, Any]) -> None:
        if not hasattr(self, "_presence_subscribers_lock"):
            self._presence_subscribers_lock = threading.Lock()
        if not hasattr(self, "_presence_subscribers"):
            self._presence_subscribers = []
        with self._presence_subscribers_lock:
            subscribers = list(self._presence_subscribers)
        for callback in subscribers:
            try:
                callback(dict(snapshot or {}))
            except Exception:
                logger.exception("Face presence subscriber failed")

    def _clear_presence_if_expired(self, now: float) -> bool:
        if not self._presence_cache.clear_if_expired(now):
            return False
        stability = getattr(self, "_recognition_stability", None)
        if stability is not None:
            stability.reset()
        self._reset_unknown_stability()
        self._notify_presence_subscribers(self.get_presence_snapshot())
        return True

    def _reset_unknown_stability(self) -> None:
        self._unknown_stability_frames = 0
        self._attentive_unknown_stability_frames = 0

    def _update_unknown_stability(
        self,
        *,
        unknown_count: int,
        attentive_unknown_count: int,
    ) -> tuple[int, int]:
        if int(unknown_count or 0) > 0:
            self._unknown_stability_frames = int(
                getattr(self, "_unknown_stability_frames", 0) or 0
            ) + 1
        else:
            self._unknown_stability_frames = 0

        if int(attentive_unknown_count or 0) > 0:
            self._attentive_unknown_stability_frames = int(
                getattr(self, "_attentive_unknown_stability_frames", 0) or 0
            ) + 1
        else:
            self._attentive_unknown_stability_frames = 0

        return (
            self._unknown_stability_frames,
            self._attentive_unknown_stability_frames,
        )

    # ------------------------------------------------------------------
    # Image capture
    # ------------------------------------------------------------------

    def get_latest_image(self, camera_resource_id: str, timeout: float = 2.0):
        """Receive one frame via the configured robot camera backend."""
        if self.robot_client is None:
            return None
        frame = self.robot_client.get_latest_image(
            resource_id=camera_resource_id,
            timeout=timeout,
        )
        if frame is None:
            return None
        return getattr(frame, "image", frame)

    def _cache_latest_loop_frame(
        self,
        *,
        image,
        camera_resource_id: str,
        captured_at: float,
    ) -> None:
        """Store the most recent color frame seen by the background face loop."""
        if image is None:
            return
        with self._latest_loop_frame_lock:
            self._latest_loop_frame = image.copy()
            self._latest_loop_frame_resource_id = camera_resource_id
            self._latest_loop_frame_at = float(captured_at)

    def get_cached_latest_frame(
        self,
        *,
        camera_resource_id: str | None = None,
        max_age_sec: float = 2.0,
    ) -> tuple[object | None, str | None, float | None]:
        """Return the latest cached face-loop color frame when it is recent enough."""
        lock = getattr(self, "_latest_loop_frame_lock", None)
        if lock is None:
            return None, None, None

        with lock:
            image = getattr(self, "_latest_loop_frame", None)
            cached_resource_id = getattr(self, "_latest_loop_frame_resource_id", None)
            captured_at = float(getattr(self, "_latest_loop_frame_at", 0.0))
            if image is None or cached_resource_id is None or captured_at <= 0.0:
                return None, None, None
            if camera_resource_id and camera_resource_id != cached_resource_id:
                return None, None, None
            if max_age_sec >= 0.0 and (time.time() - captured_at) > max_age_sec:
                return None, None, None
            return image.copy(), cached_resource_id, captured_at

    # ------------------------------------------------------------------
    # Detection + embedding
    # ------------------------------------------------------------------

    def detect_and_extract_faces(self, image):
        """Detect all faces in image and return embeddings + bounding boxes."""
        return self._run_pipeline_operation("detect_and_extract_faces", image)

    def detect_faces(self, image):
        """Detect face boxes and landmarks without embedding extraction."""
        return self._run_pipeline_operation("detect_faces", image)

    def extract_face_embedding(self, image, face):
        """Extract an embedding for one detected face."""
        return self._run_pipeline_operation("extract_embedding", image, face)

    def _ensure_pipeline(self) -> FaceEmbeddingPipeline:
        """Build the embedding pipeline only when face inference is actually needed."""
        if self._pipeline is None:
            self._pipeline = FaceEmbeddingPipeline(self.device)
            self.mtcnn = self._pipeline.mtcnn
            self.resnet = self._pipeline.resnet
            logger.info("Face embedding pipeline initialized")
        return self._pipeline

    def _run_pipeline_operation(self, operation: str, *args):
        pipeline = self._ensure_pipeline()
        method = getattr(pipeline, operation)
        try:
            return method(*args)
        except FacePipelineCudaUnavailable as exc:
            if self.device.type != "cuda":
                raise
            logger.warning(
                "Face pipeline CUDA execution failed; falling back to CPU. error=%s",
                exc,
            )
            self.device = torch.device("cpu")
            self._pipeline = None
            self.mtcnn = None
            self.resnet = None
            pipeline = self._ensure_pipeline()
            method = getattr(pipeline, operation)
            return method(*args)

    # ------------------------------------------------------------------
    # Full recognition pipeline
    # ------------------------------------------------------------------

    def _capture_for_recognition(
        self,
        camera_resource_id: str,
        *,
        timeout: float,
    ) -> tuple[object | None, object | None]:
        """Capture the frame(s) needed for recognition."""
        if self._depth_gate_settings is None:
            try:
                return self.get_latest_image(camera_resource_id, timeout=timeout), None
            except Exception as exc:
                if is_provider_error(exc):
                    self._log_loop_heartbeat(
                        "camera_image_provider_unavailable",
                        "Provider image unavailable from camera resource %s: %s",
                        camera_resource_id,
                        exc,
                        interval_sec=10.0,
                        level=logging.WARNING,
                    )
                    return None, None
                raise

        if self.robot_client is None:
            return None, None
        try:
            rgbd = self.robot_client.get_latest_rgbd(
                resource_id=camera_resource_id,
                timeout=min(timeout, self._depth_gate_settings.capture_timeout_sec),
                sync_slop_sec=self._depth_gate_settings.sync_slop_sec,
                queue_size=self._depth_gate_settings.sync_queue_size,
            )
        except Exception as exc:
            if is_provider_error(exc):
                self._log_loop_heartbeat(
                    "camera_rgbd_provider_unavailable",
                    "Provider RGBD unavailable from camera resource %s: %s",
                    camera_resource_id,
                    exc,
                    interval_sec=10.0,
                    level=logging.WARNING,
                )
                return None, None
            raise
        if rgbd is None:
            return None, None

        return rgbd.color_image, rgbd.depth_m

    def _prepare_faces_for_recognition(
        self,
        image,
        depth_m,
    ) -> list[dict[str, Any]]:
        """Run detection, optional depth gating, then embedding extraction."""
        return self._prepare_faces_for_recognition_result(
            image,
            depth_m,
            min_face_area=self._recognition_min_face_area(),
        ).faces

    def _prepare_faces_for_recognition_result(
        self,
        image,
        depth_m,
        *,
        min_face_area: int | None = None,
    ) -> FacePreparationResult:
        """Run detection, optional depth gating, and embedding extraction with diagnostics."""
        detected_faces = self.detect_faces(image)
        if not detected_faces:
            return FacePreparationResult(
                faces=[],
                reason="no_detection",
                detected_count=0,
            )

        detected_count = len(detected_faces)
        rejected_count = 0
        rejection_details: list[str] = []
        if min_face_area is not None and int(min_face_area) > 0:
            min_area = int(min_face_area)
            area_rejected_faces = [
                face
                for face in detected_faces
                if self._bbox_area(face) < min_area
            ]
            kept_faces = [
                face
                for face in detected_faces
                if self._bbox_area(face) >= min_area
            ]
            area_rejected_count = len(area_rejected_faces)
            for index, face in enumerate(area_rejected_faces):
                rejection_details.append(
                    f"face{index}:face_too_small area={self._bbox_area(face)} "
                    f"min_face_area={min_area}"
                )
            rejected_count += area_rejected_count
            detected_faces = kept_faces
            if area_rejected_count and not detected_faces:
                return FacePreparationResult(
                    faces=[],
                    reason="face_too_small",
                    detected_count=detected_count,
                    rejected_count=rejected_count,
                    rejection_details=rejection_details,
                )
        if depth_m is not None and self._depth_gate_settings is not None:
            gated_faces, depth_rejected_count = filter_detections_by_depth(
                detected_faces,
                depth_m,
                self._depth_gate_settings,
            )
            rejected_count += depth_rejected_count
            if depth_rejected_count:
                kept = len(gated_faces)
                total = len(detected_faces)
                if kept == 0:
                    self._log_loop_heartbeat(
                        "depth_gate_rejected_all",
                        "[FaceLoop] depth gate kept 0/%s face(s); likely no valid aligned depth samples or face beyond %.2fm",
                        total,
                        self._depth_gate_settings.max_face_depth_m,
                        level=logging.DEBUG,
                    )
                else:
                    logger.debug(
                        "[FaceLoop] depth gate kept %s/%s face(s)",
                        kept,
                        total,
                    )
            detected_faces = gated_faces
            if rejected_count and not detected_faces:
                return FacePreparationResult(
                    faces=[],
                    reason="depth_rejected",
                    detected_count=detected_count,
                    rejected_count=rejected_count,
                    rejection_details=rejection_details,
                )

        faces_with_embeddings: list[dict[str, Any]] = []
        for detection in detected_faces:
            embedding = self.extract_face_embedding(image, detection)
            if embedding is None:
                continue
            enriched = dict(detection)
            enriched["embedding"] = embedding
            faces_with_embeddings.append(enriched)
        if not faces_with_embeddings:
            return FacePreparationResult(
                faces=[],
                reason="no_embedding",
                detected_count=detected_count,
                rejected_count=rejected_count,
                rejection_details=rejection_details,
            )
        return FacePreparationResult(
            faces=faces_with_embeddings,
            detected_count=detected_count,
            rejected_count=rejected_count,
            rejection_details=rejection_details,
        )

    def _recognition_min_face_area(self) -> int:
        policy = getattr(self, "_enrollment_policy", DEFAULT_FACE_ENROLLMENT_POLICY)
        return int(
            getattr(
                policy,
                "min_face_area",
                DEFAULT_FACE_ENROLLMENT_POLICY.min_face_area,
            )
        )

    @staticmethod
    def _bbox_area(face: dict[str, Any]) -> int:
        bbox = face["bbox"]
        return int(bbox["w"]) * int(bbox["h"])

    @staticmethod
    def _bbox_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
        left_bbox = left["bbox"]
        right_bbox = right["bbox"]
        left_x1 = int(left_bbox["x"])
        left_y1 = int(left_bbox["y"])
        left_x2 = left_x1 + int(left_bbox["w"])
        left_y2 = left_y1 + int(left_bbox["h"])
        right_x1 = int(right_bbox["x"])
        right_y1 = int(right_bbox["y"])
        right_x2 = right_x1 + int(right_bbox["w"])
        right_y2 = right_y1 + int(right_bbox["h"])

        inter_x1 = max(left_x1, right_x1)
        inter_y1 = max(left_y1, right_y1)
        inter_x2 = min(left_x2, right_x2)
        inter_y2 = min(left_y2, right_y2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        left_area = max(1, (left_x2 - left_x1) * (left_y2 - left_y1))
        right_area = max(1, (right_x2 - right_x1) * (right_y2 - right_y1))
        union_area = left_area + right_area - inter_area
        return float(inter_area) / float(max(1, union_area))

    def _select_enrollment_face(
        self,
        detected_faces: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, bool]:
        """Return one enrollment face from the already-usable face list.

        Callers should pass only faces that survived any active depth gate.
        Enrollment should block only when more than one enrollment-sized face remains.
        """
        if not detected_faces:
            return None, False
        min_face_area = int(
            getattr(
                getattr(self, "_enrollment_policy", DEFAULT_FACE_ENROLLMENT_POLICY),
                "min_face_area",
                DEFAULT_FACE_ENROLLMENT_POLICY.min_face_area,
            )
        )
        ordered_all = sorted(
            detected_faces,
            key=lambda candidate: (
                -float(candidate.get("confidence", 0.0)),
                -float(self._bbox_area(candidate)),
            ),
        )
        enrollment_sized = [
            candidate
            for candidate in ordered_all
            if self._bbox_area(candidate) >= min_face_area
        ]
        if not enrollment_sized:
            return ordered_all[0], False
        if len(enrollment_sized) == 1:
            ignored = len(ordered_all) - 1
            if ignored > 0:
                logger.debug(
                    "Enrollment ignored %s below-min-area face detection(s) and kept the strongest enrollment-sized candidate.",
                    ignored,
                )
            return enrollment_sized[0], False

        ordered = sorted(
            enrollment_sized,
            key=lambda candidate: (
                -float(candidate.get("confidence", 0.0)),
                -float(self._bbox_area(candidate)),
            ),
        )
        primary = ordered[0]
        primary_confidence = float(primary.get("confidence", 0.0))
        primary_area = max(1, self._bbox_area(primary))
        significant_others: list[dict[str, Any]] = []
        for other in ordered[1:]:
            overlap = self._bbox_iou(primary, other)
            other_area_ratio = float(self._bbox_area(other)) / float(primary_area)
            other_confidence = float(other.get("confidence", 0.0))
            if overlap >= 0.4:
                continue
            if other_area_ratio < 0.35 and other_confidence <= max(
                MIN_FACE_DETECTION_CONFIDENCE,
                primary_confidence - 0.15,
            ):
                continue
            significant_others.append(other)
        if significant_others:
            return None, True
        logger.debug(
            "Enrollment ignored %s extra face detection(s) and kept the strongest enrollment-sized candidate.",
            len(ordered_all) - 1,
        )
        return primary, False

    @staticmethod
    def _center_distance(face: dict[str, Any], image_shape: tuple[int, ...]) -> float:
        height, width = image_shape[:2]
        bbox = face["bbox"]
        center_x = float(bbox["x"]) + (float(bbox["w"]) / 2.0)
        center_y = float(bbox["y"]) + (float(bbox["h"]) / 2.0)
        image_center_x = float(width) / 2.0
        image_center_y = float(height) / 2.0
        return float(
            np.hypot(center_x - image_center_x, center_y - image_center_y)
        )

    @staticmethod
    def _quality_response_for_reason(reason: str) -> tuple[str, str]:
        guidance_by_reason = {
            "face_too_small": "Come a little closer so I can see your face clearly.",
            "face_clipped": "Please center your whole face in view.",
            "too_dark": "Please move to better light.",
            "too_bright": "Please move away from the bright light.",
            "low_contrast": "Please move to better light.",
            "embedding_inconsistent": "Hold still and face me directly for a second.",
        }
        return reason, guidance_by_reason.get(
            reason,
            "Please face me directly and hold still for a second.",
        )

    @classmethod
    def _enrollment_similarity_diagnostics(
        cls,
        *,
        accepted_faces: list[dict[str, Any]],
        reference_item: dict[str, Any],
        threshold: float,
    ) -> dict[str, Any]:
        reference_face = reference_item["face"]
        similarities: list[float] = []
        failed_similarities: list[float] = []
        consistent_count = 0
        for item in accepted_faces:
            face = item["face"]
            similarity = cls._embedding_similarity(
                reference_face["embedding"],
                face["embedding"],
            )
            rounded_similarity = round(float(similarity), 3)
            similarities.append(rounded_similarity)
            if similarity >= threshold:
                consistent_count += 1
            else:
                failed_similarities.append(float(similarity))

        best_failed_similarity = (
            max(failed_similarities) if failed_similarities else None
        )
        return {
            "burst_frames": ENROLLMENT_BURST_FRAMES,
            "required_stable_frames": ENROLLMENT_REQUIRED_STABLE_FRAMES,
            "accepted_frame_count": len(accepted_faces),
            "consistent_frame_count": consistent_count,
            "min_embedding_similarity": round(float(threshold), 3),
            "similarities_to_reference": similarities,
            "best_failed_similarity": round(best_failed_similarity, 3)
            if best_failed_similarity is not None
            else None,
            "best_failed_shortfall": round(
                max(0.0, float(threshold) - best_failed_similarity),
                3,
            )
            if best_failed_similarity is not None
            else None,
        }

    @staticmethod
    def _embedding_similarity(left: Any, right: Any) -> float:
        left_vec = np.asarray(left, dtype=np.float32).reshape(-1)
        right_vec = np.asarray(right, dtype=np.float32).reshape(-1)
        left_norm = float(np.linalg.norm(left_vec))
        right_norm = float(np.linalg.norm(right_vec))
        if left_norm <= 1e-8 or right_norm <= 1e-8:
            return 0.0
        return float(np.dot(left_vec / left_norm, right_vec / right_norm))

    @classmethod
    def _average_embeddings(cls, embeddings: list[Any]) -> np.ndarray:
        vectors = [np.asarray(embedding, dtype=np.float32).reshape(-1) for embedding in embeddings]
        if not vectors:
            return np.asarray([], dtype=np.float32)
        normalized_vectors = []
        for vector in vectors:
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-8:
                continue
            normalized_vectors.append(vector / norm)
        if not normalized_vectors:
            return np.asarray([], dtype=np.float32)
        averaged = np.mean(np.stack(normalized_vectors, axis=0), axis=0)
        averaged_norm = float(np.linalg.norm(averaged))
        if averaged_norm <= 1e-8:
            return averaged.astype(np.float32)
        return (averaged / averaged_norm).astype(np.float32)

    def _measure_enrollment_face_quality(
        self,
        image,
        face: dict[str, Any],
    ) -> FaceEnrollmentQualityMetrics:
        bbox = face.get("bbox") or {}
        height, width = image.shape[:2]
        x = int(bbox.get("x", 0) or 0)
        y = int(bbox.get("y", 0) or 0)
        w = int(bbox.get("w", 0) or 0)
        h = int(bbox.get("h", 0) or 0)
        bbox_area = int(w * h)
        clipped = x <= 1 or y <= 1 or (x + w) >= (width - 1) or (y + h) >= (height - 1)

        crop = image[y : y + h, x : x + w]
        crop_valid = bool(crop.size > 0)
        brightness = 0.0
        contrast = 0.0
        if crop_valid:
            gray = crop.astype(np.float32).mean(axis=2) if crop.ndim == 3 else crop.astype(np.float32)
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
        return FaceEnrollmentQualityMetrics(
            bbox_area=bbox_area,
            clipped=clipped,
            crop_valid=crop_valid,
            brightness=brightness,
            contrast=contrast,
        )

    @staticmethod
    def _enrollment_preview_image(
        image: Any,
        face: dict[str, Any],
        padding_ratio: float = 0.45,
    ) -> Any:
        if image is None or not hasattr(image, "shape"):
            return image
        try:
            height, width = image.shape[:2]
            if height <= 0 or width <= 0:
                return image
            bbox = face.get("bbox") or {}
            x = float(bbox.get("x", 0) or 0)
            y = float(bbox.get("y", 0) or 0)
            w = float(bbox.get("w", 0) or 0)
            h = float(bbox.get("h", 0) or 0)
            if w <= 0 or h <= 0:
                return image.copy()

            center_x = x + (w / 2.0)
            center_y = y + (h / 2.0)
            side = max(w, h) * (1.0 + (2.0 * max(0.0, float(padding_ratio))))
            x1 = max(0, int(round(center_x - (side / 2.0))))
            y1 = max(0, int(round(center_y - (side / 2.0))))
            x2 = min(width, int(round(center_x + (side / 2.0))))
            y2 = min(height, int(round(center_y + (side / 2.0))))
            if x2 <= x1 or y2 <= y1:
                return image.copy()
            return image[y1:y2, x1:x2].copy()
        except Exception:
            logger.exception("Failed to prepare enrollment preview crop")
            return image.copy() if hasattr(image, "copy") else image

    def _assess_enrollment_face_quality(
        self,
        image,
        face: dict[str, Any],
        metrics: FaceEnrollmentQualityMetrics | None = None,
    ) -> FaceEnrollmentQuality:
        policy = getattr(self, "_enrollment_policy", DEFAULT_FACE_ENROLLMENT_POLICY)
        metrics = metrics or self._measure_enrollment_face_quality(image, face)
        if metrics.bbox_area < policy.min_face_area:
            reason, guidance = self._quality_response_for_reason("face_too_small")
            return FaceEnrollmentQuality(False, reason, guidance, float(metrics.bbox_area))
        if metrics.clipped:
            reason, guidance = self._quality_response_for_reason("face_clipped")
            return FaceEnrollmentQuality(False, reason, guidance, 0.0)
        if not metrics.crop_valid:
            reason, guidance = self._quality_response_for_reason("face_clipped")
            return FaceEnrollmentQuality(False, reason, guidance, 0.0)
        if metrics.brightness < policy.min_brightness:
            reason, guidance = self._quality_response_for_reason("too_dark")
            return FaceEnrollmentQuality(False, reason, guidance, metrics.brightness)
        if metrics.brightness > policy.max_brightness:
            reason, guidance = self._quality_response_for_reason("too_bright")
            return FaceEnrollmentQuality(False, reason, guidance, metrics.brightness)
        if metrics.contrast < policy.min_contrast:
            reason, guidance = self._quality_response_for_reason("low_contrast")
            return FaceEnrollmentQuality(False, reason, guidance, metrics.contrast)
        return FaceEnrollmentQuality(True)

    def _recognize_face_match_with_diagnostics(
        self,
        face: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        match_override = self.__dict__.get("_recognize_face_match")
        if callable(match_override):
            match = match_override(face)
            if match is None:
                return None, {"status": "rejected", "reason": "no_match"}
            similarity = float(match.get("similarity", 0.0) or 0.0)
            return match, {
                "status": "accepted",
                "reason": "matched",
                "name": str(match.get("name", "") or ""),
                "person_id": str(match.get("person_id", "") or ""),
                "similarity": similarity,
                "threshold": 0.0,
                "runner_up_similarity": float(
                    match.get("runner_up_similarity", 0.0) or 0.0
                ),
                "margin": float(match.get("similarity_margin", 0.0) or 0.0),
                "margin_threshold": 0.0,
            }

        identity_memory = getattr(self, "identity_memory_client", None)
        if identity_memory is None:
            return None, {"status": "rejected", "reason": "identity_memory_unavailable"}
        result = identity_memory.search_face(
            embedding=face["embedding"],
            model="facenet-vggface2",
            limit=2,
        )
        matches = [
            {
                "person_id": candidate.person_id,
                "name": candidate.display_name or candidate.person_id,
                "similarity": float(candidate.score),
                "metadata": dict(candidate.metadata or {}),
            }
            for candidate in tuple(getattr(result, "candidates", ()) or ())
        ]
        if not matches:
            return None, {"status": "rejected", "reason": "no_db_match"}
        top_match = matches[0]
        top_similarity = float(getattr(result, "top_score", 0.0) or top_match.get("similarity", 0.0) or 0.0)
        runner_up_similarity = float(getattr(result, "runner_up_score", 0.0) or 0.0)
        margin = float(getattr(result, "margin", top_similarity - runner_up_similarity) or 0.0)
        result_status = str(getattr(result, "status", "") or "")
        result_reason = str(getattr(result, "reason", "") or "")
        threshold = float(getattr(result, "threshold", 0.0) or 0.0)
        margin_threshold = float(getattr(result, "margin_threshold", 0.0) or 0.0)
        if result_status != "accepted" and result_reason == "below_threshold":
            return None, {
                "status": "rejected",
                "reason": "below_threshold",
                "name": str(top_match.get("name", "") or ""),
                "person_id": str(top_match.get("person_id", "") or ""),
                "similarity": top_similarity,
                "threshold": threshold,
                "runner_up_similarity": runner_up_similarity,
                "margin": margin,
                "margin_threshold": margin_threshold,
            }
        if result_status != "accepted" and result_reason == "margin_too_small":
            return None, {
                "status": "rejected",
                "reason": "margin_too_small",
                "name": str(top_match.get("name", "") or ""),
                "person_id": str(top_match.get("person_id", "") or ""),
                "similarity": top_similarity,
                "threshold": threshold,
                "runner_up_similarity": runner_up_similarity,
                "margin": margin,
                "margin_threshold": margin_threshold,
            }
        if result_status != "accepted":
            return None, {
                "status": "rejected",
                "reason": result_reason or "no_match",
                "name": str(top_match.get("name", "") or ""),
                "person_id": str(top_match.get("person_id", "") or ""),
                "similarity": top_similarity,
                "threshold": threshold,
                "runner_up_similarity": runner_up_similarity,
                "margin": margin,
                "margin_threshold": margin_threshold,
            }
        top_match["runner_up_similarity"] = runner_up_similarity
        top_match["similarity_margin"] = margin
        return top_match, {
            "status": "accepted",
            "reason": "matched",
            "name": str(top_match.get("name", "") or ""),
            "person_id": str(top_match.get("person_id", "") or ""),
            "similarity": top_similarity,
            "threshold": threshold,
            "runner_up_similarity": runner_up_similarity,
            "margin": margin,
            "margin_threshold": margin_threshold,
        }

    def _recognize_face_match(self, face: dict[str, Any]) -> dict[str, Any] | None:
        match, _diagnostics = self._recognize_face_match_with_diagnostics(face)
        return match

    def _build_scene_state(
        self,
        *,
        image=None,
        detected_faces: list[dict[str, Any]],
        image_shape: tuple[int, ...],
        now: float,
    ) -> tuple[list[PersonContext], int, set[str], Any]:
        current_ids: set[str] = set()
        unknown_count = 0
        persons: list[PersonContext] = []
        candidates: list[FaceSceneCandidate] = []
        intrinsics = self._get_camera_intrinsics()
        attention_gate = getattr(self, "_attention_gate", None)
        if attention_gate is None:
            attention_gate = FaceAttentionGate(AttentionGateSettings(enabled=False))
            self._attention_gate = attention_gate

        for face in detected_faces:
            bbox_area = self._bbox_area(face)
            center_distance = self._center_distance(face, image_shape)
            bearing_rad = estimate_robot_yaw_error_rad(
                face,
                intrinsics=intrinsics,
                camera_yaw_offset_rad=float(
                    getattr(self, "_camera_yaw_offset_rad", 0.0) or 0.0
                ),
            )
            face_center = face_center_px(face)
            match, recognition = self._recognize_face_match_with_diagnostics(face)
            face["recognition"] = recognition
            pid = match["person_id"] if match is not None else ""
            track_id = pid or self._unknown_attention_track_id(face, image_shape)
            attention = attention_gate.evaluate(
                image,
                face,
                image_shape=image_shape,
                track_id=track_id,
                now=now,
            )
            face["attention"] = attention
            if match is None:
                unknown_count += 1
                candidates.append(
                    FaceSceneCandidate(
                        kind="unknown",
                        bbox_area=bbox_area,
                        center_distance=center_distance,
                        depth_m=face.get("depth_m"),
                        attentive=attention.attentive,
                        attention_confidence=attention.confidence,
                    )
                )
                continue

            current_ids.add(pid)
            face["recognized_name"] = match["name"]
            self._presence_cache.mark_person_seen(pid, now)
            meta = match["metadata"]
            directory_profile_lines = _profile_directory_lines(meta)
            person = PersonContext(
                person_id=pid,
                name=match["name"],
                interaction_count=meta.get("interaction_count", 0),
                confidence=match["similarity"],
                bbox_area=bbox_area,
                timestamp=now,
                recognition_status=str(recognition.get("status") or ""),
                recognition_reason=str(recognition.get("reason") or ""),
                recognition_threshold=float(recognition.get("threshold", 0.0) or 0.0),
                runner_up_confidence=float(
                    recognition.get("runner_up_similarity", 0.0) or 0.0
                ),
                confidence_margin=float(recognition.get("margin", 0.0) or 0.0),
                margin_threshold=float(
                    recognition.get("margin_threshold", 0.0) or 0.0
                ),
                depth_m=face.get("depth_m"),
                bearing_rad=bearing_rad,
                face_center_x_px=face_center[0] if face_center is not None else None,
                face_center_y_px=face_center[1] if face_center is not None else None,
                center_distance=center_distance,
                attentive=attention.attentive,
                attention_confidence=attention.confidence,
                head_yaw_deg=attention.yaw_deg,
                head_pitch_deg=attention.pitch_deg,
                head_roll_deg=attention.roll_deg,
                directory_profile_lines=directory_profile_lines,
            )
            persons.append(person)
            candidates.append(
                FaceSceneCandidate(
                    kind="recognized",
                    bbox_area=bbox_area,
                    center_distance=center_distance,
                    depth_m=face.get("depth_m"),
                    person_id=pid,
                    name=match["name"],
                    attentive=attention.attentive,
                    attention_confidence=attention.confidence,
                )
            )
            face["recognized_person_id"] = pid
            self._remember_recent_face_observation(
                person_id=pid,
                embedding=face.get("embedding"),
                metadata={
                    "score": float(recognition.get("similarity", match["similarity"]) or 0.0),
                    "runner_up_score": float(
                        recognition.get("runner_up_similarity", 0.0) or 0.0
                    ),
                    "margin": float(recognition.get("margin", 0.0) or 0.0),
                    "threshold": float(recognition.get("threshold", 0.0) or 0.0),
                    "margin_threshold": float(
                        recognition.get("margin_threshold", 0.0) or 0.0
                    ),
                    "bbox_area": int(bbox_area),
                    "center_distance": float(center_distance),
                    "depth_m": face.get("depth_m"),
                    "attentive": bool(attention.attentive),
                    "attention_confidence": float(attention.confidence),
                },
            )

        analysis = analyze_face_scene(candidates)
        return persons, unknown_count, current_ids, analysis

    @staticmethod
    def _best_face_match_evidence(
        detected_faces: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return the strongest visible recognition attempt for dashboard evidence."""
        best: tuple[float, int, dict[str, Any]] | None = None
        for index, face in enumerate(detected_faces):
            recognition = dict(face.get("recognition") or {})
            if not recognition:
                continue
            bbox = dict(face.get("bbox") or {})
            try:
                area = int(float(bbox.get("w", 0) or 0) * float(bbox.get("h", 0) or 0))
            except Exception:
                area = 0
            try:
                similarity = float(recognition.get("similarity", 0.0) or 0.0)
            except Exception:
                similarity = 0.0
            candidate = (similarity, area, recognition)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            return {}
        return dict(best[2])

    def _ensure_recognition_stability(self) -> RecognitionStabilityWindow:
        stability = getattr(self, "_recognition_stability", None)
        if stability is None:
            stability = RecognitionStabilityWindow()
            self._recognition_stability = stability
        return stability

    def _stable_scene_state(
        self,
        *,
        detected_faces: list[dict[str, Any]],
        raw_persons: list[PersonContext],
        image_shape: tuple[int, ...],
        now: float,
    ) -> tuple[list[PersonContext], int, set[str], Any]:
        stable_persons, stable_ids = self._ensure_recognition_stability().update(
            raw_persons
        )
        stable_by_id = {
            str(person.person_id or "").strip(): person
            for person in stable_persons
            if str(person.person_id or "").strip()
        }
        current_stable_ids: set[str] = set()
        candidates: list[FaceSceneCandidate] = []
        unknown_count = 0
        seen_stable_candidates: set[str] = set()
        single_face_stable_fallback = (
            len(detected_faces) == 1
            and len(stable_by_id) == 1
            and not str(detected_faces[0].get("recognized_person_id") or "").strip()
        )

        for face in detected_faces:
            pid = str(face.get("recognized_person_id") or "").strip()
            attention = face.get("attention")
            bbox_area = self._bbox_area(face)
            center_distance = self._center_distance(face, image_shape)
            stable_person = None
            if pid and pid in stable_ids and pid in stable_by_id:
                stable_person = stable_by_id[pid]
            elif single_face_stable_fallback:
                fallback_pid, fallback_person = next(iter(stable_by_id.items()))
                if fallback_pid in stable_ids:
                    pid = fallback_pid
                    stable_person = fallback_person

            if stable_person is not None:
                person = replace(
                    stable_person,
                    bbox_area=bbox_area,
                    timestamp=now,
                    depth_m=face.get("depth_m"),
                    center_distance=center_distance,
                    attentive=bool(getattr(attention, "attentive", False)),
                    attention_confidence=float(
                        getattr(attention, "confidence", 0.0) or 0.0
                    ),
                    head_yaw_deg=getattr(attention, "yaw_deg", None),
                    head_pitch_deg=getattr(attention, "pitch_deg", None),
                    head_roll_deg=getattr(attention, "roll_deg", None),
                )
                stable_by_id[pid] = person
                current_stable_ids.add(pid)
                seen_stable_candidates.add(pid)
                candidates.append(
                    FaceSceneCandidate(
                        kind="recognized",
                        bbox_area=int(person.bbox_area),
                        center_distance=float(person.center_distance),
                        depth_m=person.depth_m,
                        person_id=person.person_id,
                        name=person.name,
                        attentive=bool(person.attentive),
                        attention_confidence=float(person.attention_confidence),
                    )
                )
                continue

            unknown_count += 1
            candidates.append(
                FaceSceneCandidate(
                    kind="unknown",
                    bbox_area=bbox_area,
                    center_distance=center_distance,
                    depth_m=face.get("depth_m"),
                    attentive=bool(getattr(attention, "attentive", False)),
                    attention_confidence=float(
                        getattr(attention, "confidence", 0.0) or 0.0
                    ),
                )
            )

        current_stable_persons = [
            stable_by_id[person_id]
            for person_id in current_stable_ids
            if person_id in stable_by_id
        ]
        current_stable_persons.sort(
            key=lambda person: int(person.bbox_area),
            reverse=True,
        )
        return current_stable_persons, unknown_count, current_stable_ids, analyze_face_scene(candidates)

    @staticmethod
    def _unknown_attention_track_id(
        face: dict[str, Any],
        image_shape: tuple[int, ...],
    ) -> str:
        """Return a coarse stable key for unknown attention tracks."""
        center = face_center_px(face)
        if center is None:
            return "unknown"
        height, width = image_shape[:2]
        bucket_w = max(1, int(width / 8)) if width else 1
        bucket_h = max(1, int(height / 6)) if height else 1
        return f"unknown:{int(center[0] // bucket_w)}:{int(center[1] // bucket_h)}"

    def _get_camera_intrinsics(self) -> CameraIntrinsics | None:
        """Return cached color camera intrinsics when available."""
        cached = getattr(self, "_camera_intrinsics", None)
        if cached is not None:
            return cached
        resource_id = str(getattr(self, "_camera_resource_id", "") or "").strip()
        if not resource_id or self.robot_client is None:
            return None
        try:
            intrinsics = self.robot_client.get_latest_intrinsics(
                resource_id=resource_id,
                timeout=0.02,
            )
        except Exception as exc:
            if is_provider_error(exc):
                self._log_loop_heartbeat(
                    "camera_intrinsics_provider_unavailable",
                    "Provider camera intrinsics unavailable from camera resource %s: %s",
                    resource_id,
                    exc,
                    interval_sec=10.0,
                    level=logging.WARNING,
                )
            else:
                logger.exception(
                    "Failed to capture camera intrinsics from resource %s",
                    resource_id,
                )
            return None
        if intrinsics is not None:
            self._camera_intrinsics = intrinsics
        return intrinsics

    def recognize_faces(self, camera_resource_id: str | None = None) -> dict[str, Any]:
        """
        Capture frame → detect faces → match against DB.

        Returns:
            {
                "success": bool,
                "faces_detected": int,
                "faces_recognized": int,
                "people": [{"name", "person_id", "confidence", "bbox",
                             "last_seen", "interaction_count"}, ...],
                "unknown_faces": int,
                "error": str   # only if failed
            }
        """
        result: dict[str, Any] = {
            "success": False,
            "faces_detected": 0,
            "faces_recognized": 0,
            "people": [],
            "unknown_faces": 0,
        }

        resource_id = str(
            camera_resource_id or getattr(self, "_camera_resource_id", "") or ""
        ).strip()
        image, depth_m = self._capture_for_recognition(resource_id, timeout=2.0)
        if image is None:
            if self._depth_gate_settings is None:
                result["error"] = f"Failed to get image from camera resource {resource_id}"
            else:
                result["error"] = (
                    f"Failed to get synced RGBD from camera resource {resource_id}"
                )
            return result

        prepared = self._prepare_faces_for_recognition_result(
            image,
            depth_m,
            min_face_area=self._recognition_min_face_area(),
        )
        detected_faces = prepared.faces
        result["faces_detected"] = len(detected_faces)

        if not detected_faces:
            result["success"] = True
            result["error"] = "No faces detected in image"
            if prepared.reason:
                result["failure_reason"] = prepared.reason
            return result

        for face in detected_faces:
            match = self._recognize_face_match(face)
            if match is not None:
                updated_meta = match["metadata"]
                result["people"].append({
                    "name": match["name"],
                    "person_id": match["person_id"],
                    "confidence": match["similarity"],
                    "bbox": face["bbox"],
                    "depth_m": face.get("depth_m"),
                    "last_seen": updated_meta.get("last_seen", "unknown"),
                    "interaction_count": updated_meta.get("interaction_count", 0),
                })
                result["faces_recognized"] += 1
                logger.debug(
                    "Recognized face name=%s similarity=%.2f",
                    match["name"],
                    match["similarity"],
                )
            else:
                result["unknown_faces"] += 1
                logger.debug("Detected unknown face")

        result["success"] = True
        return result

    @staticmethod
    def _enrollment_response(
        *,
        success: bool,
        status: str,
        message: str,
        recognized_name: str = "",
        person_id: str = "",
        next_step_hint: str = "",
        failure_reason: str = "",
        enrollment_diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "success": success,
            "status": status,
            "message": message,
        }
        if recognized_name:
            payload["recognized_name"] = recognized_name
        if person_id:
            payload["person_id"] = person_id
        if next_step_hint:
            payload["next_step_hint"] = next_step_hint
        if failure_reason:
            payload["failure_reason"] = failure_reason
        if enrollment_diagnostics:
            payload["enrollment_diagnostics"] = enrollment_diagnostics
        return payload

    def enroll_visible_person(
        self,
        *,
        official_name: str = "",
        username: str = "",
        employee_profile: dict[str, Any] | None = None,
        camera_resource_id: str | None = None,
        display_runtime: Any | None = None,
    ) -> dict[str, Any]:
        """Validate a short enrollment burst and save exactly one new visible person."""
        candidate, failure = self._prepare_visible_person_enrollment(
            official_name=official_name,
            username=username,
            employee_profile=employee_profile,
            camera_resource_id=camera_resource_id,
            display_runtime=display_runtime,
        )
        if failure is not None:
            return failure
        if candidate is None:
            return self._enrollment_response(
                success=False,
                status="error",
                message="I couldn't prepare a face registration candidate.",
                failure_reason="candidate_unavailable",
            )

        if display_runtime is not None and bool(
            getattr(display_runtime, "is_configured", False)
        ):
            preview_url = self._enrollment_preview_data_url(candidate.preview_image)
            if not preview_url:
                return self._enrollment_response(
                    success=False,
                    status="display_unavailable",
                    message=(
                        "I captured a candidate face, but couldn't prepare the screen preview. "
                        "Please try again."
                    ),
                    failure_reason="preview_encoding_failed",
                )
            try:
                review = display_runtime.review_face_capture(
                    request_id=f"enroll-{uuid4().hex[:12]}",
                    image_url=preview_url,
                    title="Face Capture Preview",
                    accept_label="Accept",
                    reject_label="Reject",
                )
            except Exception as exc:
                logger.warning("Enrollment display review failed: %s", exc)
                review = {
                    "available": False,
                    "accepted": False,
                    "status": "display_unavailable",
                }
            status = str(review.get("status", "") or "").strip()
            if not bool(review.get("available", False)):
                return self._enrollment_response(
                    success=False,
                    status="display_unavailable",
                    message=(
                        "I captured a candidate face, but the review screen isn't available. "
                        "Please try again when the screen is ready."
                    ),
                    failure_reason="display_unavailable",
                )
            if status == "review_timeout":
                return self._enrollment_response(
                    success=False,
                    status="review_timeout",
                    message=(
                        "I didn't get an accept or reject on the screen in time. "
                        "Please tell me if you'd like to try again."
                    ),
                    failure_reason="review_timeout",
                )
            if not bool(review.get("accepted", False)):
                return self._enrollment_response(
                    success=False,
                    status="user_rejected_preview",
                    message=(
                        "No problem, I won't save that face capture. "
                        "Please tell me if you'd like to try again."
                    ),
                    failure_reason="user_rejected_preview",
                )

        return self._commit_visible_person_enrollment(candidate)

    def _prepare_visible_person_enrollment(
        self,
        *,
        official_name: str = "",
        username: str = "",
        employee_profile: dict[str, Any] | None = None,
        camera_resource_id: str | None = None,
        display_runtime: Any | None = None,
    ) -> tuple[FaceEnrollmentCandidate | None, dict[str, Any] | None]:
        verified_profile = {
            key: str(value or "").strip()
            for key, value in dict(employee_profile or {}).items()
            if str(value or "").strip()
        }
        cleaned_name = str(
            verified_profile.get("official_name") or official_name or ""
        ).strip()
        if not cleaned_name:
            return (
                None,
                self._enrollment_response(
                    success=False,
                    status="error",
                    message="I still need your name before I can save a new face.",
                    failure_reason="missing_name",
                ),
            )

        accepted_faces: list[dict[str, Any]] = []
        single_face_seen = False
        last_quality: FaceEnrollmentQuality | None = None
        last_prepare_reason = ""
        resource_id = str(
            camera_resource_id or getattr(self, "_camera_resource_id", "") or ""
        ).strip()

        for attempt in range(ENROLLMENT_BURST_FRAMES):
            image, depth_m = self._capture_for_recognition(resource_id, timeout=1.5)
            if image is None:
                return (
                    None,
                    self._enrollment_response(
                        success=False,
                        status="error",
                        message="I couldn't get a clear camera view right now. Please try again in a moment.",
                        failure_reason="capture_failed",
                    ),
                )

            prepared = self._prepare_faces_for_recognition_result(image, depth_m)
            usable_faces = list(prepared.faces)
            raw_detected_count = int(prepared.detected_count or 0)
            rejected_count = int(prepared.rejected_count or 0)
            if raw_detected_count > len(usable_faces):
                logger.debug(
                    "Enrollment ignored %s face(s) outside the usable registration scene raw_detected=%s usable=%s rejected=%s",
                    raw_detected_count - len(usable_faces),
                    raw_detected_count,
                    len(usable_faces),
                    rejected_count,
                )
            if not usable_faces:
                last_prepare_reason = prepared.reason
                if attempt < (ENROLLMENT_BURST_FRAMES - 1):
                    time.sleep(ENROLLMENT_BURST_SLEEP_SEC)
                continue

            face, multiple_people_visible = self._select_enrollment_face(usable_faces)
            if multiple_people_visible:
                logger.info(
                    "Enrollment blocked by multiple usable faces raw_detected=%s usable=%s rejected=%s",
                    raw_detected_count,
                    len(usable_faces),
                    rejected_count,
                )
                self._show_multiple_faces_enrollment_preview(
                    display_runtime,
                    image=image,
                    faces=usable_faces,
                )
                return (
                    None,
                    self._enrollment_response(
                        success=False,
                        status="retry_single_face",
                        message="I can still see more than one face. Please make sure you're the only person in view and try again.",
                        failure_reason="multiple_faces",
                    ),
                )
            if face is None:
                if attempt < (ENROLLMENT_BURST_FRAMES - 1):
                    time.sleep(ENROLLMENT_BURST_SLEEP_SEC)
                continue

            single_face_seen = True
            quality = self._assess_enrollment_face_quality(image, face)
            if not quality.accepted:
                last_quality = quality
                logger.debug(
                    "Enrollment frame rejected reason=%s metric=%.3f guidance=%s",
                    quality.reason,
                    quality.metric,
                    quality.guidance,
                )
                if attempt < (ENROLLMENT_BURST_FRAMES - 1):
                    time.sleep(ENROLLMENT_BURST_SLEEP_SEC)
                continue
            match = self._recognize_face_match(face)
            if match is not None:
                return (
                    None,
                    self._enrollment_response(
                        success=False,
                        status="retry_already_known",
                        message=f"I think I already know you as {match['name']}.",
                        recognized_name=match["name"],
                        failure_reason="already_known",
                    ),
                )

            accepted_faces.append(
                {
                    "image_shape": image.shape,
                    "image": image.copy(),
                    "face": face,
                    "bbox_area": self._bbox_area(face),
                    "center_distance": self._center_distance(face, image.shape),
                }
            )
            if attempt < (ENROLLMENT_BURST_FRAMES - 1):
                time.sleep(ENROLLMENT_BURST_SLEEP_SEC)

        if len(accepted_faces) < ENROLLMENT_REQUIRED_STABLE_FRAMES:
            failure_reason = (
                (last_quality.reason if last_quality is not None else last_prepare_reason)
                or "unstable_face"
            )
            if last_quality is not None and last_quality.guidance:
                message = last_quality.guidance
            elif last_prepare_reason == "depth_rejected":
                message = (
                    "I can see a face, but I need a closer face view with valid depth. "
                    "Please stand within about two meters and face me."
                )
            elif last_prepare_reason == "no_embedding":
                message = (
                    "I can see a face, but I couldn't encode it clearly. "
                    "Please face me directly and hold still for a second."
                )
            elif single_face_seen:
                message = (
                    "Please come a little closer and look at the camera so I can get a stable face view."
                )
            else:
                message = (
                    "I couldn't get a stable face view. Please stand in front of me, "
                    "look at the camera, and try again."
                )
            return (
                None,
                self._enrollment_response(
                    success=False,
                    status="retry_quality",
                    message=message,
                    failure_reason=failure_reason,
                ),
            )

        reference_item = min(
            accepted_faces,
            key=lambda item: (
                item["face"].get("depth_m")
                if item["face"].get("depth_m") is not None
                else float("inf"),
                -float(item["bbox_area"]),
                float(item["center_distance"]),
            ),
        )
        reference_face = reference_item["face"]
        similarity_threshold = getattr(
            self,
            "_enrollment_policy",
            DEFAULT_FACE_ENROLLMENT_POLICY,
        ).min_embedding_similarity
        consistent_faces = [
            item["face"]
            for item in accepted_faces
            if self._embedding_similarity(
                reference_face["embedding"],
                item["face"]["embedding"],
            )
            >= similarity_threshold
        ]
        if len(consistent_faces) < ENROLLMENT_REQUIRED_STABLE_FRAMES:
            reason, guidance = self._quality_response_for_reason("embedding_inconsistent")
            diagnostics = self._enrollment_similarity_diagnostics(
                accepted_faces=accepted_faces,
                reference_item=reference_item,
                threshold=float(similarity_threshold),
            )
            logger.info(
                "Enrollment burst rejected reason=%s accepted=%s consistent=%s threshold=%.3f best_failed_similarity=%s shortfall=%s similarities=%s",
                reason,
                len(accepted_faces),
                len(consistent_faces),
                float(similarity_threshold),
                diagnostics.get("best_failed_similarity"),
                diagnostics.get("best_failed_shortfall"),
                diagnostics.get("similarities_to_reference"),
            )
            return (
                None,
                self._enrollment_response(
                    success=False,
                    status="retry_quality",
                    message=guidance,
                    failure_reason=reason,
                    enrollment_diagnostics=diagnostics,
                ),
            )
        averaged_embedding = self._average_embeddings(
            [face_payload["embedding"] for face_payload in consistent_faces]
        )
        if averaged_embedding.size <= 0:
            reason, guidance = self._quality_response_for_reason("embedding_inconsistent")
            logger.info("Enrollment burst rejected reason=%s empty averaged embedding", reason)
            return (
                None,
                self._enrollment_response(
                    success=False,
                    status="retry_quality",
                    message=guidance,
                    failure_reason=reason,
                ),
            )

        base_profile = {
            "official_name": str(official_name or "").strip(),
            "username": str(username or "").strip(),
        }
        verified_durable = {
            key: value
            for key, value in {**base_profile, **verified_profile}.items()
            if value
        }
        return (
            FaceEnrollmentCandidate(
                cleaned_name=cleaned_name,
                verified_durable=verified_durable,
                averaged_embedding=averaged_embedding,
                reference_face=reference_face,
                image_shape=reference_item["image_shape"],
                preview_image=self._enrollment_preview_image(
                    reference_item["image"],
                    reference_face,
                ),
            ),
            None,
        )

    def _commit_visible_person_enrollment(
        self,
        candidate: FaceEnrollmentCandidate,
    ) -> dict[str, Any]:
        identity_memory = getattr(self, "identity_memory_client", None)
        if identity_memory is None:
            return self._enrollment_response(
                success=False,
                status="error",
                message="Identity memory is unavailable, so I can't save this face yet.",
                failure_reason="identity_memory_unavailable",
            )
        person_id = (
            str(candidate.verified_durable.get("person_id") or "").strip()
            or _person_id_from_profile(candidate.cleaned_name, candidate.verified_durable)
        )
        metadata = {
            **dict(candidate.verified_durable or {}),
            "display_name": candidate.cleaned_name,
            "name": candidate.cleaned_name,
        }
        result = identity_memory.enroll_face_reference(
            person_id=person_id,
            embedding=candidate.averaged_embedding,
            model="facenet-vggface2",
            metadata=metadata,
            consent_status="consented",
        )
        if not getattr(result, "saved", False):
            return self._enrollment_response(
                success=False,
                status="error",
                message="I couldn't save the face reference.",
                failure_reason=str(getattr(result, "reason", "") or "save_failed"),
            )
        self._prime_presence_cache_after_enrollment(
            person_id=person_id,
            name=candidate.cleaned_name,
            face=candidate.reference_face,
            image_shape=candidate.image_shape,
            metadata=metadata,
        )
        return self._enrollment_response(
            success=True,
            status="enrolled",
            message=f"You're all set, {candidate.cleaned_name}. I'll remember you next time.",
            person_id=person_id,
            next_step_hint=(
                "Now continue with one short social follow-up to learn durable context. "
                "Pets are usually the best default topic, then preferred name, team, or current work."
            ),
        )

    def _show_multiple_faces_enrollment_preview(
        self,
        display_runtime: Any | None,
        *,
        image: Any,
        faces: list[dict[str, Any]],
    ) -> None:
        if display_runtime is None or not bool(
            getattr(display_runtime, "is_configured", False)
        ):
            return
        show_preview = getattr(display_runtime, "show_image_message_preview", None)
        if not callable(show_preview):
            return
        image_url = self._enrollment_preview_data_url(
            self._enrollment_diagnostic_image(image, faces)
        )
        if not image_url:
            return
        try:
            show_preview(
                image_url=image_url,
                title="Multiple Faces Detected",
                message=(
                    "I can see more than one face. Please make sure you are the only "
                    "person in view before enrollment."
                ),
                hold_sec=5.0,
            )
        except Exception as exc:
            logger.warning("Multiple-face enrollment preview failed: %s", exc)

    @staticmethod
    def _enrollment_diagnostic_image(image: Any, faces: list[dict[str, Any]]) -> Any:
        if image is None or not hasattr(image, "shape"):
            return image
        try:
            annotated = image.copy()
            height, width = annotated.shape[:2]
            for idx, face in enumerate(faces, start=1):
                bbox = face.get("bbox") or {}
                x = max(0, min(int(bbox.get("x", 0) or 0), width - 1))
                y = max(0, min(int(bbox.get("y", 0) or 0), height - 1))
                w = max(0, int(bbox.get("w", 0) or 0))
                h = max(0, int(bbox.get("h", 0) or 0))
                x2 = max(0, min(x + w, width - 1))
                y2 = max(0, min(y + h, height - 1))
                if x2 <= x or y2 <= y:
                    continue
                color = (255, 221, 0)
                cv2.rectangle(annotated, (x, y), (x2, y2), color, 2)
                confidence = float(face.get("confidence", 0.0) or 0.0)
                label = f"{idx}: {confidence:.2f}"
                text_y = max(14, y - 6)
                cv2.putText(
                    annotated,
                    label,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            return annotated
        except Exception:
            logger.exception("Failed to prepare multiple-face enrollment preview")
            return image.copy() if hasattr(image, "copy") else image

    @staticmethod
    def _enrollment_preview_data_url(image: Any) -> str:
        if image is None:
            return ""
        try:
            return "data:image/png;base64," + preprocess_image(image)
        except Exception:
            logger.exception("Failed to encode enrollment preview image")
            return ""

    @staticmethod
    def _live_frame_data_url(image: Any) -> str:
        if image is None:
            return ""
        try:
            return "data:image/png;base64," + preprocess_image(image)
        except Exception:
            logger.exception("Failed to encode live camera frame for display")
            return ""

    def _publish_live_image_frame(
        self,
        image: Any,
        *,
        faces: list[dict[str, Any]] | None = None,
    ) -> None:
        if not bool(getattr(self, "_live_image_enabled", True)):
            return
        display = getattr(self, "_display_runtime", None)
        if display is None or not bool(getattr(display, "is_configured", False)):
            return
        display_image = image
        if faces:
            try:
                display_image = draw_attention_overlay(image, faces)
            except Exception:
                logger.debug("Failed to draw attention overlay", exc_info=True)
                display_image = image
        data_url = self._live_frame_data_url(display_image)
        if not data_url:
            return
        show_live_image = getattr(display, "show_live_image", None)
        if not callable(show_live_image):
            return
        try:
            show_live_image(
                data_url=data_url,
                title=getattr(self, "_live_image_title", "Camera"),
                ttl_ms=int(getattr(self, "_live_image_ttl_ms", 1000)),
            )
        except Exception:
            logger.debug("Display live image update failed", exc_info=True)

    def _clear_live_image_frame(self) -> None:
        if not bool(getattr(self, "_live_image_enabled", True)):
            return
        display = getattr(self, "_display_runtime", None)
        if display is None or not bool(getattr(display, "is_configured", False)):
            return
        clear_live_image = getattr(display, "clear_live_image", None)
        if not callable(clear_live_image):
            return
        try:
            clear_live_image()
        except Exception:
            logger.debug("Display live image clear failed", exc_info=True)

    def _prime_presence_cache_after_enrollment(
        self,
        *,
        person_id: str,
        name: str,
        face: dict[str, Any],
        image_shape: tuple[int, ...],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Seed the live presence cache so post-enrollment turns see a clean recognized face."""
        cache = getattr(self, "_presence_cache", None)
        if cache is None:
            return

        now = time.time()
        bbox_area = self._bbox_area(face)
        center_distance = self._center_distance(face, image_shape)
        depth_m = face.get("depth_m")
        person = PersonContext(
            person_id=str(person_id or "").strip(),
            name=str(name or "").strip(),
            interaction_count=0,
            confidence=1.0,
            bbox_area=bbox_area,
            timestamp=now,
            depth_m=depth_m,
            center_distance=center_distance,
            directory_profile_lines=(),
        )
        attention_target = AttentionTarget(
            kind="recognized",
            depth_m=depth_m,
            bbox_area=bbox_area,
            center_distance=center_distance,
            person_id=person.person_id,
            name=person.name,
        )
        social_scene = SocialSceneContext(
            has_unrecognized_people=False,
            closest_person_kind="recognized",
            nearest_recognized_name=person.name,
        )
        cache.mark_faces_seen(now)
        cache.mark_person_seen(person.person_id, now)
        cache.update(
            persons=[person],
            faces_detected=1,
            unknown_count=0,
            attention_target=attention_target,
            social_scene=social_scene,
            now=now,
        )
        self._notify_presence_subscribers(self.get_presence_snapshot())

    # ------------------------------------------------------------------
    # Background loop (continuous recognition, zero latency at response time)
    # ------------------------------------------------------------------

    def _emit_loop_timing(
        self,
        *,
        tick_started: float,
        camera_resource_id: str,
        interval_sec: float | None,
        outcome: str,
        capture_s: float | None = None,
        prepare_s: float | None = None,
        scene_s: float | None = None,
        publish_s: float | None = None,
        detected_count: int | None = None,
        recognized_count: int | None = None,
        unknown_count: int | None = None,
        rejected_count: int | None = None,
        reason: str | None = None,
        prepare_details: list[str] | None = None,
        recognition_details: list[str] | None = None,
    ) -> None:
        latency = getattr(self, "_loop_latency", None)
        if latency is None:
            return
        heartbeat_at = getattr(self, "_loop_metric_heartbeat_at", None)
        if heartbeat_at is None:
            heartbeat_at = {}
            self._loop_metric_heartbeat_at = heartbeat_at
        now = time.time()
        key = f"timing:{outcome}"
        last = float(heartbeat_at.get(key, 0.0))
        if last and (now - last) < LOOP_HEARTBEAT_LOG_SEC:
            return
        heartbeat_at[key] = now
        latency.timing(
            "tick",
            perf_now() - tick_started,
            camera_resource=camera_resource_id,
            interval_s=interval_sec,
            outcome=outcome,
            capture_s=capture_s,
            prepare_s=prepare_s,
            scene_s=scene_s,
            publish_s=publish_s,
            detected=detected_count,
            recognized=recognized_count,
            unknown=unknown_count,
            rejected=rejected_count,
            reason=reason,
            prepare_details=prepare_details,
            recognition_details=recognition_details,
        )

    def _loop_tick(
        self,
        camera_resource_id: str,
        *,
        interval_sec: float | None = None,
    ) -> None:
        """Run one recognition cycle; update cache and session state."""
        tick_started = perf_now()
        now = time.time()
        capture_started = perf_now()
        image, depth_m = self._capture_for_recognition(camera_resource_id, timeout=1.5)
        capture_s = perf_now() - capture_started
        if image is None:
            if self._clear_presence_if_expired(now):
                logger.debug("[FaceLoop] no image, cache expired and cleared")
            self._emit_loop_timing(
                tick_started=tick_started,
                camera_resource_id=camera_resource_id,
                interval_sec=interval_sec,
                outcome="no_image",
                capture_s=capture_s,
            )
            return

        self._cache_latest_loop_frame(
            image=image,
            camera_resource_id=camera_resource_id,
            captured_at=time.time(),
        )
        prepare_started = perf_now()
        prepared = self._prepare_faces_for_recognition_result(
            image,
            depth_m,
            min_face_area=self._recognition_min_face_area(),
        )
        prepare_s = perf_now() - prepare_started
        detected_faces = prepared.faces
        if not detected_faces:
            self._reset_unknown_stability()
            publish_started = perf_now()
            self._publish_live_image_frame(image)
            publish_s = perf_now() - publish_started
            if self._clear_presence_if_expired(now):
                logger.debug("[FaceLoop] no faces, cache expired and cleared")
            self._emit_loop_timing(
                tick_started=tick_started,
                camera_resource_id=camera_resource_id,
                interval_sec=interval_sec,
                outcome="no_faces",
                capture_s=capture_s,
                prepare_s=prepare_s,
                publish_s=publish_s,
                detected_count=prepared.detected_count,
                recognized_count=0,
                unknown_count=0,
                rejected_count=prepared.rejected_count,
                reason=prepared.reason,
                prepare_details=prepared.rejection_details or None,
            )
            return

        self._presence_cache.mark_faces_seen(now)
        scene_started = perf_now()
        raw_persons, raw_unknown_count, _, _ = self._build_scene_state(
            image=image,
            detected_faces=detected_faces,
            image_shape=image.shape,
            now=now,
        )
        persons, unknown_count, current_ids, analysis = self._stable_scene_state(
            detected_faces=detected_faces,
            raw_persons=raw_persons,
            image_shape=image.shape,
            now=now,
        )
        unknown_stability_frames, attentive_unknown_stability_frames = (
            self._update_unknown_stability(
                unknown_count=unknown_count,
                attentive_unknown_count=analysis.attentive_unknown_count,
            )
        )
        scene_s = perf_now() - scene_started
        publish_started = perf_now()
        self._publish_live_image_frame(image, faces=detected_faces)
        publish_s = perf_now() - publish_started

        self._presence_cache.expire_inactive(current_ids, now)
        self._presence_cache.update(
            persons=persons,
            faces_detected=len(detected_faces),
            unknown_count=unknown_count,
            attentive_unknown_count=analysis.attentive_unknown_count,
            unknown_stability_frames=unknown_stability_frames,
            attentive_unknown_stability_frames=attentive_unknown_stability_frames,
            attention_target=analysis.attention_target,
            primary_attention_target=analysis.primary_attention_target,
            face_match_evidence=self._best_face_match_evidence(detected_faces),
            social_scene=analysis.social_scene,
            now=now,
        )
        self._notify_presence_subscribers(self.get_presence_snapshot())
        attentive_names = [p.name for p in persons if bool(p.attentive)]
        recognized_details = self._format_recognition_log_details(persons)
        recognition_rejection_details = self._format_recognition_attempt_log_details(
            detected_faces
        )
        primary_attention = analysis.primary_attention_target
        primary_face = (
            analysis.attention_target.person_id if analysis.attention_target else None
        )
        primary_attention_label = (
            primary_attention.person_id
            if primary_attention and primary_attention.person_id
            else (primary_attention.kind if primary_attention else None)
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[FaceLoop] detected %s face(s), recognized=%s unknown=%s "
                "raw_recognized=%s raw_unknown=%s attentive=%s attentive_unknown=%s primary_face=%s primary_attention=%s "
                "recognition_rejections=%s attention_details=%s",
                len(detected_faces),
                recognized_details,
                unknown_count,
                len(raw_persons),
                raw_unknown_count,
                attentive_names,
                analysis.attentive_unknown_count,
                primary_face,
                primary_attention_label,
                recognition_rejection_details,
                self._format_attention_log_details(detected_faces),
            )
        self._emit_loop_timing(
            tick_started=tick_started,
            camera_resource_id=camera_resource_id,
            interval_sec=interval_sec,
            outcome="ok",
            capture_s=capture_s,
            prepare_s=prepare_s,
            scene_s=scene_s,
            publish_s=publish_s,
            detected_count=len(detected_faces),
            recognized_count=len(persons),
            unknown_count=unknown_count,
            rejected_count=prepared.rejected_count,
            reason="ok",
            prepare_details=prepared.rejection_details or None,
            recognition_details=recognition_rejection_details or None,
        )
        self._log_loop_heartbeat(
            "attention_summary",
            "[FaceLoop] summary detected=%s rejected=%s recognized=%s unknown=%s "
            "attentive=%s attentive_unknown=%s primary_face=%s primary_attention=%s",
            len(detected_faces),
            prepared.rejected_count,
            len(persons),
            unknown_count,
            len(attentive_names),
            analysis.attentive_unknown_count,
            primary_face,
            primary_attention_label,
        )

    def _format_recognition_log_details(self, persons: list[PersonContext]) -> list[str]:
        details: list[str] = []
        for person in persons:
            label = str(person.name or person.person_id).replace(" ", "_")
            threshold = float(getattr(person, "recognition_threshold", 0.0) or 0.0)
            details.append(f"{label}:sim={person.confidence:.2f},threshold={threshold:.2f}")
        return details

    @staticmethod
    def _format_recognition_attempt_log_details(faces: list[dict[str, Any]]) -> list[str]:
        details: list[str] = []
        for index, face in enumerate(faces):
            recognition = dict(face.get("recognition") or {})
            reason = str(recognition.get("reason") or "not_evaluated")
            if reason == "matched":
                continue
            label = str(
                recognition.get("name")
                or face.get("recognized_name")
                or f"face{index}"
            ).replace(" ", "_")
            similarity = recognition.get("similarity")
            threshold = recognition.get("threshold")
            runner_up = recognition.get("runner_up_similarity")
            margin = recognition.get("margin")
            margin_threshold = recognition.get("margin_threshold")
            parts = [f"{label}:{reason}"]
            if similarity is not None:
                parts.append(f"sim={float(similarity):.2f}")
            if threshold is not None:
                parts.append(f"threshold={float(threshold):.2f}")
            if runner_up is not None:
                parts.append(f"runner_up={float(runner_up):.2f}")
            if margin is not None:
                parts.append(f"margin={float(margin):.2f}")
            if margin_threshold is not None:
                parts.append(f"margin_threshold={float(margin_threshold):.2f}")
            details.append(",".join(parts))
        return details

    @staticmethod
    def _format_attention_log_details(faces: list[dict[str, Any]]) -> list[str]:
        details: list[str] = []
        for index, face in enumerate(faces):
            attention = face.get("attention")
            if attention is None:
                details.append(f"face{index}:missing")
                continue
            label = str(face.get("recognized_name") or f"face{index}").replace(" ", "_")
            reason = str(getattr(attention, "reason", "") or "unknown")
            attentive = "yes" if bool(getattr(attention, "attentive", False)) else "no"
            raw = "yes" if bool(getattr(attention, "raw_attentive", False)) else "no"
            confidence = float(getattr(attention, "confidence", 0.0) or 0.0)
            raw_confidence = float(getattr(attention, "raw_confidence", 0.0) or 0.0)
            yaw = _format_optional_float(getattr(attention, "yaw_deg", None))
            pitch = _format_optional_float(getattr(attention, "pitch_deg", None))
            roll = _format_optional_float(getattr(attention, "roll_deg", None))
            details.append(
                f"{label}:att={attentive},raw={raw},reason={reason},"
                f"conf={confidence:.2f},raw_conf={raw_confidence:.2f},"
                f"yaw={yaw},pitch={pitch},roll={roll}"
            )
        return details

    def _log_loop_heartbeat(
        self,
        key: str,
        message: str,
        *args: object,
        interval_sec: float = LOOP_HEARTBEAT_LOG_SEC,
        level: int = logging.INFO,
    ) -> None:
        """Log noisy recurring face-loop states at a throttled cadence."""
        now = time.time()
        heartbeat_at = getattr(self, "_loop_log_heartbeat_at", None)
        if heartbeat_at is None:
            heartbeat_at = {}
            self._loop_log_heartbeat_at = heartbeat_at
        last = float(heartbeat_at.get(key, 0.0))
        if last and (now - last) < interval_sec:
            return
        heartbeat_at[key] = now
        logger.log(level, message, *args)

    def start_loop(
        self,
        camera_resource_id: str | None = None,
        interval: float = 1.0,
    ) -> None:
        """Start background daemon thread that runs recognition every interval seconds."""
        if self._loop_thread is not None and self._loop_thread.is_alive():
            logger.warning("[FaceLoop] already running")
            return
        self._loop_stop.clear()
        resource_id = str(
            camera_resource_id or getattr(self, "_camera_resource_id", "") or ""
        ).strip()
        self._camera_resource_id = resource_id

        def run() -> None:
            while not self._loop_stop.wait(interval):
                try:
                    self._loop_tick(resource_id, interval_sec=interval)
                except Exception as e:
                    if is_provider_error(e):
                        self._log_loop_heartbeat(
                            "robot_provider_unavailable",
                            "[FaceLoop] robot provider capability unavailable: %s",
                            e,
                            interval_sec=10.0,
                            level=logging.WARNING,
                        )
                    else:
                        logger.exception("[FaceLoop] tick failed: %s", e)

        self._loop_thread = threading.Thread(target=run, daemon=True)
        self._loop_thread.start()
        logger.info(
            "[FaceLoop] started (interval=%ss, camera_resource=%s)",
            interval,
            resource_id,
        )

    def stop_loop(self) -> None:
        """Stop the background recognition loop."""
        self._loop_stop.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=3.0)
            self._loop_thread = None
        self._clear_live_image_frame()
        logger.info("[FaceLoop] stopped")

    def shutdown(self) -> None:
        """Stop the face loop."""
        self.stop_loop()

    def get_cached_persons(self) -> list[PersonContext]:
        """
        Thread-safe read of cache: persons in current frame or last seen within CACHE_EXPIRE_SEC.
        Used for owner-scoped prompt context and recognized-person personalization.
        """
        return self._presence_cache.get_cached_persons()

    def get_attention_target_person_id(self) -> str | None:
        """Return the current recognized attention target person id, if any."""
        return self._presence_cache.get_attention_target_person_id()

    def get_primary_face_person_id(self) -> str | None:
        """Return the current recognized primary visible person id, if any."""
        return self._presence_cache.get_primary_face_person_id()

    def get_primary_attention_person_id(self) -> str | None:
        """Return the current recognized primary attentive person id, if any."""
        return self._presence_cache.get_primary_attention_person_id()

    def get_face_turn_target(self, person_id: str | None = None):
        """Return the current recognized face-bearing target for a person id."""
        return self._presence_cache.get_face_turn_target(person_id)

    def get_presence_snapshot(self) -> dict[str, Any]:
        """Thread-safe read of canonical face presence state."""
        return self._presence_cache.get_presence_snapshot()

    def get_recent_face_observation(
        self,
        person_id: str,
        *,
        max_age_sec: float = 8.0,
    ) -> dict[str, Any] | None:
        """Return the latest accepted face embedding for a person in this session."""
        rendered = str(person_id or "").strip()
        if not rendered:
            return None
        now = time.time()
        if not hasattr(self, "_recent_face_observations_lock"):
            self._recent_face_observations_lock = threading.Lock()
        if not hasattr(self, "_recent_face_observations"):
            self._recent_face_observations = {}
        with self._recent_face_observations_lock:
            observation = dict(self._recent_face_observations.get(rendered) or {})
        if not observation:
            return None
        observed_at = float(observation.get("observed_at", 0.0) or 0.0)
        if max_age_sec > 0.0 and (now - observed_at) > float(max_age_sec):
            return None
        return observation

    def _remember_recent_face_observation(
        self,
        *,
        person_id: str,
        embedding: Any,
        metadata: dict[str, Any],
    ) -> None:
        rendered = str(person_id or "").strip()
        if not rendered or embedding is None:
            return
        try:
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1).copy()
        except Exception:
            return
        if vector.size <= 0:
            return
        if not hasattr(self, "_recent_face_observations_lock"):
            self._recent_face_observations_lock = threading.Lock()
        if not hasattr(self, "_recent_face_observations"):
            self._recent_face_observations = {}
        with self._recent_face_observations_lock:
            self._recent_face_observations[rendered] = {
                "person_id": rendered,
                "embedding": vector,
                "model": "facenet-vggface2",
                "metadata": dict(metadata or {}),
                "observed_at": time.time(),
            }


def _person_id_from_profile(name: str, profile: dict[str, Any]) -> str:
    username = str(profile.get("username") or "").strip().lower()
    if username:
        return f"person_{username}"
    email = str(profile.get("email") or "").strip().lower()
    if email and "@" in email:
        return f"person_{email.split('@', 1)[0]}"
    slug = "_".join(
        part
        for part in re.sub(r"[^a-zA-Z0-9]+", " ", str(name or "").casefold()).split()
        if part
    )
    if slug:
        return f"person_{slug}"
    return f"person_{uuid4().hex[:12]}"


def _profile_directory_lines(profile: Any) -> tuple[str, ...]:
    if isinstance(profile, dict):
        lines = profile.get("directory_profile_lines") or ()
    else:
        lines = getattr(profile, "directory_profile_lines", ()) or ()
    return tuple(str(line) for line in lines if str(line or "").strip())
