"""Face recognition service orchestration for Go2."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import torch

from argos_src.face_recognition.bearing import estimate_robot_yaw_error_rad, face_center_px
from argos_src.face_recognition.constants import (
    DEFAULT_FACE_DB_PATH,
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
from argos_src.face_recognition.store import FaceRecognitionStore
from argos_src.identity.prompting import format_identity_profile_lines
from argos_src.memory_provider.encounters import build_encounter_metadata
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
    min_face_area: int = 1600
    min_sharpness: float = 20.0
    min_brightness: float = 30.0
    max_brightness: float = 220.0
    min_contrast: float = 14.0
    max_eye_tilt: float = 0.25
    max_nose_center_offset: float = 0.10
    min_embedding_similarity: float = 0.70


DEFAULT_FACE_ENROLLMENT_POLICY = FaceEnrollmentPolicy()


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
    has_required_landmarks: bool = False
    eye_tilt: float = 0.0
    nose_center_offset: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    sharpness: float = 0.0


@dataclass(frozen=True)
class FacePreparationResult:
    faces: list[dict[str, Any]]
    reason: str = ""
    detected_count: int = 0
    rejected_count: int = 0


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
    - ChromaDB: persistent storage + similarity search
    """

    def __init__(
        self,
        db_path: str = DEFAULT_FACE_DB_PATH,
        recognition_threshold: float = 0.6,
        robot_client: Any | None = None,
        depth_gate_settings: Optional[DepthGateSettings] = None,
        attention_gate_settings: AttentionGateSettings | None = None,
        enrollment_policy: FaceEnrollmentPolicy | None = None,
        identity_db_path: str | None = None,
        identity_store: Any | None = None,
        memory_store: Any | None = None,
        site_code: str = "",
        camera_resource_id: str = "",
        camera_yaw_offset_rad: float = 0.0,
        display_runtime: Any | None = None,
        live_image_title: str = "Camera",
        live_image_ttl_ms: int = 1000,
    ):
        self.device = FaceEmbeddingPipeline.resolve_device()
        logger.info(f"Face recognition running on: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        self._pipeline: Optional[FaceEmbeddingPipeline] = None
        self.mtcnn = None
        self.resnet = None

        self.robot_client = robot_client
        self._recognition_threshold = float(recognition_threshold)
        self.db = FaceRecognitionStore(
            db_path=db_path,
            identity_db_path=identity_db_path,
            identity_store=identity_store,
        )
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
        self._enrollment_policy = enrollment_policy or DEFAULT_FACE_ENROLLMENT_POLICY
        self._presence_cache = FacePresenceCache(cache_expire_sec=CACHE_EXPIRE_SEC)
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_stop = threading.Event()
        self._latest_loop_frame_lock = threading.Lock()
        self._latest_loop_frame = None
        self._latest_loop_frame_resource_id: str | None = None
        self._latest_loop_frame_at = 0.0
        self._loop_log_heartbeat_at: dict[str, float] = {}
        self._loop_metric_heartbeat_at: dict[str, float] = {}
        self._loop_latency = LatencyLogger("face_loop")

        logger.info("Face Recognition Service initialized")
        logger.info(f"Device: {self.device} | DB: {self.db.collection.count()} people enrolled")

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

        logger.debug(
            "Captured synced RGBD pair for face recognition (delta_ms=%.2f)",
            rgbd.delta_ms,
        )
        return rgbd.color_image, rgbd.depth_m

    def _prepare_faces_for_recognition(
        self,
        image,
        depth_m,
    ) -> list[dict[str, Any]]:
        """Run detection, optional depth gating, then embedding extraction."""
        return self._prepare_faces_for_recognition_result(image, depth_m).faces

    def _prepare_faces_for_recognition_result(
        self,
        image,
        depth_m,
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
        if depth_m is not None and self._depth_gate_settings is not None:
            gated_faces, rejected_count = filter_detections_by_depth(
                detected_faces,
                depth_m,
                self._depth_gate_settings,
            )
            if rejected_count:
                kept = len(gated_faces)
                total = len(detected_faces)
                if kept == 0:
                    self._log_loop_heartbeat(
                        "depth_gate_rejected_all",
                        "[FaceLoop] depth gate kept 0/%s face(s); likely no valid aligned depth samples or face beyond %.2fm",
                        total,
                        self._depth_gate_settings.max_face_depth_m,
                    )
                else:
                    logger.info(
                        "[FaceLoop] depth gate kept %s/%s face(s)",
                        kept,
                        total,
                    )
            detected_faces = gated_faces
            if rejected_count and not detected_faces:
                logger.debug("[FaceLoop] depth gate rejected all detected faces")
                return FacePreparationResult(
                    faces=[],
                    reason="depth_rejected",
                    detected_count=detected_count,
                    rejected_count=rejected_count,
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
            )
        return FacePreparationResult(
            faces=faces_with_embeddings,
            detected_count=detected_count,
            rejected_count=rejected_count,
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
        Enrollment should block only when more than one usable face remains.
        """
        if not detected_faces:
            return None, False
        if len(detected_faces) == 1:
            return detected_faces[0], False

        ordered = sorted(
            detected_faces,
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
        logger.info(
            "Enrollment ignored %s extra face detection(s) and kept the strongest candidate.",
            len(ordered) - 1,
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
            "missing_landmarks": "Please face me directly.",
            "side_face": "Please face me directly.",
            "too_blurry": "Hold still for a second so the image is clear.",
            "too_dark": "Please move to better light.",
            "too_bright": "Please move away from the bright light.",
            "low_contrast": "Please move to better light.",
            "embedding_inconsistent": "Hold still and face me directly for a second.",
        }
        return reason, guidance_by_reason.get(
            reason,
            "Please face me directly and hold still for a second.",
        )

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

        landmarks = face.get("landmarks") or {}
        required = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
        has_required_landmarks = all(name in landmarks for name in required)
        eye_tilt = 0.0
        nose_offset = 0.0
        if has_required_landmarks:
            left_eye = np.asarray(landmarks["left_eye"], dtype=np.float32)
            right_eye = np.asarray(landmarks["right_eye"], dtype=np.float32)
            nose = np.asarray(landmarks["nose"], dtype=np.float32)
            eye_distance = float(np.linalg.norm(right_eye - left_eye))
            if eye_distance > 1e-6:
                eye_tilt = abs(float(left_eye[1] - right_eye[1])) / eye_distance
                nose_offset = abs(float(nose[0] - ((left_eye[0] + right_eye[0]) / 2.0))) / eye_distance
            else:
                has_required_landmarks = False

        crop = image[y : y + h, x : x + w]
        crop_valid = bool(crop.size > 0)
        brightness = 0.0
        contrast = 0.0
        sharpness = 0.0
        if crop_valid:
            gray = crop.astype(np.float32).mean(axis=2) if crop.ndim == 3 else crop.astype(np.float32)
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            if gray.shape[0] >= 2 and gray.shape[1] >= 2:
                dy, dx = np.gradient(gray)
                sharpness = float(np.var(dx) + np.var(dy))
        return FaceEnrollmentQualityMetrics(
            bbox_area=bbox_area,
            clipped=clipped,
            crop_valid=crop_valid,
            has_required_landmarks=has_required_landmarks,
            eye_tilt=eye_tilt,
            nose_center_offset=nose_offset,
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
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
        if not metrics.has_required_landmarks:
            reason, guidance = self._quality_response_for_reason("missing_landmarks")
            return FaceEnrollmentQuality(False, reason, guidance, 0.0)
        if (
            metrics.eye_tilt > policy.max_eye_tilt
            or metrics.nose_center_offset > policy.max_nose_center_offset
        ):
            reason, guidance = self._quality_response_for_reason("side_face")
            return FaceEnrollmentQuality(
                False,
                reason,
                guidance,
                max(metrics.eye_tilt, metrics.nose_center_offset),
            )
        if not metrics.crop_valid:
            reason, guidance = self._quality_response_for_reason("face_clipped")
            return FaceEnrollmentQuality(False, reason, guidance, 0.0)
        if metrics.sharpness < policy.min_sharpness:
            reason, guidance = self._quality_response_for_reason("too_blurry")
            return FaceEnrollmentQuality(False, reason, guidance, metrics.sharpness)
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

    def _recognize_face_match(self, face: dict[str, Any]) -> dict[str, Any] | None:
        matches = self.db.recognize_face(
            face_embedding=face["embedding"],
            threshold=self._recognition_threshold,
            top_k=1,
        )
        if not matches:
            return None
        return matches[0]

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
            match = self._recognize_face_match(face)
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
            should_record_interaction = self._presence_cache.should_record_interaction(
                pid,
                now,
            )
            self._presence_cache.mark_person_seen(pid, now)
            meta = match["metadata"]
            if should_record_interaction:
                try:
                    updated_meta = self.db.update_interaction(pid)
                except Exception:
                    logger.exception(
                        "Failed to update interaction metadata for person_id=%s",
                        pid,
                    )
                else:
                    if updated_meta is not None:
                        meta = {**dict(meta or {}), **dict(updated_meta or {})}
                        memory_store = getattr(self, "memory_store", None)
                        if memory_store is not None:
                            try:
                                site_code = str(getattr(self, "site_code", "") or "").strip()
                                memory_store.record_encounter(
                                    person_id=pid,
                                    name=match["name"],
                                    site_code=site_code,
                                    metadata=build_encounter_metadata(
                                        name=match["name"],
                                        site_code=site_code,
                                        identity_metadata=meta,
                                    ),
                                )
                            except Exception:
                                logger.exception(
                                    "Failed to record encounter for person_id=%s",
                                    pid,
                                )
                finally:
                    self._presence_cache.mark_interaction_recorded(pid, now)
            person = PersonContext(
                person_id=pid,
                name=match["name"],
                interaction_count=meta.get("interaction_count", 0),
                confidence=match["similarity"],
                bbox_area=bbox_area,
                timestamp=now,
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
                directory_profile_lines=format_identity_profile_lines(meta),
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

        analysis = analyze_face_scene(candidates)
        return persons, unknown_count, current_ids, analysis

    @staticmethod
    def _unknown_attention_track_id(
        face: dict[str, Any],
        image_shape: tuple[int, ...],
    ) -> str:
        """Return a coarse stable key for smoothing unknown attentive faces."""
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

        prepared = self._prepare_faces_for_recognition_result(image, depth_m)
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
                now = time.time()
                updated_meta = match["metadata"]
                if self._presence_cache.should_record_interaction(match["person_id"], now):
                    try:
                        updated = self.db.update_interaction(match["person_id"])
                    except Exception:
                        logger.exception(
                            "Failed to update interaction metadata for person_id=%s",
                            match["person_id"],
                        )
                    else:
                        if updated:
                            updated_meta = {**dict(updated_meta or {}), **dict(updated or {})}
                    finally:
                        self._presence_cache.mark_interaction_recorded(match["person_id"], now)
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
                logger.info(f"Recognized: {match['name']} (similarity: {match['similarity']:.2f})")
            else:
                result["unknown_faces"] += 1
                logger.info("Detected unknown face")

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
                logger.info(
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
                logger.info(
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
        consistent_faces = [
            item["face"]
            for item in accepted_faces
            if self._embedding_similarity(
                reference_face["embedding"],
                item["face"]["embedding"],
            )
            >= getattr(
                self,
                "_enrollment_policy",
                DEFAULT_FACE_ENROLLMENT_POLICY,
            ).min_embedding_similarity
        ]
        if len(consistent_faces) < ENROLLMENT_REQUIRED_STABLE_FRAMES:
            reason, guidance = self._quality_response_for_reason("embedding_inconsistent")
            logger.info(
                "Enrollment burst rejected reason=%s accepted=%s consistent=%s",
                reason,
                len(accepted_faces),
                len(consistent_faces),
            )
            return (
                None,
                self._enrollment_response(
                    success=False,
                    status="retry_quality",
                    message=guidance,
                    failure_reason=reason,
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
        person_id = self.db.add_person(
            name=candidate.cleaned_name,
            face_embedding=candidate.averaged_embedding,
            metadata=candidate.verified_durable,
        )
        self._prime_presence_cache_after_enrollment(
            person_id=person_id,
            name=candidate.cleaned_name,
            face=candidate.reference_face,
            image_shape=candidate.image_shape,
            metadata=candidate.verified_durable,
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
            directory_profile_lines=format_identity_profile_lines(metadata),
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
            if self._depth_gate_settings is None:
                self._log_loop_heartbeat(
                    "no_color_frame",
                    "[FaceLoop] no color frame available from camera resource %s within %.2fs",
                    camera_resource_id,
                    1.5,
                )
            else:
                self._log_loop_heartbeat(
                    "no_rgbd_pair",
                    "[FaceLoop] no synced RGBD pair available from camera resource %s within %.2fs",
                    camera_resource_id,
                    min(1.5, self._depth_gate_settings.capture_timeout_sec),
                )
            if self._presence_cache.clear_if_expired(now):
                logger.info("[FaceLoop] no image, cache expired and cleared")
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
        prepared = self._prepare_faces_for_recognition_result(image, depth_m)
        prepare_s = perf_now() - prepare_started
        detected_faces = prepared.faces
        if not detected_faces:
            publish_started = perf_now()
            self._publish_live_image_frame(image)
            publish_s = perf_now() - publish_started
            if prepared.reason:
                self._log_loop_heartbeat(
                    f"prepare_{prepared.reason}",
                    "[FaceLoop] summary reason=%s detected=%s rejected=%s "
                    "recognized=%s unknown=%s attentive=%s attentive_unknown=%s "
                    "primary_face=%s primary_attention=%s",
                    prepared.reason,
                    prepared.detected_count,
                    prepared.rejected_count,
                    [],
                    0,
                    [],
                    0,
                    None,
                    None,
                )
            if self._presence_cache.clear_if_expired(now):
                logger.info("[FaceLoop] no faces, cache expired and cleared")
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
            )
            return

        self._presence_cache.mark_faces_seen(now)
        scene_started = perf_now()
        persons, unknown_count, current_ids, analysis = self._build_scene_state(
            image=image,
            detected_faces=detected_faces,
            image_shape=image.shape,
            now=now,
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
            attention_target=analysis.attention_target,
            primary_attention_target=analysis.primary_attention_target,
            social_scene=analysis.social_scene,
            now=now,
        )
        attentive_names = [p.name for p in persons if bool(p.attentive)]
        primary_attention = analysis.primary_attention_target
        attention_details = self._format_attention_log_details(detected_faces)
        logger.debug(
            "[FaceLoop] detected %s face(s), recognized=%s unknown=%s "
            "attentive=%s attentive_unknown=%s primary_face=%s primary_attention=%s "
            "attention_details=%s",
            len(detected_faces),
            [p.name for p in persons],
            unknown_count,
            attentive_names,
            analysis.attentive_unknown_count,
            analysis.attention_target.person_id if analysis.attention_target else None,
            (
                primary_attention.person_id
                if primary_attention and primary_attention.person_id
                else (primary_attention.kind if primary_attention else None)
            ),
            attention_details,
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
        )
        self._log_loop_heartbeat(
            "attention_summary",
            "[FaceLoop] summary reason=%s detected=%s rejected=%s "
            "recognized=%s unknown=%s attentive=%s attentive_unknown=%s "
            "primary_face=%s primary_attention=%s attention_details=%s",
            "ok",
            len(detected_faces),
            prepared.rejected_count,
            [p.name for p in persons],
            unknown_count,
            attentive_names,
            analysis.attentive_unknown_count,
            analysis.attention_target.person_id if analysis.attention_target else None,
            (
                primary_attention.person_id
                if primary_attention and primary_attention.person_id
                else (primary_attention.kind if primary_attention else None)
            ),
            attention_details,
        )

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
        Used for [PEOPLE IN VIEW] and for recognized-person personalization.
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
