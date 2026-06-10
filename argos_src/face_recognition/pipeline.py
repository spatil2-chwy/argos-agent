"""Face detection and embedding extraction pipeline."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image as PILImage

from argos_src.face_recognition.constants import MIN_FACE_DETECTION_CONFIDENCE


logger = logging.getLogger(__name__)


class FacePipelineCudaUnavailable(RuntimeError):
    """Raised when the configured CUDA device cannot execute face kernels."""


def _is_cuda_runtime_failure(exc: Exception) -> bool:
    message = str(exc).casefold()
    return "cuda error" in message or "no kernel image is available" in message


class FaceEmbeddingPipeline:
    """Wrap MTCNN detection and FaceNet embedding extraction."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            post_process=False,
        )
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    @staticmethod
    def resolve_device() -> torch.device:
        """Select the best available inference device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_pil_image(self, image: np.ndarray) -> PILImage:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(rgb)

    def _clip_box(self, box, image_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
        height, width = image_shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def detect_faces(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect faces in a BGR image and return bounding boxes plus landmarks."""
        pil_img = self._to_pil_image(image)

        try:
            boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)
        except Exception as exc:
            if self.device.type == "cuda" and _is_cuda_runtime_failure(exc):
                raise FacePipelineCudaUnavailable(str(exc)) from exc
            logger.warning(f"MTCNN detection failed: {exc}")
            return []

        if boxes is None or len(boxes) == 0:
            return []

        landmark_names = (
            "left_eye",
            "right_eye",
            "nose",
            "mouth_left",
            "mouth_right",
        )
        faces: list[dict[str, Any]] = []
        landmarks_list = landmarks if landmarks is not None else [None] * len(boxes)
        for box, prob, face_landmarks in zip(boxes, probs, landmarks_list):
            if prob is None or prob < MIN_FACE_DETECTION_CONFIDENCE:
                continue

            clipped = self._clip_box(box, image.shape)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            landmark_dict: dict[str, tuple[float, float]] = {}
            if face_landmarks is not None and len(face_landmarks) == len(landmark_names):
                landmark_dict = {
                    name: (float(point[0]), float(point[1]))
                    for name, point in zip(landmark_names, face_landmarks)
                }

            faces.append(
                {
                    "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                    "confidence": float(prob),
                    "landmarks": landmark_dict,
                }
            )

        logger.debug(f"Detected {len(faces)} face(s) in image")
        return faces

    def extract_embedding(
        self,
        image: np.ndarray,
        face: dict[str, Any],
    ) -> np.ndarray | None:
        """Extract one embedding for a detected face."""
        pil_img = self._to_pil_image(image)
        bbox = face["bbox"]
        x1 = int(bbox["x"])
        y1 = int(bbox["y"])
        x2 = x1 + int(bbox["w"])
        y2 = y1 + int(bbox["h"])
        if x2 <= x1 or y2 <= y1:
            return None

        face_crop = pil_img.crop((x1, y1, x2, y2)).resize((160, 160))
        face_tensor = torch.tensor(np.array(face_crop)).permute(2, 0, 1).float()
        face_tensor = (face_tensor / 127.5) - 1.0
        face_tensor = face_tensor.unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                return self.resnet(face_tensor).squeeze(0).cpu().numpy()
        except Exception as exc:
            if self.device.type == "cuda" and _is_cuda_runtime_failure(exc):
                raise FacePipelineCudaUnavailable(str(exc)) from exc
            logger.warning(f"Face embedding extraction failed: {exc}")
            return None

    def detect_and_extract_faces(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect faces in a BGR image and return embeddings plus boxes."""
        faces: list[dict[str, Any]] = []
        for detection in self.detect_faces(image):
            embedding = self.extract_embedding(image, detection)
            if embedding is None:
                continue
            enriched = dict(detection)
            enriched["embedding"] = embedding
            faces.append(enriched)
        return faces
