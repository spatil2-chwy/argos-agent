from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image

from argos_src.face_recognition.face_recognition_service import (
    FaceEnrollmentCandidate,
    FaceRecognitionService,
)


class _IdentityMemory:
    def __init__(self):
        self.enrollments = []

    def enroll_face_reference(self, **kwargs):
        self.enrollments.append(kwargs)
        return type("Enrollment", (), {"saved": True, "reason": "saved"})()


class _Display:
    is_configured = True

    def __init__(self, response):
        self.response = response
        self.reviews = []

    def review_face_capture(self, **kwargs):
        self.reviews.append(kwargs)
        return dict(self.response)


def _service_with_candidate():
    service = object.__new__(FaceRecognitionService)
    service.identity_memory_client = _IdentityMemory()
    candidate = FaceEnrollmentCandidate(
        cleaned_name="Sakshee Patil",
        verified_durable={"official_name": "Sakshee Patil", "username": "spatil2"},
        averaged_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        reference_face={
            "bbox": {"x": 0, "y": 0, "w": 10, "h": 10},
            "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        },
        image_shape=(20, 20, 3),
        preview_image=np.zeros((20, 20, 3), dtype=np.uint8),
    )
    service._prepare_visible_person_enrollment = lambda **_kwargs: (candidate, None)
    service._prime_presence_cache_after_enrollment = lambda **_kwargs: None
    return service


def test_enrollment_display_accept_commits_person():
    service = _service_with_candidate()
    display = _Display(
        {"available": True, "accepted": True, "status": "accepted", "action": "accept"}
    )

    result = service.enroll_visible_person(
        official_name="Sakshee Patil",
        username="spatil2",
        display_runtime=display,
    )

    assert result["success"] is True
    assert result["status"] == "enrolled"
    assert len(service.identity_memory_client.enrollments) == 1
    assert display.reviews[0]["image_url"].startswith("data:image/png;base64,")


def test_enrollment_preview_data_url_converts_internal_bgr_to_rgb():
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    image[0, 0] = [0, 0, 255]

    data_url = FaceRecognitionService._enrollment_preview_data_url(image)
    encoded = data_url.removeprefix("data:image/png;base64,")
    decoded = Image.open(BytesIO(base64.b64decode(encoded)))

    assert decoded.getpixel((0, 0)) == (255, 0, 0)


def test_live_frame_data_url_converts_internal_bgr_to_rgb():
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    image[0, 0] = [255, 0, 0]

    data_url = FaceRecognitionService._live_frame_data_url(image)
    encoded = data_url.removeprefix("data:image/png;base64,")
    decoded = Image.open(BytesIO(base64.b64decode(encoded)))

    assert decoded.getpixel((0, 0)) == (0, 0, 255)


def test_enrollment_display_reject_does_not_commit():
    service = _service_with_candidate()
    display = _Display(
        {
            "available": True,
            "accepted": False,
            "status": "rejected",
            "action": "reject",
        }
    )

    result = service.enroll_visible_person(
        official_name="Sakshee Patil",
        display_runtime=display,
    )

    assert result["success"] is False
    assert result["status"] == "user_rejected_preview"
    assert service.identity_memory_client.enrollments == []


def test_enrollment_display_timeout_does_not_commit():
    service = _service_with_candidate()
    display = _Display(
        {"available": True, "accepted": False, "status": "review_timeout"}
    )

    result = service.enroll_visible_person(
        official_name="Sakshee Patil",
        display_runtime=display,
    )

    assert result["success"] is False
    assert result["status"] == "review_timeout"
    assert service.identity_memory_client.enrollments == []


def test_enrollment_display_unavailable_does_not_commit():
    service = _service_with_candidate()
    display = _Display(
        {"available": False, "accepted": False, "status": "display_unavailable"}
    )

    result = service.enroll_visible_person(
        official_name="Sakshee Patil",
        display_runtime=display,
    )

    assert result["success"] is False
    assert result["status"] == "display_unavailable"
    assert service.identity_memory_client.enrollments == []


def test_enrollment_without_display_preserves_commit_behavior():
    service = _service_with_candidate()

    result = service.enroll_visible_person(official_name="Sakshee Patil")

    assert result["success"] is True
    assert result["status"] == "enrolled"
    assert len(service.identity_memory_client.enrollments) == 1
