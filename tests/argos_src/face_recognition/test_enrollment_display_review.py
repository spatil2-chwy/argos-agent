from __future__ import annotations

import numpy as np

from argos_src.face_recognition.face_recognition_service import (
    FaceEnrollmentCandidate,
    FaceRecognitionService,
)


class _DB:
    def __init__(self):
        self.saved = []

    def add_person(self, *, name, face_embedding, metadata):
        self.saved.append(
            {
                "name": name,
                "face_embedding": face_embedding,
                "metadata": metadata,
            }
        )
        return "person-1"


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
    service.db = _DB()
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
    assert len(service.db.saved) == 1
    assert display.reviews[0]["image_url"].startswith("data:image/png;base64,")


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
    assert service.db.saved == []


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
    assert service.db.saved == []


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
    assert service.db.saved == []


def test_enrollment_without_display_preserves_commit_behavior():
    service = _service_with_candidate()

    result = service.enroll_visible_person(official_name="Sakshee Patil")

    assert result["success"] is True
    assert result["status"] == "enrolled"
    assert len(service.db.saved) == 1
