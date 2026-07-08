from __future__ import annotations

from argos_src.face_recognition.models import (
    AttentionTarget,
    PersonContext,
    SocialSceneContext,
)
from argos_src.face_recognition.presence_cache import FacePresenceCache


def test_should_record_interaction_uses_interaction_dedupe_window():
    cache = FacePresenceCache(cache_expire_sec=5.0, interaction_dedupe_sec=60.0)

    assert cache.should_record_interaction("person-1", 100.0) is True

    cache.mark_person_seen("person-1", 100.0)
    cache.mark_interaction_recorded("person-1", 100.0)

    assert cache.should_record_interaction("person-1", 101.0) is False
    assert cache.should_record_interaction("person-1", 106.1) is False
    assert cache.should_record_interaction("person-1", 160.1) is True


def test_expired_presence_cache_does_not_return_stale_identity(monkeypatch):
    now = 1000.0
    monkeypatch.setattr("argos_src.face_recognition.presence_cache.time.time", lambda: now)
    cache = FacePresenceCache(cache_expire_sec=1.0)
    person = PersonContext(
        person_id="person-1",
        name="Alice",
        interaction_count=0,
        confidence=0.9,
        bbox_area=2500,
        timestamp=now,
    )
    attention = AttentionTarget(
        kind="recognized",
        depth_m=1.2,
        bbox_area=2500,
        center_distance=10.0,
        person_id="person-1",
        name="Alice",
    )
    scene = SocialSceneContext(
        has_unrecognized_people=False,
        closest_person_kind="recognized",
        nearest_recognized_name="Alice",
    )

    cache.mark_faces_seen(now)
    cache.mark_person_seen("person-1", now)
    cache.update(
        persons=[person],
        faces_detected=1,
        unknown_count=0,
        attention_target=attention,
        primary_attention_target=attention,
        face_match_evidence={
            "status": "accepted",
            "reason": "matched",
            "name": "Alice",
            "person_id": "person-1",
            "similarity": 0.91,
            "threshold": 0.6,
            "runner_up_similarity": 0.2,
            "margin": 0.71,
            "margin_threshold": 0.2,
        },
        social_scene=scene,
        now=now,
    )

    now = 1002.0

    assert cache.get_cached_persons() == []
    assert cache.get_primary_face_person_id() is None
    snapshot = cache.get_presence_snapshot()
    assert snapshot["status"] == "none"
    assert snapshot["recognized_count"] == 0
    assert snapshot["attention_status"] == "none"


def test_presence_snapshot_includes_attentive_people():
    cache = FacePresenceCache(cache_expire_sec=5.0)
    person = PersonContext(
        person_id="person-1",
        name="Alice",
        interaction_count=0,
        confidence=0.9,
        bbox_area=2500,
        timestamp=100.0,
        attentive=True,
        attention_confidence=0.82,
    )
    attention = AttentionTarget(
        kind="recognized",
        depth_m=1.2,
        bbox_area=2500,
        center_distance=10.0,
        person_id="person-1",
        name="Alice",
    )
    scene = SocialSceneContext(
        has_unrecognized_people=True,
        closest_person_kind="recognized",
        nearest_recognized_name="Alice",
    )

    cache.update(
        persons=[person],
        faces_detected=2,
        unknown_count=1,
        attentive_unknown_count=0,
        attention_target=None,
        primary_attention_target=attention,
        social_scene=scene,
        now=100.0,
    )

    snapshot = cache.get_presence_snapshot()
    assert snapshot["attention_status"] == "attentive"
    assert snapshot["attention_count"] == 1
    assert snapshot["attentive_recognized_count"] == 1
    assert snapshot["attentive_unknown_count"] == 0
    assert snapshot["primary_attention_person_id"] == "person-1"
    assert snapshot["primary_attention_name"] == "Alice"
    assert snapshot["attention_confidence"] == 0.82
    assert cache.get_primary_attention_person_id() == "person-1"


def test_presence_snapshot_includes_face_match_evidence():
    cache = FacePresenceCache(cache_expire_sec=5.0)
    scene = SocialSceneContext(
        has_unrecognized_people=False,
        closest_person_kind="unknown",
        nearest_recognized_name=None,
    )

    cache.update(
        persons=[],
        faces_detected=1,
        unknown_count=1,
        attention_target=None,
        primary_attention_target=None,
        face_match_evidence={
            "status": "rejected",
            "reason": "below_threshold",
            "name": "Alice",
            "person_id": "person-1",
            "similarity": 0.42,
            "threshold": 0.6,
            "runner_up_similarity": 0.31,
            "margin": 0.11,
            "margin_threshold": 0.2,
        },
        social_scene=scene,
        now=100.0,
    )

    snapshot = cache.get_presence_snapshot()

    assert snapshot["face_match_status"] == "rejected"
    assert snapshot["face_match_reason"] == "below_threshold"
    assert snapshot["face_match_name"] == "Alice"
    assert snapshot["face_match_person_id"] == "person-1"
    assert snapshot["face_score"] == 0.42
    assert snapshot["face_score_threshold"] == 0.6
    assert snapshot["face_runner_up_score"] == 0.31
    assert snapshot["face_score_margin"] == 0.11
    assert snapshot["face_margin_threshold"] == 0.2
