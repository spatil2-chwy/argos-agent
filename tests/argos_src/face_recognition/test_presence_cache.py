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
        social_scene=scene,
        now=now,
    )

    now = 1002.0

    assert cache.get_cached_persons() == []
    assert cache.get_primary_face_person_id() is None
    snapshot = cache.get_presence_snapshot()
    assert snapshot["status"] == "none"
    assert snapshot["recognized_count"] == 0
