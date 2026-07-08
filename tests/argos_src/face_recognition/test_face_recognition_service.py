from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import numpy as np
from argos_src.face_recognition.attention_gate.models import FaceAttentionObservation


def _load_face_service_module(monkeypatch):
    rclpy_mod = types.ModuleType("rclpy")
    rclpy_mod.ok = lambda: True
    rclpy_mod.init = lambda: None
    monkeypatch.setitem(sys.modules, "rclpy", rclpy_mod)

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        get_device_name=lambda _idx: "cpu",
    )
    torch_mod.device = lambda name: types.SimpleNamespace(type=name)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    capture_mod = types.ModuleType("argos_src.face_recognition.camera_capture")
    capture_mod.CameraIntrinsics = object
    capture_mod.ROSCameraInfoCapture = object
    capture_mod.ROSImageCapture = object
    capture_mod.ROSSyncedRGBDCapture = object
    monkeypatch.setitem(
        sys.modules,
        "argos_src.face_recognition.camera_capture",
        capture_mod,
    )

    db_mod = types.ModuleType("argos_src.face_recognition.store")
    db_mod.FaceRecognitionStore = object
    monkeypatch.setitem(
        sys.modules,
        "argos_src.face_recognition.store",
        db_mod,
    )

    pipeline_mod = types.ModuleType("argos_src.face_recognition.pipeline")
    pipeline_mod.FaceEmbeddingPipeline = types.SimpleNamespace(
        resolve_device=lambda: types.SimpleNamespace(type="cpu")
    )
    pipeline_mod.FacePipelineCudaUnavailable = RuntimeError
    monkeypatch.setitem(
        sys.modules,
        "argos_src.face_recognition.pipeline",
        pipeline_mod,
    )

    module_name = "test_argos_face_recognition_service_module"
    module_path = (
        Path(__file__).resolve().parents[3]
        / "argos_src/face_recognition/face_recognition_service.py"
    )
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _good_image(size: int = 128) -> np.ndarray:
    y, x = np.indices((size, size))
    base = ((x * 7 + y * 11) % 160 + 50).astype(np.uint8)
    return np.stack([base, np.roll(base, 1, axis=0), np.roll(base, 1, axis=1)], axis=2)


def _face(
    area: int = 1600,
    depth_m: float | None = 1.0,
    *,
    x: int = 10,
    y: int = 10,
    embedding=None,
):
    width = int(area ** 0.5)
    landmarks = {
        "left_eye": (x + width * 0.32, y + width * 0.38),
        "right_eye": (x + width * 0.68, y + width * 0.38),
        "nose": (x + width * 0.50, y + width * 0.55),
        "mouth_left": (x + width * 0.38, y + width * 0.72),
        "mouth_right": (x + width * 0.62, y + width * 0.72),
    }
    return {
        "bbox": {"x": x, "y": y, "w": width, "h": width},
        "confidence": 0.99,
        "landmarks": landmarks,
        "embedding": embedding if embedding is not None else [0.1, 0.2, 0.3],
        "depth_m": depth_m,
    }


def _person(
    person_id: str = "person-1",
    name: str = "Alex",
    *,
    attentive: bool = False,
    bbox_area: int = 1600,
) -> object:
    from argos_src.face_recognition.models import PersonContext

    return PersonContext(
        person_id=person_id,
        name=name,
        interaction_count=1,
        confidence=0.93,
        bbox_area=bbox_area,
        timestamp=100.0,
        center_distance=0.0,
        attentive=attentive,
        attention_confidence=1.0 if attentive else 0.0,
    )


def test_recognition_stability_promotes_after_min_hits(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    window = module.RecognitionStabilityWindow(
        module.FaceRecognitionStabilitySettings(window_frames=5, min_hits=2)
    )
    person = _person()

    first, first_ids = window.update([person])
    second, second_ids = window.update([person])

    assert first == []
    assert first_ids == set()
    assert [p.person_id for p in second] == ["person-1"]
    assert second_ids == {"person-1"}


def test_recognition_stability_keeps_multiple_people(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    window = module.RecognitionStabilityWindow(
        module.FaceRecognitionStabilitySettings(window_frames=5, min_hits=2)
    )
    alice = _person("person-1", "Alice", bbox_area=1200)
    bob = _person("person-2", "Bob", bbox_area=1800)

    window.update([alice, bob])
    stable, stable_ids = window.update([alice, bob])

    assert [p.person_id for p in stable] == ["person-2", "person-1"]
    assert stable_ids == {"person-1", "person-2"}


def test_recognition_stability_decays_when_hits_leave_window(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    window = module.RecognitionStabilityWindow(
        module.FaceRecognitionStabilitySettings(window_frames=5, min_hits=2)
    )
    person = _person()

    window.update([person])
    stable, _ = window.update([person])
    assert [p.person_id for p in stable] == ["person-1"]

    for _ in range(4):
        stable, stable_ids = window.update([])

    assert stable == []
    assert stable_ids == set()


def test_stable_scene_treats_one_hit_attentive_recognition_as_unknown(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._recognition_stability = module.RecognitionStabilityWindow(
        module.FaceRecognitionStabilitySettings(window_frames=5, min_hits=2)
    )
    face = _face()
    face["recognized_person_id"] = "person-1"
    face["attention"] = FaceAttentionObservation(
        attentive=True,
        confidence=0.8,
        reason="attentive",
    )

    persons, unknown_count, current_ids, analysis = module.FaceRecognitionService._stable_scene_state(
        service,
        detected_faces=[face],
        raw_persons=[_person(attentive=True)],
        image_shape=(100, 100, 3),
        now=100.0,
    )

    assert persons == []
    assert unknown_count == 1
    assert current_ids == set()
    assert analysis.primary_attention_target is not None
    assert analysis.primary_attention_target.kind == "unknown"
    assert analysis.attentive_unknown_count == 1


def test_stable_scene_single_face_miss_keeps_one_stable_identity(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._recognition_stability = module.RecognitionStabilityWindow(
        module.FaceRecognitionStabilitySettings(window_frames=5, min_hits=2)
    )
    stable_person = _person("person-1", "Alice", attentive=True)
    service._recognition_stability.update([stable_person])
    service._recognition_stability.update([stable_person])

    face = _face()
    face["attention"] = FaceAttentionObservation(
        attentive=True,
        confidence=0.9,
        reason="attentive",
    )

    persons, unknown_count, current_ids, analysis = module.FaceRecognitionService._stable_scene_state(
        service,
        detected_faces=[face],
        raw_persons=[],
        image_shape=(100, 100, 3),
        now=101.0,
    )

    assert [person.person_id for person in persons] == ["person-1"]
    assert unknown_count == 0
    assert current_ids == {"person-1"}
    assert analysis.attention_target is not None
    assert analysis.attention_target.person_id == "person-1"


def test_stable_scene_ambiguous_miss_does_not_create_extra_faces(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._recognition_stability = module.RecognitionStabilityWindow(
        module.FaceRecognitionStabilitySettings(window_frames=5, min_hits=2)
    )
    stable_person = _person("person-1", "Alice", attentive=True)
    service._recognition_stability.update([stable_person])
    service._recognition_stability.update([stable_person])
    face_one = _face(x=10)
    face_two = _face(x=80)
    face_one["attention"] = FaceAttentionObservation(
        attentive=True,
        confidence=0.9,
        reason="attentive",
    )
    face_two["attention"] = FaceAttentionObservation(
        attentive=False,
        confidence=0.0,
        reason="head_pose_outside_threshold",
    )

    persons, unknown_count, current_ids, analysis = module.FaceRecognitionService._stable_scene_state(
        service,
        detected_faces=[face_one, face_two],
        raw_persons=[],
        image_shape=(140, 140, 3),
        now=101.0,
    )

    assert persons == []
    assert unknown_count == 2
    assert current_ids == set()
    assert analysis.attention_target is None
    assert analysis.attentive_unknown_count == 1


def test_unknown_stability_tracks_consecutive_unknown_frames(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)

    assert module.FaceRecognitionService._update_unknown_stability(
        service,
        unknown_count=1,
        attentive_unknown_count=1,
    ) == (1, 1)
    assert module.FaceRecognitionService._update_unknown_stability(
        service,
        unknown_count=1,
        attentive_unknown_count=0,
    ) == (2, 0)
    assert module.FaceRecognitionService._update_unknown_stability(
        service,
        unknown_count=0,
        attentive_unknown_count=0,
    ) == (0, 0)


def test_face_presence_subscriber_receives_updates(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._presence_cache = module.FacePresenceCache(cache_expire_sec=5.0)
    seen = []

    unsubscribe = module.FaceRecognitionService.subscribe_presence(
        service,
        lambda snapshot: seen.append(snapshot),
        replay_latest=False,
    )
    module.FaceRecognitionService._notify_presence_subscribers(
        service,
        {"status": "unknown"},
    )
    unsubscribe()
    module.FaceRecognitionService._notify_presence_subscribers(
        service,
        {"status": "recognized"},
    )

    assert seen == [{"status": "unknown"}]


def test_publish_live_image_frame_sends_data_url_to_display(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    updates = []

    class _Display:
        is_configured = True

        def show_live_image(self, **kwargs):
            updates.append(kwargs)
            return True

    service._display_runtime = _Display()
    service._live_image_title = "Camera"
    service._live_image_ttl_ms = 1000
    service._live_image_enabled = True

    module.FaceRecognitionService._publish_live_image_frame(service, _good_image(8))

    assert len(updates) == 1
    assert updates[0]["title"] == "Camera"
    assert updates[0]["ttl_ms"] == 1000
    assert updates[0]["data_url"].startswith("data:image/png;base64,")


def test_publish_live_image_frame_skips_display_when_disabled(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    updates = []

    class _Display:
        is_configured = True

        def show_live_image(self, **kwargs):
            updates.append(kwargs)
            return True

    service._display_runtime = _Display()
    service._live_image_title = "Camera"
    service._live_image_ttl_ms = 1000
    service._live_image_enabled = False

    module.FaceRecognitionService._publish_live_image_frame(service, _good_image(8))

    assert updates == []


def test_publish_live_image_frame_draws_attention_overlay(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    updates = []

    class _Display:
        is_configured = True

        def show_live_image(self, **kwargs):
            updates.append(kwargs)
            return True

    service._display_runtime = _Display()
    service._live_image_title = "Camera"
    service._live_image_ttl_ms = 1000
    service._live_image_enabled = True
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    face = {
        "bbox": {"x": 20, "y": 20, "w": 30, "h": 30},
        "landmarks": {"nose": (35.0, 35.0)},
        "attention": FaceAttentionObservation(
            attentive=True,
            confidence=0.9,
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
        ),
    }

    module.FaceRecognitionService._publish_live_image_frame(
        service,
        image,
        faces=[face],
    )

    assert len(updates) == 1
    assert updates[0]["data_url"].startswith("data:image/png;base64,")


def test_attention_log_details_include_reason_pose_and_raw_state(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    details = module.FaceRecognitionService._format_attention_log_details(
        [
            {
                "recognized_name": "Sakshee Patil",
                "attention": FaceAttentionObservation(
                    attentive=False,
                    confidence=0.74,
                    reason="head_pose_outside_threshold",
                    yaw_deg=8.25,
                    pitch_deg=-3.5,
                    roll_deg=1.0,
                    raw_attentive=True,
                    raw_confidence=0.74,
                ),
            }
        ]
    )

    assert details == [
        "Sakshee_Patil:att=no,raw=yes,reason=head_pose_outside_threshold,"
        "conf=0.74,raw_conf=0.74,yaw=8.2,pitch=-3.5,roll=1.0"
    ]


def test_recognition_log_details_include_similarity_and_threshold(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._recognition_threshold = 0.65
    details = module.FaceRecognitionService._format_recognition_log_details(
        service,
        [
            module.PersonContext(
                person_id="person-1",
                name="Sakshee Patil",
                interaction_count=1,
                confidence=0.734,
                bbox_area=1600,
                timestamp=100.0,
            )
        ],
    )

    assert details == ["Sakshee_Patil:sim=0.73,threshold=0.65"]


def test_recognize_face_match_accepts_clear_top_match(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._recognition_threshold = 0.6
    service._recognition_margin_threshold = 0.20
    calls = []

    def recognize_face(**kwargs):
        calls.append(kwargs)
        return [
            {"person_id": "arushi", "name": "Arushi", "similarity": 0.82},
            {"person_id": "sakshee", "name": "Sakshee", "similarity": 0.55},
        ]

    service.db = types.SimpleNamespace(recognize_face=recognize_face)

    match = module.FaceRecognitionService._recognize_face_match(
        service,
        {"embedding": [0.1, 0.2, 0.3]},
    )

    assert match["person_id"] == "arushi"
    assert match["runner_up_similarity"] == 0.55
    assert abs(match["similarity_margin"] - 0.27) < 1e-6
    assert calls[0]["threshold"] == -1.0
    assert calls[0]["top_k"] == 2


def test_recognize_face_match_rejects_low_similarity(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._recognition_threshold = 0.6
    service._recognition_margin_threshold = 0.20
    service.db = types.SimpleNamespace(
        recognize_face=lambda **_kwargs: [
            {"person_id": "arushi", "name": "Arushi", "similarity": 0.59},
            {"person_id": "sakshee", "name": "Sakshee", "similarity": 0.10},
        ]
    )

    match = module.FaceRecognitionService._recognize_face_match(
        service,
        {"embedding": [0.1, 0.2, 0.3]},
    )

    assert match is None


def test_recognize_face_match_diagnostics_explain_low_similarity(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._recognition_threshold = 0.6
    service._recognition_margin_threshold = 0.20
    service.db = types.SimpleNamespace(
        recognize_face=lambda **_kwargs: [
            {"person_id": "arushi", "name": "Arushi", "similarity": 0.59},
            {"person_id": "sakshee", "name": "Sakshee", "similarity": 0.10},
        ]
    )

    match, diagnostics = (
        module.FaceRecognitionService._recognize_face_match_with_diagnostics(
            service,
            {"embedding": [0.1, 0.2, 0.3]},
        )
    )

    assert match is None
    assert diagnostics["reason"] == "below_threshold"
    assert diagnostics["name"] == "Arushi"
    assert diagnostics["similarity"] == 0.59
    assert diagnostics["threshold"] == 0.6


def test_recognize_face_match_rejects_small_margin(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._recognition_threshold = 0.6
    service._recognition_margin_threshold = 0.20
    service.db = types.SimpleNamespace(
        recognize_face=lambda **_kwargs: [
            {"person_id": "arushi", "name": "Arushi", "similarity": 0.75},
            {"person_id": "sakshee", "name": "Sakshee", "similarity": 0.62},
        ]
    )

    match = module.FaceRecognitionService._recognize_face_match(
        service,
        {"embedding": [0.1, 0.2, 0.3]},
    )

    assert match is None


def test_format_recognition_attempt_log_details(monkeypatch):
    module = _load_face_service_module(monkeypatch)

    details = module.FaceRecognitionService._format_recognition_attempt_log_details(
        [
            {
                "recognition": {
                    "reason": "below_threshold",
                    "name": "Arushi",
                    "similarity": 0.59,
                    "threshold": 0.6,
                    "runner_up_similarity": 0.10,
                    "margin": 0.49,
                    "margin_threshold": 0.20,
                }
            }
        ]
    )

    assert details == [
        "Arushi:below_threshold,sim=0.59,threshold=0.60,"
        "runner_up=0.10,margin=0.49,margin_threshold=0.20"
    ]


def test_loop_tick_emits_timing_metric(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    image = _good_image()
    events = []

    class _FakeLatency:
        def timing(self, metric, duration_s, **fields):
            events.append({"metric": metric, "duration_s": duration_s, **fields})

    service._loop_latency = _FakeLatency()
    service._loop_metric_heartbeat_at = {}
    service._depth_gate_settings = None
    service._presence_cache = module.FacePresenceCache(cache_expire_sec=5.0)
    service._latest_loop_frame_lock = module.threading.Lock()
    service._latest_loop_frame = None
    service._latest_loop_frame_resource_id = None
    service._latest_loop_frame_at = 0.0
    service._display_runtime = None
    service._capture_for_recognition = lambda *_args, **_kwargs: (image, None)
    service._prepare_faces_for_recognition_result = (
        lambda *_args, **_kwargs: module.FacePreparationResult(
            faces=[],
            reason="no_detection",
            detected_count=0,
        )
    )
    service._log_loop_heartbeat = lambda *_args, **_kwargs: None

    module.FaceRecognitionService._loop_tick(
        service,
        "head_realsense",
        interval_sec=0.3,
    )

    assert len(events) == 1
    event = events[0]
    assert event["metric"] == "tick"
    assert event["duration_s"] >= 0.0
    assert event["camera_resource"] == "head_realsense"
    assert event["interval_s"] == 0.3
    assert event["outcome"] == "no_faces"
    assert event["reason"] == "no_detection"
    assert event["detected"] == 0
    assert event["recognized"] == 0
    assert event["unknown"] == 0
    assert event["capture_s"] >= 0.0
    assert event["prepare_s"] >= 0.0
    assert event["publish_s"] >= 0.0
    assert service._latest_loop_frame_resource_id == "head_realsense"
    assert service._latest_loop_frame is not None


def test_build_scene_state_dedupes_interaction_updates(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._presence_cache = module.FacePresenceCache(
        cache_expire_sec=5.0,
        interaction_dedupe_sec=60.0,
    )

    interaction_state = {"count": 2}
    update_calls: list[str] = []

    def recognize_face_match(_face_payload):
        return {
            "person_id": "person-1",
            "name": "Alex",
            "similarity": 0.93,
            "metadata": {
                "interaction_count": interaction_state["count"],
                "last_seen": "before",
            },
        }

    def update_interaction(person_id: str):
        update_calls.append(person_id)
        interaction_state["count"] += 1
        return {
            "name": "Alex",
            "interaction_count": interaction_state["count"],
            "last_seen": f"count-{interaction_state['count']}",
        }

    service._recognize_face_match = recognize_face_match
    service.db = types.SimpleNamespace(
        update_interaction=update_interaction,
    )

    persons, unknown_count, current_ids, analysis = module.FaceRecognitionService._build_scene_state(
        service,
        detected_faces=[_face()],
        image_shape=(100, 100, 3),
        now=100.0,
    )

    assert update_calls == ["person-1"]
    assert unknown_count == 0
    assert current_ids == {"person-1"}
    assert analysis.attention_target is not None
    assert analysis.attention_target.person_id == "person-1"
    assert persons[0].interaction_count == 3
    assert persons[0].recognition_status == "accepted"
    assert persons[0].recognition_reason == "matched"
    assert persons[0].recognition_threshold >= 0.0

    persons, _, _, _ = module.FaceRecognitionService._build_scene_state(
        service,
        detected_faces=[_face()],
        image_shape=(100, 100, 3),
        now=101.0,
    )

    assert update_calls == ["person-1"]
    assert persons[0].interaction_count == 3

    persons, _, _, _ = module.FaceRecognitionService._build_scene_state(
        service,
        detected_faces=[_face()],
        image_shape=(100, 100, 3),
        now=106.1,
    )

    assert update_calls == ["person-1"]
    assert persons[0].interaction_count == 3

    persons, _, _, _ = module.FaceRecognitionService._build_scene_state(
        service,
        detected_faces=[_face()],
        image_shape=(100, 100, 3),
        now=160.1,
    )

    assert update_calls == ["person-1", "person-1"]
    assert persons[0].interaction_count == 4


def test_best_face_match_evidence_prefers_highest_similarity(monkeypatch):
    module = _load_face_service_module(monkeypatch)

    evidence = module.FaceRecognitionService._best_face_match_evidence(
        [
            {
                "bbox": {"w": 80, "h": 80},
                "recognition": {
                    "status": "rejected",
                    "reason": "below_threshold",
                    "name": "Alex",
                    "person_id": "person-alex",
                    "similarity": 0.42,
                    "threshold": 0.6,
                    "runner_up_similarity": 0.31,
                    "margin": 0.11,
                    "margin_threshold": 0.2,
                },
            },
            {
                "bbox": {"w": 40, "h": 40},
                "recognition": {
                    "status": "accepted",
                    "reason": "matched",
                    "name": "Blair",
                    "person_id": "person-blair",
                    "similarity": 0.82,
                    "threshold": 0.6,
                    "runner_up_similarity": 0.2,
                    "margin": 0.62,
                    "margin_threshold": 0.2,
                },
            },
        ]
    )

    assert evidence["person_id"] == "person-blair"
    assert evidence["status"] == "accepted"
    assert evidence["similarity"] == 0.82


def test_build_scene_state_records_encounter_once_per_presence_episode(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._presence_cache = module.FacePresenceCache(cache_expire_sec=5.0)
    service.site_code = "BOS3"
    encounters = []
    service.memory_store = types.SimpleNamespace(
        record_encounter=lambda **kwargs: encounters.append(kwargs) or "mem-1"
    )

    def recognize_face_match(_face_payload):
        return {
            "person_id": "person-1",
            "name": "Alex",
            "similarity": 0.93,
            "metadata": {"interaction_count": 1},
        }

    service._recognize_face_match = recognize_face_match
    service.db = types.SimpleNamespace(
        update_interaction=lambda _person_id: {
            "interaction_count": 2,
        },
    )

    module.FaceRecognitionService._build_scene_state(
        service,
        detected_faces=[_face()],
        image_shape=(100, 100, 3),
        now=100.0,
    )
    module.FaceRecognitionService._build_scene_state(
        service,
        detected_faces=[_face()],
        image_shape=(100, 100, 3),
        now=101.0,
    )

    assert len(encounters) == 1
    assert encounters[0]["person_id"] == "person-1"
    assert encounters[0]["name"] == "Alex"
    assert encounters[0]["site_code"] == "BOS3"
    assert encounters[0]["metadata"]["site_code"] == "BOS3"


def test_build_scene_state_skips_encounter_when_identity_update_misses(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._presence_cache = module.FacePresenceCache(cache_expire_sec=5.0)
    service.site_code = "BOS3"
    encounters = []
    service.memory_store = types.SimpleNamespace(
        record_encounter=lambda **kwargs: encounters.append(kwargs) or "mem-1"
    )
    service._recognize_face_match = lambda _face_payload: {
        "person_id": "person-missing",
        "name": "Alex",
        "similarity": 0.93,
        "metadata": {"interaction_count": 1},
    }
    service.db = types.SimpleNamespace(update_interaction=lambda _person_id: None)

    persons, _, _, _ = module.FaceRecognitionService._build_scene_state(
        service,
        detected_faces=[_face()],
        image_shape=(100, 100, 3),
        now=100.0,
    )

    assert encounters == []
    assert persons[0].person_id == "person-missing"
    assert persons[0].interaction_count == 1


def test_build_scene_state_has_no_primary_face_id_with_multiple_usable_faces(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._presence_cache = module.FacePresenceCache(cache_expire_sec=5.0)
    matches = iter(
        [
            {
                "person_id": "person-1",
                "name": "Alex",
                "similarity": 0.93,
                "metadata": {"interaction_count": 1},
            },
            None,
        ]
    )
    service._recognize_face_match = lambda _face_payload: next(matches)
    service.db = types.SimpleNamespace(update_interaction=lambda _person_id: {})

    persons, unknown_count, current_ids, analysis = module.FaceRecognitionService._build_scene_state(
        service,
        detected_faces=[_face(x=10), _face(x=80)],
        image_shape=(140, 140, 3),
        now=100.0,
    )

    assert [person.person_id for person in persons] == ["person-1"]
    assert unknown_count == 1
    assert current_ids == {"person-1"}
    assert analysis.attention_target is None


def test_build_scene_state_adds_robot_yaw_bearing(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._presence_cache = module.FacePresenceCache(cache_expire_sec=5.0)
    service._camera_intrinsics = types.SimpleNamespace(fx=100.0, cx=50.0)
    service._camera_yaw_offset_rad = 0.0
    service._camera_resource_id = "head_realsense"
    service._camera_info_capture = None
    service.site_code = ""
    service.memory_store = None
    service.db = types.SimpleNamespace(update_interaction=lambda _person_id: {})
    service._recognize_face_match = lambda _face_payload: {
        "person_id": "person-1",
        "name": "Alex",
        "similarity": 0.93,
        "metadata": {"interaction_count": 1},
    }

    persons, _, _, _ = module.FaceRecognitionService._build_scene_state(
        service,
        detected_faces=[_face(x=40, area=400)],
        image_shape=(100, 100, 3),
        now=100.0,
    )

    assert persons[0].face_center_x_px == 50.0
    assert persons[0].bearing_rad == 0.0

    persons, _, _, _ = module.FaceRecognitionService._build_scene_state(
        service,
        detected_faces=[_face(x=50, area=400)],
        image_shape=(100, 100, 3),
        now=101.0,
    )

    assert persons[0].face_center_x_px == 60.0
    assert persons[0].bearing_rad < 0.0


def test_recognize_faces_continues_when_interaction_update_fails(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._presence_cache = module.FacePresenceCache(
        cache_expire_sec=5.0,
        interaction_dedupe_sec=60.0,
    )
    service._depth_gate_settings = None
    image = _good_image()
    face = _face(area=6400, depth_m=0.8)

    class _FakeDb:
        def update_interaction(self, _person_id):
            raise RuntimeError("db unavailable")

    service.db = _FakeDb()
    service._capture_for_recognition = lambda *_args, **_kwargs: (image, None)
    service._prepare_faces_for_recognition_result = (
        lambda *_args, **_kwargs: module.FacePreparationResult(faces=[dict(face)])
    )
    service._recognize_face_match = lambda _face_payload: {
        "person_id": "person-1",
        "name": "Alex",
        "similarity": 0.93,
        "metadata": {
            "interaction_count": 7,
            "last_seen": "before",
        },
    }

    result = module.FaceRecognitionService.recognize_faces(service)

    assert result["success"] is True
    assert result["faces_detected"] == 1
    assert result["faces_recognized"] == 1
    assert result["people"] == [
        {
            "name": "Alex",
            "person_id": "person-1",
            "confidence": 0.93,
            "bbox": face["bbox"],
            "depth_m": 0.8,
            "last_seen": "before",
            "interaction_count": 7,
        }
    ]
    assert not service._presence_cache.should_record_interaction("person-1", time.time())


def test_enroll_visible_person_seeds_verified_profile_fields(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    image = _good_image()
    face = _face(area=6400, depth_m=0.8)
    added = {}

    class _FakeDb:
        def add_person(self, *, name, face_embedding, metadata=None):
            added["name"] = name
            added["face_embedding"] = face_embedding
            added["metadata"] = metadata
            return "person-123"

    service.db = _FakeDb()
    service._capture_for_recognition = lambda *_args, **_kwargs: (image, None)
    service._prepare_faces_for_recognition_result = (
        lambda *_args, **_kwargs: module.FacePreparationResult(faces=[dict(face)])
    )
    service._recognize_face_match = lambda *_args, **_kwargs: None
    service._bbox_area = (
        lambda face_payload: face_payload["bbox"]["w"] * face_payload["bbox"]["h"]
    )
    service._center_distance = lambda *_args, **_kwargs: 0.0

    result = module.FaceRecognitionService.enroll_visible_person(
        service,
        official_name="Sakshee Patil",
        username="spatil2",
        employee_profile={
            "official_name": "Sakshee Patil",
            "employee_name": "Sakshee Patil",
            "username": "spatil2",
            "business_title": "AI Technologist II",
            "job_family": "Artificial Intelligence",
            "job_family_group": "Information Technology",
            "job_level": "Analyst",
            "c_level": "C05",
            "manager_name": "Dan Burns",
            "cost_center": "AI and Data Innovation",
            "senior_leadership_team": "Jeff Greenfield",
            "business_function": "AI & Data",
            "tenure": "0 year(s), 3 month(s), 5 day(s)",
        },
        camera_resource_id="head_realsense",
    )

    assert result["success"] is True
    assert result["status"] == "enrolled"
    assert added["name"] == "Sakshee Patil"
    assert added["metadata"] == {
        "official_name": "Sakshee Patil",
        "employee_name": "Sakshee Patil",
        "username": "spatil2",
        "business_title": "AI Technologist II",
        "job_family": "Artificial Intelligence",
        "job_family_group": "Information Technology",
        "job_level": "Analyst",
        "c_level": "C05",
        "manager_name": "Dan Burns",
        "cost_center": "AI and Data Innovation",
        "senior_leadership_team": "Jeff Greenfield",
        "business_function": "AI & Data",
        "tenure": "0 year(s), 3 month(s), 5 day(s)",
    }


def test_enroll_visible_person_primes_presence_cache_for_voice_followup(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._presence_cache = module.FacePresenceCache(cache_expire_sec=5.0)
    image = _good_image()
    face = _face(area=6400, depth_m=0.8)

    class _FakeDb:
        def add_person(self, *, name, face_embedding, metadata=None):
            return "person-voice-ready"

    service.db = _FakeDb()
    service._capture_for_recognition = lambda *_args, **_kwargs: (image, None)
    service._prepare_faces_for_recognition_result = (
        lambda *_args, **_kwargs: module.FacePreparationResult(faces=[dict(face)])
    )
    service._recognize_face_match = lambda *_args, **_kwargs: None
    service._bbox_area = lambda face_payload: face_payload["bbox"]["w"] * face_payload["bbox"]["h"]
    service._center_distance = lambda *_args, **_kwargs: 0.0

    result = module.FaceRecognitionService.enroll_visible_person(
        service,
        official_name="Sakshee Patil",
        camera_resource_id="head_realsense",
    )

    snapshot = service.get_presence_snapshot()
    cached = service.get_cached_persons()

    assert result["success"] is True
    assert snapshot["recognized_count"] == 1
    assert snapshot["unknown_count"] == 0
    assert snapshot["primary_face_kind"] == "recognized"
    assert cached[0].person_id == "person-voice-ready"
    assert cached[0].name == "Sakshee Patil"


def test_enrollment_face_selection_ignores_small_weak_extra_detection(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    primary = {
        "bbox": {"x": 10, "y": 10, "w": 80, "h": 80},
        "confidence": 0.99,
    }
    ghost = {
        "bbox": {"x": 120, "y": 15, "w": 20, "h": 20},
        "confidence": 0.81,
    }

    face, multiple_people_visible = module.FaceRecognitionService._select_enrollment_face(
        service,
        [primary, ghost],
    )

    assert multiple_people_visible is False
    assert face == primary


def test_enrollment_face_selection_ignores_below_min_area_extra_detection(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._enrollment_policy = module.FaceEnrollmentPolicy(min_face_area=1300)
    primary = {
        "bbox": {"x": 10, "y": 10, "w": 80, "h": 80},
        "confidence": 0.99,
    }
    small_extra = {
        "bbox": {"x": 120, "y": 15, "w": 30, "h": 30},
        "confidence": 0.999,
    }

    face, multiple_people_visible = module.FaceRecognitionService._select_enrollment_face(
        service,
        [primary, small_extra],
    )

    assert multiple_people_visible is False
    assert face == primary


def test_enrollment_face_selection_returns_small_face_for_quality_rejection(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._enrollment_policy = module.FaceEnrollmentPolicy(min_face_area=1300)
    small = {
        "bbox": {"x": 10, "y": 10, "w": 30, "h": 30},
        "confidence": 0.99,
    }

    face, multiple_people_visible = module.FaceRecognitionService._select_enrollment_face(
        service,
        [small],
    )

    assert multiple_people_visible is False
    assert face == small


def test_enrollment_face_selection_rejects_two_distinct_strong_faces(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._enrollment_policy = module.FaceEnrollmentPolicy(min_face_area=1300)
    left = {
        "bbox": {"x": 10, "y": 10, "w": 80, "h": 80},
        "confidence": 0.99,
    }
    right = {
        "bbox": {"x": 140, "y": 10, "w": 76, "h": 76},
        "confidence": 0.98,
    }

    face, multiple_people_visible = module.FaceRecognitionService._select_enrollment_face(
        service,
        [left, right],
    )

    assert multiple_people_visible is True
    assert face is None


def test_prepare_faces_for_recognition_reports_no_embedding(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._depth_gate_settings = None
    service.detect_faces = lambda _image: [
        {
            "bbox": {"x": 10, "y": 10, "w": 40, "h": 40},
            "confidence": 0.99,
        }
    ]
    service.extract_face_embedding = lambda *_args, **_kwargs: None

    result = module.FaceRecognitionService._prepare_faces_for_recognition_result(
        service,
        np.zeros((32, 32, 3), dtype=np.uint8),
        None,
    )

    assert result.faces == []
    assert result.reason == "no_embedding"


def test_prepare_faces_for_recognition_filters_faces_below_min_area(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._depth_gate_settings = None
    small = {
        "bbox": {"x": 10, "y": 10, "w": 20, "h": 20},
        "confidence": 0.99,
    }
    usable = {
        "bbox": {"x": 50, "y": 10, "w": 50, "h": 50},
        "confidence": 0.98,
    }
    service.detect_faces = lambda _image: [small, usable]
    service.extract_face_embedding = lambda _image, _face: np.ones(512, dtype=np.float32)

    result = module.FaceRecognitionService._prepare_faces_for_recognition_result(
        service,
        np.zeros((128, 128, 3), dtype=np.uint8),
        None,
        min_face_area=1300,
    )

    assert result.reason == ""
    assert result.detected_count == 2
    assert result.rejected_count == 1
    assert result.rejection_details == [
        "face0:face_too_small area=400 min_face_area=1300"
    ]
    assert len(result.faces) == 1
    assert result.faces[0]["bbox"] == usable["bbox"]


def test_prepare_faces_for_recognition_reports_all_faces_below_min_area(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    service._depth_gate_settings = None
    service.detect_faces = lambda _image: [
        {
            "bbox": {"x": 10, "y": 10, "w": 20, "h": 20},
            "confidence": 0.99,
        }
    ]

    result = module.FaceRecognitionService._prepare_faces_for_recognition_result(
        service,
        np.zeros((128, 128, 3), dtype=np.uint8),
        None,
        min_face_area=1300,
    )

    assert result.faces == []
    assert result.reason == "face_too_small"
    assert result.detected_count == 1
    assert result.rejected_count == 1
    assert result.rejection_details == [
        "face0:face_too_small area=400 min_face_area=1300"
    ]


def test_enrollment_face_quality_rejects_low_contrast_frame(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)

    result = module.FaceRecognitionService._assess_enrollment_face_quality(
        service,
        np.full((128, 128, 3), 120, dtype=np.uint8),
        _face(area=6400),
    )

    assert result.accepted is False
    assert result.reason == "low_contrast"


def test_enrollment_face_quality_accepts_face_without_landmarks(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    face = _face(area=6400)
    face.pop("landmarks")

    result = module.FaceRecognitionService._assess_enrollment_face_quality(
        service,
        _good_image(),
        face,
    )

    assert result.accepted is True


def test_enrollment_face_quality_rejects_clipped_bbox(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)

    result = module.FaceRecognitionService._assess_enrollment_face_quality(
        service,
        _good_image(),
        _face(area=6400, x=0, y=10),
    )

    assert result.accepted is False
    assert result.reason == "face_clipped"


def test_enrollment_face_quality_rejects_small_face(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)

    result = module.FaceRecognitionService._assess_enrollment_face_quality(
        service,
        _good_image(),
        _face(area=400),
    )

    assert result.accepted is False
    assert result.reason == "face_too_small"


def test_enrollment_preview_image_uses_padded_reference_bbox(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    image = np.arange(100 * 120 * 3, dtype=np.uint8).reshape((100, 120, 3))
    face = {"bbox": {"x": 40, "y": 30, "w": 20, "h": 20}}

    preview = module.FaceRecognitionService._enrollment_preview_image(
        image,
        face,
        padding_ratio=0.5,
    )

    assert preview.shape == (40, 40, 3)
    np.testing.assert_array_equal(preview, image[20:60, 30:70])
    assert not np.shares_memory(preview, image)


def test_prepare_visible_person_enrollment_preview_is_padded_face_crop(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    image = _good_image(size=160)
    face = _face(area=6400, depth_m=0.8, x=50, y=40)

    service._capture_for_recognition = lambda *_args, **_kwargs: (image, None)
    service._prepare_faces_for_recognition_result = (
        lambda *_args, **_kwargs: module.FacePreparationResult(faces=[dict(face)])
    )
    service._recognize_face_match = lambda *_args, **_kwargs: None
    service._bbox_area = lambda face_payload: face_payload["bbox"]["w"] * face_payload["bbox"]["h"]
    service._center_distance = lambda *_args, **_kwargs: 0.0

    candidate, failure = module.FaceRecognitionService._prepare_visible_person_enrollment(
        service,
        official_name="Sakshee Patil",
    )

    assert failure is None
    assert candidate is not None
    assert candidate.preview_image.shape[0] < image.shape[0]
    assert candidate.preview_image.shape[1] < image.shape[1]
    assert not np.shares_memory(candidate.preview_image, image)


def test_enroll_visible_person_reports_missing_name_failure(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)

    result = module.FaceRecognitionService.enroll_visible_person(
        service,
        official_name="",
    )

    assert result["success"] is False
    assert result["status"] == "error"
    assert result["failure_reason"] == "missing_name"


def test_enroll_visible_person_reports_already_known_failure(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    image = _good_image()

    service._capture_for_recognition = lambda *_args, **_kwargs: (image, None)
    service._prepare_faces_for_recognition_result = (
        lambda *_args, **_kwargs: module.FacePreparationResult(
            faces=[_face(area=6400, depth_m=0.8)]
        )
    )
    service._recognize_face_match = lambda *_args, **_kwargs: {
        "name": "Sakshee Patil",
        "person_id": "person-known",
        "similarity": 0.91,
    }

    result = module.FaceRecognitionService.enroll_visible_person(
        service,
        official_name="Sakshee Patil",
    )

    assert result["success"] is False
    assert result["status"] == "retry_already_known"
    assert result["failure_reason"] == "already_known"
    assert result["recognized_name"] == "Sakshee Patil"


def test_enroll_visible_person_shows_multiple_face_warning_preview(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    image = _good_image(size=240)
    previews = []

    class _Display:
        is_configured = True

        def show_image_message_preview(self, **kwargs):
            previews.append(kwargs)
            return True

    service._capture_for_recognition = lambda *_args, **_kwargs: (image, None)
    service._prepare_faces_for_recognition_result = (
        lambda *_args, **_kwargs: module.FacePreparationResult(
            faces=[
                _face(area=6400, x=20, y=20),
                _face(area=6400, x=140, y=20),
            ],
            detected_count=2,
        )
    )

    result = module.FaceRecognitionService.enroll_visible_person(
        service,
        official_name="Sakshee Patil",
        display_runtime=_Display(),
    )

    assert result["success"] is False
    assert result["status"] == "retry_single_face"
    assert result["failure_reason"] == "multiple_faces"
    assert len(previews) == 1
    assert previews[0]["title"] == "Multiple Faces Detected"
    assert previews[0]["hold_sec"] == 5.0
    assert previews[0]["image_url"].startswith("data:image/png;base64,")


def test_enroll_visible_person_rejects_inconsistent_face_embeddings(monkeypatch):
    module = _load_face_service_module(monkeypatch)
    service = object.__new__(module.FaceRecognitionService)
    image = _good_image()
    embeddings = iter(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    added = []

    class _FakeDb:
        def add_person(self, *, name, face_embedding):
            added.append((name, face_embedding))
            return "person-bad"

    service.db = _FakeDb()
    service._capture_for_recognition = lambda *_args, **_kwargs: (image, None)
    service._prepare_faces_for_recognition_result = (
        lambda *_args, **_kwargs: module.FacePreparationResult(
            faces=[_face(area=6400, embedding=next(embeddings))]
        )
    )
    service._recognize_face_match = lambda *_args, **_kwargs: None
    service._bbox_area = lambda face_payload: face_payload["bbox"]["w"] * face_payload["bbox"]["h"]
    service._center_distance = lambda *_args, **_kwargs: 0.0

    result = module.FaceRecognitionService.enroll_visible_person(
        service,
        official_name="Sakshee Patil",
    )

    assert result["success"] is False
    assert result["status"] == "retry_quality"
    assert added == []


def test_detect_faces_falls_back_to_cpu_when_cuda_pipeline_is_unusable(monkeypatch):
    module = _load_face_service_module(monkeypatch)

    class _CudaFailure(RuntimeError):
        pass

    class _FakePipeline:
        def __init__(self, device):
            self.device = device
            self.mtcnn = f"mtcnn-{device.type}"
            self.resnet = f"resnet-{device.type}"

        def detect_faces(self, _image):
            if self.device.type == "cuda":
                raise _CudaFailure("CUDA error: no kernel image is available for execution on the device")
            return [{"bbox": {"x": 1, "y": 2, "w": 3, "h": 4}}]

    module.FacePipelineCudaUnavailable = _CudaFailure
    module.FaceEmbeddingPipeline = _FakePipeline

    service = object.__new__(module.FaceRecognitionService)
    service.device = types.SimpleNamespace(type="cuda")
    service._pipeline = None
    service.mtcnn = None
    service.resnet = None
    service.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    torch_mod = sys.modules["torch"]
    torch_mod.device = lambda name: types.SimpleNamespace(type=name)

    detections = module.FaceRecognitionService.detect_faces(
        service,
        np.zeros((8, 8, 3), dtype=np.uint8),
    )

    assert detections == [{"bbox": {"x": 1, "y": 2, "w": 3, "h": 4}}]
    assert service.device.type == "cpu"
    assert service.mtcnn == "mtcnn-cpu"
    assert service.resnet == "resnet-cpu"
