import importlib.util
import sys
import types
from pathlib import Path


def _load_bridges_module(monkeypatch):
    module_name = "test_argos_bridges_module"
    module_path = Path(__file__).resolve().parents[3] / "argos_src/agent/bridges.py"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "argos_src.agent"
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_face_bridge_suppresses_proactive_events_while_recording(monkeypatch):
    bridges_module = _load_bridges_module(monkeypatch)
    submitted = []
    wake_calls = []

    bridge = object.__new__(bridges_module.FaceEventBridge)
    bridge._connector = types.SimpleNamespace(send_message=lambda *args, **kwargs: None)
    bridge._coalescer = types.SimpleNamespace(
        submit=lambda text, metadata: submitted.append((text, metadata))
    )
    bridge._engagement = types.SimpleNamespace(
        state=bridges_module.EngagementMode.IDLE,
        on_face_or_wake=lambda: wake_calls.append("wake"),
        is_recording_active=lambda: True,
    )
    bridge._nav_state = types.SimpleNamespace(
        get_active_goal=lambda: None,
        allows_proactive_face_attention=lambda: True,
    )
    bridge._recognized_greet_enabled = True
    bridge._unknown_greet_enabled = True
    bridge._recognized_greet_cooldown_sec = 45.0
    bridge._unknown_greet_cooldown_sec = 30.0
    bridge._last_unknown_greet_s = 0.0
    bridge._recognized_last_greet_s = {}
    bridge._previous_ids = set()
    bridge._previous_unknown_count = 0

    bridges_module.FaceEventBridge._maybe_enqueue_face_events(
        bridge,
        snapshot={
            "unknown_count": 1,
            "has_mixed_scene": True,
            "nearest_recognized_name": "Sakshee",
        },
        persons=[types.SimpleNamespace(person_id="p1", name="Sakshee")],
        now=100.0,
    )

    assert submitted == []
    assert wake_calls == []


def test_face_bridge_requires_attention_when_configured(monkeypatch):
    bridges_module = _load_bridges_module(monkeypatch)
    submitted = []
    wake_calls = []

    bridge = object.__new__(bridges_module.FaceEventBridge)
    bridge._coalescer = types.SimpleNamespace(
        submit=lambda text, metadata: submitted.append((text, metadata))
    )
    bridge._engagement = types.SimpleNamespace(
        state=bridges_module.EngagementMode.IDLE,
        on_face_or_wake=lambda: wake_calls.append("wake"),
        is_recording_active=lambda: False,
    )
    bridge._nav_state = types.SimpleNamespace(
        get_active_goal=lambda: None,
        allows_proactive_face_attention=lambda: True,
    )
    bridge._recognized_greet_enabled = True
    bridge._unknown_greet_enabled = True
    bridge._require_attention = True
    bridge._recognized_greet_cooldown_sec = 45.0
    bridge._unknown_greet_cooldown_sec = 30.0
    bridge._last_unknown_greet_s = 0.0
    bridge._recognized_last_greet_s = {}
    bridge._previous_ids = set()
    bridge._previous_unknown_count = 0

    bridges_module.FaceEventBridge._maybe_enqueue_face_events(
        bridge,
        snapshot={
            "unknown_count": 1,
            "attentive_unknown_count": 0,
            "has_attentive_mixed_scene": False,
        },
        persons=[types.SimpleNamespace(person_id="p1", name="Sakshee", attentive=False)],
        now=100.0,
    )
    assert submitted == []

    bridges_module.FaceEventBridge._maybe_enqueue_face_events(
        bridge,
        snapshot={
            "unknown_count": 1,
            "attentive_unknown_count": 0,
            "has_attentive_mixed_scene": False,
            "primary_attention_name": "Sakshee",
        },
        persons=[types.SimpleNamespace(person_id="p1", name="Sakshee", attentive=True)],
        now=101.0,
    )

    assert len(submitted) == 1
    assert "recognized person 'Sakshee' appeared" in submitted[0][0]
    assert submitted[0][1]["person_id"] == "p1"
    assert wake_calls == ["wake"]


def test_face_bridge_waits_for_unknown_stability_before_greeting(monkeypatch):
    bridges_module = _load_bridges_module(monkeypatch)
    submitted = []
    wake_calls = []

    bridge = object.__new__(bridges_module.FaceEventBridge)
    bridge._coalescer = types.SimpleNamespace(
        submit=lambda text, metadata: submitted.append((text, metadata))
    )
    bridge._engagement = types.SimpleNamespace(
        state=bridges_module.EngagementMode.IDLE,
        on_face_or_wake=lambda: wake_calls.append("wake"),
        is_recording_active=lambda: False,
    )
    bridge._nav_state = types.SimpleNamespace(
        get_active_goal=lambda: None,
        allows_proactive_face_attention=lambda: True,
    )
    bridge._recognized_greet_enabled = True
    bridge._unknown_greet_enabled = True
    bridge._require_attention = True
    bridge._recognized_greet_cooldown_sec = 45.0
    bridge._unknown_greet_cooldown_sec = 30.0
    bridge._last_unknown_greet_s = 0.0
    bridge._recognized_last_greet_s = {}
    bridge._previous_ids = set()
    bridge._previous_unknown_count = 0
    bridge._previous_unknown_greet_ready = False
    bridge._unknown_greet_stability_frames = 3

    for stability_frames, now in ((1, 100.0), (2, 101.0)):
        bridges_module.FaceEventBridge._maybe_enqueue_face_events(
            bridge,
            snapshot={
                "attentive_unknown_count": 1,
                "attentive_unknown_stability_frames": stability_frames,
                "has_attentive_mixed_scene": False,
            },
            persons=[],
            now=now,
        )

    assert submitted == []
    assert wake_calls == []

    bridges_module.FaceEventBridge._maybe_enqueue_face_events(
        bridge,
        snapshot={
            "attentive_unknown_count": 1,
            "attentive_unknown_stability_frames": 3,
            "has_attentive_mixed_scene": False,
        },
        persons=[],
        now=102.0,
    )

    assert len(submitted) == 1
    assert submitted[0][1]["face_status"] == "unknown"
    assert wake_calls == ["wake"]


def test_patrol_bridge_drops_delayed_hop_when_active_goal_appears(monkeypatch):
    bridges_module = _load_bridges_module(monkeypatch)
    submitted = []

    class _NavState:
        def __init__(self):
            self.active_goal = None
            self.patrol = {"enabled": True, "awaiting_target": "desk"}

        def get_patrol(self):
            return dict(self.patrol)

        def get_active_goal(self):
            return self.active_goal

        def patrol_mark_arrived_and_get_next(self, arrived_target):
            assert arrived_target == "lobby"
            self.active_goal = {"goal_id": "new-goal"}
            return "desk"

    class _ImmediateThread:
        def __init__(self, *, target, daemon):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(bridges_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(bridges_module.threading, "Thread", _ImmediateThread)
    bridge = bridges_module.PatrolLoopBridge(
        coalescer=types.SimpleNamespace(
            submit=lambda text, metadata: submitted.append((text, metadata))
        ),
        nav_state=_NavState(),
        next_hop_delay_sec=0.0,
    )

    bridge.on_nav_event(
        {
            "event_type": "goal_result",
            "outcome": "succeeded",
            "target_label": "lobby",
        }
    )

    assert submitted == []


def test_patrol_bridge_drops_delayed_hop_when_awaiting_target_changes(monkeypatch):
    bridges_module = _load_bridges_module(monkeypatch)
    submitted = []

    class _NavState:
        def __init__(self):
            self.patrol = {"enabled": True, "awaiting_target": "desk"}

        def get_patrol(self):
            return dict(self.patrol)

        def get_active_goal(self):
            return None

        def patrol_mark_arrived_and_get_next(self, arrived_target):
            assert arrived_target == "lobby"
            self.patrol["awaiting_target"] = "atrium"
            return "desk"

    class _ImmediateThread:
        def __init__(self, *, target, daemon):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(bridges_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(bridges_module.threading, "Thread", _ImmediateThread)
    bridge = bridges_module.PatrolLoopBridge(
        coalescer=types.SimpleNamespace(
            submit=lambda text, metadata: submitted.append((text, metadata))
        ),
        nav_state=_NavState(),
        next_hop_delay_sec=0.0,
    )

    bridge.on_nav_event(
        {
            "event_type": "goal_result",
            "outcome": "succeeded",
            "target_label": "lobby",
        }
    )

    assert submitted == []
