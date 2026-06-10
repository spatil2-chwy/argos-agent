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
        state=bridges_module.EngagementState.IDLE,
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
