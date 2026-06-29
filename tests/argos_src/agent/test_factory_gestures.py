import importlib
from pathlib import Path
import sys
import threading
import types

from argos_src.profile_config import _parse_profile as _raw_parse_profile


def _parse_profile(payload, *, profile_path, framework_config):
    merged = {"manifest": "puffle"}
    merged.update(payload)
    return _raw_parse_profile(
        merged,
        profile_path=profile_path,
        framework_config=framework_config,
    )


class _FakeProviderClient:
    pass


class _FakeEngagement:
    def snapshot(self):
        return types.SimpleNamespace(state="idle")


def _load_factory_module(monkeypatch):
    sys.modules.pop("argos_src.agent.factory", None)

    realtime_mod = types.ModuleType("argos_src.agent.agent_runtime")
    realtime_mod.RealtimeRobotAgent = object
    monkeypatch.setitem(sys.modules, "argos_src.agent.agent_runtime", realtime_mod)

    prompt_loader_mod = types.ModuleType("argos_src.resource_paths")
    prompt_loader_mod.load_system_prompt = lambda *_args, **_kwargs: "prompt"
    monkeypatch.setitem(sys.modules, "argos_src.resource_paths", prompt_loader_mod)

    battery_mod = types.ModuleType("argos_src.runtime.battery_state")
    battery_mod.BatteryStateCache = object
    monkeypatch.setitem(sys.modules, "argos_src.runtime.battery_state", battery_mod)

    nav_mod = types.ModuleType("argos_src.nav_support.locations")
    nav_mod.LocationStore = object
    nav_mod.NavigationState = object
    monkeypatch.setitem(sys.modules, "argos_src.nav_support.locations", nav_mod)

    tools_mod = types.ModuleType("argos_src.tools")
    tools_mod.MEMORY_TOOL_NAMES = ()
    tools_mod.NAVIGATION_TOOL_NAMES = ()
    tools_mod.build_builtin_tools = lambda **_kwargs: []
    tools_mod.build_knowledge_tools = lambda *_args, **_kwargs: []
    tools_mod.resolve_builtin_tool_name = lambda name, **_kwargs: name
    tools_mod.resolve_builtin_tool_names = lambda names, **_kwargs: tuple(names)
    monkeypatch.setitem(sys.modules, "argos_src.tools", tools_mod)

    bridges_mod = types.ModuleType("argos_src.agent.bridges")
    bridges_mod.FaceEventBridge = object
    bridges_mod.PatrolLoopBridge = object
    monkeypatch.setitem(sys.modules, "argos_src.agent.bridges", bridges_mod)

    startup_mod = types.ModuleType("argos_src.agent.startup")
    startup_mod.derive_initial_robot_posture = lambda **_kwargs: "standing"
    startup_mod.prepare_robot_for_agent_session = lambda *_args, **_kwargs: []
    monkeypatch.setitem(sys.modules, "argos_src.agent.startup", startup_mod)

    return importlib.import_module("argos_src.agent.factory")


def test_factory_does_not_create_gesture_runtime_when_disabled(monkeypatch):
    factory_mod = _load_factory_module(monkeypatch)
    profile = _parse_profile(
        {"name": "factory-disabled"},
        profile_path=Path("/tmp/factory-disabled.yaml"),
        framework_config={},
    )

    runtime = factory_mod._create_gesture_runtime(
        scenario_profile=profile,
        robot_client=_FakeProviderClient(),
        engagement=_FakeEngagement(),
    )

    assert runtime is None


def test_factory_creates_go2_gesture_runtime_when_enabled(monkeypatch):
    factory_mod = _load_factory_module(monkeypatch)
    created = {}

    class _FakeGestureRuntime:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(factory_mod, "GestureRuntime", _FakeGestureRuntime)
    profile = _parse_profile(
        {
            "name": "factory-enabled",
            "embodiment": {
                "gestures": {
                    "enabled": True,
                    "preset": "auto",
                }
            },
        },
        profile_path=Path("/tmp/factory-enabled.yaml"),
        framework_config={},
    )
    robot_client = _FakeProviderClient()
    engagement = _FakeEngagement()

    runtime = factory_mod._create_gesture_runtime(
        scenario_profile=profile,
        robot_client=robot_client,
        engagement=engagement,
    )

    assert isinstance(runtime, _FakeGestureRuntime)
    assert created["connector"] is robot_client
    assert created["engagement"] is engagement
    assert created["preset_name"] == "go2_pose_v1"
    assert created["enabled_states"] == ("idle", "listening")


def test_factory_passes_individual_gesture_state_toggles(monkeypatch):
    factory_mod = _load_factory_module(monkeypatch)
    created = {}

    class _FakeGestureRuntime:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(factory_mod, "GestureRuntime", _FakeGestureRuntime)
    profile = _parse_profile(
        {
            "name": "factory-enabled-tilt-only",
            "embodiment": {
                "gestures": {
                    "enabled": True,
                    "preset": "auto",
                    "tilt_enabled": True,
                    "nodding_enabled": False,
                }
            },
        },
        profile_path=Path("/tmp/factory-enabled-tilt-only.yaml"),
        framework_config={},
    )

    runtime = factory_mod._create_gesture_runtime(
        scenario_profile=profile,
        robot_client=_FakeProviderClient(),
        engagement=_FakeEngagement(),
    )

    assert isinstance(runtime, _FakeGestureRuntime)
    assert created["enabled_states"] == ("idle",)


def test_factory_skips_runtime_when_all_gesture_states_are_disabled(monkeypatch):
    factory_mod = _load_factory_module(monkeypatch)
    profile = _parse_profile(
        {
            "name": "factory-all-states-disabled",
            "embodiment": {
                "gestures": {
                    "enabled": True,
                    "preset": "auto",
                    "tilt_enabled": False,
                    "nodding_enabled": False,
                }
            },
        },
        profile_path=Path("/tmp/factory-all-states-disabled.yaml"),
        framework_config={},
    )

    runtime = factory_mod._create_gesture_runtime(
        scenario_profile=profile,
        robot_client=_FakeProviderClient(),
        engagement=_FakeEngagement(),
    )

    assert runtime is None


def test_factory_startup_does_not_block_when_employee_directory_warmup_fails(monkeypatch):
    factory_mod = _load_factory_module(monkeypatch)
    created_services = []

    class _FakeProviderClient:
        def start(self):
            return None

        def shutdown(self):
            return None

    class _FakeRealtimeRobotAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

        def shutdown(self):
            return None

    class _FakeEngagementStateMachine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._coalescer = None
            self._battery_low_submit = None

        def shutdown(self):
            return None

    class _FakeEventCoalescer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def submit(self, *args, **kwargs):
            return None

    class _FailingEmployeeDirectoryService:
        def __init__(self, *, site_code, email_domain=""):
            self.site_code = site_code
            self.email_domain = email_domain
            self.started = False
            self.stopped = False
            created_services.append(self)

        def start_background(self):
            self.started = True
            thread = threading.Thread(
                target=lambda: None,
                daemon=True,
            )
            thread.start()

        def shutdown(self):
            self.stopped = True

    monkeypatch.setattr(
        factory_mod,
        "create_provider_client",
        lambda **_kwargs: _FakeProviderClient(),
    )
    monkeypatch.setattr(factory_mod, "RealtimeRobotAgent", _FakeRealtimeRobotAgent)
    monkeypatch.setattr(
        factory_mod,
        "EngagementStateMachine",
        _FakeEngagementStateMachine,
    )
    monkeypatch.setattr(factory_mod, "EventCoalescer", _FakeEventCoalescer)
    monkeypatch.setitem(
        sys.modules,
        "argos_src.employee_directory",
        types.SimpleNamespace(
            EmployeeDirectoryService=_FailingEmployeeDirectoryService,
        ),
    )

    profile = _parse_profile(
        {
            "name": "employee-directory-startup",
            "employee_directory": {
                "enabled": True,
                "site_code": "BOS3",
                "email_domain": "chewy.com",
            },
            "battery": {
                "enabled": False,
            },
            "face_recognition": {
                "enabled": False,
            },
            "speaker_recognition": {
                "enabled": False,
            },
        },
        profile_path=Path("/tmp/employee-directory-startup.yaml"),
        framework_config={},
    )

    agent = factory_mod.create_agent(scenario_profile=profile)

    assert isinstance(agent, _FakeRealtimeRobotAgent)
    assert created_services and created_services[0].started is True
    assert created_services[0].email_domain == "chewy.com"
    assert agent.kwargs["employee_directory_service"] is created_services[0]
