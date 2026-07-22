import base64
import logging
from collections import deque
import importlib
from pathlib import Path
import queue
from types import SimpleNamespace
import threading
import time
import sys
import types

from argos_src.agent.runtime_context import (
    format_people_context,
)
from argos_src.observability.observability import get_request_context
from argos_src.speaker_recognition.models import (
    SpeakerRecognitionPolicy,
    SpeakerResolutionResult,
    VoiceEnrollmentResult,
)

_TEMP_STUBS = set()

if "sounddevice" not in sys.modules:
    sys.modules["sounddevice"] = types.SimpleNamespace(
        InputStream=object,
        OutputStream=object,
        CallbackFlags=object,
    )
    _TEMP_STUBS.add("sounddevice")
if "websocket" not in sys.modules:
    sys.modules["websocket"] = types.SimpleNamespace(
        WebSocket=object,
        WebSocketConnectionClosedException=RuntimeError,
        create_connection=lambda *args, **kwargs: object(),
    )
    _TEMP_STUBS.add("websocket")
if "argos_src.media.audio_detection" not in sys.modules:
    audio_mod = types.ModuleType("argos_src.media.audio_detection")
    audio_mod.OpenWakeWord = lambda *args, **kwargs: SimpleNamespace(threshold=0.5)
    audio_mod.SileroVAD = lambda *args, **kwargs: (lambda audio, ctx: (False, {}))
    sys.modules["argos_src.media.audio_detection"] = audio_mod
    _TEMP_STUBS.add("argos_src.media.audio_detection")
if "argos_src.agent.factory" not in sys.modules:
    factory_mod = types.ModuleType("argos_src.agent.factory")
    factory_mod.create_ros2_agent = None
    sys.modules["argos_src.agent.factory"] = factory_mod
    _TEMP_STUBS.add("argos_src.agent.factory")

realtime_mod = importlib.import_module("argos_src.agent.agent_runtime")
for _name in _TEMP_STUBS:
    sys.modules.pop(_name, None)


class _FakeLatency:
    def __init__(self):
        self.events = []

    def emit(self, *args, **kwargs):
        self.events.append(dict(kwargs))
        return None

    def timing(self, metric, duration_s, **kwargs):
        event = dict(kwargs)
        event["metric"] = metric
        event["duration_s"] = duration_s
        self.events.append(event)
        return None


class _FakeEngagement:
    def __init__(self):
        self.human_inputs = []
        self.output_started = []
        self.done = []
        self.playback_events = []

    def on_human_input(self, req_id):
        self.human_inputs.append(req_id)

    def on_agent_output_started(self, req_id, *, stream_id=None):
        self.output_started.append((req_id, stream_id))

    def on_agent_done(self, *, has_reply, req_id):
        self.done.append((req_id, has_reply))

    def on_playback_event(self, event, req_id, *, stream_id=None):
        self.playback_events.append((event, req_id, stream_id))

    def snapshot(self):
        return SimpleNamespace(
            state="idle",
            req_id="",
            entered_at=0.0,
            expires_at=None,
            nav_active=False,
            nav_source="",
            nav_interruptible=True,
            nav_passive_listen_allowed=True,
        )


class _ImmediateExecutor:
    def submit(self, fn):
        fn()


class _RecordingExecutor:
    def __init__(self):
        self.submitted = []

    def submit(self, fn):
        self.submitted.append(fn)


class _FakeConnector:
    def __init__(self):
        self.messages = []

    def send_message(self, message, target=None, msg_type=None):
        self.messages.append((message, target, msg_type))


class _FakeTool:
    name = "fake_tool"
    description = "fake"

    def invoke(self, arguments):
        return {"success": True, "arguments": arguments}


def _parse_factory_profile(payload):
    from argos_src.profile_config import _parse_profile as _raw_parse_profile

    merged = {"manifest": "puffle"}
    merged.update(payload)
    return _raw_parse_profile(
        merged,
        profile_path=Path(f"/tmp/{merged['name']}.yaml"),
        framework_config={},
    )


def _load_factory_for_memory_tests(monkeypatch, *, created):
    monkeypatch.delitem(sys.modules, "argos_src.agent.factory", raising=False)

    class _FakeRealtimeRobotAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.owner_turn_controller = kwargs.get("owner_turn_controller")

        def is_recording_active(self):
            return False

        def update_face_presence_snapshot(self, snapshot):
            return None

        def shutdown(self):
            return None

    class _FakeEngagementStateMachine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.setdefault("engagements", []).append(self)

        def attach_coalescer(self, coalescer):
            self.coalescer = coalescer

        def attach_battery_low_submitter(self, submitter):
            self.battery_low_submitter = submitter

        def attach_recording_state_provider(self, provider):
            self.recording_state_provider = provider

        def shutdown(self):
            return None

    class _FakeEventCoalescer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.setdefault("coalescers", []).append(self)

        def submit(self, *args, **kwargs):
            return None

    class _FakeIdentityMemoryClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.retention_class = kwargs["retention_class"]
            created["identity_memory_clients"].append(self)

        def close(self):
            return None

    class _FakeFaceRecognitionService:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.start_calls = []
            created["face_services"].append(self)

        def start_loop(self, **kwargs):
            self.start_calls.append(kwargs)

    class _FakeFaceEnrollmentPolicy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeFaceRecognitionStabilitySettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeNavigationPolicy:
        def __init__(
            self,
            *,
            source,
            interruptible=True,
            passive_listen_allowed=True,
        ):
            self.source = source
            self.interruptible = interruptible
            self.passive_listen_allowed = passive_listen_allowed

    class _FakeNavigationState:
        def __init__(self, store):
            self.store = store

        def get_patrol(self):
            return {}

        def get_active_goal(self):
            return None

    class _FakeFaceEventBridge:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False
            created["face_event_bridges"].append(self)

        def start(self):
            self.started = True

        def stop(self):
            return None

    class _FakePatrolLoopBridge:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def on_nav_event(self, event):
            return None

    realtime_stub = types.ModuleType("argos_src.agent.agent_runtime")
    realtime_stub.RealtimeRobotAgent = _FakeRealtimeRobotAgent
    monkeypatch.setitem(sys.modules, "argos_src.agent.agent_runtime", realtime_stub)

    prompt_loader_mod = types.ModuleType("argos_src.resource_paths")
    prompt_loader_mod.load_system_prompt = lambda *_args, **_kwargs: "prompt"
    monkeypatch.setitem(sys.modules, "argos_src.resource_paths", prompt_loader_mod)

    battery_mod = types.ModuleType("argos_src.runtime.battery_state")
    battery_mod.BatteryStateCache = object
    battery_mod.LOW_BATTERY_NAVIGATION_MSG = "Battery is low. Cannot navigate."
    monkeypatch.setitem(sys.modules, "argos_src.runtime.battery_state", battery_mod)

    nav_mod = types.ModuleType("argos_src.nav_support.locations")
    nav_mod.LocationStore = lambda **kwargs: SimpleNamespace(kwargs=kwargs)
    nav_mod.CHARGE_DOCK_LOCATION_NAME = "charge_dock"
    nav_mod.NavigationPolicy = _FakeNavigationPolicy
    nav_mod.INTERRUPTIBLE_NAVIGATION_POLICY = _FakeNavigationPolicy(
        source="general_navigation"
    )
    nav_mod.FOCUSED_NAVIGATION_POLICY = _FakeNavigationPolicy(
        source="human_task",
        interruptible=False,
        passive_listen_allowed=False,
    )
    nav_mod.NavigationState = _FakeNavigationState
    monkeypatch.setitem(sys.modules, "argos_src.nav_support.locations", nav_mod)

    tools_mod = types.ModuleType("argos_src.tools")
    tools_mod.__path__ = []
    tools_mod.MEMORY_TOOL_NAMES = (
        "search_memory_semantic",
    )
    tools_mod.NAVIGATION_TOOL_NAMES = ()
    def _fake_build_builtin_tools(**kwargs):
        created.setdefault("build_builtin_tools_kwargs", []).append(kwargs)
        return []

    def _fake_resolve_builtin_tool_name(name, **_kwargs):
        if name == "memory.search_semantic":
            return "search_memory_semantic"
        return name

    tools_mod.build_builtin_tools = _fake_build_builtin_tools
    tools_mod.build_knowledge_tools = lambda *_args, **_kwargs: []
    tools_mod.resolve_builtin_tool_name = _fake_resolve_builtin_tool_name
    tools_mod.resolve_builtin_tool_names = lambda names, **kwargs: tuple(
        _fake_resolve_builtin_tool_name(name, **kwargs) for name in names
    )
    monkeypatch.setitem(sys.modules, "argos_src.tools", tools_mod)
    unitree_tools_mod = types.ModuleType("argos_src.tools.unitree_go2")
    unitree_tools_mod.__path__ = []
    navigation_tools_mod = types.ModuleType("argos_src.tools.unitree_go2.navigation")
    navigation_tools_mod.__path__ = []
    navigation_toolset_mod = types.ModuleType(
        "argos_src.tools.unitree_go2.navigation.toolset"
    )
    navigation_toolset_mod.process_navigation_event = lambda **_kwargs: None
    monkeypatch.setitem(sys.modules, "argos_src.tools.unitree_go2", unitree_tools_mod)
    monkeypatch.setitem(
        sys.modules,
        "argos_src.tools.unitree_go2.navigation",
        navigation_tools_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "argos_src.tools.unitree_go2.navigation.toolset",
        navigation_toolset_mod,
    )

    bridges_mod = types.ModuleType("argos_src.agent.bridges")
    bridges_mod.FaceEventBridge = _FakeFaceEventBridge
    bridges_mod.PatrolLoopBridge = _FakePatrolLoopBridge
    monkeypatch.setitem(sys.modules, "argos_src.agent.bridges", bridges_mod)

    startup_mod = types.ModuleType("argos_src.agent.startup")
    startup_mod.derive_initial_robot_posture = lambda **_kwargs: "standing"
    startup_mod.prepare_robot_for_agent_session = lambda *_args, **_kwargs: []
    monkeypatch.setitem(sys.modules, "argos_src.agent.startup", startup_mod)

    identity_memory_mod = types.ModuleType("argos_src.identity_memory")
    identity_memory_mod.TailwagHttpIdentityMemoryClient = _FakeIdentityMemoryClient
    identity_memory_mod.NoopIdentityMemoryClient = _FakeIdentityMemoryClient
    monkeypatch.setitem(sys.modules, "argos_src.identity_memory", identity_memory_mod)

    attention_mod = types.ModuleType("argos_src.face_recognition.attention_gate")
    attention_mod.AttentionGateSettings = lambda **kwargs: SimpleNamespace(**kwargs)
    attention_mod.AttentionSmoothingSettings = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(
        sys.modules,
        "argos_src.face_recognition.attention_gate",
        attention_mod,
    )

    depth_mod = types.ModuleType("argos_src.face_recognition.depth_gate")
    depth_mod.DepthGateSettings = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "argos_src.face_recognition.depth_gate", depth_mod)

    face_service_mod = types.ModuleType(
        "argos_src.face_recognition.face_recognition_service"
    )
    face_service_mod.FaceEnrollmentPolicy = _FakeFaceEnrollmentPolicy
    face_service_mod.FaceRecognitionStabilitySettings = (
        _FakeFaceRecognitionStabilitySettings
    )
    face_service_mod.FaceRecognitionService = _FakeFaceRecognitionService
    monkeypatch.setitem(
        sys.modules,
        "argos_src.face_recognition.face_recognition_service",
        face_service_mod,
    )

    factory_mod = importlib.import_module("argos_src.agent.factory")
    monkeypatch.setattr(
        factory_mod,
        "create_provider_client",
        lambda **_kwargs: SimpleNamespace(shutdown=lambda: None),
    )
    monkeypatch.setattr(factory_mod, "RealtimeRobotAgent", _FakeRealtimeRobotAgent)
    monkeypatch.setattr(factory_mod, "EngagementStateMachine", _FakeEngagementStateMachine)
    monkeypatch.setattr(factory_mod, "EventCoalescer", _FakeEventCoalescer)
    return factory_mod


class _FakeGestureRuntime:
    def __init__(self):
        self.recording_active = []
        self.shutdown_calls = 0

    def set_recording_active(self, active):
        self.recording_active.append(bool(active))

    def shutdown(self):
        self.shutdown_calls += 1


class _FakeOwnerTurnController:
    def __init__(self):
        self.requests = []
        self.cancellations = []

    def request_turn(self, *, person_id, req_id="", owner_source=""):
        self.requests.append(
            {
                "person_id": person_id,
                "req_id": req_id,
                "owner_source": owner_source,
            }
        )
        return True

    def cancel_request(self, *, req_id, reason=""):
        self.cancellations.append({"req_id": req_id, "reason": reason})


class _FakeRawDataCapture:
    def __init__(self):
        self.exchanges = []
        self.closed = False

    def save_exchange(self, **kwargs):
        self.exchanges.append(kwargs)

    def close(self):
        self.closed = True


class _FakeFaceService:
    def __init__(self, *, persons=None, snapshot=None):
        self._persons = list(persons or [])
        self._snapshot = dict(snapshot or {})

    def get_cached_persons(self):
        return list(self._persons)

    def get_presence_snapshot(self):
        return dict(self._snapshot)

    def get_primary_face_person_id(self):
        if len(self._persons) != 1 or int(self._snapshot.get("unknown_count", 0) or 0) != 0:
            return None
        return str(getattr(self._persons[0], "person_id", "") or "") or None


class _FakeSpeakerService:
    def __init__(self, *, policy=None, enrollment_results=None, references=None):
        self.policy = policy or SpeakerRecognitionPolicy()
        self._enrollment_results = deque(enrollment_results or [])
        self._references = set(references or [])
        self.calls = []
        self.trim_calls = []
        self.resolve_calls = []

    def has_reference(self, person_id):
        return str(person_id or "").strip() in self._references

    def trim_turn_audio(self, audio_pcm16, *, vad=None):
        self.trim_calls.append({"audio_pcm16": audio_pcm16, "vad": vad})
        return audio_pcm16

    def resolve_turn_owner(
        self,
        *,
        audio_pcm16,
        primary_face_person_id,
        visible_face_person_ids,
        face_evidence=None,
        log_fields=None,
    ):
        self.resolve_calls.append(
            {
                "audio_pcm16": audio_pcm16,
                "primary_face_person_id": primary_face_person_id,
                "visible_face_person_ids": tuple(visible_face_person_ids or ()),
                "face_evidence": dict(face_evidence or {}),
                "log_fields": dict(log_fields or {}),
            }
        )
        owner_id = str(primary_face_person_id or "").strip() or None
        visible_ids = {
            str(person_id or "").strip()
            for person_id in (visible_face_person_ids or ())
            if str(person_id or "").strip()
        }
        return SpeakerResolutionResult(
            audio_speaker_id=None,
            top_score=0.0,
            runner_up_score=0.0,
            margin=0.0,
            speaker_visible=bool(owner_id and (owner_id in visible_ids or not visible_ids)),
            owner_id=owner_id,
            owner_source="face" if owner_id else "unknown",
            owner_confidence=0.0,
        )

    def try_store_reference(self, *, person_id, audio_pcm16, attempt_kind):
        self.calls.append(
            {
                "person_id": person_id,
                "audio_pcm16": audio_pcm16,
                "attempt_kind": attempt_kind,
            }
        )
        if self._enrollment_results:
            result = self._enrollment_results.popleft()
        else:
            result = VoiceEnrollmentResult(
                saved=False,
                reason="reject_clipped",
                person_id=str(person_id or "").strip(),
                attempt_kind=attempt_kind,
            )
        if result.saved:
            self._references.add(str(person_id or "").strip())
        return result


def _make_agent():
    agent = realtime_mod.RealtimeRobotAgent.__new__(realtime_mod.RealtimeRobotAgent)
    agent.logger = logging.getLogger("test.argos.agent_runtime")
    agent.realtime_profile = SimpleNamespace(
        prompt_file="static_interaction_prompt.md",
        model="gpt-realtime-1.5",
        input_sample_rate=24000,
        output_sample_rate=24000,
        silence_grace_period=0.1,
        max_output_tokens=None,
        audio_output_speed=0.9,
        voice="cedar",
        transcription_model="gpt-4o-mini-transcribe",
        noise_reduction="near_field",
        language="en",
    )
    agent._stop_event = threading.Event()
    agent._turn_lock = threading.RLock()
    agent._recording_lock = threading.RLock()
    agent._audio_send_queue = queue.Queue()
    agent._turn_queue = queue.Queue()
    agent._tool_queue = queue.Queue()
    agent._playback_buffer = realtime_mod.PlaybackBuffer()
    agent._capture_state = "admission_closed"
    agent._played_output_frames = 0
    agent._playback_req_id = ""
    agent._playback_stream_id = ""
    agent._playback_item_id = ""
    agent._active_turn = None
    agent._turns_by_req_id = {}
    agent._response_id_to_req_id = {}
    agent._item_id_to_req_id = {}
    agent._call_id_to_req_id = {}
    agent._pending_function_args = {}
    agent._pending_response_turn_req_ids = deque()
    agent._expired_stale_response_turn_req_ids = deque()
    agent._stale_response_deadlines_by_req_id = {}
    agent._pending_audio_turn_req_ids = deque()
    agent._pending_audio_item_ids = deque()
    agent._pending_input_transcription_events = {}
    agent._pending_local_created_items = deque()
    agent._history_item_order = deque()
    agent._known_history_item_ids = set()
    agent._history_item_owner_req_id = {}
    agent._history_items = {}
    agent._history_item_snapshots = {}
    agent._active_inference_owner_key = ""
    agent._active_inference_scope_id = ""
    agent._pending_anonymous_inference_scope_id = ""
    agent._anonymous_inference_patch_index = 0
    agent._known_owner_names_by_owner_id = {}
    agent._ignored_voice_commands = deque()
    agent._latency = _FakeLatency()
    agent._tool_latency = _FakeLatency()
    agent.engagement = _FakeEngagement()
    agent.gesture_runtime = _FakeGestureRuntime()
    agent.owner_turn_controller = None
    agent.ros2_connector = _FakeConnector()
    agent._tool_registry = {"fake_tool": _FakeTool()}
    agent._tool_schemas = []
    agent.base_system_prompt = "You are Puffle."
    agent._last_tool_name = None
    agent._last_tool_summary = None
    agent._robot_posture = "standing"
    agent._stand_tool_name = "stand"
    agent._supports_navigation = False
    agent._current_office_location = ""
    agent.face_service = None
    agent.speaker_service = None
    agent.battery_cache = None
    agent.location_store = None
    agent._preference_segments = realtime_mod._PreferenceSegmentCoordinator()
    agent._pending_preference_segment_ids = set()
    agent._pending_lock = threading.Lock()
    agent._preference_idle_flush_lock = threading.Lock()
    agent._preference_idle_flush_timer = None
    agent._preference_idle_flush_delay_sec = 0.05
    agent._preference_executor = _ImmediateExecutor()
    agent.preference_extraction_enabled = True
    agent.preference_extractor = None
    agent._session_id = ""
    agent._session_estimated_cost_usd = 0.0
    agent._current_face_evidence_fields = {}
    agent._ws = object()
    agent._ws_lock = threading.Lock()
    agent.coalescer = None
    agent._vad = None
    sent = []

    def _send_event(payload):
        sent.append(payload)

    agent._send_event = _send_event
    agent._sent_events = sent
    agent._get_current_primary_face_person_id = lambda: "person-1"
    agent._current_primary_face_person_id = None
    agent._current_visible_face_person_ids = ()
    agent._current_turn_audio_chunks = []
    agent._current_turn_vad_positive_blocks = 0
    agent._pending_voice_enrollments = {}
    agent._voice_enrollment_lock = threading.Lock()
    agent._candidate_voice_blocks = 0
    agent._recording_preroll_chunks = deque()
    agent._recording_gesture_queue = queue.Queue()
    agent._recording_gesture_lock = threading.Lock()
    agent._recording_gesture_thread = None
    agent._display_queue = queue.Queue()
    agent._display_mode_lock = threading.Lock()
    agent._display_mode = ""
    return agent


def _make_turn(req_id: str, **kwargs):
    audio_speaker_id = kwargs.pop("audio_speaker_id", None)
    return realtime_mod.QueuedTurn(
        kind=kwargs.pop("kind", "audio"),
        req_id=req_id,
        speech_end_perf_s=kwargs.pop("speech_end_perf_s", 1.0),
        speech_end_unix_s=kwargs.pop("speech_end_unix_s", 2.0),
        transcript_perf_s=kwargs.pop("transcript_perf_s", 3.0),
        primary_face_person_id=kwargs.pop("primary_face_person_id", "person-1"),
        owner_id=kwargs.pop("owner_id", audio_speaker_id or "person-1"),
        audio_speaker_id=audio_speaker_id,
        context_snapshot=kwargs.pop(
            "context_snapshot",
            realtime_mod.FrozenTurnContext(
                primary_face_person_id=kwargs.pop("context_primary_face_person_id", "person-1"),
                owner_id=audio_speaker_id or "person-1",
                audio_speaker_id=audio_speaker_id,
            ),
        ),
        **kwargs,
    )


def test_build_turn_instructions_includes_current_office_location_block():
    agent = _make_agent()
    agent._current_office_location = "BOS1"

    instructions = agent._build_turn_instructions(_make_turn("rt-office"))

    assert "[CURRENT OFFICE LOCATION] BOS1" in instructions


def test_unknown_turn_instructions_forbid_identity_inference():
    agent = _make_agent()
    turn = _make_turn(
        "rt-unknown",
        owner_id=None,
        primary_face_person_id=None,
        context_snapshot=realtime_mod.FrozenTurnContext(owner_id=None),
    )

    instructions = agent._build_turn_instructions(turn)

    assert "[IDENTITY STATUS]" in instructions
    assert "Current speaker is not safely identified" in instructions
    assert "prior session history" in instructions
    assert "Only use a person's name when a current [PERSON SPEAKING TO YOU] block is present" in instructions


def test_anonymous_assistant_name_leak_is_quarantined_from_inference():
    agent = _make_agent()
    agent._known_owner_names_by_owner_id = {"person-1": {"Sakshee"}}
    turn = _make_turn(
        "rt-anon-leak",
        owner_id=None,
        primary_face_person_id=None,
        context_snapshot=realtime_mod.FrozenTurnContext(owner_id=None),
    )
    turn.inference_owner_key = "anonymous"
    turn.inference_scope_id = "anonymous:1"
    turn.assistant_item_id = "asst-leak"
    turn.assistant_item_ids.add("asst-leak")
    turn.assistant_transcript = "Hey Sakshee, Puffle remembers that smile!"
    agent._turns_by_req_id[turn.req_id] = turn
    agent._register_turn_history_item(
        turn,
        "asst-leak",
        item_type="message",
        role="assistant",
        status="done",
        permitted_for_inference=True,
    )

    assert agent._quarantine_anonymous_history_if_needed(turn)

    assert agent._history_items["asst-leak"].permitted_for_inference is False
    assert any(
        event.get("event") == "anonymous_history_quarantined"
        for event in agent._latency.events
    )


def test_quarantined_anonymous_assistant_item_is_excluded_from_tool_followup_input():
    agent = _make_agent()
    agent._known_owner_names_by_owner_id = {"person-1": {"Sakshee"}}
    turn = _make_turn(
        "rt-anon-tool-leak",
        owner_id=None,
        primary_face_person_id=None,
        context_snapshot=realtime_mod.FrozenTurnContext(owner_id=None),
    )
    turn.inference_owner_key = "anonymous"
    turn.inference_scope_id = "anonymous:1"
    turn.user_item_id = "anon-user"
    turn.assistant_item_id = "asst-leak"
    turn.assistant_item_ids.add("asst-leak")
    turn.assistant_transcript = "Hey Sakshee, let me check that."
    agent._turns_by_req_id[turn.req_id] = turn
    agent._register_turn_history_item(
        turn,
        "anon-user",
        item_type="message",
        role="user",
        status="done",
        permitted_for_inference=True,
    )
    agent._register_turn_history_item(
        turn,
        "asst-leak",
        item_type="message",
        role="assistant",
        status="done",
        permitted_for_inference=True,
    )
    agent._register_turn_history_item(
        turn,
        "call-item",
        item_type="function_call",
        status="done",
        permitted_for_inference=True,
        input_item={
            "id": "call-item",
            "type": "function_call",
            "name": "fake_tool",
            "call_id": "call-1",
            "arguments": "{}",
            "status": "completed",
        },
    )
    turn.function_call_item_ids.add("call-item")

    assert agent._quarantine_anonymous_history_if_needed(turn)

    assert agent._response_input_items_for_turn(turn) == [
        {"type": "item_reference", "id": "anon-user"},
        {
            "id": "call-item",
            "type": "function_call",
            "name": "fake_tool",
            "call_id": "call-1",
            "arguments": "{}",
            "status": "completed",
        },
    ]


def test_response_create_waits_for_audio_item_id_before_sending():
    agent = _make_agent()
    turn = _make_turn("rt-audio-wait")
    turn.inference_scope_id = "owner:person-1"
    turn.inference_owner_key = "owner:person-1"
    original_timeout = realtime_mod.CURRENT_AUDIO_ITEM_ACK_TIMEOUT_SEC
    realtime_mod.CURRENT_AUDIO_ITEM_ACK_TIMEOUT_SEC = 0.0
    try:
        agent._send_response_create(turn)
    finally:
        realtime_mod.CURRENT_AUDIO_ITEM_ACK_TIMEOUT_SEC = original_timeout

    assert not [event for event in agent._sent_events if event.get("type") == "response.create"]
    assert turn.phase == realtime_mod.TURN_PHASE_CANCELED
    assert any(event.get("event") == "audio_item_bind_timeout" for event in agent._latency.events)


def test_early_audio_commit_item_becomes_future_selectable():
    agent = _make_agent()
    agent._handle_input_audio_buffer_committed(
        {"type": "input_audio_buffer.committed", "item_id": "early-audio"}
    )
    first_turn = _make_turn("rt-first")
    agent._turns_by_req_id[first_turn.req_id] = first_turn

    agent._register_pending_audio_turn(first_turn)

    assert first_turn.user_item_id == "early-audio"
    item = agent._history_items["early-audio"]
    assert item.scope_id == "owner:person-1"
    assert item.status == "done"
    assert item.permitted_for_inference is True
    first_turn.phase = realtime_mod.TURN_PHASE_FINALIZED
    first_turn.finalized = True

    second_turn = _make_turn("rt-second")
    second_turn.inference_scope_id = "owner:person-1"
    second_turn.inference_owner_key = "owner:person-1"
    agent._turns_by_req_id[second_turn.req_id] = second_turn

    assert agent._response_input_items_for_turn(second_turn) == [
        {"type": "item_reference", "id": "early-audio"}
    ]


def test_exchange_log_fields_include_speaker_resolution_scores():
    agent = _make_agent()
    turn = _make_turn(
        "rt-score",
        metadata={
            "audio_score": 0.62,
            "audio_runner_up_score": 0.21,
            "audio_score_margin": 0.41,
        },
    )

    fields = agent._exchange_log_fields(turn)

    assert fields["audio_score"] == 0.62
    assert fields["audio_runner_up_score"] == 0.21
    assert fields["audio_score_margin"] == 0.41


def test_build_turn_instructions_includes_memory_context_blocks():
    agent = _make_agent()
    turn = _make_turn(
        "rt-memory-block",
        context_snapshot=realtime_mod.FrozenTurnContext(
            memory_context_blocks=("[OFFICE CONTEXT]\n- Free snacks today.",),
        ),
    )

    instructions = agent._build_turn_instructions(turn)

    assert "[OFFICE CONTEXT]" in instructions
    assert "Free snacks today." in instructions


def test_send_event_stops_runtime_when_websocket_is_already_closed():
    agent = _make_agent()

    class _ClosedSocket:
        def send(self, _payload):
            raise RuntimeError("socket is already closed.")

    agent._send_event = realtime_mod.AgentStateRuntime._send_event.__get__(
        agent,
        realtime_mod.RealtimeRobotAgent,
    )
    agent._ws = _ClosedSocket()
    agent._stop_event.clear()

    agent._send_event({"type": "response.create"})

    assert agent._stop_event.is_set()
    assert agent._ws is None


def test_handle_server_event_routes_output_text_delta_through_dispatch():
    agent = _make_agent()
    calls = []

    def _handle_output_text_delta(event):
        calls.append(event)

    agent._handle_output_text_delta = _handle_output_text_delta

    agent._handle_server_event({"type": "response.output_text.delta", "delta": "hello"})

    assert calls == [{"type": "response.output_text.delta", "delta": "hello"}]


def test_forced_idle_display_mode_requeues_even_when_cached():
    agent = _make_agent()
    agent.display_runtime = object()
    agent._display_queue = queue.Queue()
    agent._display_mode_lock = threading.Lock()
    agent._display_mode = "idle"

    agent._set_display_mode_async("idle", force=True)

    assert agent._display_queue.get_nowait() == ("mode", "idle")


def test_display_subtitle_window_has_no_character_limit():
    text = " ".join(f"word{i}" for i in range(80))

    assert realtime_mod.RealtimeRobotAgent._display_subtitle_window(text) == text


def test_superseded_turn_is_canceled_and_old_audio_is_ignored():
    agent = _make_agent()
    old_turn = _make_turn("rt-old")
    old_turn.response_id = "resp-old"
    agent._turns_by_req_id[old_turn.req_id] = old_turn
    agent._active_turn = old_turn
    agent._bind_response_id(old_turn, old_turn.response_id)

    new_turn = _make_turn("rt-new")
    agent._supersede_unanswered_turn(new_turn)

    assert old_turn.phase == realtime_mod.TURN_PHASE_SUPERSEDED
    assert old_turn.response_finished.is_set()
    assert old_turn.playback_finished.is_set()
    assert any(evt["type"] == "response.cancel" for evt in agent._sent_events)

    before = agent._playback_buffer.buffered_frames()
    agent._handle_output_audio_delta(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-old",
            "item_id": "asst-old",
            "delta": "AQI=",
        }
    )
    assert agent._playback_buffer.buffered_frames() == before


def test_response_watchdog_cancels_stalled_turn():
    agent = _make_agent()
    turn = _make_turn("rt-stall")
    turn.phase = realtime_mod.TURN_PHASE_RESPONSE_REQUESTED
    turn.response_requested_at = time.time() - 10.0
    agent._turns_by_req_id[turn.req_id] = turn

    old_timeout = realtime_mod.RESPONSE_STALL_TIMEOUT_SEC
    try:
        realtime_mod.RESPONSE_STALL_TIMEOUT_SEC = 0.1
        watchdog = threading.Thread(target=agent._watchdog_loop, daemon=True)
        watchdog.start()
        time.sleep(0.35)
        agent._stop_event.set()
        watchdog.join(timeout=1.0)
    finally:
        realtime_mod.RESPONSE_STALL_TIMEOUT_SEC = old_timeout

    assert turn.phase == realtime_mod.TURN_PHASE_CANCELED
    assert turn.response_finished.is_set()
    assert turn.playback_finished.is_set()


def test_no_active_response_cancel_error_does_not_terminate_active_turn():
    agent = _make_agent()
    turn = _make_turn("rt-cancel-race")
    turn.phase = realtime_mod.TURN_PHASE_WAITING_FIRST_AUDIO
    agent._turns_by_req_id[turn.req_id] = turn
    agent._active_turn = turn

    agent._handle_server_error(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Cancellation failed: no active response found",
            },
        }
    )

    assert turn.phase == realtime_mod.TURN_PHASE_WAITING_FIRST_AUDIO
    assert not turn.finalized
    assert not any(event.get("event") == "exchange_terminal" for event in agent._latency.events)


def test_non_cancel_race_server_error_terminates_active_turn():
    agent = _make_agent()
    turn = _make_turn("rt-server-error")
    turn.phase = realtime_mod.TURN_PHASE_WAITING_FIRST_AUDIO
    agent._turns_by_req_id[turn.req_id] = turn
    agent._active_turn = turn

    agent._handle_server_error(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Some other realtime request failed",
            },
        }
    )

    assert turn.phase == realtime_mod.TURN_PHASE_CANCELED
    assert turn.finalized
    terminal = next(event for event in agent._latency.events if event.get("event") == "exchange_terminal")
    assert terminal["terminal_reason"] == "server_error"
    assert terminal["error_source"] == "openai_realtime"
    assert terminal["error_type"] == "invalid_request_error"
    assert terminal["error_message"] == "Some other realtime request failed"
    assert terminal["server_error_type"] == "invalid_request_error"
    assert terminal["server_error_message"] == "Some other realtime request failed"


def test_terminated_runtime_turn_logs_generic_error_fields():
    agent = _make_agent()
    turn = _make_turn("rt-timeout")
    turn.phase = realtime_mod.TURN_PHASE_WAITING_FIRST_AUDIO
    agent._turns_by_req_id[turn.req_id] = turn

    agent._terminate_turn(
        turn,
        realtime_mod.TURN_PHASE_CANCELED,
        "response_timeout",
        send_cancel=False,
    )

    terminal = next(event for event in agent._latency.events if event.get("event") == "exchange_terminal")
    assert terminal["terminal_reason"] == "response_timeout"
    assert terminal["error_source"] == "runtime"
    assert terminal["error_type"] == "response_timeout"
    assert terminal["error_message"] == "response_timeout"


def test_tool_barrier_waits_for_all_tool_results_before_followup_response():
    agent = _make_agent()
    owner_turn = _FakeOwnerTurnController()
    agent.owner_turn_controller = owner_turn
    turn = _make_turn("rt-tools")
    turn.pending_tool_calls = 2
    turn.pending_call_ids = {"call-1", "call-2"}
    agent._turns_by_req_id[turn.req_id] = turn
    followups = []
    agent._send_response_create = lambda queued_turn: followups.append(queued_turn.req_id)

    agent._execute_tool_call(
        realtime_mod.PendingToolCall(
            turn_req_id=turn.req_id,
            call_id="call-1",
            tool_name="fake_tool",
            arguments_json='{"a": 1}',
        )
    )
    assert turn.pending_tool_calls == 1
    assert followups == []
    assert owner_turn.cancellations == []

    agent._execute_tool_call(
        realtime_mod.PendingToolCall(
            turn_req_id=turn.req_id,
            call_id="call-2",
            tool_name="fake_tool",
            arguments_json='{"b": 2}',
        )
    )
    assert turn.pending_tool_calls == 0
    assert followups == [turn.req_id]
    assert owner_turn.cancellations == []

def test_tool_invocation_context_includes_owner_id():
    agent = _make_agent()
    seen = {}

    class _ContextTool:
        name = "context_tool"

        def invoke(self, _arguments):
            seen.update(get_request_context())
            return {"success": True}

    agent._tool_registry = {"context_tool": _ContextTool()}
    turn = _make_turn("rt-owner-tool", owner_id="person-owner", owner_source="audio")
    turn.pending_tool_calls = 1
    turn.pending_call_ids = {"call-owner"}
    agent._turns_by_req_id[turn.req_id] = turn
    agent._send_response_create = lambda _turn: None

    agent._execute_tool_call(
        realtime_mod.PendingToolCall(
            turn_req_id=turn.req_id,
            call_id="call-owner",
            tool_name="context_tool",
            arguments_json="{}",
        )
    )

    assert seen["owner_id"] == "person-owner"
    assert seen["owner_source"] == "audio"


def test_recording_hooks_update_gesture_runtime():
    agent = _make_agent()

    agent._start_recording_locked(now_s=10.0)
    agent._finalize_recording_locked(now_s=11.0)

    deadline = time.time() + 1.0
    while len(agent.gesture_runtime.recording_active) < 2 and time.time() < deadline:
        time.sleep(0.01)

    assert agent.gesture_runtime.recording_active == [True, False]


def test_recording_started_freezes_face_match_evidence():
    agent = _make_agent()
    agent.face_service = _FakeFaceService(
        persons=[SimpleNamespace(person_id="person-1", visible=True)],
        snapshot={
            "face_match_status": "rejected",
            "face_match_reason": "below_threshold",
            "face_match_name": "Alice",
            "face_match_person_id": "person-1",
            "face_score": 0.42,
            "face_score_threshold": 0.6,
            "face_runner_up_score": 0.31,
            "face_score_margin": 0.11,
            "face_margin_threshold": 0.2,
        },
    )

    agent._start_recording_locked(now_s=10.0)

    started = next(
        event for event in agent._latency.events if event.get("event") == "recording_started"
    )
    assert started["face_match_status"] == "rejected"
    assert started["face_match_reason"] == "below_threshold"
    assert started["face_score"] == 0.42
    assert agent._current_face_evidence_fields["face_score_margin"] == 0.11


def test_recording_display_moves_from_recording_to_thinking():
    agent = _make_agent()
    agent.display_runtime = object()
    agent._display_queue = queue.Queue()
    agent._display_mode_lock = threading.Lock()
    agent._display_mode = ""

    agent._start_recording_locked(now_s=10.0)
    agent._finalize_recording_locked(now_s=11.0)

    assert agent._display_queue.get_nowait() == ("mode", "recording")
    assert agent._display_queue.get_nowait() == ("mode", "thinking")


def test_recording_gesture_does_not_block_audio_finalize():
    class _SlowGestureRuntime:
        def __init__(self):
            self.recording_active = []

        def set_recording_active(self, active):
            time.sleep(0.25)
            self.recording_active.append(bool(active))

    agent = _make_agent()
    agent.gesture_runtime = _SlowGestureRuntime()

    started = time.monotonic()
    agent._finalize_recording_locked(now_s=11.0)
    elapsed = time.monotonic() - started

    assert elapsed < 0.1
    assert any(evt.get("event") == "speech_end" for evt in agent._latency.events)


def test_speech_end_claims_engagement_before_speaker_resolution_finishes():
    import numpy as np

    resolution_started = threading.Event()
    release_resolution = threading.Event()

    class _StatefulEngagement(_FakeEngagement):
        def __init__(self):
            super().__init__()
            self.state = "alert"

        def on_human_input(self, req_id):
            super().on_human_input(req_id)
            self.state = "engaged"

    class _BlockingSpeakerService(_FakeSpeakerService):
        def resolve_turn_owner(self, **kwargs):
            resolution_started.set()
            release_resolution.wait(timeout=2.0)
            return super().resolve_turn_owner(**kwargs)

    agent = _make_agent()
    agent.engagement = _StatefulEngagement()
    agent.speaker_service = _BlockingSpeakerService()
    agent.gesture_runtime = None
    agent._current_primary_face_person_id = "person-1"
    agent._current_visible_face_person_ids = ("person-1",)
    agent._current_turn_audio_chunks = [b"\x01\x02"]
    agent._session_ready = threading.Event()
    agent._session_ready.set()
    req_id = ""

    try:
        agent._finalize_recording_locked(now_s=11.0)

        assert resolution_started.wait(timeout=1.0)
        assert agent.engagement.state == "engaged"
        assert len(agent.engagement.human_inputs) == 1
        req_id = agent.engagement.human_inputs[0]
        assert req_id.startswith("rt-")
        speech_end = next(
            event for event in agent._latency.events if event.get("event") == "speech_end"
        )
        assert speech_end["req_id"] == req_id
        assert req_id not in agent._turns_by_req_id
        assert agent._audio_turn_pending_registration is True
        assert agent.is_recording_active() is True

        sent_count = len(agent._sent_events)
        agent._capture_callback(
            np.zeros((1600, 1), dtype=np.int16),
            1600,
            None,
            None,
        )
        assert agent._recording_active is False
        assert len(agent._sent_events) == sent_count
    finally:
        release_resolution.set()

    deadline = time.time() + 1.0
    while req_id not in agent._turns_by_req_id and time.time() < deadline:
        time.sleep(0.01)
    assert req_id in agent._turns_by_req_id
    assert agent.engagement.human_inputs == [req_id]
    assert agent._audio_turn_pending_registration is False


def test_audio_commit_failure_releases_early_engagement_claim():
    agent = _make_agent()
    agent._send_event = lambda _event: (_ for _ in ()).throw(RuntimeError("commit failed"))

    agent._commit_audio_turn(
        primary_face_person_id="person-1",
        visible_face_person_ids=("person-1",),
        audio_pcm16=b"\x01\x02",
        capture_vad_positive_blocks=1,
        speech_end_perf_s=1.0,
        speech_end_unix_s=2.0,
        req_id="rt-commit-failed",
    )

    assert agent._capture_state == "admission_closed"
    assert agent._audio_turn_pending_registration is False
    assert agent.engagement.human_inputs == ["rt-commit-failed"]
    assert agent.engagement.done == [("rt-commit-failed", False)]


def test_audio_commit_worker_start_failure_releases_pending_registration(monkeypatch):
    agent = _make_agent()
    agent.gesture_runtime = None

    def fail_start(_thread):
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", fail_start)

    agent._finalize_recording_locked(now_s=11.0)

    assert agent._audio_turn_pending_registration is False
    assert agent._capture_state == "admission_closed"
    assert agent.engagement.human_inputs == []


def test_owner_resolution_failure_releases_pending_registration_and_engagement():
    agent = _make_agent()
    agent._audio_turn_pending_registration = True
    agent._face_owner_resolution = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("owner resolution failed")
    )

    agent._commit_audio_turn(
        primary_face_person_id="person-1",
        visible_face_person_ids=("person-1",),
        audio_pcm16=b"\x01\x02",
        capture_vad_positive_blocks=1,
        speech_end_perf_s=1.0,
        speech_end_unix_s=2.0,
        req_id="rt-owner-failed",
    )

    assert {event["type"] for event in agent._sent_events} == {
        "input_audio_buffer.commit"
    }
    assert agent._audio_turn_pending_registration is False
    assert agent.engagement.human_inputs == ["rt-owner-failed"]
    assert agent.engagement.done == [("rt-owner-failed", False)]


def test_audio_face_uses_only_strict_primary_face_id():
    agent = _make_agent()
    agent.speaker_service = _FakeSpeakerService(
        policy=SpeakerRecognitionPolicy(query_match_threshold=0.60)
    )

    result = agent._face_owner_resolution(
        primary_face_person_id=None,
    )

    assert result.owner_id is None
    assert result.owner_source == "unknown"

    result = agent._face_owner_resolution(
        primary_face_person_id="person-1",
    )

    assert result.owner_id == "person-1"
    assert result.owner_source == "face"


def test_input_transcription_binds_to_exact_audio_turn_by_item_id():
    agent = _make_agent()
    turn = _make_turn("rt-audio")
    agent._turns_by_req_id[turn.req_id] = turn
    agent._pending_audio_turn_req_ids.append(turn.req_id)

    agent._handle_input_transcription_completed(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "user-item-1",
            "transcript": "hello there",
        }
    )

    assert turn.user_item_id == "user-item-1"
    assert turn.user_transcript == "hello there"
    assert agent._item_id_to_req_id["user-item-1"] == turn.req_id


def test_input_audio_buffer_committed_can_arrive_before_turn_registration():
    agent = _make_agent()

    agent._handle_input_audio_buffer_committed(
        {
            "type": "input_audio_buffer.committed",
            "item_id": "early-user-item",
        }
    )
    turn = _make_turn("rt-audio-early-item")
    agent._turns_by_req_id[turn.req_id] = turn
    agent._register_pending_audio_turn(turn)

    assert turn.user_item_id == "early-user-item"
    assert agent._item_id_to_req_id["early-user-item"] == turn.req_id
    assert list(agent._pending_audio_turn_req_ids) == []


def test_input_transcription_completed_before_turn_registration_is_replayed():
    agent = _make_agent()

    agent._handle_input_audio_buffer_committed(
        {
            "type": "input_audio_buffer.committed",
            "item_id": "early-transcribed-user-item",
        }
    )
    agent._handle_input_transcription_completed(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "early-transcribed-user-item",
            "transcript": "the game was canceled",
        }
    )

    assert "early-transcribed-user-item" in agent._pending_input_transcription_events

    turn = _make_turn("rt-audio-early-transcript")
    agent._turns_by_req_id[turn.req_id] = turn
    agent._register_pending_audio_turn(turn)

    assert turn.user_item_id == "early-transcribed-user-item"
    assert turn.user_transcript == "the game was canceled"
    assert agent._history_item_snapshots[turn.user_item_id]["text"] == "the game was canceled"
    assert agent._pending_input_transcription_events == {}


def test_input_transcription_failed_is_logged_for_bound_turn():
    agent = _make_agent()
    warnings = []
    agent.logger = SimpleNamespace(
        warning=lambda *args: warnings.append(args),
        debug=lambda *args: None,
    )
    turn = _make_turn("rt-transcribe-failed")
    agent._turns_by_req_id[turn.req_id] = turn
    agent._bind_item_id_to_turn(turn, "failed-user-item")

    agent._handle_input_transcription_failed(
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "item_id": "failed-user-item",
            "error": {
                "type": "transcription_error",
                "code": "audio_unintelligible",
                "message": "not enough signal",
            },
        }
    )

    assert warnings == [
        (
            "Input transcription failed req_id=%s item_id=%s type=%s code=%s message=%s",
            "rt-transcribe-failed",
            "failed-user-item",
            "transcription_error",
            "audio_unintelligible",
            "not enough signal",
        )
    ]


def test_input_transcription_logs_usage_cost():
    agent = _make_agent()
    turn = _make_turn("rt-transcribe")
    agent._turns_by_req_id[turn.req_id] = turn
    agent._pending_audio_turn_req_ids.append(turn.req_id)
    agent._session_id = "sess-123"

    agent._handle_input_transcription_completed(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "user-item-2",
            "transcript": "hello there",
            "usage": {
                "input_tokens": 120,
                "output_tokens": 20,
                "total_tokens": 140,
                "input_token_details": {"audio_tokens": 120},
                "output_token_details": {"text_tokens": 20},
            },
        }
    )

    usage_event = next(
        evt for evt in agent._latency.events if evt.get("event") == "transcription_usage"
    )
    assert usage_event["req_id"] == turn.req_id
    assert usage_event["session_id"] == "sess-123"
    assert usage_event["model"] == "gpt-4o-mini-transcribe"
    assert usage_event["input_audio_tokens"] == 120
    assert usage_event["output_text_tokens"] == 20
    assert usage_event["estimated_cost_usd"] == 0.00025
    assert usage_event["session_total_cost_usd"] == 0.00025


def test_output_audio_and_transcript_resolve_by_response_and_item_id():
    agent = _make_agent()
    turn = _make_turn("rt-output")
    agent._turns_by_req_id[turn.req_id] = turn
    agent._bind_response_id(turn, "resp-1")

    agent._handle_output_audio_delta(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-1",
            "item_id": "asst-1",
            "delta": "AQI=",
        }
    )
    agent._handle_output_transcript_delta(
        {
            "type": "response.output_audio_transcript.delta",
            "response_id": "resp-1",
            "item_id": "asst-1",
            "delta": "hi",
        }
    )

    assert turn.audio_started is True
    assert turn.phase == realtime_mod.TURN_PHASE_PLAYING
    assert turn.assistant_item_id == "asst-1"
    assert turn.assistant_transcript == "hi"
    assert agent.engagement.output_started == [(turn.req_id, "resp-1")]


def test_response_done_flushes_complete_audio_transcript_to_display():
    agent = _make_agent()
    agent.display_runtime = object()
    agent._display_queue = queue.Queue()
    turn = _make_turn("rt-final-display")
    turn.response_id = "resp-final-display"
    turn.audio_started = True
    turn.assistant_transcript = "partial"
    agent._turns_by_req_id[turn.req_id] = turn
    agent._bind_response_id(turn, turn.response_id)

    agent._handle_response_done(
        {
            "type": "response.done",
            "response": {
                "id": "resp-final-display",
                "status": "completed",
                "output": [
                    {
                        "id": "asst-final-display",
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "partial transcript completed",
                            }
                        ],
                    }
                ],
            },
        }
    )

    assert turn.assistant_transcript == "partial transcript completed"
    assert agent._display_queue.get_nowait() == (
        "subtitle",
        {"text": "partial transcript completed", "duration_ms": 5000},
    )
    assert agent._history_items["asst-final-display"].input_item == {
        "id": "asst-final-display",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": "partial transcript completed",
            }
        ],
    }


def test_owner_turn_is_requested_when_audio_turn_commits():
    agent = _make_agent()
    owner_turn = _FakeOwnerTurnController()
    agent.owner_turn_controller = owner_turn

    agent._commit_audio_turn(
        primary_face_person_id="person-7",
        visible_face_person_ids=("person-7",),
        audio_pcm16=b"\x01\x02",
        capture_vad_positive_blocks=1,
        speech_end_perf_s=1.0,
        speech_end_unix_s=2.0,
    )

    assert len(owner_turn.requests) == 1
    assert owner_turn.requests[0]["person_id"] == "person-7"
    assert owner_turn.requests[0]["req_id"].startswith("rt-")
    assert owner_turn.requests[0]["owner_source"] == "face"


def test_audio_turn_uses_raw_speaker_audio_without_trim_pass():
    agent = _make_agent()
    agent.speaker_service = _FakeSpeakerService()
    agent._vad = object()

    agent._commit_audio_turn(
        primary_face_person_id="person-7",
        visible_face_person_ids=("person-7",),
        audio_pcm16=b"\x01\x02",
        capture_vad_positive_blocks=1,
        speech_end_perf_s=1.0,
        speech_end_unix_s=2.0,
    )

    assert agent.speaker_service.trim_calls == []
    assert agent.speaker_service.resolve_calls[0]["audio_pcm16"] == b"\x01\x02"


def test_audio_commit_queues_raw_exchange_artifacts_after_owner_resolution():
    agent = _make_agent()
    raw_capture = _FakeRawDataCapture()
    agent.raw_data_capture = raw_capture

    agent._commit_audio_turn(
        primary_face_person_id="person-7",
        visible_face_person_ids=("person-7",),
        audio_pcm16=b"\x01\x02",
        capture_vad_positive_blocks=3,
        speech_end_perf_s=1.0,
        speech_end_unix_s=2.0,
        face_evidence_fields={"face_match_status": "matched"},
    )

    assert len(raw_capture.exchanges) == 1
    saved = raw_capture.exchanges[0]
    assert saved["exchange_id"] == agent._current_exchange_id
    assert saved["owner_id"] == "person-7"
    assert saved["owner_source"] == "face"
    assert saved["audio_pcm16"] == b"\x01\x02"
    assert saved["sample_rate_hz"] == 16000
    assert saved["metadata"]["face_match_status"] == "matched"
    assert saved["metadata"]["capture_vad_positive_blocks"] == 3


def test_output_audio_does_not_request_duplicate_owner_turn():
    agent = _make_agent()
    owner_turn = _FakeOwnerTurnController()
    agent.owner_turn_controller = owner_turn
    turn = _make_turn("rt-owner-audio", owner_id="person-7", owner_source="speaker")
    agent._turns_by_req_id[turn.req_id] = turn
    agent._bind_response_id(turn, "resp-owner")

    agent._handle_output_audio_delta(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-owner",
            "item_id": "asst-owner",
            "delta": "AQI=",
        }
    )

    assert owner_turn.requests == []


def test_playback_completion_returns_display_to_idle():
    agent = _make_agent()
    display_modes = []
    agent._set_display_mode_async = lambda mode: display_modes.append(mode)
    turn = _make_turn("rt-playback-display")
    turn.response_id = "resp-playback-display"
    turn.response_finished.set()

    agent._wait_for_playback_and_complete(turn, "resp-playback-display")

    assert agent.engagement.playback_events[-1] == (
        "playback_completed",
        turn.req_id,
        "resp-playback-display",
    )
    assert display_modes == ["idle"]


def test_playback_stopped_returns_display_to_idle():
    agent = _make_agent()
    display_modes = []
    agent._set_display_mode_async = lambda mode: display_modes.append(mode)
    turn = _make_turn("rt-playback-stopped-display")
    turn.response_id = "resp-playback-stopped-display"
    with agent._turn_lock:
        agent._playback_req_id = turn.req_id

    agent._force_complete_stalled_playback(turn, reason="test")

    assert agent.engagement.playback_events[-1] == (
        "playback_stopped",
        turn.req_id,
        "resp-playback-stopped-display",
    )
    assert display_modes == ["idle"]


def test_function_call_cancels_queued_owner_turn():
    agent = _make_agent()
    owner_turn = _FakeOwnerTurnController()
    agent.owner_turn_controller = owner_turn
    turn = _make_turn("rt-owner-cancel")
    agent._turns_by_req_id[turn.req_id] = turn
    agent._bind_response_id(turn, "resp-cancel")

    agent._handle_function_call_done(
        {
            "type": "response.function_call_arguments.done",
            "response_id": "resp-cancel",
            "item_id": "func-1",
            "call_id": "call-1",
            "name": "move_robot",
            "arguments": "{}",
        }
    )

    assert owner_turn.cancellations == [
        {"req_id": "rt-owner-cancel", "reason": "tool:move_robot"}
    ]


def test_first_audio_latency_is_not_logged_for_text_turns():
    agent = _make_agent()
    turn = _make_turn(
        "evt-first-audio-text",
        kind="text",
        speech_end_perf_s=0.0,
        speech_end_unix_s=time.time(),
    )
    turn.response_id = "resp-text-first-audio"
    agent._turns_by_req_id[turn.req_id] = turn
    agent._bind_response_id(turn, turn.response_id)

    agent._handle_output_audio_delta(
        {
            "type": "response.output_audio.delta",
            "response_id": turn.response_id,
            "item_id": "asst-text-first-audio",
            "delta": "AQI=",
        }
    )

    assert not any(
        evt.get("metric") == "first_audio_latency_s"
        for evt in agent._latency.events
    )


def test_unknown_output_audio_does_not_bind_pending_turn():
    agent = _make_agent()
    turn = _make_turn("rt-pending")
    agent._turns_by_req_id[turn.req_id] = turn
    agent._pending_response_turn_req_ids.append(turn.req_id)

    before = agent._playback_buffer.buffered_frames()
    agent._handle_output_audio_delta(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-stale",
            "item_id": "asst-stale",
            "delta": "AQI=",
        }
    )

    assert agent._playback_buffer.buffered_frames() == before
    assert "resp-stale" not in agent._response_id_to_req_id
    assert turn.audio_started is False


def test_completed_response_without_audio_retries_once_and_deletes_silent_item():
    agent = _make_agent()
    turn = _make_turn("rt-silent-retry")
    turn.response_id = "resp-silent-1"
    agent._turns_by_req_id[turn.req_id] = turn
    agent._bind_response_id(turn, turn.response_id)
    followups = []
    agent._send_response_create = lambda queued_turn: followups.append(queued_turn.req_id)

    agent._handle_response_done(
        {
            "type": "response.done",
            "response": {
                "id": "resp-silent-1",
                "status": "completed",
                "output": [
                    {
                        "id": "asst-silent-1",
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hello"}],
                    }
                ],
            },
        }
    )

    assert turn.no_audio_retry_count == 1
    assert turn.response_finished.is_set() is False
    assert turn.playback_finished.is_set() is False
    assert followups == [turn.req_id]
    assert turn.response_id == ""
    assert turn.assistant_transcript == ""
    assert "resp-silent-1" not in agent._response_id_to_req_id
    delete_events = [evt for evt in agent._sent_events if evt["type"] == "conversation.item.delete"]
    assert delete_events == [{"type": "conversation.item.delete", "item_id": "asst-silent-1"}]


def test_completed_response_without_audio_after_retry_is_canceled_without_fake_playback_stop():
    agent = _make_agent()
    turn = _make_turn("rt-silent-fail")
    turn.response_id = "resp-silent-2"
    turn.no_audio_retry_count = realtime_mod.NO_AUDIO_RESPONSE_RETRY_LIMIT
    agent._turns_by_req_id[turn.req_id] = turn
    agent._bind_response_id(turn, turn.response_id)

    agent._handle_response_done(
        {
            "type": "response.done",
            "response": {
                "id": "resp-silent-2",
                "status": "completed",
                "output": [
                    {
                        "id": "asst-silent-2",
                        "type": "message",
                        "content": [{"type": "output_text", "text": "still silent"}],
                    }
                ],
            },
        }
    )

    assert turn.phase == realtime_mod.TURN_PHASE_CANCELED
    assert turn.finalized_reason == "response_completed_without_audio"
    assert turn.response_finished.is_set()
    assert turn.playback_finished.is_set()
    assert agent.engagement.done == [(turn.req_id, False)]
    assert agent.ros2_connector.messages == []


def test_response_done_logs_cache_usage_details():
    agent = _make_agent()
    turn = _make_turn("rt-usage")
    turn.response_id = "resp-usage"
    turn.audio_started = True
    turn.audio_started_at = time.time()
    agent._session_id = "sess-usage"
    agent._turns_by_req_id[turn.req_id] = turn
    agent._bind_response_id(turn, turn.response_id)

    agent._handle_response_done(
        {
            "type": "response.done",
            "response": {
                "id": "resp-usage",
                "status": "completed",
                "usage": {
                    "input_tokens": 200,
                    "output_tokens": 40,
                    "total_tokens": 240,
                    "input_token_details": {
                        "text_tokens": 190,
                        "audio_tokens": 10,
                        "image_tokens": 0,
                        "cached_tokens": 120,
                        "cached_tokens_details": {
                            "audio_tokens": 10,
                            "text_tokens": 110,
                            "image_tokens": 0,
                        },
                    },
                    "output_token_details": {
                        "text_tokens": 8,
                        "audio_tokens": 32,
                    },
                },
                "output": [
                    {
                        "id": "asst-usage",
                        "type": "message",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
            },
        }
    )

    usage_event = next(
        evt for evt in agent._latency.events if evt.get("event") == "response_usage"
    )
    assert usage_event["req_id"] == turn.req_id
    assert usage_event["session_id"] == "sess-usage"
    assert usage_event["input_tokens"] == 200
    assert usage_event["cached_tokens"] == 120
    assert usage_event["uncached_input_tokens"] == 80
    assert usage_event["cache_hit_ratio"] == 0.6
    assert usage_event["cached_audio_tokens"] == 10
    assert usage_event["cached_text_tokens"] == 110
    assert usage_event["input_text_tokens"] == 190
    assert usage_event["input_audio_tokens"] == 10
    assert usage_event["output_text_tokens"] == 8
    assert usage_event["output_audio_tokens"] == 32
    assert usage_event["estimated_cost_usd"] == 0.002544
    assert usage_event["estimated_cached_savings_usd"] == 0.000712
    assert usage_event["session_total_cost_usd"] == 0.002544


def test_incomplete_response_with_audio_finishes_playback_instead_of_canceling():
    agent = _make_agent()
    turn = _make_turn("rt-incomplete-audio")
    turn.response_id = "resp-incomplete"
    turn.audio_started = True
    turn.audio_started_at = time.time()
    agent._turns_by_req_id[turn.req_id] = turn
    agent._bind_response_id(turn, turn.response_id)

    agent._handle_response_done(
        {
            "type": "response.done",
            "response": {
                "id": "resp-incomplete",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [
                    {
                        "id": "asst-incomplete",
                        "type": "message",
                        "content": [{"type": "output_text", "text": "complete reply."}],
                    }
                ],
            },
        }
    )

    assert turn.phase != realtime_mod.TURN_PHASE_CANCELED
    assert turn.response_finished.is_set()
    assert turn.playback_finished.wait(timeout=0.2)
    assert agent.engagement.done == [(turn.req_id, True)]


def test_truncated_incomplete_audio_reply_requests_one_continuation():
    agent = _make_agent()
    turn = _make_turn("rt-incomplete-continue")
    turn.response_id = "resp-incomplete-continue"
    turn.audio_started = True
    turn.audio_started_at = time.time()
    agent._turns_by_req_id[turn.req_id] = turn
    agent._bind_response_id(turn, turn.response_id)
    followups = []
    agent._send_response_create = lambda queued_turn: followups.append(queued_turn.req_id)

    agent._handle_response_done(
        {
            "type": "response.done",
            "response": {
                "id": "resp-incomplete-continue",
                "status": "incomplete",
                "output": [
                    {
                        "id": "asst-incomplete-continue",
                        "type": "message",
                        "content": [{"type": "output_text", "text": "cut off mid sentence"}],
                    }
                ],
            },
        }
    )

    assert turn.incomplete_audio_continuation_count == 1
    assert turn.response_id == ""
    assert turn.response_finished.is_set() is False
    assert followups == [turn.req_id]


def test_completed_output_item_arms_playback_completion_before_response_done():
    agent = _make_agent()
    turn = _make_turn("rt-item-done")
    turn.response_id = "resp-item-done"
    turn.audio_started = True
    turn.audio_started_at = time.time()
    agent._turns_by_req_id[turn.req_id] = turn
    agent._bind_response_id(turn, turn.response_id)

    agent._handle_output_item_done(
        {
            "type": "response.output_item.done",
            "response_id": "resp-item-done",
            "item": {
                "id": "asst-item-done",
                "type": "message",
                "role": "assistant",
                "status": "completed",
            },
        }
    )

    assert turn.playback_completion_armed is True
    assert agent.engagement.done == [(turn.req_id, True)]
    assert turn.playback_finished.wait(timeout=0.1) is False
    assert not any(event[0] == "playback_completed" for event in agent.engagement.playback_events)

    agent._handle_response_done(
        {
            "type": "response.done",
            "response": {
                "id": "resp-item-done",
                "status": "completed",
                "output": [
                    {
                        "id": "asst-item-done",
                        "type": "message",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
            },
        }
    )

    assert agent.engagement.done == [(turn.req_id, True)]
    assert turn.response_finished.is_set()
    assert turn.playback_finished.wait(timeout=0.2)
    assert agent.engagement.playback_events[-1][0] == "playback_completed"


def test_terminated_precreated_turn_stays_as_stale_response_queue_head():
    agent = _make_agent()
    old_turn = _make_turn("rt-old-pending")
    new_turn = _make_turn("rt-new-pending")
    new_turn.user_item_id = "new-user-item"
    new_turn.user_item_id = "new-user-item"
    agent._turns_by_req_id[old_turn.req_id] = old_turn
    agent._turns_by_req_id[new_turn.req_id] = new_turn
    agent._pending_response_turn_req_ids.extend([old_turn.req_id, new_turn.req_id])

    agent._terminate_turn(
        old_turn,
        realtime_mod.TURN_PHASE_CANCELED,
        "test_cancel",
        send_cancel=False,
    )

    assert list(agent._pending_response_turn_req_ids) == [
        old_turn.req_id,
        new_turn.req_id,
    ]

    stale = agent._consume_pending_response_turn("resp-stale-old")
    assert stale is old_turn
    assert old_turn.response_id == "resp-stale-old"

    resolved = agent._consume_pending_response_turn("resp-new")
    assert resolved is new_turn
    assert new_turn.response_id == "resp-new"
    assert agent._response_id_to_req_id["resp-new"] == new_turn.req_id


def test_expired_stale_pending_response_is_not_bound_to_later_turn(monkeypatch):
    agent = _make_agent()
    old_turn = _make_turn("rt-old-pending")
    new_turn = _make_turn("rt-new-pending")
    new_turn.user_item_id = "new-user-item"
    agent._turns_by_req_id[old_turn.req_id] = old_turn
    agent._turns_by_req_id[new_turn.req_id] = new_turn
    agent._pending_response_turn_req_ids.extend([old_turn.req_id, new_turn.req_id])

    now = 100.0
    monkeypatch.setattr(realtime_mod.time, "time", lambda: now)
    agent._terminate_turn(
        old_turn,
        realtime_mod.TURN_PHASE_CANCELED,
        "test_cancel_before_response_created",
        send_cancel=False,
    )
    now += realtime_mod.RESPONSE_STALL_TIMEOUT_SEC + 1.0

    agent._handle_response_created(
        {
            "type": "response.created",
            "response": {"id": "resp-ambiguous-late"},
        }
    )

    assert old_turn.response_id == "resp-ambiguous-late"
    assert new_turn.response_id == ""
    assert agent._response_id_to_req_id["resp-ambiguous-late"] == old_turn.req_id
    assert list(agent._pending_response_turn_req_ids) == [new_turn.req_id]
    assert agent._sent_events[-2] == {
        "type": "response.cancel",
        "response_id": "resp-ambiguous-late",
    }
    assert agent._sent_events[-1]["type"] == "response.create"


def test_expired_stale_response_reissues_pending_live_turn(monkeypatch):
    agent = _make_agent()
    old_turn = _make_turn("rt-old-pending")
    new_turn = _make_turn("rt-new-pending")
    new_turn.user_item_id = "new-user-item"
    agent._turns_by_req_id[old_turn.req_id] = old_turn
    agent._turns_by_req_id[new_turn.req_id] = new_turn
    agent._pending_response_turn_req_ids.append(old_turn.req_id)

    now = 100.0
    monkeypatch.setattr(realtime_mod.time, "time", lambda: now)
    agent._terminate_turn(
        old_turn,
        realtime_mod.TURN_PHASE_CANCELED,
        "test_cancel_before_response_created",
        send_cancel=False,
    )
    now += realtime_mod.RESPONSE_STALL_TIMEOUT_SEC + 1.0

    agent._send_response_create(new_turn)
    assert new_turn.pending_response_requests == 1
    assert list(agent._pending_response_turn_req_ids) == [new_turn.req_id]
    assert list(agent._expired_stale_response_turn_req_ids) == [old_turn.req_id]

    agent._handle_response_created(
        {
            "type": "response.created",
            "response": {"id": "resp-ambiguous-late"},
        }
    )

    assert old_turn.response_id == "resp-ambiguous-late"
    assert new_turn.response_id == ""
    assert new_turn.pending_response_requests == 1
    assert list(agent._pending_response_turn_req_ids) == [new_turn.req_id]
    assert agent._sent_events[-3]["type"] == "response.create"
    assert agent._sent_events[-2] == {
        "type": "response.cancel",
        "response_id": "resp-ambiguous-late",
    }
    assert agent._sent_events[-1]["type"] == "response.create"


def test_stale_response_created_after_pending_cancel_is_not_bound_to_next_turn():
    agent = _make_agent()
    old_turn = _make_turn("rt-old-pending")
    new_turn = _make_turn("rt-new-pending")
    agent._turns_by_req_id[old_turn.req_id] = old_turn
    agent._turns_by_req_id[new_turn.req_id] = new_turn
    agent._pending_response_turn_req_ids.extend([old_turn.req_id, new_turn.req_id])

    agent._terminate_turn(
        old_turn,
        realtime_mod.TURN_PHASE_CANCELED,
        "test_cancel_before_response_created",
        send_cancel=False,
    )

    agent._handle_response_created(
        {
            "type": "response.created",
            "response": {"id": "resp-stale-old"},
        }
    )

    assert new_turn.response_id == ""
    assert agent._response_id_to_req_id["resp-stale-old"] == old_turn.req_id
    assert list(agent._pending_response_turn_req_ids) == [new_turn.req_id]
    assert agent._sent_events[-1] == {
        "type": "response.cancel",
        "response_id": "resp-stale-old",
    }

    agent._handle_response_created(
        {
            "type": "response.created",
            "response": {"id": "resp-new"},
        }
    )

    assert new_turn.response_id == "resp-new"
    assert agent._response_id_to_req_id["resp-new"] == new_turn.req_id


def test_canceled_tool_followup_response_stays_as_stale_queue_head():
    agent = _make_agent()
    old_turn = _make_turn("rt-tool-followup")
    old_turn.response_id = "resp-initial-tool-call"
    old_turn.pending_response_requests = 1
    new_turn = _make_turn("rt-new-after-tool")
    new_turn.user_item_id = "new-user-item"
    new_turn.user_item_id = "new-user-item"
    agent._turns_by_req_id[old_turn.req_id] = old_turn
    agent._turns_by_req_id[new_turn.req_id] = new_turn
    agent._pending_response_turn_req_ids.extend([old_turn.req_id, new_turn.req_id])

    agent._terminate_turn(
        old_turn,
        realtime_mod.TURN_PHASE_CANCELED,
        "test_cancel_tool_followup_before_response_created",
        send_cancel=False,
    )

    assert list(agent._pending_response_turn_req_ids) == [
        old_turn.req_id,
        new_turn.req_id,
    ]

    agent._handle_response_created(
        {
            "type": "response.created",
            "response": {"id": "resp-stale-followup"},
        }
    )

    assert old_turn.response_id == "resp-stale-followup"
    assert new_turn.response_id == ""
    assert agent._response_id_to_req_id["resp-stale-followup"] == old_turn.req_id
    assert list(agent._pending_response_turn_req_ids) == [new_turn.req_id]
    assert agent._sent_events[-1] == {
        "type": "response.cancel",
        "response_id": "resp-stale-followup",
    }


def test_expired_stale_tool_followup_reissues_pending_live_turn(monkeypatch):
    agent = _make_agent()
    old_turn = _make_turn("rt-tool-followup")
    old_turn.response_id = "resp-initial-tool-call"
    old_turn.pending_response_requests = 1
    new_turn = _make_turn("rt-new-after-tool")
    new_turn.user_item_id = "new-user-item"
    agent._turns_by_req_id[old_turn.req_id] = old_turn
    agent._turns_by_req_id[new_turn.req_id] = new_turn
    agent._pending_response_turn_req_ids.append(old_turn.req_id)

    now = 200.0
    monkeypatch.setattr(realtime_mod.time, "time", lambda: now)
    agent._terminate_turn(
        old_turn,
        realtime_mod.TURN_PHASE_CANCELED,
        "test_cancel_tool_followup_before_response_created",
        send_cancel=False,
    )
    now += realtime_mod.RESPONSE_STALL_TIMEOUT_SEC + 1.0

    agent._send_response_create(new_turn)
    agent._handle_response_created(
        {
            "type": "response.created",
            "response": {"id": "resp-ambiguous-followup"},
        }
    )

    assert old_turn.response_id == "resp-ambiguous-followup"
    assert new_turn.response_id == ""
    assert new_turn.pending_response_requests == 1
    assert list(agent._pending_response_turn_req_ids) == [new_turn.req_id]
    assert agent._sent_events[-2] == {
        "type": "response.cancel",
        "response_id": "resp-ambiguous-followup",
    }
    assert agent._sent_events[-1]["type"] == "response.create"


def test_response_request_omits_output_budget_when_uncapped():
    agent = _make_agent()
    turn = _make_turn("rt-request")

    request = agent._build_response_request(turn)

    assert "max_output_tokens" not in request
    assert request["output_modalities"] == ["audio"]
    assert "modalities" not in request


def test_response_request_starts_with_base_prompt_before_dynamic_context():
    agent = _make_agent()
    agent.base_system_prompt = "STATIC RULES"
    agent._current_office_location = "BOS3"
    turn = _make_turn("rt-request-order")

    request = agent._build_response_request(turn)

    instructions = request["instructions"]
    assert instructions.startswith("STATIC RULES\n\n")
    assert "[CURRENT TIME]" in instructions
    assert instructions.index("STATIC RULES") < instructions.index("[CURRENT TIME]")


def test_conversation_item_created_records_history_snapshot():
    agent = _make_agent()
    turn = _make_turn("rt-history-snapshot")
    agent._turns_by_req_id[turn.req_id] = turn
    agent._queue_pending_local_created_item(turn.req_id, "message", "system")

    agent._handle_conversation_item_created(
        {
            "type": "conversation.item.created",
            "item": {
                "id": "sys-item-1",
                "type": "message",
                "role": "system",
                "status": "completed",
                "content": [
                    {
                        "type": "input_text",
                        "text": "[INTERNAL EVENT]\nBattery is low.",
                    }
                ],
            },
        }
    )

    assert list(agent._history_item_order) == ["sys-item-1"]
    assert turn.history_item_ids == {"sys-item-1"}
    assert agent._history_item_snapshots["sys-item-1"] == {
        "type": "message",
        "role": "system",
        "status": "completed",
        "text": "[INTERNAL EVENT]\nBattery is low.",
    }


def test_conversation_item_deleted_forgets_cleanup_item():
    agent = _make_agent()
    turn = _make_turn("rt-history-delete")
    agent._turns_by_req_id[turn.req_id] = turn
    agent._register_turn_history_item(turn, "old-user-item")
    agent._history_item_snapshots["old-user-item"] = {
        "type": "message",
        "role": "user",
        "text": "hello",
    }
    agent._handle_server_event(
        {
            "type": "conversation.item.deleted",
            "item_id": "old-user-item",
        }
    )

    assert list(agent._history_item_order) == []
    assert "old-user-item" not in agent._known_history_item_ids
    assert "old-user-item" not in turn.history_item_ids
    assert "old-user-item" not in agent._history_item_snapshots


def test_response_create_logs_model_prompt_snapshot():
    agent = _make_agent()
    agent.base_system_prompt = "STATIC RULES"
    agent._current_office_location = "BOS3"
    agent._history_item_snapshots["prior-user-item"] = {
        "type": "message",
        "role": "user",
        "status": "transcribed",
        "text": "Where are you right now?",
    }
    agent._history_item_snapshots["current-user-item"] = {
        "type": "message",
        "role": "user",
        "text": "[audio input]",
    }
    turn = _make_turn("rt-prompt-log")
    prior_turn = _make_turn("rt-prior")
    prior_turn.inference_scope_id = "owner:person-1"
    prior_turn.inference_owner_key = "owner:person-1"
    prior_turn.phase = realtime_mod.TURN_PHASE_FINALIZED
    prior_turn.finalized = True
    agent._turns_by_req_id[prior_turn.req_id] = prior_turn
    agent._register_turn_history_item(
        prior_turn,
        "prior-user-item",
        item_type="message",
        role="user",
        status="done",
        permitted_for_inference=True,
    )
    turn.inference_scope_id = "owner:person-1"
    turn.inference_owner_key = "owner:person-1"
    turn.user_item_id = "current-user-item"
    agent._turns_by_req_id[turn.req_id] = turn
    agent._register_turn_history_item(
        turn,
        "current-user-item",
        item_type="message",
        role="user",
        status="done",
        permitted_for_inference=True,
    )

    agent._send_response_create(turn)

    sent_request = agent._sent_events[-1]["response"]
    response_log = next(
        event for event in agent._latency.events if event.get("event") == "response_create"
    )
    logged_prompt = base64.b64decode(response_log["model_prompt_b64"]).decode("utf-8")
    logged_dynamic = base64.b64decode(
        response_log["model_dynamic_context_b64"]
    ).decode("utf-8")
    logged_history = base64.b64decode(
        response_log["model_history_snapshot_b64"]
    ).decode("utf-8")
    assert logged_prompt == sent_request["instructions"]
    assert logged_prompt.startswith("STATIC RULES\n\n")
    assert "[CURRENT OFFICE LOCATION] BOS3" in logged_dynamic
    assert "1. user type=message item_id=prior-user-item status=transcribed" in logged_history
    assert "Where are you right now?" in logged_history
    assert "2. user type=message item_id=current-user-item" in logged_history
    assert "[audio input]" in logged_history
    assert sent_request["conversation"] == "auto"
    assert sent_request["input"] == [
        {"type": "item_reference", "id": "prior-user-item"},
        {"type": "item_reference", "id": "current-user-item"},
    ]
    assert "model_dynamic_context_b64" in response_log["_console_omit_fields"]
    assert "model_history_snapshot_b64" in response_log["_console_omit_fields"]
    assert "model_prompt_b64" in response_log["_console_omit_fields"]
    assert response_log["model_prompt_chars"] == len(sent_request["instructions"])
    assert response_log["model_static_prompt_chars"] == len("STATIC RULES")
    assert response_log["model_dynamic_context_chars"] == len(logged_dynamic)
    assert response_log["model_history_snapshot_chars"] == len(logged_history)
    assert response_log["model_inference_owner_key"] == "owner:person-1"
    assert response_log["model_inference_scope_id"] == "owner:person-1"
    assert response_log["model_selected_history_item_count"] == 1
    assert response_log["model_selected_current_turn_item_count"] == 1
    assert response_log["model_selected_item_ids"] == "prior-user-item,current-user-item"
    assert response_log["model_selected_current_turn_item_ids"] == "current-user-item"


def test_response_input_caps_prior_scope_history_but_keeps_current_turn_items():
    agent = _make_agent()
    original_limit = realtime_mod.INFERENCE_HISTORY_MAX_ITEMS
    realtime_mod.INFERENCE_HISTORY_MAX_ITEMS = 2
    try:
        for index in range(3):
            prior_turn = _make_turn(f"rt-prior-{index}")
            prior_turn.inference_scope_id = "owner:person-1"
            prior_turn.inference_owner_key = "owner:person-1"
            prior_turn.phase = realtime_mod.TURN_PHASE_FINALIZED
            prior_turn.finalized = True
            agent._turns_by_req_id[prior_turn.req_id] = prior_turn
            agent._register_turn_history_item(
                prior_turn,
                f"prior-{index}",
                item_type="message",
                role="user",
                status="done",
                permitted_for_inference=True,
            )
        turn = _make_turn("rt-current")
        turn.inference_scope_id = "owner:person-1"
        turn.inference_owner_key = "owner:person-1"
        turn.user_item_id = "current-user"
        agent._turns_by_req_id[turn.req_id] = turn
        agent._register_turn_history_item(
            turn,
            "current-user",
            item_type="message",
            role="user",
            status="done",
            permitted_for_inference=True,
        )

        inputs = agent._response_input_items_for_turn(turn)
    finally:
        realtime_mod.INFERENCE_HISTORY_MAX_ITEMS = original_limit

    assert inputs == [
        {"type": "item_reference", "id": "prior-1"},
        {"type": "item_reference", "id": "prior-2"},
        {"type": "item_reference", "id": "current-user"},
    ]
    assert turn.selected_inference_history_item_ids == ["prior-1", "prior-2"]
    assert turn.selected_inference_current_item_ids == ["current-user"]


def test_tool_followup_response_input_includes_current_call_chain_in_order():
    agent = _make_agent()
    turn = _make_turn("rt-tool-chain")
    turn.inference_scope_id = "owner:person-1"
    turn.inference_owner_key = "owner:person-1"
    turn.user_item_id = "current-user"
    agent._turns_by_req_id[turn.req_id] = turn
    agent._register_turn_history_item(
        turn,
        "current-user",
        item_type="message",
        role="user",
        status="done",
        permitted_for_inference=True,
    )
    agent._register_turn_history_item(
        turn,
        "call-item",
        item_type="function_call",
        status="done",
        permitted_for_inference=True,
        input_item={
            "id": "call-item",
            "type": "function_call",
            "name": "fake_tool",
            "call_id": "call-1",
            "arguments": "{}",
            "status": "completed",
        },
    )
    turn.function_call_item_ids.add("call-item")
    agent._register_turn_history_item(
        turn,
        "tool-output",
        item_type="function_call_output",
        status="done",
        permitted_for_inference=True,
        input_item={
            "id": "tool-output",
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "{\"success\": true}",
            "status": "completed",
        },
    )

    assert agent._response_input_items_for_turn(turn) == [
        {"type": "item_reference", "id": "current-user"},
        {
            "id": "call-item",
            "type": "function_call",
            "name": "fake_tool",
            "call_id": "call-1",
            "arguments": "{}",
            "status": "completed",
        },
        {
            "id": "tool-output",
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "{\"success\": true}",
            "status": "completed",
        },
    ]


def test_playback_watchdog_does_not_fire_while_playback_progress_is_recent():
    agent = _make_agent()
    turn = _make_turn("rt-playback-progress")
    turn.phase = realtime_mod.TURN_PHASE_PLAYING
    turn.audio_started = True
    turn.audio_started_at = time.time() - 10.0
    turn.last_playback_progress_at = time.time()
    turn.response_finished.set()
    agent._turns_by_req_id[turn.req_id] = turn

    watchdog = threading.Thread(target=agent._watchdog_loop, daemon=True)
    watchdog.start()
    time.sleep(0.35)
    agent._stop_event.set()
    watchdog.join(timeout=1.0)

    assert turn.finalized is False
    assert turn.playback_finished.is_set() is False


def test_playback_watchdog_forces_completion_instead_of_canceling():
    agent = _make_agent()
    turn = _make_turn("rt-playback-stall")
    turn.phase = realtime_mod.TURN_PHASE_PLAYING
    turn.audio_started = True
    turn.response_id = "resp-stall"
    turn.audio_started_at = time.time() - 10.0
    turn.last_playback_progress_at = time.time() - 10.0
    turn.response_finished.set()
    agent._turns_by_req_id[turn.req_id] = turn
    agent._playback_req_id = turn.req_id
    agent._playback_stream_id = turn.response_id

    old_timeout = realtime_mod.PLAYBACK_STALL_TIMEOUT_SEC
    try:
        realtime_mod.PLAYBACK_STALL_TIMEOUT_SEC = 0.1
        watchdog = threading.Thread(target=agent._watchdog_loop, daemon=True)
        watchdog.start()
        time.sleep(0.35)
        agent._stop_event.set()
        watchdog.join(timeout=1.0)
    finally:
        realtime_mod.PLAYBACK_STALL_TIMEOUT_SEC = old_timeout

    assert turn.phase == realtime_mod.TURN_PHASE_PLAYING
    assert turn.finalized is False
    assert turn.playback_finished.is_set()
    assert agent.engagement.playback_events[-1][0] == "playback_stopped"


def test_interrupt_current_response_truncates_played_audio():
    agent = _make_agent()
    turn = _make_turn("rt-interrupt")
    turn.response_id = "resp-interrupt"
    turn.assistant_item_id = "asst-interrupt"
    turn.audio_started = True
    agent._turns_by_req_id[turn.req_id] = turn
    agent._active_turn = turn
    agent._playback_req_id = turn.req_id
    agent._playback_item_id = turn.assistant_item_id
    agent._played_output_frames = 2400

    agent.interrupt_current_response(reason="voice_command")

    sent_types = [evt["type"] for evt in agent._sent_events]
    assert "response.cancel" in sent_types
    assert "conversation.item.truncate" in sent_types
    truncate_event = next(evt for evt in agent._sent_events if evt["type"] == "conversation.item.truncate")
    assert truncate_event["item_id"] == "asst-interrupt"
    assert truncate_event["audio_end_ms"] > 0


def test_self_published_stop_voice_command_is_ignored():
    agent = _make_agent()
    turn = _make_turn("rt-voice-self")
    agent._turns_by_req_id[turn.req_id] = turn
    agent._active_turn = turn

    agent.note_local_voice_command("stop")
    agent._on_voice_command(SimpleNamespace(data="stop"))

    assert turn.finalized is False
    assert agent._sent_events == []


def test_external_stop_voice_command_interrupts_active_turn():
    agent = _make_agent()
    turn = _make_turn("rt-voice-external")
    turn.response_id = "resp-external"
    turn.assistant_item_id = "asst-external"
    turn.audio_started = True
    agent._turns_by_req_id[turn.req_id] = turn
    agent._active_turn = turn
    agent._playback_req_id = turn.req_id
    agent._playback_item_id = turn.assistant_item_id
    agent._played_output_frames = 1200

    agent._on_voice_command(SimpleNamespace(data="stop"))

    assert turn.phase == realtime_mod.TURN_PHASE_CANCELED
    sent_types = [evt["type"] for evt in agent._sent_events]
    assert "response.cancel" in sent_types
    assert "conversation.item.truncate" in sent_types


def test_configure_session_uses_realtime_ga_shape():
    agent = _make_agent()

    agent._configure_session()

    update_event = next(evt for evt in agent._sent_events if evt["type"] == "session.update")
    session = update_event["session"]
    assert session["type"] == "realtime"
    assert session["model"] == "gpt-realtime-1.5"
    assert session["output_modalities"] == ["audio"]
    assert session["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
    }
    assert session["audio"]["input"]["turn_detection"] is None
    assert session["audio"]["input"]["noise_reduction"] == {"type": "near_field"}
    assert session["audio"]["input"]["transcription"] == {
        "model": "gpt-4o-mini-transcribe",
        "language": "en",
    }
    assert session["audio"]["output"]["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
    }
    assert session["audio"]["output"]["voice"] == "cedar"
    assert session["audio"]["output"]["speed"] == 0.9
    assert "input_audio_format" not in session
    assert "modalities" not in session
    assert "temperature" not in session


def test_preference_turn_uses_captured_speaker_and_transcripts():
    agent = _make_agent()
    seen = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment: seen.append(segment)
    )
    turn = _make_turn("rt-pref", audio_speaker_id="person-9")
    turn.user_transcript = "my dog is luna"
    turn.assistant_transcript = "luna sounds adorable"

    agent._maybe_note_preference_turn(turn)
    agent.flush_preference_segments(reason="speaker_change")

    assert len(seen) == 1
    assert seen[0].person_id == "person-9"
    assert seen[0].turns[0].user_text == "my dog is luna"
    assert seen[0].turns[0].assistant_text == "luna sounds adorable"


def test_preference_turn_is_claimed_once_when_called_concurrently():
    agent = _make_agent()
    seen = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment: seen.append(segment)
    )
    turn = _make_turn("rt-pref-race", audio_speaker_id="person-9")
    turn.user_transcript = "my dog is luna"
    turn.assistant_transcript = "luna sounds adorable"
    barrier = threading.Barrier(2)

    def note_turn():
        barrier.wait(timeout=1.0)
        agent._maybe_note_preference_turn(turn)

    threads = [threading.Thread(target=note_turn) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    agent.flush_preference_segments(reason="speaker_change")

    assert len(seen) == 1
    assert [item.turn_id for item in seen[0].turns] == ["rt-pref-race"]


def test_idle_preference_flush_is_debounced():
    agent = _make_agent()
    seen = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment: seen.append(segment)
    )
    turn = _make_turn("rt-pref-idle", audio_speaker_id="person-9")
    turn.user_transcript = "my dog is luna"
    turn.assistant_transcript = "luna sounds adorable"

    agent._maybe_note_preference_turn(turn)
    agent.flush_preference_segments(reason="idle")

    assert seen == []
    time.sleep(0.08)

    assert len(seen) == 1
    assert seen[0].person_id == "person-9"


def test_same_speaker_resume_cancels_idle_preference_flush():
    agent = _make_agent()
    seen = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment: seen.append(segment)
    )
    first_turn = _make_turn("rt-pref-resume-1", audio_speaker_id="person-9")
    first_turn.user_transcript = "my dog is luna"
    first_turn.assistant_transcript = "luna sounds adorable"

    agent._maybe_note_preference_turn(first_turn)
    agent.flush_preference_segments(reason="idle")
    time.sleep(0.01)
    agent._start_recording_locked(now_s=10.0)
    time.sleep(0.08)

    assert seen == []

    second_turn = _make_turn("rt-pref-resume-2", audio_speaker_id="person-9")
    second_turn.user_transcript = "she likes fetch"
    second_turn.assistant_transcript = "fetch is a great game"
    agent._maybe_note_preference_turn(second_turn)
    agent.flush_preference_segments(reason="shutdown")

    assert len(seen) == 1
    assert [item.turn_id for item in seen[0].turns] == [
        "rt-pref-resume-1",
        "rt-pref-resume-2",
    ]


def test_shutdown_preference_flush_runs_extraction_synchronously():
    agent = _make_agent()
    executor = _RecordingExecutor()
    agent._preference_executor = executor
    seen = []
    reasons = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment, reason="": (
            seen.append(segment),
            reasons.append(reason),
        )
    )
    turn = _make_turn("rt-pref-shutdown", audio_speaker_id="person-9")
    turn.user_transcript = "I am reading a new book"
    turn.assistant_transcript = "that sounds fun"

    agent._maybe_note_preference_turn(turn)
    agent.flush_preference_segments(reason="shutdown")

    assert executor.submitted == []
    assert len(seen) == 1
    assert seen[0].person_id == "person-9"
    assert [item.turn_id for item in seen[0].turns] == ["rt-pref-shutdown"]
    assert reasons == ["shutdown"]


def test_unattributed_turn_flushes_active_preference_segment():
    agent = _make_agent()
    seen = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment: seen.append(segment)
    )
    first_turn = _make_turn("rt-pref-known", audio_speaker_id="person-9")
    first_turn.user_transcript = "my dog is luna"
    first_turn.assistant_transcript = "luna sounds adorable"
    agent._maybe_note_preference_turn(first_turn)

    second_turn = _make_turn("rt-pref-unknown", audio_speaker_id=None)
    second_turn.user_transcript = "hello again"
    second_turn.assistant_transcript = "hi there"
    agent._maybe_note_preference_turn(second_turn)

    assert len(seen) == 1
    assert seen[0].person_id == "person-9"


def test_missing_owner_turn_flushes_previous_preference_segment_without_claiming_turn():
    agent = _make_agent()
    log_messages = []
    agent.logger = SimpleNamespace(
        info=lambda message, *args: log_messages.append(message % args),
        debug=lambda *_args, **_kwargs: None,
        exception=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )
    seen = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment: seen.append(segment)
    )
    first_turn = _make_turn("rt-pref-known-before-unknown", audio_speaker_id="person-9")
    first_turn.user_transcript = "my dog is luna"
    first_turn.assistant_transcript = "luna sounds adorable"
    agent._maybe_note_preference_turn(first_turn)

    unknown_turn = _make_turn(
        "rt-pref-owner-missing",
        owner_id=None,
        audio_speaker_id=None,
        context_snapshot=realtime_mod.FrozenTurnContext(
            primary_face_person_id="person-1",
            owner_id=None,
            audio_speaker_id=None,
        ),
    )
    unknown_turn.user_transcript = "hello again"
    unknown_turn.assistant_transcript = "hi there"
    agent._maybe_note_preference_turn(unknown_turn)

    assert len(seen) == 1
    assert seen[0].person_id == "person-9"
    assert unknown_turn.preference_noted is False
    assert any(
        "Preference extraction skipped unattributed turn" in message
        for message in log_messages
    )
    assert all("Preference extraction waiting" not in message for message in log_messages)


def test_active_recording_survives_admission_closing_mid_capture():
    import numpy as np

    agent = _make_agent()
    agent.realtime_profile.input_sample_rate = 16000
    agent.realtime_profile.wake_window_sec = 5.0
    agent.realtime_profile.admission = SimpleNamespace(
        block_during_speaking=True,
        block_during_engaged=False,
        open_on_face_presence=True,
        open_on_interaction_states=("alert", "cooldown"),
        open_on_wake_window=True,
    )
    agent._session_ready = threading.Event()
    agent._session_ready.set()
    agent._resample_state = None
    agent._wake_window_until = 0.0
    agent._recording_active = False
    agent._recording_started_at = 0.0
    agent._last_voice_at = 0.0
    agent._current_turn_vad_positive_blocks = 0
    agent._face_gate = SimpleNamespace(is_face_present=lambda: False)
    agent._vad = lambda *_args, **_kwargs: (True, {})
    agent._wake_word = lambda *_args, **_kwargs: (False, {})

    states = deque(["cooldown", "cooldown", "idle"])

    class _Engagement:
        def snapshot(self):
            return SimpleNamespace(
                state=states.popleft() if states else "idle",
                req_id="",
                entered_at=0.0,
                expires_at=None,
                nav_active=False,
                nav_source="",
                nav_interruptible=True,
                nav_passive_listen_allowed=True,
            )

    agent.engagement = _Engagement()
    chunk = np.zeros((1600, 1), dtype=np.int16)

    agent._capture_callback(chunk, 1600, None, None)

    assert agent._recording_active is True
    assert len(agent._current_turn_audio_chunks) == 1
    assert agent._current_turn_vad_positive_blocks == 1

    agent._capture_callback(chunk, 1600, None, None)
    agent._capture_callback(chunk, 1600, None, None)

    assert agent._recording_active is True
    assert len(agent._current_turn_audio_chunks) == 3
    assert agent._current_turn_vad_positive_blocks == 3


def test_stale_cooldown_admission_does_not_override_idle_display():
    import numpy as np

    agent = _make_agent()
    agent.realtime_profile.input_sample_rate = 16000
    agent.realtime_profile.wake_window_sec = 5.0
    agent.realtime_profile.admission = SimpleNamespace(
        block_during_speaking=True,
        block_during_engaged=False,
        open_on_face_presence=False,
        open_on_attention_presence=False,
        open_on_interaction_states=("alert", "cooldown"),
        open_on_wake_window=False,
    )
    agent._session_ready = threading.Event()
    agent._session_ready.set()
    agent._resample_state = None
    agent._wake_window_until = 0.0
    agent._recording_active = False
    agent._recording_started_at = 0.0
    agent._last_voice_at = 0.0
    agent._face_gate = SimpleNamespace(
        is_face_present=lambda: False,
        is_attention_present=lambda: False,
    )
    agent._vad = lambda *_args, **_kwargs: (False, {})
    agent._wake_word = lambda *_args, **_kwargs: (False, {})
    display_modes = []
    agent._set_display_mode_async = lambda mode: display_modes.append(mode)

    states = deque(["cooldown", "idle"])

    class _Engagement:
        @property
        def state_name(self):
            return "idle"

        def snapshot(self):
            return SimpleNamespace(
                state=states.popleft() if states else "idle",
                req_id="",
                entered_at=0.0,
                expires_at=None,
                nav_active=False,
                nav_source="",
                nav_interruptible=True,
                nav_passive_listen_allowed=True,
            )

    agent.engagement = _Engagement()
    chunk = np.zeros((1600, 1), dtype=np.int16)

    agent._capture_callback(chunk, 1600, None, None)

    assert display_modes == []


def test_closed_admission_clears_passive_alert_display():
    import numpy as np

    agent = _make_agent()
    agent.realtime_profile.input_sample_rate = 16000
    agent.realtime_profile.wake_window_sec = 5.0
    agent.realtime_profile.admission = SimpleNamespace(
        block_during_speaking=True,
        block_during_engaged=False,
        open_on_face_presence=False,
        open_on_attention_presence=True,
        open_on_interaction_states=("alert",),
        open_on_wake_window=False,
    )
    agent._session_ready = threading.Event()
    agent._session_ready.set()
    agent._resample_state = None
    agent._wake_window_until = 0.0
    agent._recording_active = False
    agent._recording_started_at = 0.0
    agent._last_voice_at = 0.0
    agent._face_gate = SimpleNamespace(
        is_face_present=lambda: False,
        is_attention_present=lambda: False,
    )
    agent._vad = lambda *_args, **_kwargs: (False, {})
    agent._wake_word = lambda *_args, **_kwargs: (False, {})
    agent.display_runtime = object()
    agent._display_queue = queue.Queue()
    agent._display_mode_lock = threading.Lock()
    agent._display_mode = "alert"

    chunk = np.zeros((1600, 1), dtype=np.int16)
    agent._capture_callback(chunk, 1600, None, None)

    assert agent._display_queue.get_nowait() == ("mode", "idle")
    assert agent._display_mode == "idle"


def test_wakeword_debug_uses_environment_without_callback_error(monkeypatch):
    agent = _make_agent()
    messages = []
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: messages.append((args, kwargs)),
        warning=lambda *args, **kwargs: None,
        exception=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )
    agent._last_wake_debug_log_s = 0.0
    agent._wake_word = SimpleNamespace(threshold=0.5)
    monkeypatch.setenv("ARGOS_WAKEWORD_DEBUG", "1")

    agent._log_wakeword_debug(
        wake_detected=False,
        wake_output={"open_wake_word": {"predictions": {"hey_puffle": 0.1}}},
    )

    assert messages


def test_wake_word_during_speaking_does_not_interrupt_response():
    import numpy as np

    agent = _make_agent()
    agent.realtime_profile.input_sample_rate = 16000
    agent._session_ready = threading.Event()
    agent._session_ready.set()
    agent._resample_state = None
    agent._input_suppressed_until_s = 0.0
    agent._playback_req_id = ""
    agent._recording_active = False
    agent._vad = lambda *_args, **_kwargs: (True, {})
    agent._wake_word = lambda *_args, **_kwargs: (True, {})
    interruptions = []
    agent.interrupt_current_response = lambda *, reason: interruptions.append(reason)

    class _SpeakingEngagement:
        def __init__(self):
            self.face_or_wake_calls = 0

        def snapshot(self):
            return SimpleNamespace(
                state="speaking",
                req_id="rt-speaking",
                entered_at=0.0,
                expires_at=None,
                nav_active=False,
                nav_source="",
                nav_interruptible=True,
                nav_passive_listen_allowed=True,
            )

        def on_face_or_wake(self):
            self.face_or_wake_calls += 1

    engagement = _SpeakingEngagement()
    agent.engagement = engagement
    chunk = np.zeros((1600, 1), dtype=np.int16)

    agent._capture_callback(chunk, 1600, None, None)

    assert interruptions == []
    assert engagement.face_or_wake_calls == 0
    assert agent._recording_active is False


def test_closed_admission_does_not_clear_thinking_display():
    agent = _make_agent()
    agent.display_runtime = object()
    agent._display_queue = queue.Queue()
    agent._display_mode_lock = threading.Lock()
    agent._display_mode = "thinking"

    agent._clear_passive_alert_display_if_needed()

    assert agent._display_queue.empty()
    assert agent._display_mode == "thinking"


def test_repeated_missing_owner_turn_flushes_preference_segment_only_once():
    agent = _make_agent()
    seen = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment: seen.append(segment)
    )
    known_turn = _make_turn("rt-pref-known-before-missing", audio_speaker_id="person-9")
    known_turn.user_transcript = "my dog is luna"
    known_turn.assistant_transcript = "luna sounds adorable"
    agent._maybe_note_preference_turn(known_turn)

    missing_owner_turn = _make_turn(
        "rt-pref-owner-still-missing",
        owner_id=None,
        audio_speaker_id=None,
        context_snapshot=realtime_mod.FrozenTurnContext(
            primary_face_person_id="person-1",
            owner_id=None,
            audio_speaker_id=None,
        ),
    )
    missing_owner_turn.user_transcript = "hello again"
    missing_owner_turn.assistant_transcript = "hi there"
    agent._maybe_note_preference_turn(missing_owner_turn)

    next_known_turn = _make_turn("rt-pref-next-known", audio_speaker_id="person-10")
    next_known_turn.user_transcript = "my cat is milo"
    next_known_turn.assistant_transcript = "milo sounds sweet"
    agent._maybe_note_preference_turn(next_known_turn)

    agent._maybe_note_preference_turn(missing_owner_turn)

    assert len(seen) == 1
    assert seen[0].person_id == "person-9"
    assert missing_owner_turn.preference_noted is False
    assert missing_owner_turn.preference_unattributed_flushed is True

    agent.flush_preference_segments(reason="shutdown")

    assert len(seen) == 2
    assert seen[1].person_id == "person-10"


def test_internal_text_turn_uses_system_role_message():
    agent = _make_agent()
    followups = []

    def _send_response_create(queued_turn):
        followups.append(queued_turn.req_id)
        queued_turn.response_finished.set()
        queued_turn.playback_finished.set()

    agent._send_response_create = _send_response_create
    turn = _make_turn(
        "evt-system",
        kind="text",
        input_text="NAV_EVENT: target reached",
        source_is_internal=True,
    )

    agent._run_turn(turn)

    assert not [
        evt for evt in agent._sent_events if evt["type"] == "conversation.item.create"
    ]
    item_id = next(iter(turn.history_item_ids))
    assert agent._history_item_snapshots[item_id]["role"] == "system"
    assert followups == [turn.req_id]


def test_internal_recognized_face_event_resolves_owner_context():
    agent = _make_agent()
    person = SimpleNamespace(
        person_id="person-9",
        name="Alex",
        interaction_count=1,
        confidence=0.9,
        bbox_area=100,
        timestamp=1.0,
        memory_profile_lines=(),
        preferred_language="",
        potential_followups=(),
        visible=True,
    )
    agent.face_service = _FakeFaceService(
        persons=[person],
        snapshot={"recognized_count": 1, "unknown_count": 0},
    )

    agent.enqueue_internal_event(
        "FACE_EVENT: recognized person 'Alex' appeared in front of you.",
        metadata={
            "internal": True,
            "internal_event": "face",
            "face_status": "recognized",
            "person_id": "person-9",
            "person_name": "Alex",
            "req_id": "evt-face-alex",
        },
    )

    turn = agent._turn_queue.get_nowait()

    assert turn.source_is_internal is True
    assert turn.owner_id == "person-9"
    assert turn.owner_source == "face"
    assert turn.context_snapshot.owner_id == "person-9"


def test_external_text_turn_resolves_single_visible_owner_for_live_chat_memory():
    agent = _make_agent()
    person = SimpleNamespace(
        person_id="person-9",
        name="Alex",
        interaction_count=1,
        confidence=0.9,
        bbox_area=100,
        timestamp=1.0,
        memory_profile_lines=(),
        preferred_language="",
        potential_followups=(),
    )
    agent.face_service = _FakeFaceService(
        persons=[person],
        snapshot={"recognized_count": 1, "unknown_count": 0},
    )
    seen = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment: seen.append(segment)
    )
    def _send_response_create(queued_turn):
        queued_turn.assistant_transcript = "nice"
        queued_turn.response_finished.set()
        queued_turn.playback_finished.set()

    agent._send_response_create = _send_response_create

    agent.enqueue_internal_event(
        "[PENDING EVENTS]\n- NAV_EVENT: reached desk\n[HUMAN INPUT]\nmy dog is luna",
        metadata={"req_id": "text-pref"},
    )
    turn = agent._turn_queue.get_nowait()
    agent._run_turn(turn)
    agent.flush_preference_segments(reason="shutdown")

    assert turn.owner_id == "person-9"
    assert turn.context_snapshot.owner_id == "person-9"
    assert len(seen) == 1
    assert seen[0].person_id == "person-9"
    assert seen[0].turns[0].user_text == "my dog is luna"


def test_external_text_turn_does_not_use_face_owner_with_multiple_visible_people():
    agent = _make_agent()
    seen = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment: seen.append(segment)
    )
    first_turn = _make_turn("rt-pref-known", audio_speaker_id="person-9")
    first_turn.user_transcript = "my dog is luna"
    first_turn.assistant_transcript = "luna sounds adorable"
    agent._maybe_note_preference_turn(first_turn)

    agent.face_service = _FakeFaceService(
        persons=[
            SimpleNamespace(person_id="person-9"),
            SimpleNamespace(person_id="person-10"),
        ],
        snapshot={"recognized_count": 2, "unknown_count": 0},
    )

    agent.enqueue_internal_event("hello", metadata={"req_id": "text-ambiguous"})
    turn = agent._turn_queue.get_nowait()
    turn.user_transcript = turn.input_text
    turn.assistant_transcript = "hi"
    agent._maybe_note_preference_turn(turn)

    assert turn.owner_id is None
    agent.flush_preference_segments(reason="shutdown")
    assert len(seen) == 1
    assert seen[0].person_id == "person-9"
    assert [item.turn_id for item in seen[0].turns] == ["rt-pref-known"]


def test_capture_turn_context_enriches_memory_only_for_owner():
    agent = _make_agent()
    calls = []

    class _MemoryCompiler:
        def person_context(self, person_id, **_kwargs):
            calls.append(person_id)
            return SimpleNamespace(
                context_markdown=(
                    "[PERSON MEMORY]\n"
                    "Preferences:\n"
                    f"- memory for {person_id}\n"
                    "Potential Follow-Ups:\n"
                    f"- followup for {person_id}"
                ),
                preferred_language=f"language-{person_id}",
            )

        def site_blocks(self, *_args, **_kwargs):
            return ()

    agent.memory_context_compiler = _MemoryCompiler()
    agent.face_service = _FakeFaceService(
        persons=[
            SimpleNamespace(
                person_id="person-1",
                name="Alice",
                interaction_count=2,
                confidence=0.93,
                bbox_area=20,
                timestamp=100.0,
                directory_profile_lines=("title: Engineer",),
                memory_profile_lines=(),
                potential_followups=(),
                preferred_language="",
                visible=True,
            ),
            SimpleNamespace(
                person_id="person-2",
                name="Bob",
                interaction_count=1,
                confidence=0.91,
                bbox_area=15,
                timestamp=100.0,
                directory_profile_lines=("title: PM",),
                memory_profile_lines=(),
                potential_followups=(),
                preferred_language="",
                visible=True,
            ),
        ],
        snapshot={"recognized_count": 2, "unknown_count": 0},
    )

    context = agent._capture_turn_context(
        primary_face_person_id="person-1",
        owner_id="person-2",
        owner_source="audio",
        speaker_visible=True,
    )

    assert calls == ["person-2"]
    alice, bob = context.persons
    assert alice.directory_profile_lines == ("title: Engineer",)
    assert alice.memory_profile_lines == ()
    assert alice.potential_followups == ()
    assert alice.__dict__.get("context_markdown", "") == ""
    assert bob.directory_profile_lines == ("title: PM",)
    assert bob.memory_profile_lines == ()
    assert bob.potential_followups == ()
    assert bob.context_markdown == (
        "[PERSON MEMORY]\n"
        "Preferences:\n"
        "- memory for person-2\n"
        "Potential Follow-Ups:\n"
        "- followup for person-2"
    )
    assert bob.preferred_language == "language-person-2"


def test_capture_turn_context_continues_without_memory_when_tailwag_fails():
    agent = _make_agent()
    agent._current_office_location = "BOS3"
    log_messages = []
    agent.logger = SimpleNamespace(
        exception=lambda message, *args: log_messages.append(message % args),
    )

    class _FailingMemoryCompiler:
        def person_context(self, *_args, **_kwargs):
            raise RuntimeError("tailwag unavailable")

        def site_blocks(self, *_args, **_kwargs):
            raise RuntimeError("tailwag unavailable")

    agent.memory_context_compiler = _FailingMemoryCompiler()
    agent.face_service = _FakeFaceService(
        persons=[
            SimpleNamespace(
                person_id="person-1",
                name="Alice",
                interaction_count=2,
                confidence=0.93,
                bbox_area=20,
                timestamp=100.0,
                directory_profile_lines=("title: Engineer",),
                memory_profile_lines=("local fallback memory",),
                potential_followups=("local fallback followup",),
                preferred_language="",
                visible=True,
            ),
        ],
        snapshot={"recognized_count": 1, "unknown_count": 0},
    )

    context = agent._capture_turn_context(
        primary_face_person_id="person-1",
        owner_id="person-1",
        owner_source="face",
        speaker_visible=True,
    )

    assert context.memory_context_blocks == ()
    assert len(context.persons) == 1
    assert context.persons[0].memory_profile_lines == ("local fallback memory",)
    assert context.persons[0].potential_followups == ("local fallback followup",)
    assert "Failed to compile site memory context" in log_messages
    assert "Failed to compile memory context for person-1" in log_messages


def test_factory_wires_identity_memory_into_prompt_extraction_and_face(monkeypatch):
    created = {
        "identity_memory_clients": [],
        "face_services": [],
        "face_event_bridges": [],
    }
    profile = _parse_factory_profile(
        {
            "name": "tailwag-factory-wiring",
            "identity_memory": {
                "enabled": True,
                "retention_class": "priority",
                "place_room_id": "lab",
                "record_live_episodes": True,
                "extract_live_turn_memory": False,
            },
            "face_recognition": {
                "enabled": True,
            },
            "speaker_recognition": {"enabled": False},
            "battery": {"enabled": False},
            "display": {"enabled": False},
        }
    )
    factory_mod = _load_factory_for_memory_tests(monkeypatch, created=created)

    agent = factory_mod.create_agent(scenario_profile=profile)

    assert len(created["identity_memory_clients"]) == 1
    client = created["identity_memory_clients"][0]
    assert client.kwargs["robot_id"] == "puffle"
    assert client.kwargs["robot_display_name"] == "Puffle"
    assert client.kwargs["site_code"] == ""
    assert client.kwargs["place_room_id"] == "lab"
    assert client.kwargs["retention_class"] == "priority"
    assert client.kwargs["extract_live_turn_memory"] is False
    assert client.kwargs["resource_id"] == "memory"
    assert client.kwargs["provider_client"] is not None
    assert agent.kwargs["identity_memory_client"] is client
    assert agent.kwargs["memory_context_compiler"] is client
    assert agent.kwargs["preference_extractor"] is client
    assert agent.kwargs["preference_extraction_enabled"] is True
    assert len(created["face_services"]) == 1
    assert created["face_services"][0].kwargs["identity_memory_client"] is client
    assert created["face_services"][0].kwargs["memory_store"] is client
    assert agent.kwargs["state_observer"] is not None
    assert created["engagements"][0].kwargs["state_observer"] is agent.kwargs["state_observer"]
    assert created["coalescers"][0].kwargs["state_observer"] is agent.kwargs["state_observer"]


def test_factory_identity_memory_disabled_omits_client_from_runtime_surfaces(monkeypatch):
    created = {
        "identity_memory_clients": [],
        "face_services": [],
        "face_event_bridges": [],
    }
    profile = _parse_factory_profile(
        {
            "name": "tailwag-factory-memory-disabled",
            "identity_memory": {"enabled": False, "record_live_episodes": True},
            "face_recognition": {
                "enabled": True,
            },
            "speaker_recognition": {"enabled": False},
            "battery": {"enabled": False},
            "display": {"enabled": False},
        }
    )
    factory_mod = _load_factory_for_memory_tests(monkeypatch, created=created)

    agent = factory_mod.create_agent(scenario_profile=profile)

    assert created["identity_memory_clients"] == []
    assert agent.kwargs["identity_memory_client"] is None
    assert agent.kwargs["memory_context_compiler"] is None
    assert agent.kwargs["preference_extractor"] is None
    assert agent.kwargs["preference_extraction_enabled"] is False
    assert len(created["face_services"]) == 1
    assert created["face_services"][0].kwargs["memory_store"] is None


def test_factory_memory_query_tools_create_tailwag_provider_without_face_runtime(monkeypatch):
    created = {
        "identity_memory_clients": [],
        "sqlite_memory_stores": [],
        "face_services": [],
        "face_event_bridges": [],
    }
    profile = _parse_factory_profile(
        {
            "name": "tailwag-memory-tools-only",
            "identity_memory": {"enabled": True},
            "tools": {
                "enabled_tool_ids": [
                    "memory.search_semantic",
                ],
            },
            "face_recognition": {"enabled": False},
            "speaker_recognition": {"enabled": False},
            "battery": {"enabled": False},
            "display": {"enabled": False},
        }
    )
    factory_mod = _load_factory_for_memory_tests(monkeypatch, created=created)

    agent = factory_mod.create_agent(scenario_profile=profile)

    assert len(created["identity_memory_clients"]) == 1
    client = created["identity_memory_clients"][0]
    assert agent.kwargs["identity_memory_client"] is client
    assert agent.kwargs["memory_context_compiler"] is client
    build_kwargs = created["build_builtin_tools_kwargs"][0]
    assert build_kwargs["memory_provider"] is client


def test_audio_turn_pending_internal_text_uses_system_role_message():
    agent = _make_agent()
    followups = []

    def _send_response_create(queued_turn):
        followups.append(queued_turn.req_id)
        queued_turn.response_finished.set()
        queued_turn.playback_finished.set()

    agent._send_response_create = _send_response_create
    turn = _make_turn(
        "rt-audio-system",
        kind="audio",
        pending_internal_text="[PENDING EVENTS]\n- BATTERY_EVENT: charging complete",
    )

    agent._run_turn(turn)

    assert not [
        evt for evt in agent._sent_events if evt["type"] == "conversation.item.create"
    ]
    item_id = next(iter(turn.history_item_ids))
    assert agent._history_item_snapshots[item_id]["role"] == "system"
    assert followups == [turn.req_id]


def test_people_context_reports_audio_face_mismatch():
    persons = [
        SimpleNamespace(
            person_id="person-1",
            name="Alice",
            bbox_area=20.0,
            interaction_count=2,
            memory_profile_lines=(),
            potential_followups=(),
            preferred_language="",
        ),
        SimpleNamespace(
            person_id="person-2",
            name="Bob",
            bbox_area=15.0,
            interaction_count=1,
            memory_profile_lines=(),
            potential_followups=(),
            preferred_language="",
        ),
    ]

    rendered = format_people_context(
        persons,
        primary_face_person_id="person-1",
        face_snapshot={
            "recognized_count": 2,
            "unknown_count": 0,
            "primary_face_kind": "recognized",
            "primary_face_name": "Alice",
            "recognized_names": ["Alice", "Bob"],
        },
        audio_speaker_id="person-2",
        owner_id="person-2",
        owner_source="audio",
        speaker_visible=True,
    )

    assert "[PERSON SPEAKING TO YOU]" in rendered
    assert "- Bob (met once before)" in rendered
    assert "Speaker resolution:" not in rendered
    assert "[OTHER PEOPLE IN VIEW]" in rendered
    assert "- Alice" in rendered
    assert "[talking to you]" not in rendered
    assert "primary visible person" not in rendered


def test_people_context_includes_directory_profile_lines():
    persons = [
        SimpleNamespace(
            person_id="person-1",
            name="Alice",
            bbox_area=20.0,
            interaction_count=2,
            directory_profile_lines=(
                "title: AI Technologist II (Analyst)",
                "manager: Dan Burns",
                "tenure: 0 year(s), 3 month(s), 5 day(s)",
            ),
            context_markdown="[PERSON MEMORY]\nPreferences:\n- preferred name: sash",
            preferred_language="",
        ),
    ]

    rendered = format_people_context(
        persons,
        primary_face_person_id="person-1",
        face_snapshot={
            "recognized_count": 1,
            "unknown_count": 0,
            "primary_face_kind": "recognized",
            "primary_face_name": "Alice",
            "recognized_names": ["Alice"],
        },
        audio_speaker_id="person-1",
        owner_id="person-1",
        owner_source="audio_face_agree",
        speaker_visible=True,
    )

    assert (
        "Directory: title: AI Technologist II (Analyst); manager: Dan Burns; "
        "tenure: 0 year(s), 3 month(s), 5 day(s)"
    ) in rendered
    assert "[PERSON SPEAKING TO YOU]" in rendered
    assert "- Alice (met 2 times)" in rendered
    assert "[PERSON MEMORY]\nPreferences:\n- preferred name: sash" in rendered


def test_people_context_normalizes_observed_stringified_directory_profile():
    persons = [
        SimpleNamespace(
            person_id="person-1",
            name="Alice",
            bbox_area=20.0,
            interaction_count=2,
            directory_profile_lines=(
                "['Title: Robotics Software Engineer I Co-op', "
                "'Manager: Brian Waite', 'Tenure: ...', "
                "'Function: Administration']"
            ),
            context_markdown="[PERSON MEMORY]\nPreferences:\n- preferred name: sash",
            preferred_language="",
        ),
    ]

    rendered = format_people_context(
        persons,
        primary_face_person_id="person-1",
        face_snapshot={
            "recognized_count": 1,
            "unknown_count": 0,
            "primary_face_kind": "recognized",
            "primary_face_name": "Alice",
            "recognized_names": ["Alice"],
        },
        audio_speaker_id="person-1",
        owner_id="person-1",
        owner_source="audio_face_agree",
        speaker_visible=True,
    )

    assert (
        "Directory: Title: Robotics Software Engineer I Co-op; "
        "Manager: Brian Waite; Tenure: ...; Function: Administration"
    ) in rendered
    assert "Directory: [; '; T; i; t; l; e" not in rendered


def test_people_context_includes_directory_only_for_visible_non_owner_people():
    persons = [
        SimpleNamespace(
            person_id="person-1",
            name="Alice",
            bbox_area=20.0,
            interaction_count=2,
            directory_profile_lines=("title: Robotics Engineer",),
            context_markdown="[PERSON MEMORY]\nPets:\n- Luna is her dog.",
            preferred_language="",
        ),
        SimpleNamespace(
            person_id="person-2",
            name="Bob",
            bbox_area=15.0,
            interaction_count=1,
            directory_profile_lines=("title: Product Manager",),
            context_markdown="[PERSON MEMORY]\nPreferences:\n- preferred name: Bobby",
            preferred_language="",
        ),
    ]

    rendered = format_people_context(
        persons,
        primary_face_person_id="person-1",
        face_snapshot={
            "recognized_count": 2,
            "unknown_count": 0,
            "primary_face_kind": "recognized",
            "primary_face_name": "Alice",
            "recognized_names": ["Alice", "Bob"],
        },
        audio_speaker_id="person-2",
        owner_id="person-2",
        owner_source="audio",
        speaker_visible=True,
    )

    assert "[OTHER PEOPLE IN VIEW]" in rendered
    assert "- Alice" in rendered
    assert "Directory: title: Robotics Engineer" not in rendered
    assert "Pets:\n- Luna is her dog." not in rendered
    assert "[PERSON SPEAKING TO YOU]" in rendered
    assert "- Bob (met once before)" in rendered
    assert "Directory: title: Product Manager" in rendered
    assert "[PERSON MEMORY]\nPreferences:\n- preferred name: Bobby" in rendered


def test_people_context_omits_audio_face_agreement_logistics():
    persons = [
        SimpleNamespace(
            person_id="person-1",
            name="Alice",
            bbox_area=20.0,
            interaction_count=2,
            memory_profile_lines=(),
            potential_followups=(),
            preferred_language="",
        ),
    ]

    rendered = format_people_context(
        persons,
        primary_face_person_id="person-1",
        face_snapshot={
            "recognized_count": 1,
            "unknown_count": 0,
            "primary_face_kind": "recognized",
            "primary_face_name": "Alice",
            "recognized_names": ["Alice"],
        },
        audio_speaker_id="person-1",
        owner_id="person-1",
        owner_source="audio_face_agree",
        speaker_visible=True,
    )

    assert "[PERSON SPEAKING TO YOU]" in rendered
    assert "- Alice (met 2 times)" in rendered
    assert "Speaker resolution:" not in rendered
    assert "primary visible person" not in rendered


def test_people_context_omits_offscreen_audio_speaker_logistics():
    persons = [
        SimpleNamespace(
            person_id="person-1",
            name="Alice",
            bbox_area=20.0,
            interaction_count=2,
            memory_profile_lines=(),
            potential_followups=(),
            preferred_language="",
        ),
        SimpleNamespace(
            person_id="person-2",
            name="Bob",
            bbox_area=15.0,
            interaction_count=1,
            memory_profile_lines=(),
            potential_followups=(),
            preferred_language="",
        ),
    ]

    rendered = format_people_context(
        persons,
        primary_face_person_id="person-1",
        face_snapshot={
            "recognized_count": 1,
            "unknown_count": 0,
            "primary_face_kind": "recognized",
            "primary_face_name": "Alice",
        },
        audio_speaker_id="person-2",
        owner_id="person-2",
        owner_source="audio",
        speaker_visible=False,
    )

    assert "[PERSON SPEAKING TO YOU]" in rendered
    assert "- Bob (met once before)" in rendered
    assert "Speaker resolution:" not in rendered
    assert "[OTHER PEOPLE IN VIEW]" in rendered
    assert "- Alice" in rendered
    assert "Attribute this turn to Bob, not Alice." not in rendered


def test_people_context_reports_unresolved_speaker():
    persons = [
        SimpleNamespace(
            person_id="person-1",
            name="Alice",
            bbox_area=20.0,
            interaction_count=2,
            memory_profile_lines=(),
            potential_followups=(),
            preferred_language="",
        ),
    ]

    rendered = format_people_context(
        persons,
        primary_face_person_id="person-1",
        face_snapshot={
            "recognized_count": 1,
            "unknown_count": 0,
            "primary_face_kind": "recognized",
            "primary_face_name": "Alice",
        },
        audio_speaker_id=None,
        owner_id=None,
        owner_source="unknown",
        speaker_visible=False,
    )

    assert rendered == ""


def test_people_context_falls_back_to_owner_id_when_owner_person_missing():
    rendered = format_people_context(
        [],
        audio_speaker_id="person-7",
        owner_id="person-7",
        owner_source="audio",
        speaker_visible=False,
    )

    assert "[PERSON SPEAKING TO YOU]" in rendered
    assert "- person-7 (first time; not visible)" in rendered
    assert "Speaker resolution:" not in rendered


def test_people_context_emits_preferred_language_directive():
    persons = [
        SimpleNamespace(
            person_id="person-1",
            name="Alice",
            bbox_area=20.0,
            interaction_count=2,
            memory_profile_lines=("preferred language: Spanish",),
            potential_followups=(),
            preferred_language="Spanish",
        ),
    ]

    rendered = format_people_context(
        persons,
        primary_face_person_id="person-1",
        face_snapshot={
            "recognized_count": 1,
            "unknown_count": 0,
            "primary_face_kind": "recognized",
            "primary_face_name": "Alice",
        },
        audio_speaker_id="person-1",
        owner_id="person-1",
        owner_source="audio_face_agree",
        speaker_visible=True,
    )

    assert "Prioritize talking in this language to this user: Spanish." in rendered


def test_tool_side_effect_arms_pending_voice_enrollment():
    agent = _make_agent()
    agent.speaker_service = _FakeSpeakerService()

    agent._maybe_handle_tool_side_effects(
        "enroll_visible_person",
        {"success": True, "person_id": "person-7"},
    )

    assert "person-7" in agent._pending_voice_enrollments


def test_voice_reference_capture_arms_one_prompt_after_two_quality_failures():
    agent = _make_agent()
    agent.speaker_service = _FakeSpeakerService(
        policy=SpeakerRecognitionPolicy(explicit_prompt_after_silent_failures=2),
        enrollment_results=[
            VoiceEnrollmentResult(
                saved=False,
                reason="reject_empty",
                person_id="person-1",
                attempt_kind="silent",
            ),
            VoiceEnrollmentResult(
                saved=False,
                reason="reject_clipped",
                person_id="person-1",
                attempt_kind="silent",
            ),
        ],
    )
    agent._arm_pending_voice_enrollment("person-1")
    turn = _make_turn(
        "rt-voice-enroll",
        primary_face_person_id="person-1",
        input_audio_pcm16=b"\x01\x00" * 1600,
        trimmed_input_audio_pcm16=b"\x01\x00" * 1600,
        context_snapshot=realtime_mod.FrozenTurnContext(
            primary_face_person_id="person-1",
            owner_id="person-1",
            face_snapshot={"recognized_count": 1, "unknown_count": 0},
        ),
    )
    turn.user_transcript = "hello my name is alice and i build robots today"

    agent._maybe_capture_voice_reference(turn)
    pending = agent._pending_voice_enrollments["person-1"]
    assert pending.silent_failures == 1
    assert pending.explicit_prompt_armed is False

    agent._maybe_capture_voice_reference(turn)
    pending = agent._pending_voice_enrollments["person-1"]
    assert pending.silent_failures == 2
    assert pending.explicit_prompt_armed is True

    prompt_note = agent._consume_voice_enrollment_prompt_note(turn)
    assert "[VOICE ENROLLMENT]" in prompt_note
    assert agent._consume_voice_enrollment_prompt_note(turn) == ""


def test_build_turn_instructions_uses_people_context_without_extra_speaker_block():
    agent = _make_agent()
    agent.speaker_service = _FakeSpeakerService()
    turn = _make_turn(
        "rt-guidance-enabled",
        context_snapshot=realtime_mod.FrozenTurnContext(
            persons=[
                SimpleNamespace(
                    person_id="person-1",
                    name="Alice",
                    interaction_count=2,
                    memory_profile_lines=(),
                    potential_followups=(),
                    preferred_language="",
                ),
                SimpleNamespace(
                    person_id="person-2",
                    name="Bob",
                    interaction_count=1,
                    memory_profile_lines=(),
                    potential_followups=(),
                    preferred_language="",
                    visible=False,
                ),
            ],
            primary_face_person_id="person-1",
            audio_speaker_id="person-2",
            owner_id="person-2",
            owner_source="audio",
            speaker_visible=False,
            face_snapshot={
                "recognized_count": 1,
                "unknown_count": 0,
                "primary_face_kind": "recognized",
                "primary_face_name": "Alice",
            },
        ),
    )

    rendered = agent._build_turn_instructions(turn)

    assert "[PERSON SPEAKING TO YOU]" in rendered
    assert "- Bob (met once before; not visible)" in rendered
    assert "[OTHER PEOPLE IN VIEW]" in rendered
    assert "- Alice" in rendered
    assert "Current speaker is not safely identified." not in rendered


def test_build_turn_instructions_omits_person_context_without_owner_id():
    agent = _make_agent()
    turn = _make_turn(
        "rt-owner-unknown",
        owner_id=None,
        primary_face_person_id=None,
        context_snapshot=realtime_mod.FrozenTurnContext(
            persons=[
                SimpleNamespace(
                    person_id="person-1",
                    name="Alice",
                    interaction_count=2,
                    memory_profile_lines=("likes robotics",),
                    potential_followups=("Ask about the perception demo.",),
                    preferred_language="",
                )
            ],
            primary_face_person_id=None,
            owner_id=None,
            owner_source="unknown",
            face_snapshot={
                "recognized_count": 1,
                "unknown_count": 0,
                "recognized_names": ["Alice"],
            },
        ),
    )

    rendered = agent._build_turn_instructions(turn)

    assert "[IDENTITY STATUS] Current speaker is not safely identified." in rendered
    assert "Do not use any person's name, claim to recognize them" in rendered
    assert "[OTHER PEOPLE IN VIEW]" not in rendered
    assert "Alice" not in rendered
    assert "likes robotics" not in rendered


def test_post_enrollment_voice_reference_save_clears_pending_state_and_supports_memory():
    agent = _make_agent()
    seen = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment: seen.append(segment)
    )
    agent.speaker_service = _FakeSpeakerService(
        enrollment_results=[
            VoiceEnrollmentResult(
                saved=True,
                reason="saved",
                person_id="person-1",
                attempt_kind="silent",
            )
        ]
    )
    agent._maybe_handle_tool_side_effects(
        "enroll_visible_person",
        {"success": True, "person_id": "person-1"},
    )
    turn = _make_turn(
        "rt-post-enroll",
        primary_face_person_id="person-1",
        owner_id="person-1",
        input_audio_pcm16=b"\x10\x00" * 32000,
        trimmed_input_audio_pcm16=b"\x10\x00" * 32000,
        context_snapshot=realtime_mod.FrozenTurnContext(
            primary_face_person_id="person-1",
            owner_id="person-1",
            face_snapshot={"recognized_count": 1, "unknown_count": 0},
        ),
    )
    turn.user_transcript = "hello my name is alice and i work on robot perception systems"
    turn.assistant_transcript = "nice to meet you"

    agent._maybe_capture_voice_reference(turn)
    agent._maybe_note_preference_turn(turn)
    agent.flush_preference_segments(reason="shutdown")

    assert "person-1" not in agent._pending_voice_enrollments
    assert agent.speaker_service.has_reference("person-1") is True
    assert len(seen) == 1
    assert seen[0].person_id == "person-1"


def test_post_enrollment_voice_reference_save_does_not_require_transcript():
    agent = _make_agent()
    agent.speaker_service = _FakeSpeakerService(
        enrollment_results=[
            VoiceEnrollmentResult(
                saved=True,
                reason="saved",
                person_id="person-1",
                attempt_kind="silent",
            )
        ]
    )
    agent._maybe_handle_tool_side_effects(
        "enroll_visible_person",
        {"success": True, "person_id": "person-1"},
    )
    turn = _make_turn(
        "rt-post-enroll-no-transcript",
        primary_face_person_id="person-1",
        owner_id="person-1",
        input_audio_pcm16=b"\x10\x00" * 32000,
        trimmed_input_audio_pcm16=b"\x10\x00" * 32000,
        context_snapshot=realtime_mod.FrozenTurnContext(
            primary_face_person_id="person-1",
            owner_id="person-1",
            face_snapshot={"recognized_count": 1, "unknown_count": 0},
        ),
    )

    agent._maybe_capture_voice_reference(turn)

    assert "person-1" not in agent._pending_voice_enrollments
    assert agent.speaker_service.has_reference("person-1") is True
    assert "transcript" not in agent.speaker_service.calls[0]


def test_voice_reference_capture_recovers_from_live_face_cache_when_frozen_primary_missing():
    agent = _make_agent()
    agent.face_service = _FakeFaceService(
        persons=[
            SimpleNamespace(
                person_id="person-1",
                name="Alice",
                bbox_area=42,
            )
        ],
        snapshot={"recognized_count": 1, "unknown_count": 0},
    )
    agent.speaker_service = _FakeSpeakerService(
        enrollment_results=[
            VoiceEnrollmentResult(
                saved=True,
                reason="saved",
                person_id="person-1",
                attempt_kind="silent",
            )
        ]
    )
    agent._maybe_handle_tool_side_effects(
        "enroll_visible_person",
        {"success": True, "person_id": "person-1"},
    )
    turn = _make_turn(
        "rt-live-face-recover",
        primary_face_person_id=None,
        owner_id=None,
        input_audio_pcm16=b"\x10\x00" * 32000,
        trimmed_input_audio_pcm16=b"\x10\x00" * 32000,
        context_snapshot=realtime_mod.FrozenTurnContext(
            primary_face_person_id=None,
            owner_id=None,
            face_snapshot={"recognized_count": 0, "unknown_count": 0},
        ),
    )
    turn.user_transcript = "hello my name is alice and i work on robot perception systems"

    agent._maybe_capture_voice_reference(turn)

    assert "person-1" not in agent._pending_voice_enrollments
    assert agent.speaker_service.has_reference("person-1") is True


def test_input_transcription_completed_does_not_retry_voice_enrollment_for_finalized_turn():
    agent = _make_agent()
    agent.speaker_service = _FakeSpeakerService(
        enrollment_results=[
            VoiceEnrollmentResult(
                saved=True,
                reason="saved",
                person_id="person-1",
                attempt_kind="silent",
            )
        ]
    )
    agent._maybe_handle_tool_side_effects(
        "enroll_visible_person",
        {"success": True, "person_id": "person-1"},
    )
    turn = _make_turn(
        "rt-enroll-late-transcript",
        primary_face_person_id="person-1",
        owner_id="person-1",
        input_audio_pcm16=b"\x10\x00" * 32000,
        trimmed_input_audio_pcm16=b"\x10\x00" * 32000,
        context_snapshot=realtime_mod.FrozenTurnContext(
            primary_face_person_id="person-1",
            owner_id="person-1",
            face_snapshot={"recognized_count": 1, "unknown_count": 0},
        ),
    )
    turn.phase = realtime_mod.TURN_PHASE_FINALIZED
    agent._turns_by_req_id[turn.req_id] = turn
    agent._bind_item_id_to_turn(turn, "user-item-enroll")
    turn.user_item_id = "user-item-enroll"
    agent._maybe_capture_voice_reference(turn)
    assert "person-1" not in agent._pending_voice_enrollments
    assert agent.speaker_service.has_reference("person-1") is True
    assert len(agent.speaker_service.calls) == 1

    agent._handle_input_transcription_completed(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "user-item-enroll",
            "transcript": "hello my name is alice and i build robots today",
        }
    )

    assert turn.user_transcript == "hello my name is alice and i build robots today"
    assert "person-1" not in agent._pending_voice_enrollments
    assert agent.speaker_service.has_reference("person-1") is True
    assert len(agent.speaker_service.calls) == 1


def test_late_input_transcription_adds_finalized_audio_turn_to_preference_segment():
    agent = _make_agent()
    seen = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment: seen.append(segment)
    )
    turn = _make_turn(
        "rt-pref-late-transcript",
        primary_face_person_id="person-1",
        owner_id="person-1",
        audio_speaker_id=None,
    )
    turn.phase = realtime_mod.TURN_PHASE_FINALIZED
    turn.assistant_transcript = "Mochi sounds sweet."
    agent._turns_by_req_id[turn.req_id] = turn
    agent._bind_item_id_to_turn(turn, "user-item-pref")
    turn.user_item_id = "user-item-pref"

    agent._maybe_note_preference_turn(turn)
    agent.flush_preference_segments(reason="shutdown")
    assert seen == []
    assert turn.preference_noted is False

    agent._handle_input_transcription_completed(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "user-item-pref",
            "transcript": "my dog is named mochi",
        }
    )
    agent.flush_preference_segments(reason="shutdown")

    assert len(seen) == 1
    assert seen[0].person_id == "person-1"
    assert seen[0].turns[0].user_text == "my dog is named mochi"
    assert seen[0].turns[0].assistant_text == "Mochi sounds sweet."


def test_late_input_transcription_can_bind_finalized_pending_audio_turn():
    agent = _make_agent()
    seen = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment: seen.append(segment)
    )
    turn = _make_turn(
        "rt-pref-unbound-late-transcript",
        primary_face_person_id="person-1",
        owner_id="person-1",
        audio_speaker_id=None,
    )
    turn.phase = realtime_mod.TURN_PHASE_FINALIZED
    turn.finalized = True
    turn.assistant_transcript = "Got it, Mochi."
    agent._turns_by_req_id[turn.req_id] = turn
    agent._pending_audio_turn_req_ids.append(turn.req_id)

    agent._maybe_note_preference_turn(turn)
    assert turn.preference_noted is False

    agent._handle_input_transcription_completed(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "late-user-item-pref",
            "transcript": "my dog is named mochi",
        }
    )
    agent.flush_preference_segments(reason="shutdown")

    assert turn.user_item_id == "late-user-item-pref"
    assert agent._item_id_to_req_id["late-user-item-pref"] == turn.req_id
    assert len(seen) == 1
    assert seen[0].person_id == "person-1"


def test_attributed_superseded_user_only_turn_is_persisted():
    agent = _make_agent()
    seen = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment: seen.append(segment)
    )
    turn = _make_turn(
        "rt-pref-superseded",
        primary_face_person_id="person-1",
        owner_id="person-1",
        audio_speaker_id=None,
    )
    turn.phase = realtime_mod.TURN_PHASE_SUPERSEDED
    turn.finalized = True
    turn.user_transcript = "the game was canceled"
    agent._turns_by_req_id[turn.req_id] = turn

    agent._maybe_note_preference_turn(turn)
    agent.flush_preference_segments(reason="shutdown")

    assert len(seen) == 1
    assert seen[0].person_id == "person-1"
    assert seen[0].turns[0].user_text == "the game was canceled"
    assert seen[0].turns[0].assistant_text == ""


def test_late_input_transcription_persists_superseded_user_only_turn():
    agent = _make_agent()
    seen = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment: seen.append(segment)
    )
    turn = _make_turn(
        "rt-pref-superseded-late-transcript",
        primary_face_person_id="person-1",
        owner_id="person-1",
        audio_speaker_id=None,
    )
    turn.phase = realtime_mod.TURN_PHASE_SUPERSEDED
    turn.finalized = True
    agent._turns_by_req_id[turn.req_id] = turn
    agent._bind_item_id_to_turn(turn, "superseded-user-item")
    turn.user_item_id = "superseded-user-item"

    agent._handle_input_transcription_completed(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "superseded-user-item",
            "transcript": "the game was canceled",
        }
    )
    agent.flush_preference_segments(reason="shutdown")

    assert len(seen) == 1
    assert seen[0].turns[0].user_text == "the game was canceled"
    assert seen[0].turns[0].assistant_text == ""


def test_preference_turn_without_owner_id_writes_nothing():
    agent = _make_agent()
    seen = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment: seen.append(segment)
    )
    turn = _make_turn(
        "rt-pref-none",
        owner_id=None,
        audio_speaker_id=None,
        context_snapshot=realtime_mod.FrozenTurnContext(
            primary_face_person_id="person-1",
            owner_id=None,
            audio_speaker_id=None,
        ),
    )
    turn.user_transcript = "hello there"
    turn.assistant_transcript = "hi"

    agent._maybe_note_preference_turn(turn)
    agent.flush_preference_segments(reason="shutdown")

    assert seen == []
    assert turn.preference_noted is False


def test_preference_turn_missing_owner_can_be_retried_after_resolution():
    agent = _make_agent()
    seen = []
    agent.preference_extractor = SimpleNamespace(
        extract_and_store_segment=lambda segment: seen.append(segment)
    )
    turn = _make_turn(
        "rt-pref-owner-late",
        owner_id=None,
        audio_speaker_id=None,
        context_snapshot=realtime_mod.FrozenTurnContext(
            primary_face_person_id="person-1",
            owner_id=None,
            audio_speaker_id=None,
        ),
    )
    turn.phase = realtime_mod.TURN_PHASE_FINALIZED
    turn.user_transcript = "my dog is luna"
    turn.assistant_transcript = "Luna sounds sweet."
    agent._turns_by_req_id[turn.req_id] = turn

    agent._maybe_note_preference_turn(turn)
    assert turn.preference_noted is False

    turn.owner_id = "person-1"
    turn.context_snapshot.owner_id = "person-1"
    agent.flush_preference_segments(reason="shutdown")

    assert len(seen) == 1
    assert seen[0].person_id == "person-1"
    assert seen[0].turns[0].user_text == "my dog is luna"
