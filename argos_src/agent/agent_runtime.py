"""Realtime speech-first Argos agent runtime."""

from __future__ import annotations

import atexit
import base64
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
import queue
from collections import deque
import threading
import time
from typing import Any, Optional
from uuid import uuid4

import websocket

from argos_src.agent.control.coalescer import EventCoalescer
from argos_src.agent.control.display_controller import DisplayController
from argos_src.agent.control.engagement_runtime import EngagementStateMachine
from argos_src.agent.preference_segments import _PreferenceSegmentCoordinator
from argos_src.agent.control.audio_runtime import (
    AudioRuntime,
    VAD_SAMPLE_RATE,
)
from argos_src.agent.control.state_runtime import AgentStateRuntime
from argos_src.agent.control.server_event_runtime import ServerEventRuntime
from argos_src.agent.realtime_turns import (
    NO_AUDIO_RESPONSE_RETRY_LIMIT,
    PLAYBACK_STALL_TIMEOUT_SEC,
    RESPONSE_STALL_TIMEOUT_SEC,
    TURN_PHASE_CANCELED,
    TURN_PHASE_FINALIZED,
    TURN_PHASE_PLAYING,
    TURN_PHASE_QUEUED,
    TURN_PHASE_RESPONSE_REQUESTED,
    TURN_PHASE_REQUESTING_FOLLOWUP,
    TURN_PHASE_SUPERSEDED,
    TURN_PHASE_WAITING_FIRST_AUDIO,
    TURN_PHASE_WAITING_TOOLS,
    WATCHDOG_POLL_SEC,
    FrozenTurnContext,
    PendingToolCall,
    PlaybackBuffer,
    QueuedTurn,
)
from argos_src.agent.control.history_store import OwnerScopedHistoryIndex
from argos_src.agent.control.event_adapter import RealtimeEventAdapter
from argos_src.agent.control.observers import safe_transition
from argos_src.agent.control.playback_runtime import PlaybackRuntime
from argos_src.agent.control.preference_runtime import PreferenceRuntime
from argos_src.agent.control.turn_store import PendingResponseBindingStore
from argos_src.agent.control.tool_runtime import ToolRuntime
from argos_src.agent.control.turn_runner import TurnRunner
from argos_src.agent.control.types import (
    SessionState,
    StateAxis,
    StateTransition,
    TranscriptionState,
)
from argos_src.agent.control.watchdog_runtime import TurnWatchdogRuntime
from argos_src.agent.control.voice_command_runtime import VoiceCommandRuntime
from argos_src.identity_memory.biometric_updates import AdaptiveBiometricObservation
from argos_src.observability.state_observer import StructuredStateObserver
from argos_src.agent.runtime_context import (
    format_current_office_location_block,
    format_current_time_block,
    format_people_context,
    format_robot_state_block,
    format_saved_locations,
)
from argos_src.observability.observability import (
    LatencyLogger,
    perf_now,
)
from argos_src.observability.pricing import (
    estimate_realtime_response_cost,
)
from argos_src.openai_realtime import (
    realtime_audio_session_payload,
    realtime_auth_headers,
    realtime_response_payload,
    realtime_websocket_url,
)
from argos_src.profile_config import RealtimeProfile, ScenarioProfile
from argos_src.runtime.audio_admission import FacePresenceGate
from argos_src.provider_api.client import ProviderClient
from argos_src.speaker_recognition.models import (
    PendingVoiceEnrollment,
    SpeakerResolutionResult,
)
from argos_src.speaker_recognition.policy import resolve_owner_id
from argos_src.media.audio_detection import OpenWakeWord, SileroVAD


logger = logging.getLogger(__name__)

__all__ = [
    "FrozenTurnContext",
    "NO_AUDIO_RESPONSE_RETRY_LIMIT",
    "RealtimeRobotAgent",
]

PREFERENCE_IDLE_FLUSH_DELAY_SEC = 60.0


def _log_text_b64(value: str) -> str:
    rendered = str(value or "")
    if not rendered:
        return ""
    return base64.b64encode(rendered.encode("utf-8")).decode("ascii")


def _human_text_from_text_turn(text: str) -> str:
    rendered = str(text or "").strip()
    marker = "[HUMAN INPUT]"
    if marker not in rendered:
        return rendered
    return rendered.split(marker, 1)[1].strip()


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


class RealtimeRobotAgent:
    """Persistent realtime speech-to-speech robot agent runtime."""

    def __init__(
        self,
        *,
        scenario_profile: ScenarioProfile,
        robot_client: ProviderClient,
        tools: list[Any],
        base_system_prompt: str,
        engagement: EngagementStateMachine,
        coalescer: Optional[EventCoalescer] = None,
        face_service: Any = None,
        speaker_service: Any = None,
        memory_context_compiler: Any = None,
        preference_extractor: Any = None,
        preference_extraction_enabled: bool = False,
        location_store: Any = None,
        nav_state: Any = None,
        battery_cache: Any = None,
        gesture_runtime: Any = None,
        display_runtime: Any = None,
        owner_turn_controller: Any = None,
        initial_robot_posture: str = "standing",
        stand_tool_name: str = "move_robot",
        supports_navigation: bool = False,
        state_observer: Any = None,
        identity_memory_client: Any = None,
        adaptive_update_coordinator: Any = None,
        raw_data_capture: Any = None,
    ) -> None:
        self.logger = logging.getLogger("argos.agent_runtime")
        self.scenario_profile = scenario_profile
        self.realtime_profile: RealtimeProfile = scenario_profile.realtime
        self.robot_client = robot_client
        self.tools = list(tools)
        self.base_system_prompt = base_system_prompt
        self.engagement = engagement
        self.coalescer = coalescer
        self.face_service = face_service
        self.speaker_service = speaker_service
        self.identity_memory_client = identity_memory_client
        self.adaptive_update_coordinator = adaptive_update_coordinator
        self.raw_data_capture = raw_data_capture
        self.memory_context_compiler = memory_context_compiler
        self.preference_extractor = preference_extractor
        self.preference_extraction_enabled = preference_extraction_enabled
        self.location_store = location_store
        self.nav_state = nav_state
        self.battery_cache = battery_cache
        self.gesture_runtime = gesture_runtime
        self.display_runtime = display_runtime
        self.owner_turn_controller = owner_turn_controller
        self._robot_posture = initial_robot_posture
        self._stand_tool_name = stand_tool_name
        self._supports_navigation = supports_navigation
        self._last_tool_name: Optional[str] = None
        self._last_tool_summary: Optional[str] = None
        self._last_external_input_s = 0.0
        self._current_office_location = str(
            getattr(getattr(scenario_profile, "identity_memory", None), "site_code", "")
            or ""
        ).strip()

        self._latency = LatencyLogger("realtime")
        self._tool_latency = LatencyLogger("tool")
        self._state_observer = state_observer or StructuredStateObserver()
        self._run_id = f"run-{uuid4().hex[:12]}"
        self._exchange_counter = 0
        self._current_exchange_id = ""
        self._current_exchange_index = 0
        self._current_exchange_trigger = ""
        self._current_exchange_admission_reason = ""
        self._current_face_evidence_fields: dict[str, Any] = {}
        self._session_state = SessionState.STOPPED.value
        self._preference_segments = (
            _PreferenceSegmentCoordinator() if preference_extraction_enabled else None
        )
        self._preference_executor = ThreadPoolExecutor(max_workers=1)
        self._pending_preference_segment_ids: set[str] = set()
        self._pending_lock = threading.Lock()
        self._preference_idle_flush_lock = threading.Lock()
        self._preference_idle_flush_timer: Optional[threading.Timer] = None
        self._preference_idle_flush_delay_sec = PREFERENCE_IDLE_FLUSH_DELAY_SEC

        self._face_gate = FacePresenceGate()
        self._vad = SileroVAD(VAD_SAMPLE_RATE, self.realtime_profile.vad_threshold)
        self._wake_word = OpenWakeWord(
            self.realtime_profile.wake_word_model,
            self.realtime_profile.wake_threshold,
        )

        self._ws: Optional[websocket.WebSocket] = None
        self._ws_lock = threading.Lock()
        self._session_ready = threading.Event()
        self._receiver_thread: Optional[threading.Thread] = None
        self._sender_thread: Optional[threading.Thread] = None
        self._response_thread: Optional[threading.Thread] = None
        self._tool_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._executor_thread: Optional[threading.Thread] = None
        self._executor: Optional[Any] = None
        self._event_adapter = RealtimeEventAdapter(self)
        self._server_event_runtime = ServerEventRuntime(self)

        self._stop_event = threading.Event()
        self._audio_send_queue: queue.Queue[bytes] = queue.Queue()
        self._turn_queue: queue.Queue[QueuedTurn] = queue.Queue()
        self._tool_queue: queue.Queue[PendingToolCall] = queue.Queue()
        self._playback_buffer = PlaybackBuffer()
        self._turn_runner = TurnRunner(self)
        self._playback_runtime = PlaybackRuntime(self)
        self._turn_watchdog_runtime = TurnWatchdogRuntime(self)
        self._audio_runtime = AudioRuntime(self)
        self._state_runtime = AgentStateRuntime(self)
        self._preference_runtime = PreferenceRuntime(self)
        self._voice_command_runtime = VoiceCommandRuntime(self)
        self._display_controller = DisplayController(self)
        self._playback_state = "idle"
        self._capture_state = "not_ready"
        self._input_stream: Optional[Any] = None
        self._output_stream: Optional[Any] = None

        self._recording_lock = threading.RLock()
        self._recording_active = False
        self._recording_started_at = 0.0
        self._last_voice_at = 0.0
        self._current_primary_face_person_id: Optional[str] = None
        self._current_visible_face_person_ids: tuple[str, ...] = ()
        self._current_turn_audio_chunks: list[bytes] = []
        self._current_raw_face_snapshot: Any = None
        self._current_turn_vad_positive_blocks = 0
        self._candidate_voice_blocks = 0
        self._recording_preroll_chunks: deque[tuple[float, bytes, bytes]] = deque()
        self._recording_gesture_queue: queue.Queue[bool] = queue.Queue()
        self._recording_gesture_lock = threading.Lock()
        self._recording_gesture_thread: Optional[threading.Thread] = None
        self._display_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._display_thread: Optional[threading.Thread] = None
        self._display_mode_lock = threading.Lock()
        self._display_mode = ""
        self._resample_state: Any = None
        self._wake_window_until = 0.0
        self._input_suppressed_until_s = 0.0
        self._last_wake_debug_log_s = 0.0
        self._pending_voice_enrollments: dict[str, PendingVoiceEnrollment] = {}
        self._voice_enrollment_lock = threading.Lock()

        self._turn_lock = threading.RLock()
        self._active_turn: Optional[QueuedTurn] = None
        self._turns_by_req_id: dict[str, QueuedTurn] = {}
        self._response_id_to_req_id: dict[str, str] = {}
        self._item_id_to_req_id: dict[str, str] = {}
        self._call_id_to_req_id: dict[str, str] = {}
        self._pending_function_args: dict[str, dict[str, str]] = {}
        self._pending_response_turn_req_ids: deque[str] = deque()
        self._expired_stale_response_turn_req_ids: deque[str] = deque()
        self._stale_response_deadlines_by_req_id: dict[str, float] = {}
        self._response_binding_store = PendingResponseBindingStore(
            turns_by_req_id=self._turns_by_req_id,
            is_terminal=self._is_turn_terminal,
            pending_req_ids=self._pending_response_turn_req_ids,
            expired_stale_req_ids=self._expired_stale_response_turn_req_ids,
            stale_deadlines_by_req_id=self._stale_response_deadlines_by_req_id,
            response_id_to_req_id=self._response_id_to_req_id,
            now=time.time,
        )
        self._pending_audio_turn_req_ids: deque[str] = deque()
        self._pending_audio_item_ids: deque[str] = deque()
        self._pending_local_created_items: deque[Any] = deque()
        self._history_item_order: deque[str] = deque()
        self._known_history_item_ids: set[str] = set()
        self._history_item_owner_req_id: dict[str, str] = {}
        self._history_index_store = OwnerScopedHistoryIndex(
            item_order=self._history_item_order,
            known_item_ids=self._known_history_item_ids,
            item_owner_req_id=self._history_item_owner_req_id,
        )
        self._active_history_owner_key: str = ""
        self._playback_req_id: str = ""
        self._playback_stream_id: str = ""
        self._playback_item_id: str = ""
        self._played_output_frames = 0
        self._ignored_voice_commands: deque[tuple[str, float]] = deque()

        self._tool_registry = {str(getattr(tool, "name", "")).strip(): tool for tool in self.tools}
        self._tool_runtime_controller = ToolRuntime(self)
        self._tool_schemas = [
            self._build_tool_schema(tool)
            for tool in self.tools
            if str(getattr(tool, "name", "")).strip()
        ]
        self._session_id = ""
        self._session_estimated_cost_usd = 0.0
        if self.raw_data_capture is not None:
            try:
                self.raw_data_capture.start_session(
                    run_id=self._run_id,
                    metadata={
                        "profile": getattr(scenario_profile, "name", ""),
                        "profile_file": str(getattr(scenario_profile, "source_path", "") or ""),
                        "model": getattr(self.realtime_profile, "model", ""),
                    },
                )
            except Exception:
                self.logger.exception("Failed to start raw data capture session")

    # ------------------------------------------------------------------
    # State/history runtime
    # ------------------------------------------------------------------

    def _state_controller(self) -> AgentStateRuntime:
        runtime = getattr(self, "_state_runtime", None)
        if runtime is None or getattr(runtime, "_host", None) is not self:
            runtime = AgentStateRuntime(self)
            self._state_runtime = runtime
        return runtime

    def _enrich_person_context_with_memory(self, person: Any) -> Any:
        return self._state_controller()._enrich_person_context_with_memory(person)

    def _compile_memory_context_blocks(self, current_person_id: Optional[str]) -> tuple[str, ...]:
        return self._state_controller()._compile_memory_context_blocks(current_person_id)

    def _append_text_message_item(self, turn: QueuedTurn, text: str, *, role: str) -> None:
        self._state_controller()._append_text_message_item(turn, text, role=role)

    def _queue_pending_local_created_item(
        self,
        owner_req_id: str,
        expected_type: str,
        expected_role: str = "",
    ) -> None:
        self._state_controller()._queue_pending_local_created_item(
            owner_req_id,
            expected_type,
            expected_role,
        )

    def _consume_pending_local_created_item(
        self,
        expected_type: str,
        expected_role: str = "",
    ) -> str:
        return self._state_controller()._consume_pending_local_created_item(
            expected_type,
            expected_role,
        )

    def _register_pending_audio_turn(self, turn: QueuedTurn) -> None:
        self._state_controller()._register_pending_audio_turn(turn)

    def _consume_pending_audio_turn_req_id(self, *, include_finalized: bool = False) -> str:
        return self._state_controller()._consume_pending_audio_turn_req_id(
            include_finalized=include_finalized,
        )

    def _capture_turn_context(
        self,
        *,
        primary_face_person_id: Optional[str] = None,
        audio_speaker_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        owner_source: str = "unknown",
        owner_confidence: float = 0.0,
        speaker_visible: bool = False,
    ) -> FrozenTurnContext:
        return self._state_controller()._capture_turn_context(
            primary_face_person_id=primary_face_person_id,
            audio_speaker_id=audio_speaker_id,
            owner_id=owner_id,
            owner_source=owner_source,
            owner_confidence=owner_confidence,
            speaker_visible=speaker_visible,
        )

    def _set_turn_phase(
        self,
        turn: QueuedTurn,
        phase: str,
        *,
        trigger: str = "set_turn_phase",
    ) -> None:
        self._state_controller()._set_turn_phase(turn, phase, trigger=trigger)

    def _is_turn_terminal(self, turn: Optional[QueuedTurn]) -> bool:
        return self._state_controller()._is_turn_terminal(turn)

    def _base_log_fields(self) -> dict[str, Any]:
        self._ensure_observability_ids()
        return {
            "run_id": self._run_id,
            "openai_session_id": getattr(self, "_session_id", "") or None,
            "session_id": getattr(self, "_session_id", "") or None,
        }

    def _ensure_observability_ids(self) -> None:
        if not hasattr(self, "_run_id"):
            self._run_id = f"run-{uuid4().hex[:12]}"
        if not hasattr(self, "_exchange_counter"):
            self._exchange_counter = 0
        if not hasattr(self, "_current_exchange_id"):
            self._current_exchange_id = ""
        if not hasattr(self, "_current_exchange_index"):
            self._current_exchange_index = 0
        if not hasattr(self, "_current_exchange_trigger"):
            self._current_exchange_trigger = ""
        if not hasattr(self, "_current_exchange_admission_reason"):
            self._current_exchange_admission_reason = ""
        if not hasattr(self, "_current_face_evidence_fields"):
            self._current_face_evidence_fields = {}

    def _new_exchange_locked(
        self,
        *,
        trigger: str,
        admission_reason: str,
    ) -> tuple[str, int]:
        self._ensure_observability_ids()
        self._exchange_counter += 1
        self._current_exchange_id = f"ex-{uuid4().hex[:12]}"
        self._current_exchange_index = self._exchange_counter
        self._current_exchange_trigger = str(trigger or "").strip()
        self._current_exchange_admission_reason = str(admission_reason or "").strip()
        return self._current_exchange_id, self._current_exchange_index

    def _current_exchange_fields_locked(self) -> dict[str, Any]:
        self._ensure_observability_ids()
        return {
            **self._base_log_fields(),
            "exchange_id": self._current_exchange_id or None,
            "exchange_index": self._current_exchange_index or None,
            "turn_kind": "human_audio",
            "trigger": self._current_exchange_trigger or None,
            "admission_reason": self._current_exchange_admission_reason or None,
            **dict(self._current_face_evidence_fields or {}),
        }

    def _exchange_log_fields(self, turn: QueuedTurn | None = None) -> dict[str, Any]:
        fields = self._base_log_fields()
        if turn is None:
            recording_lock = getattr(self, "_recording_lock", None)
            if recording_lock is None:
                fields.update(self._current_exchange_fields_locked())
                return fields
            with recording_lock:
                fields.update(self._current_exchange_fields_locked())
            return fields
        metadata = turn.metadata if isinstance(turn.metadata, dict) else {}
        fields.update(
            {
                "exchange_id": turn.exchange_id or metadata.get("exchange_id") or None,
                "exchange_index": turn.exchange_index or metadata.get("exchange_index") or None,
                "turn_kind": (
                    "internal_text"
                    if bool(getattr(turn, "source_is_internal", False))
                    else ("human_audio" if turn.kind == "audio" else "human_text")
                ),
                "primary_face_person_id": turn.primary_face_person_id or None,
                "audio_speaker_id": turn.audio_speaker_id or None,
                "owner_id": turn.owner_id or None,
                "owner_source": turn.owner_source or None,
                "owner_confidence": turn.owner_confidence,
                "speaker_visible": turn.speaker_visible,
                "audio_score": metadata.get("audio_score"),
                "audio_runner_up_score": metadata.get("audio_runner_up_score"),
                "audio_score_margin": metadata.get("audio_score_margin"),
                "face_match_status": metadata.get("face_match_status"),
                "face_match_reason": metadata.get("face_match_reason"),
                "face_match_name": metadata.get("face_match_name"),
                "face_match_person_id": metadata.get("face_match_person_id"),
                "face_score": metadata.get("face_score"),
                "face_score_threshold": metadata.get("face_score_threshold"),
                "face_runner_up_score": metadata.get("face_runner_up_score"),
                "face_score_margin": metadata.get("face_score_margin"),
                "face_margin_threshold": metadata.get("face_margin_threshold"),
                "error_source": metadata.get("error_source"),
                "error_type": metadata.get("error_type"),
                "error_code": metadata.get("error_code"),
                "error_message": metadata.get("error_message"),
                "server_error_type": metadata.get("server_error_type"),
                "server_error_code": metadata.get("server_error_code"),
                "server_error_message": metadata.get("server_error_message"),
                "trigger": metadata.get("trigger") or metadata.get("admission_reason") or None,
                "admission_reason": metadata.get("admission_reason") or None,
                "pending_internal_events": bool(getattr(turn, "pending_internal_text", None)),
            }
        )
        return fields

    def _response_bindings(self) -> PendingResponseBindingStore:
        return self._state_controller()._response_bindings()

    def _history_index(self) -> OwnerScopedHistoryIndex:
        return self._state_controller()._history_index()

    def _bind_response_id(self, turn: QueuedTurn, response_id: str) -> None:
        self._state_controller()._bind_response_id(turn, response_id)

    def _bind_item_id_to_turn(self, turn: QueuedTurn, item_id: str) -> None:
        self._state_controller()._bind_item_id_to_turn(turn, item_id)

    def _req_id_for_response_id(self, response_id: str) -> str:
        return self._state_controller()._req_id_for_response_id(response_id)

    def _resolve_turn_for_item(self, item_id: str) -> Optional[QueuedTurn]:
        return self._state_controller()._resolve_turn_for_item(item_id)

    def _resolve_turn_for_output(
        self,
        *,
        response_id: str = "",
        item_id: str = "",
        call_id: str = "",
    ) -> Optional[QueuedTurn]:
        return self._state_controller()._resolve_turn_for_output(
            response_id=response_id,
            item_id=item_id,
            call_id=call_id,
        )

    def _consume_pending_response_turn(
        self,
        response_id: str,
        *,
        consume_only_if_missing: bool = True,
    ) -> Optional[QueuedTurn]:
        return self._state_controller()._consume_pending_response_turn(
            response_id,
            consume_only_if_missing=consume_only_if_missing,
        )

    def _consume_pending_response_binding(
        self,
        response_id: str,
        *,
        consume_only_if_missing: bool = True,
    ) -> Any:
        return self._state_controller()._consume_pending_response_binding(
            response_id,
            consume_only_if_missing=consume_only_if_missing,
        )

    def _queue_pending_response_turn(self, req_id: str) -> None:
        self._state_controller()._queue_pending_response_turn(req_id)

    def _mark_pending_response_turn_stale(self, req_id: str) -> bool:
        return self._state_controller()._mark_pending_response_turn_stale(req_id)

    def _pending_stale_response_deadline(self) -> float | None:
        return self._state_controller()._pending_stale_response_deadline()

    def _next_pending_response_turn(self) -> Optional[QueuedTurn]:
        return self._state_controller()._next_pending_response_turn()

    def _wait_for_stale_response_slot(self) -> bool:
        return self._state_controller()._wait_for_stale_response_slot()

    def _conversation_item_looks_like_audio_input(self, item: dict[str, Any]) -> bool:
        return self._state_controller()._conversation_item_looks_like_audio_input(item)

    def _register_history_item(self, item_id: str, *, owner_req_id: str = "") -> None:
        self._state_controller()._register_history_item(
            item_id,
            owner_req_id=owner_req_id,
        )

    def _register_turn_history_item(self, turn: QueuedTurn, item_id: str) -> None:
        self._state_controller()._register_turn_history_item(turn, item_id)

    def _forget_history_item(self, turn: Optional[QueuedTurn], item_id: str) -> None:
        self._state_controller()._forget_history_item(turn, item_id)

    def _history_owner_key_for_turn(self, turn: QueuedTurn) -> str:
        return self._state_controller()._history_owner_key_for_turn(turn)

    def _history_protected_item_ids(self, current_turn: Optional[QueuedTurn]) -> set[str]:
        return self._state_controller()._history_protected_item_ids(current_turn)

    def _forget_deleted_history_item(self, item_id: str) -> None:
        self._state_controller()._forget_deleted_history_item(item_id)

    def _maybe_rotate_history_for_turn(self, turn: QueuedTurn) -> None:
        self._state_controller()._maybe_rotate_history_for_turn(turn)

    def _forget_response_id(self, response_id: str) -> None:
        self._state_controller()._forget_response_id(response_id)

    def _discard_pending_response_turn(self, req_id: str) -> int:
        return self._state_controller()._discard_pending_response_turn(req_id)

    def _response_output_types(self, response: dict[str, Any]) -> list[str]:
        return self._state_controller()._response_output_types(response)

    def _cleanup_silent_response_items(
        self,
        turn: QueuedTurn,
        response: dict[str, Any],
    ) -> None:
        self._state_controller()._cleanup_silent_response_items(turn, response)

    def _retry_no_audio_response(
        self,
        turn: QueuedTurn,
        response: dict[str, Any],
    ) -> bool:
        return self._state_controller()._retry_no_audio_response(turn, response)

    def _transcript_looks_truncated(self, transcript: str) -> bool:
        return self._state_controller()._transcript_looks_truncated(transcript)

    def _should_continue_incomplete_audio_reply(self, turn: QueuedTurn) -> bool:
        return self._state_controller()._should_continue_incomplete_audio_reply(turn)

    def _continue_incomplete_audio_reply(self, turn: QueuedTurn) -> None:
        self._state_controller()._continue_incomplete_audio_reply(turn)

    def _stringify_tool_output(self, content: object) -> str:
        return self._state_controller()._stringify_tool_output(content)

    def _send_event(self, payload: dict[str, Any]) -> None:
        self._state_controller()._send_event(payload)

    def _transcript_from_response(self, response: dict[str, Any]) -> str:
        return self._state_controller()._transcript_from_response(response)

    def _log_wakeword_debug(
        self,
        *,
        wake_detected: bool,
        wake_output: dict[str, Any],
    ) -> None:
        self._state_controller()._log_wakeword_debug(
            wake_detected=wake_detected,
            wake_output=wake_output,
        )

    def _get_current_primary_face_person_id(self) -> Optional[str]:
        return self._state_controller()._get_current_primary_face_person_id()

    def _get_current_face_evidence_fields(self) -> dict[str, Any]:
        return self._state_controller()._get_current_face_evidence_fields()

    def _get_current_visible_face_person_ids(self) -> tuple[str, ...]:
        return self._state_controller()._get_current_visible_face_person_ids()

    # ------------------------------------------------------------------
    # Tool runtime
    # ------------------------------------------------------------------

    def _tool_runtime(self) -> ToolRuntime:
        runtime = getattr(self, "_tool_runtime_controller", None)
        if runtime is None or runtime._host is not self:
            runtime = ToolRuntime(self)
            self._tool_runtime_controller = runtime
        return runtime

    def _execute_tool_call(self, pending: PendingToolCall) -> None:
        self._tool_runtime().execute(pending)

    def _build_tool_schema(self, tool: Any) -> dict[str, Any]:
        return ToolRuntime.build_schema(tool)

    def _maybe_handle_tool_side_effects(self, tool_name: str, content: object) -> None:
        self._tool_runtime().maybe_handle_side_effects(tool_name, content)

    # ------------------------------------------------------------------
    # Playback runtime
    # ------------------------------------------------------------------

    def _playback_controller(self) -> PlaybackRuntime:
        runtime = getattr(self, "_playback_runtime", None)
        if runtime is None or runtime._host is not self:
            runtime = PlaybackRuntime(self)
            self._playback_runtime = runtime
        return runtime

    def _wait_for_playback_and_complete(self, turn: QueuedTurn, stream_id: str) -> None:
        self._playback_controller().wait_for_playback_and_complete(turn, stream_id)

    def _force_complete_stalled_playback(self, turn: QueuedTurn, *, reason: str) -> None:
        self._playback_controller().force_complete_stalled_playback(turn, reason=reason)

    def interrupt_current_response(self, *, reason: str) -> None:
        self._playback_controller().interrupt_current_response(reason=reason)

    # ------------------------------------------------------------------
    # Turn watchdog runtime
    # ------------------------------------------------------------------

    def _turn_watchdog(self) -> TurnWatchdogRuntime:
        runtime = getattr(self, "_turn_watchdog_runtime", None)
        if runtime is None or runtime._host is not self:
            runtime = TurnWatchdogRuntime(self)
            self._turn_watchdog_runtime = runtime
        return runtime

    def _turn_runner_controller(self) -> TurnRunner:
        runner = getattr(self, "_turn_runner", None)
        if runner is None or runner._host is not self:
            runner = TurnRunner(self)
            self._turn_runner = runner
        return runner

    # ------------------------------------------------------------------
    # Audio runtime
    # ------------------------------------------------------------------

    def _audio_controller(self) -> AudioRuntime:
        runtime = getattr(self, "_audio_runtime", None)
        if runtime is None or runtime._host is not self:
            runtime = AudioRuntime(self)
            self._audio_runtime = runtime
        return runtime

    def _set_capture_state(
        self,
        state: Any,
        *,
        trigger: str,
        req_id: str = "",
        reason: str = "",
    ) -> None:
        self._audio_controller()._set_capture_state(
            state,
            trigger=trigger,
            req_id=req_id,
            reason=reason,
        )

    def _start_audio_streams(self) -> None:
        self._audio_controller()._start_audio_streams()

    def _capture_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        self._audio_controller()._capture_callback(indata, frames, time_info, status)

    def _start_recording_locked(
        self,
        *,
        now_s: float,
        admission_reason: str = "",
        interaction_state: str = "",
        wake_detected: bool = False,
    ) -> None:
        self._audio_controller()._start_recording_locked(
            now_s=now_s,
            admission_reason=admission_reason,
            interaction_state=interaction_state,
            wake_detected=wake_detected,
        )

    def _finalize_recording_locked(self, *, now_s: float) -> None:
        self._audio_controller()._finalize_recording_locked(now_s=now_s)

    def _commit_audio_turn(
        self,
        primary_face_person_id: Optional[str],
        visible_face_person_ids: tuple[str, ...],
        audio_pcm16: bytes,
        capture_vad_positive_blocks: int,
        speech_end_perf_s: float,
        speech_end_unix_s: float,
        face_evidence_fields: dict[str, Any] | None = None,
        raw_face_snapshot: Any = None,
    ) -> None:
        self._audio_controller()._commit_audio_turn(
            primary_face_person_id,
            visible_face_person_ids,
            dict(face_evidence_fields or {}),
            audio_pcm16,
            capture_vad_positive_blocks,
            speech_end_perf_s,
            speech_end_unix_s,
            raw_face_snapshot,
        )

    def _maybe_submit_adaptive_face_observation(
        self,
        *,
        resolution: SpeakerResolutionResult,
        face_evidence_fields: dict[str, Any],
        log_fields: dict[str, Any] | None = None,
    ) -> None:
        coordinator = getattr(self, "adaptive_update_coordinator", None)
        face_service = getattr(self, "face_service", None)
        owner_id = str(getattr(resolution, "owner_id", "") or "").strip()
        if (
            coordinator is None
            or face_service is None
            or not owner_id
            or str(getattr(resolution, "owner_source", "") or "") != "audio_face_agree"
        ):
            return
        getter = getattr(face_service, "get_recent_face_observation", None)
        if not callable(getter):
            return
        try:
            observation = getter(owner_id)
        except Exception:
            self.logger.debug(
                "Adaptive face observation unavailable owner_id=%s",
                owner_id,
                exc_info=True,
            )
            return
        if not observation:
            return
        metadata = dict(observation.get("metadata") or {})
        evidence = {
            **dict(face_evidence_fields or {}),
            "owner_id": owner_id,
            "owner_source": str(getattr(resolution, "owner_source", "") or ""),
            "primary_face_person_id": str(face_evidence_fields.get("face_match_person_id") or owner_id),
            "audio_speaker_id": str(getattr(resolution, "audio_speaker_id", "") or "").strip(),
            "face_margin": _safe_float(
                face_evidence_fields.get("face_score_margin") or metadata.get("margin")
            ),
            "voice_margin": float(getattr(resolution, "margin", 0.0) or 0.0),
            "audio_score_margin": float(getattr(resolution, "margin", 0.0) or 0.0),
            "recognized_count": _safe_int(face_evidence_fields.get("recognized_count")),
            "unknown_count": _safe_int(face_evidence_fields.get("unknown_count")),
        }
        coordinator.submit(
            AdaptiveBiometricObservation(
                modality="face",
                person_id=owner_id,
                embedding=observation.get("embedding"),
                model=str(observation.get("model") or "facenet-vggface2"),
                evidence=evidence,
                metadata={
                    **metadata,
                    "source": "face_loop",
                    "observed_at": float(observation.get("observed_at", 0.0) or 0.0),
                },
                log_fields=dict(log_fields or {}),
            )
        )

    def _playback_callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        self._audio_controller()._playback_callback(outdata, frames, time_info, status)

    def _audio_sender_loop(self) -> None:
        self._audio_controller()._audio_sender_loop()

    def _input_playback_guard_active(self, *, now_s: float | None = None) -> bool:
        return self._audio_controller()._input_playback_guard_active(now_s=now_s)

    # ------------------------------------------------------------------
    # Preference runtime
    # ------------------------------------------------------------------

    def _preference_controller(self) -> PreferenceRuntime:
        runtime = getattr(self, "_preference_runtime", None)
        if runtime is None or runtime._host is not self:
            runtime = PreferenceRuntime(self)
            self._preference_runtime = runtime
        return runtime

    def flush_preference_segments(self, reason: str = "idle") -> None:
        """Flush any buffered speaker-owned preference segment."""
        self._preference_controller().flush_segments(reason=reason)

    def _schedule_preference_idle_flush(self) -> None:
        self._preference_controller().schedule_idle_flush()

    def _cancel_preference_idle_flush(self) -> None:
        self._preference_controller().cancel_idle_flush()

    def _maybe_note_preference_turn(self, turn: Any) -> None:
        self._preference_controller().maybe_note_turn(turn)

    def _schedule_preference_segment_extraction(self, segment: Any, *, reason: str) -> None:
        self._preference_controller().schedule_segment_extraction(segment, reason=reason)

    def _retry_ready_preference_turns(self) -> None:
        self._preference_controller().retry_ready_turns()

    # ------------------------------------------------------------------
    # State observability
    # ------------------------------------------------------------------

    def _set_session_state(
        self,
        state: SessionState | str,
        *,
        trigger: str,
        reason: str = "",
    ) -> None:
        new_state = state.value if isinstance(state, SessionState) else str(state)
        old_state = str(getattr(self, "_session_state", SessionState.STOPPED.value) or "")
        if old_state == new_state:
            return
        self._session_state = new_state
        safe_transition(
            getattr(self, "_state_observer", None),
            StateTransition(
                axis=StateAxis.SESSION,
                old_state=old_state,
                new_state=new_state,
                trigger=trigger,
                reason=reason,
                fields={"session_id": getattr(self, "_session_id", "") or ""},
            ),
        )

    def _set_transcription_state(
        self,
        turn: QueuedTurn | None,
        state: TranscriptionState | str,
        *,
        trigger: str,
        item_id: str = "",
        reason: str = "",
    ) -> None:
        if turn is None:
            return
        new_state = state.value if isinstance(state, TranscriptionState) else str(state)
        metadata = turn.metadata if isinstance(turn.metadata, dict) else {}
        old_state = str(
            metadata.get("_transcription_state", TranscriptionState.NONE.value) or ""
        )
        if old_state == new_state:
            return
        metadata["_transcription_state"] = new_state
        turn.metadata = metadata
        safe_transition(
            getattr(self, "_state_observer", None),
            StateTransition(
                axis=StateAxis.TRANSCRIPTION,
                old_state=old_state,
                new_state=new_state,
                trigger=trigger,
                req_id=turn.req_id,
                reason=reason,
                fields={"item_id": item_id},
            ),
        )

    # ------------------------------------------------------------------
    # Voice command runtime
    # ------------------------------------------------------------------

    def _voice_command_controller(self) -> VoiceCommandRuntime:
        runtime = getattr(self, "_voice_command_runtime", None)
        if runtime is None or runtime._host is not self:
            runtime = VoiceCommandRuntime(self)
            self._voice_command_runtime = runtime
        return runtime

    def note_local_voice_command(self, command: str, *, ttl_sec: float = 1.5) -> None:
        self._voice_command_controller().note_local_voice_command(
            command,
            ttl_sec=ttl_sec,
        )

    def _should_ignore_voice_command(self, command: str) -> bool:
        return self._voice_command_controller().should_ignore(command)

    def _on_voice_command(self, msg: Any) -> None:
        self._voice_command_controller().handle_message(msg)

    def handle_voice_command(self, command: str) -> None:
        """Handle a local or bridge-provided voice command."""
        self._on_voice_command(command)

    # ------------------------------------------------------------------
    # Display controller
    # ------------------------------------------------------------------

    def _display_controller_runtime(self) -> DisplayController:
        controller = getattr(self, "_display_controller", None)
        if controller is None or controller._host is not self:
            controller = DisplayController(self)
            self._display_controller = controller
        return controller

    def _set_display_mode_async(self, mode: str, *, force: bool = False) -> None:
        self._display_controller_runtime().set_mode_async(mode, force=force)

    def _clear_passive_alert_display_if_needed(self) -> None:
        self._display_controller_runtime().clear_passive_alert_if_needed()

    def _show_display_subtitle_async(self, text: str, *, duration_ms: int = 5000) -> None:
        self._display_controller_runtime().show_subtitle_async(
            text,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _display_subtitle_window(text: str) -> str:
        return DisplayController.subtitle_window(text)

    def _display_worker_loop(self) -> None:
        self._display_controller_runtime().worker_loop()

    @staticmethod
    def _apply_display_mode(display: Any, mode: str) -> None:
        DisplayController.apply_mode(display, mode)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect audio, websocket, robot transport, and worker threads."""
        if self._ws is not None:
            return
        self._set_session_state(SessionState.CONNECTING, trigger="start")

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            self._set_session_state(
                SessionState.STOPPED,
                trigger="start_failed",
                reason="missing_api_key",
            )
            raise RuntimeError("OPENAI_API_KEY must be set for realtime speech runtime.")

        headers = realtime_auth_headers(api_key)
        url = realtime_websocket_url(self.realtime_profile.model)
        self._ws = websocket.create_connection(url, header=headers, timeout=30)

        starter = getattr(self.robot_client, "start", None)
        if callable(starter):
            starter()
        display_starter = getattr(self.display_runtime, "start", None)
        if callable(display_starter):
            display_starter()
        self._start_websocket_threads()
        self._set_session_state(SessionState.CONFIGURING, trigger="session_configure")
        self._configure_session()
        self._start_audio_streams()
        self._start_workers()
        self._set_display_mode_async("idle")

        atexit.register(self.shutdown)

    def wait_until_shutdown(self) -> None:
        """Block until the runtime is asked to stop."""
        while not self._stop_event.wait(0.5):
            continue

    def shutdown(self) -> None:
        """Stop playback/capture, websocket threads, robot transport, and workers."""
        if self._stop_event.is_set():
            return
        self._set_session_state(SessionState.SHUTTING_DOWN, trigger="shutdown")
        self._stop_event.set()

        self._cancel_preference_idle_flush()
        self.flush_preference_segments(reason="shutdown")
        self._playback_buffer.clear()
        if self.gesture_runtime is not None:
            try:
                self.gesture_runtime.set_recording_active(False)
            except Exception:
                self.logger.exception("Failed to clear gesture recording state")
            try:
                self.gesture_runtime.shutdown()
            except Exception:
                self.logger.exception("Failed to stop gesture runtime cleanly")
        if self.display_runtime is not None:
            try:
                self.display_runtime.shutdown()
            except Exception:
                self.logger.exception("Failed to stop display runtime cleanly")
        if self.owner_turn_controller is not None:
            try:
                self.owner_turn_controller.shutdown()
            except Exception:
                self.logger.exception("Failed to stop owner turn controller cleanly")

        if self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
        if self._output_stream is not None:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None

        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

        for thread in (
            self._sender_thread,
            self._receiver_thread,
            self._response_thread,
            self._tool_thread,
            self._watchdog_thread,
            self._display_thread,
        ):
            if thread is not None:
                thread.join(timeout=2.0)

        if self.face_service is not None:
            try:
                self.face_service.shutdown()
            except Exception:
                self.logger.exception("Failed to stop face service cleanly")
        if self.speaker_service is not None:
            try:
                self.speaker_service.shutdown()
            except Exception:
                self.logger.exception("Failed to stop speaker service cleanly")
        if getattr(self, "adaptive_update_coordinator", None) is not None:
            try:
                self.adaptive_update_coordinator.close()
            except Exception:
                self.logger.exception("Failed to stop adaptive biometric updates cleanly")
        if getattr(self, "memory_context_compiler", None) is not None:
            try:
                close_memory = getattr(self.memory_context_compiler, "close", None)
                if callable(close_memory):
                    close_memory()
            except Exception:
                self.logger.exception("Failed to stop memory provider cleanly")
        if getattr(self, "battery_cache", None) is not None:
            try:
                shutdown_battery = getattr(self.battery_cache, "shutdown", None)
                if callable(shutdown_battery):
                    shutdown_battery()
            except Exception:
                self.logger.exception("Failed to stop battery cache cleanly")
        if getattr(self, "raw_data_capture", None) is not None:
            try:
                self.raw_data_capture.close()
            except Exception:
                self.logger.exception("Failed to close raw data capture cleanly")

        self.engagement.shutdown()
        shutdown_robot = getattr(self.robot_client, "shutdown", None)
        if callable(shutdown_robot):
            try:
                shutdown_robot()
            except Exception:
                self.logger.exception("Failed to stop robot client cleanly")
        self._preference_executor.shutdown(wait=False, cancel_futures=False)
        self._set_session_state(SessionState.STOPPED, trigger="shutdown_complete")

    # ------------------------------------------------------------------
    # Public hooks expected by orchestration code
    # ------------------------------------------------------------------

    def enqueue_internal_event(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Queue a text-only internal or coalesced turn for the response worker."""
        meta = dict(metadata or {})
        req_id = str(meta.get("req_id") or f"evt-{uuid4().hex[:12]}")
        if not meta.get("exchange_id"):
            meta["exchange_id"] = req_id
        if not meta.get("turn_kind"):
            meta["turn_kind"] = "internal_text" if bool(meta.get("internal", False)) else "human_text"
        context = self._capture_turn_context()
        resolution = None
        if not bool(meta.get("internal", False)):
            resolution = self._face_owner_resolution(
                primary_face_person_id=context.primary_face_person_id,
            )
            context = self._capture_turn_context(
                primary_face_person_id=context.primary_face_person_id,
                audio_speaker_id=resolution.audio_speaker_id,
                owner_id=resolution.owner_id,
                owner_source=resolution.owner_source,
                owner_confidence=resolution.owner_confidence,
                speaker_visible=resolution.speaker_visible,
            )
        turn = QueuedTurn(
            kind="text",
            req_id=req_id,
            speech_end_perf_s=float(meta.get("speech_end_perf_s") or 0.0),
            speech_end_unix_s=float(meta.get("speech_end_unix_s") or time.time()),
            transcript_perf_s=float(meta.get("transcript_perf_s") or perf_now()),
            source_is_internal=bool(meta.get("internal", False)),
            exchange_id=str(meta.get("exchange_id") or ""),
            exchange_index=int(meta.get("exchange_index") or 0),
            input_text=text,
            user_transcript=(
                "" if bool(meta.get("internal", False)) else _human_text_from_text_turn(text)
            ),
            metadata=meta,
            primary_face_person_id=context.primary_face_person_id,
            audio_speaker_id=resolution.audio_speaker_id if resolution is not None else None,
            owner_id=resolution.owner_id if resolution is not None else None,
            owner_source=(
                resolution.owner_source if resolution is not None else context.owner_source
            ),
            owner_confidence=(
                resolution.owner_confidence if resolution is not None else context.owner_confidence
            ),
            speaker_visible=(
                resolution.speaker_visible if resolution is not None else context.speaker_visible
            ),
            context_snapshot=context,
        )
        self._set_turn_phase(turn, TURN_PHASE_QUEUED, trigger="enqueue_text_turn")
        self._turn_queue.put(turn)

    def update_face_presence_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Update local mic admission from a face-presence snapshot."""
        self._face_gate.update_from_snapshot(dict(snapshot or {}))

    def _face_owner_resolution(
        self,
        *,
        primary_face_person_id: str | None,
        visible_face_person_ids: tuple[str, ...] | list[str] | None = None,
    ) -> SpeakerResolutionResult:
        if self.speaker_service is not None:
            return resolve_owner_id(
                policy=self.speaker_service.policy,
                primary_face_person_id=primary_face_person_id,
                audio_speaker_id=None,
                top_score=0.0,
                runner_up_score=0.0,
                visible_face_person_ids=visible_face_person_ids,
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

    def _apply_speaker_resolution(
        self,
        turn: QueuedTurn,
        result: SpeakerResolutionResult,
        *,
        update_prompt_context: bool = False,
    ) -> None:
        turn.audio_speaker_id = result.audio_speaker_id
        turn.owner_id = result.owner_id
        turn.owner_source = result.owner_source
        turn.owner_confidence = result.owner_confidence
        turn.speaker_visible = result.speaker_visible
        context = turn.context_snapshot
        context.audio_speaker_id = result.audio_speaker_id
        context.owner_id = result.owner_id
        context.owner_source = result.owner_source
        context.owner_confidence = result.owner_confidence
        context.speaker_visible = result.speaker_visible
        if update_prompt_context:
            turn.context_snapshot = context

    def _log_speaker_resolution(
        self,
        *,
        req_id: str,
        phase: str,
        primary_face_person_id: str | None,
        result: SpeakerResolutionResult,
        previous_owner_id: str | None = None,
    ) -> None:
        details = (
            "Speaker resolution %s req_id=%s primary_face_person_id=%s audio_speaker_id=%s "
            "owner_id=%s source=%s audio_score=%.3f"
        )
        args: list[object] = [
            phase,
            req_id,
            primary_face_person_id,
            result.audio_speaker_id,
            result.owner_id,
            result.owner_source,
            result.top_score,
        ]
        if previous_owner_id is not None and previous_owner_id != result.owner_id:
            details += " previous_owner_id=%s"
            args.append(previous_owner_id)
        self.logger.info(details, *args)

    def _arm_pending_voice_enrollment(self, person_id: str) -> None:
        if self.speaker_service is None:
            return
        rendered = str(person_id or "").strip()
        if not rendered:
            return
        with self._voice_enrollment_lock:
            self._pending_voice_enrollments[rendered] = PendingVoiceEnrollment(
                person_id=rendered
            )
        self.logger.info("Voice enrollment armed person_id=%s", rendered)

    def _consume_voice_enrollment_prompt_note(self, turn: QueuedTurn) -> str:
        if self.speaker_service is None:
            return ""
        primary_face_person_id = str(turn.primary_face_person_id or "").strip()
        if not primary_face_person_id:
            return ""
        if self.speaker_service.has_reference(primary_face_person_id):
            return ""
        with self._voice_enrollment_lock:
            pending = self._pending_voice_enrollments.get(primary_face_person_id)
            if pending is None or not pending.explicit_prompt_armed or pending.explicit_prompt_used:
                return ""
            pending.explicit_prompt_used = True
            pending.explicit_prompt_armed = False
        self.logger.info("Voice enrollment prompt requested person_id=%s", primary_face_person_id)
        return (
            "[VOICE ENROLLMENT] Ask the current user for one short natural phrase so you can "
            "remember their voice too. Explain briefly that you store mathematical face and voice "
            "embeddings, not raw recordings."
        )

    def _resolve_voice_enrollment_target(
        self,
        turn: QueuedTurn,
    ) -> tuple[str | None, dict[str, Any], tuple[str, ...]]:
        context = turn.context_snapshot
        frozen_snapshot = dict(context.face_snapshot or {})
        with self._voice_enrollment_lock:
            pending_targets = tuple(self._pending_voice_enrollments.keys())

        primary_face_person_id = str(turn.primary_face_person_id or context.primary_face_person_id or "").strip()
        if primary_face_person_id:
            return primary_face_person_id, frozen_snapshot, pending_targets

        if self.face_service is None:
            return None, frozen_snapshot, pending_targets

        try:
            live_snapshot = dict(self.face_service.get_presence_snapshot() or {})
            live_persons = list(self.face_service.get_cached_persons() or [])
        except Exception:
            self.logger.exception("Failed to inspect live face cache for voice enrollment")
            return None, frozen_snapshot, pending_targets

        recognized_count = int(live_snapshot.get("recognized_count", 0) or 0)
        unknown_count = int(live_snapshot.get("unknown_count", 0) or 0)
        candidate_ids = [
            str(getattr(person, "person_id", "") or "").strip()
            for person in live_persons
            if str(getattr(person, "person_id", "") or "").strip()
        ]
        unique_candidate_ids: list[str] = []
        for candidate_id in candidate_ids:
            if candidate_id not in unique_candidate_ids:
                unique_candidate_ids.append(candidate_id)
        if (
            recognized_count == 1
            and unknown_count == 0
            and len(unique_candidate_ids) == 1
        ):
            candidate_id = unique_candidate_ids[0]
            if not pending_targets or candidate_id in pending_targets:
                self.logger.info(
                    "Voice enrollment recovered person_id=%s from live face cache",
                    candidate_id,
                )
                return candidate_id, live_snapshot, pending_targets
        return None, live_snapshot or frozen_snapshot, pending_targets

    def _maybe_capture_voice_reference(self, turn: QueuedTurn) -> None:
        if self.speaker_service is None or turn.source_is_internal or turn.interrupted:
            return
        primary_face_person_id, snapshot, pending_targets = self._resolve_voice_enrollment_target(turn)
        if not primary_face_person_id:
            if pending_targets:
                self.logger.info(
                    "Voice enrollment deferred reason=no_primary_face pending_targets=%s",
                    ",".join(pending_targets),
                )
            return
        if self.speaker_service.has_reference(primary_face_person_id):
            with self._voice_enrollment_lock:
                self._pending_voice_enrollments.pop(primary_face_person_id, None)
            return
        is_pending_target = primary_face_person_id in pending_targets
        recognized_count = int(snapshot.get("recognized_count", 0) or 0)
        unknown_count = int(snapshot.get("unknown_count", 0) or 0)
        if recognized_count != 1 or unknown_count != 0:
            if is_pending_target:
                self.logger.info(
                    "Voice enrollment deferred person_id=%s reason=scene_not_clean recognized_count=%s unknown_count=%s",
                    primary_face_person_id,
                    recognized_count,
                    unknown_count,
                )
            return
        audio_bytes = turn.trimmed_input_audio_pcm16
        if audio_bytes is None:
            audio_bytes = turn.input_audio_pcm16
            if not audio_bytes:
                if is_pending_target:
                    self.logger.info(
                        "Voice enrollment deferred person_id=%s reason=no_audio_clip",
                        primary_face_person_id,
                    )
                return
        if audio_bytes is None:
            if is_pending_target:
                self.logger.info(
                    "Voice enrollment deferred person_id=%s reason=no_audio_clip",
                    primary_face_person_id,
                )
            return

        with self._voice_enrollment_lock:
            pending = self._pending_voice_enrollments.get(primary_face_person_id)
            attempt_kind = (
                "fallback"
                if pending is not None and pending.explicit_prompt_used
                else "silent"
            )
        duration_s = len(audio_bytes) / float(2 * 16000)
        speaker_audio_debug = dict((turn.metadata or {}).get("speaker_audio_debug") or {})
        self.logger.info(
            "Voice enrollment attempting person_id=%s attempt_kind=%s duration_s=%.2f",
            primary_face_person_id,
            attempt_kind,
            duration_s,
        )
        if speaker_audio_debug:
            self.logger.info(
                "Voice enrollment audio stats person_id=%s raw_duration_s=%.3f "
                "trimmed_duration_s=%.3f kept_ratio=%.4f raw_rms=%.1f trimmed_rms=%.1f "
                "capture_vad_positive_blocks=%s vad_window_samples=%s",
                primary_face_person_id,
                float(speaker_audio_debug.get("raw_duration_s", 0.0) or 0.0),
                float(speaker_audio_debug.get("trimmed_duration_s", 0.0) or 0.0),
                float(speaker_audio_debug.get("kept_ratio", 0.0) or 0.0),
                float(speaker_audio_debug.get("raw_rms_level", 0.0) or 0.0),
                float(speaker_audio_debug.get("trimmed_rms_level", 0.0) or 0.0),
                int(speaker_audio_debug.get("capture_vad_positive_blocks", 0) or 0),
                int(speaker_audio_debug.get("vad_window_samples", 0) or 0),
            )

        result = self.speaker_service.try_store_reference(
            person_id=primary_face_person_id,
            audio_pcm16=audio_bytes,
            attempt_kind=attempt_kind,
        )
        if result.saved:
            with self._voice_enrollment_lock:
                self._pending_voice_enrollments.pop(primary_face_person_id, None)
            self.logger.info(
                "Voice enrollment saved person_id=%s attempt_kind=%s",
                primary_face_person_id,
                attempt_kind,
            )
            return
        if result.reason.startswith("reject_"):
            self.logger.info(
                "Voice enrollment skipped person_id=%s attempt_kind=%s reason=%s",
                primary_face_person_id,
                attempt_kind,
                result.reason,
            )
            with self._voice_enrollment_lock:
                pending = self._pending_voice_enrollments.get(primary_face_person_id)
                if pending is None:
                    return
                pending.silent_failures += 1
                if (
                    not pending.explicit_prompt_used
                    and pending.silent_failures
                    >= self.speaker_service.policy.explicit_prompt_after_silent_failures
                ):
                    pending.explicit_prompt_armed = True
                    self.logger.info(
                        "Voice enrollment fallback prompt armed person_id=%s failures=%s",
                        primary_face_person_id,
                        pending.silent_failures,
                    )

    def seconds_since_external_input(self) -> float:
        """Report time since the last non-internal human turn was committed."""
        if self._last_external_input_s <= 0.0:
            return 1e9
        return max(0.0, time.time() - self._last_external_input_s)

    def is_recording_active(self) -> bool:
        """Return True while local speech capture is actively buffering audio."""
        with self._recording_lock:
            return bool(self._recording_active)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _start_websocket_threads(self) -> None:
        self._receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._sender_thread = threading.Thread(target=self._audio_sender_loop, daemon=True)
        self._receiver_thread.start()
        self._sender_thread.start()

    def _start_workers(self) -> None:
        self._response_thread = threading.Thread(target=self._response_loop, daemon=True)
        self._tool_thread = threading.Thread(target=self._tool_loop, daemon=True)
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._response_thread.start()
        self._tool_thread.start()
        self._watchdog_thread.start()
        if getattr(self, "display_runtime", None) is not None:
            self._display_thread = threading.Thread(
                target=self._display_worker_loop,
                daemon=True,
            )
            self._display_thread.start()

    def _configure_session(self) -> None:
        session = realtime_audio_session_payload(
            profile=self.realtime_profile,
            instructions=self.base_system_prompt,
            tools=self._tool_schemas,
        )
        self._send_event({"type": "session.update", "session": session})

    def _build_response_instruction_snapshot(self, turn: QueuedTurn) -> dict[str, str]:
        dynamic_instructions = self._build_turn_instructions(turn)
        static_instructions = str(self.base_system_prompt or "").strip()
        dynamic_instructions = str(dynamic_instructions or "").strip()
        instruction_parts = [
            part for part in (static_instructions, dynamic_instructions) if part
        ]
        instructions = "\n\n".join(instruction_parts)
        delivery_instructions = ""
        if turn.no_audio_retry_count > 0:
            delivery_instructions = (
                "[DELIVERY] Respond with spoken audio for this turn. Do not return a silent text-only reply."
            )
        if turn.incomplete_audio_continuation_count > 0:
            delivery_instructions = "\n\n".join(
                part
                for part in (
                    delivery_instructions,
                    "[DELIVERY] Continue the current answer naturally from exactly where you left off. Do not restart, repeat, or summarize what you already said.",
                )
                if part
            )
        if delivery_instructions:
            instructions = "\n\n".join(
                part for part in (instructions, delivery_instructions) if part
            )
        return {
            "instructions": instructions,
            "static_instructions": static_instructions,
            "dynamic_instructions": dynamic_instructions,
            "delivery_instructions": delivery_instructions,
        }

    def _build_response_request(self, turn: QueuedTurn) -> dict[str, Any]:
        instruction_snapshot = self._build_response_instruction_snapshot(turn)
        return realtime_response_payload(
            instructions=instruction_snapshot["instructions"],
            output_modalities=["audio"],
            max_output_tokens=self.realtime_profile.max_output_tokens,
        )

    def _model_prompt_log_fields(
        self,
        turn: QueuedTurn,
        instruction_snapshot: dict[str, str],
    ) -> dict[str, Any]:
        history_item_ids = list(getattr(self, "_history_item_order", ()) or ())
        turn_history_item_ids = sorted(str(item_id) for item_id in turn.history_item_ids)
        instructions = instruction_snapshot.get("instructions", "")
        static_instructions = instruction_snapshot.get("static_instructions", "")
        dynamic_instructions = instruction_snapshot.get("dynamic_instructions", "")
        delivery_instructions = instruction_snapshot.get("delivery_instructions", "")
        fields: dict[str, Any] = {
            "model_prompt_b64": _log_text_b64(instructions),
            "model_prompt_chars": len(instructions),
            "model_static_prompt_chars": len(static_instructions),
            "model_dynamic_context_b64": _log_text_b64(dynamic_instructions),
            "model_dynamic_context_chars": len(dynamic_instructions),
            "model_history_owner_key": self._history_owner_key_for_turn(turn),
            "model_history_item_count": len(history_item_ids),
            "model_turn_history_item_count": len(turn_history_item_ids),
        }
        if delivery_instructions:
            fields["model_delivery_instructions_b64"] = _log_text_b64(
                delivery_instructions
            )
            fields["model_delivery_instructions_chars"] = len(delivery_instructions)
        if history_item_ids:
            fields["model_history_item_ids"] = ",".join(history_item_ids[-20:])
        if turn_history_item_ids:
            fields["model_turn_history_item_ids"] = ",".join(turn_history_item_ids)
        return fields

    def _send_response_create(self, turn: QueuedTurn) -> None:
        if self._is_turn_terminal(turn):
            return
        if not self._wait_for_stale_response_slot():
            self._terminate_turn(turn, TURN_PHASE_CANCELED, "stale_response_wait_aborted")
            return
        turn.response_requested_at = time.time()
        turn.pending_response_requests += 1
        self._set_turn_phase(
            turn,
            TURN_PHASE_RESPONSE_REQUESTED,
            trigger=(
                "response_followup_create"
                if turn.phase == TURN_PHASE_REQUESTING_FOLLOWUP
                else "response_create"
            ),
        )
        self._queue_pending_response_turn(turn.req_id)
        instruction_snapshot = self._build_response_instruction_snapshot(turn)
        response_request = realtime_response_payload(
            instructions=instruction_snapshot["instructions"],
            output_modalities=["audio"],
            max_output_tokens=self.realtime_profile.max_output_tokens,
        )
        self._latency.emit(
            event="response_create",
            req_id=turn.req_id,
            **self._model_prompt_log_fields(turn, instruction_snapshot),
            **self._exchange_log_fields(turn),
        )
        self.logger.info(
            "Queueing response.create req_id=%s pending_response_requests=%s",
            turn.req_id,
            turn.pending_response_requests,
        )
        self._send_event(
            {
                "type": "response.create",
                "response": response_request,
            }
        )

    # ------------------------------------------------------------------
    # Worker loops
    # ------------------------------------------------------------------

    def _response_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                turn = self._turn_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._run_turn(turn)
            except Exception as exc:
                self.logger.exception("Turn failed req_id=%s", turn.req_id)
                metadata = turn.metadata if isinstance(turn.metadata, dict) else {}
                metadata["error_source"] = "runtime"
                metadata["error_type"] = type(exc).__name__
                metadata["error_message"] = str(exc) or type(exc).__name__
                turn.metadata = metadata
                self._terminate_turn(turn, TURN_PHASE_CANCELED, "turn_exception")
            finally:
                self._turn_queue.task_done()

    def _cancel_owner_turn_for_tool(self, turn: QueuedTurn, tool_name: str) -> None:
        owner_turn_controller = getattr(self, "owner_turn_controller", None)
        if owner_turn_controller is None:
            return
        cancel = getattr(owner_turn_controller, "cancel_request", None)
        if not callable(cancel):
            return
        try:
            cancel(req_id=turn.req_id, reason=f"tool:{tool_name}")
        except Exception:
            self.logger.exception(
                "Failed to cancel owner turn for tool req_id=%s tool=%s",
                turn.req_id,
                tool_name,
            )

    def _tool_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                pending = self._tool_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._execute_tool_call(pending)
            except Exception:
                self.logger.exception(
                    "Tool execution failed req_id=%s tool=%s",
                    pending.turn_req_id,
                    pending.tool_name,
                )
            finally:
                self._tool_queue.task_done()

    def _watchdog_loop(self) -> None:
        self._turn_watchdog().loop(
            poll_s=WATCHDOG_POLL_SEC,
            response_timeout_s=RESPONSE_STALL_TIMEOUT_SEC,
            playback_timeout_s=PLAYBACK_STALL_TIMEOUT_SEC,
        )

    # ------------------------------------------------------------------
    # Turn handling
    # ------------------------------------------------------------------

    def _run_turn(self, turn: QueuedTurn) -> None:
        self._turn_runner_controller().run(turn)

    def _wait_for_turn_settled(self, turn: QueuedTurn) -> None:
        self._turn_runner_controller().wait_for_settled(turn)

    def _bump_session_estimated_cost(self, amount_usd: Optional[float]) -> Optional[float]:
        if amount_usd is None:
            return None
        self._session_estimated_cost_usd = round(
            float(getattr(self, "_session_estimated_cost_usd", 0.0)) + float(amount_usd),
            8,
        )
        return self._session_estimated_cost_usd

    def _emit_response_usage(self, turn: QueuedTurn, response: dict[str, Any]) -> None:
        usage = response.get("usage", {}) or {}
        if not isinstance(usage, dict) or not usage:
            return
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        input_details = usage.get("input_token_details", {}) or {}
        cached_tokens = input_details.get("cached_tokens", 0)
        cached_details = input_details.get("cached_tokens_details", {}) or {}

        try:
            cached_tokens_int = int(cached_tokens or 0)
        except Exception:
            cached_tokens_int = 0
        try:
            input_tokens_int = int(input_tokens) if input_tokens is not None else None
        except Exception:
            input_tokens_int = None

        uncached_input_tokens = None
        cache_hit_ratio = None
        if input_tokens_int is not None:
            uncached_input_tokens = max(0, input_tokens_int - cached_tokens_int)
            if input_tokens_int > 0:
                cache_hit_ratio = cached_tokens_int / float(input_tokens_int)
        cost_fields = estimate_realtime_response_cost(
            usage,
            model_name=self.realtime_profile.model,
        )
        session_total_cost_usd = self._bump_session_estimated_cost(
            cost_fields.get("estimated_cost_usd")
        )

        self._latency.emit(
            event="response_usage",
            req_id=turn.req_id,
            session_id=getattr(self, "_session_id", "") or None,
            **{
                key: value
                for key, value in self._exchange_log_fields(turn).items()
                if key != "session_id"
            },
            response_id=response.get("id"),
            model=self.realtime_profile.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens_int,
            uncached_input_tokens=uncached_input_tokens,
            cache_hit_ratio=cache_hit_ratio,
            input_text_tokens=cost_fields.get("input_text_tokens"),
            input_audio_tokens=cost_fields.get("input_audio_tokens"),
            input_image_tokens=cost_fields.get("input_image_tokens"),
            output_text_tokens=cost_fields.get("output_text_tokens"),
            output_audio_tokens=cost_fields.get("output_audio_tokens"),
            uncached_input_text_tokens=cost_fields.get("uncached_input_text_tokens"),
            uncached_input_audio_tokens=cost_fields.get("uncached_input_audio_tokens"),
            uncached_input_image_tokens=cost_fields.get("uncached_input_image_tokens"),
            cached_audio_tokens=cached_details.get("audio_tokens"),
            cached_text_tokens=cached_details.get("text_tokens"),
            cached_image_tokens=cached_details.get("image_tokens"),
            estimated_cost_usd=cost_fields.get("estimated_cost_usd"),
            estimated_cached_savings_usd=cost_fields.get("estimated_cached_savings_usd"),
            session_total_cost_usd=session_total_cost_usd,
        )

    def _build_turn_instructions(self, turn: QueuedTurn) -> str:
        blocks: list[str] = []
        context = turn.context_snapshot
        persons = list(context.persons or [])
        face_snapshot = dict(context.face_snapshot or {}) if context.face_snapshot else None
        if str(context.owner_id or "").strip():
            people_context = format_people_context(
                persons,
                primary_face_person_id=context.primary_face_person_id,
                face_snapshot=face_snapshot,
                audio_speaker_id=context.audio_speaker_id,
                owner_id=context.owner_id,
                owner_source=context.owner_source,
                speaker_visible=context.speaker_visible,
            )
            if people_context:
                blocks.append(people_context)
        voice_prompt = self._consume_voice_enrollment_prompt_note(turn)
        if voice_prompt:
            blocks.append(voice_prompt)
        blocks.append(format_current_time_block())
        office_block = format_current_office_location_block(self._current_office_location)
        if office_block:
            blocks.append(office_block)
        for memory_block in tuple(getattr(context, "memory_context_blocks", ()) or ()):
            if str(memory_block or "").strip():
                blocks.append(str(memory_block).strip())
        blocks.append(
            format_robot_state_block(
                posture=self._robot_posture,
                last_tool_name=self._last_tool_name,
                last_tool_summary=self._last_tool_summary,
                stand_tool_name=self._stand_tool_name,
                supports_navigation=self._supports_navigation,
            )
        )
        if self.battery_cache is not None:
            try:
                blocks.append(self.battery_cache.format_prompt_block())
            except Exception:
                self.logger.exception("Failed to format battery prompt block")
        if self.location_store is not None:
            try:
                blocks.append(format_saved_locations(self.location_store))
            except Exception:
                self.logger.exception("Failed to format saved locations block")
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Websocket receiver
    # ------------------------------------------------------------------

    def _server_event_adapter(self) -> RealtimeEventAdapter:
        adapter = getattr(self, "_event_adapter", None)
        if adapter is None or adapter._host is not self:
            adapter = RealtimeEventAdapter(self)
            self._event_adapter = adapter
        return adapter

    def _receiver_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                message = self._ws.recv() if self._ws is not None else None
            except websocket.WebSocketConnectionClosedException:
                if not self._stop_event.is_set():
                    self.logger.warning("Realtime websocket closed during receive; stopping runtime")
                    self._stop_event.set()
                return
            except Exception:
                if not self._stop_event.is_set():
                    self.logger.exception("Realtime websocket receive failed")
                    self._stop_event.set()
                return

            if not message:
                continue

            try:
                event = json.loads(message)
            except Exception:
                self.logger.warning("Ignoring non-JSON realtime event")
                continue
            self._handle_server_event(event)

    def _handle_server_event(self, event: dict[str, Any]) -> None:
        self._server_event_adapter().handle(event)

    def _server_event_runtime_controller(self) -> ServerEventRuntime:
        runtime = getattr(self, "_server_event_runtime", None)
        if runtime is None or runtime._host is not self:
            runtime = ServerEventRuntime(self)
            self._server_event_runtime = runtime
        return runtime

    def _handle_conversation_item_created(self, event: dict[str, Any]) -> None:
        self._server_event_runtime_controller().handle_conversation_item_created(event)

    def _handle_input_audio_buffer_committed(self, event: dict[str, Any]) -> None:
        self._server_event_runtime_controller().handle_input_audio_buffer_committed(event)

    def _handle_response_created(self, event: dict[str, Any]) -> None:
        self._server_event_runtime_controller().handle_response_created(event)

    def _recover_pending_response_after_expired_stale(self, response_id: str) -> None:
        self._server_event_runtime_controller().recover_pending_response_after_expired_stale(
            response_id
        )

    def _handle_input_transcription_completed(self, event: dict[str, Any]) -> None:
        self._server_event_runtime_controller().handle_input_transcription_completed(event)

    def _handle_input_transcription_failed(self, event: dict[str, Any]) -> None:
        self._server_event_runtime_controller().handle_input_transcription_failed(event)

    def _handle_output_audio_delta(self, event: dict[str, Any]) -> None:
        self._server_event_runtime_controller().handle_output_audio_delta(event)

    def _handle_output_transcript_delta(self, event: dict[str, Any]) -> None:
        self._server_event_runtime_controller().handle_output_transcript_delta(event)

    def _handle_output_text_delta(self, event: dict[str, Any]) -> None:
        self._server_event_runtime_controller().handle_output_text_delta(event)

    def _arm_playback_completion(self, turn: QueuedTurn) -> None:
        self._server_event_runtime_controller().arm_playback_completion(turn)

    def _handle_output_item_done(self, event: dict[str, Any]) -> None:
        self._server_event_runtime_controller().handle_output_item_done(event)

    def _handle_function_call_delta(self, event: dict[str, Any]) -> None:
        self._server_event_runtime_controller().handle_function_call_delta(event)

    def _handle_function_call_done(self, event: dict[str, Any]) -> None:
        self._server_event_runtime_controller().handle_function_call_done(event)

    def _handle_response_done(self, event: dict[str, Any]) -> None:
        self._server_event_runtime_controller().handle_response_done(event)

    def _handle_server_error(self, event: dict[str, Any]) -> None:
        self._server_event_runtime_controller().handle_server_error(event)

    def _supersede_unanswered_turn(self, new_turn: QueuedTurn) -> None:
        with self._turn_lock:
            candidate = self._active_turn
        if (
            candidate is None
            or candidate is new_turn
            or self._is_turn_terminal(candidate)
            or candidate.audio_started
        ):
            return
        self.logger.info(
            "Superseding unresolved turn old_req_id=%s new_req_id=%s",
            candidate.req_id,
            new_turn.req_id,
        )
        self._terminate_turn(
            candidate,
            TURN_PHASE_SUPERSEDED,
            "superseded_by_new_human_turn",
            send_cancel=True,
        )

    def _complete_turn_success(self, turn: QueuedTurn) -> None:
        if self._is_turn_terminal(turn):
            return
        turn.finalized = True
        turn.finalized_reason = "completed"
        self._latency.emit(
            event="exchange_complete",
            req_id=turn.req_id,
            terminal_status="complete",
            terminal_reason="completed",
            **self._exchange_log_fields(turn),
        )
        self._set_turn_phase(turn, TURN_PHASE_FINALIZED, trigger="turn_completed")
        turn.response_finished.set()
        turn.playback_finished.set()
        self._discard_pending_response_turn(turn.req_id)
        with self._turn_lock:
            if self._playback_req_id == turn.req_id:
                self._clear_playback_tracking_locked()

    def _terminate_turn(
        self,
        turn: QueuedTurn,
        phase: str,
        reason: str,
        *,
        send_cancel: bool = True,
        clear_playback: bool = False,
        truncate_playback: bool = False,
    ) -> None:
        if self._is_turn_terminal(turn):
            return
        turn.interrupted = turn.interrupted or truncate_playback
        turn.finalized = True
        turn.finalized_reason = reason
        if phase == TURN_PHASE_CANCELED:
            self._ensure_terminal_error_metadata(turn, reason)
        self._latency.emit(
            event="exchange_terminal",
            req_id=turn.req_id,
            terminal_status="error" if phase == TURN_PHASE_CANCELED else phase,
            terminal_reason=reason,
            **self._exchange_log_fields(turn),
        )
        self._set_turn_phase(turn, phase, trigger=reason or phase)
        self.logger.warning(
            "Terminating turn req_id=%s phase=%s reason=%s response_id=%s audio_started=%s pending_tool_calls=%s pending_response_requests=%s",
            turn.req_id,
            phase,
            reason,
            turn.response_id,
            turn.audio_started,
            turn.pending_tool_calls,
            turn.pending_response_requests,
        )
        played_ms = 0
        if not self._mark_pending_response_turn_stale(turn.req_id):
            self._discard_pending_response_turn(turn.req_id)
        with self._turn_lock:
            if self._playback_req_id == turn.req_id:
                played_ms = int(
                    (1000.0 * self._played_output_frames)
                    / max(1, self.realtime_profile.output_sample_rate)
                )
                if clear_playback:
                    self._playback_buffer.clear()
                    self._clear_playback_tracking_locked()
        if send_cancel and turn.response_id:
            try:
                self._send_event({"type": "response.cancel", "response_id": turn.response_id})
            except Exception:
                self.logger.exception("Failed to cancel response req_id=%s", turn.req_id)
        if truncate_playback and turn.assistant_item_id:
            try:
                self._send_event(
                    {
                        "type": "conversation.item.truncate",
                        "item_id": turn.assistant_item_id,
                        "content_index": 0,
                        "audio_end_ms": played_ms,
                    }
                )
            except Exception:
                self.logger.exception("Failed to truncate assistant item req_id=%s", turn.req_id)
        self.engagement.on_agent_done(has_reply=turn.audio_started, req_id=turn.req_id)
        if turn.audio_started:
            self.engagement.on_playback_event(
                "playback_stopped",
                turn.req_id,
                stream_id=turn.response_id,
            )
            self._set_display_mode_async("idle")
        turn.response_finished.set()
        turn.playback_finished.set()

    @staticmethod
    def _ensure_terminal_error_metadata(turn: QueuedTurn, reason: str) -> None:
        metadata = turn.metadata if isinstance(turn.metadata, dict) else {}
        if not metadata.get("error_source"):
            metadata["error_source"] = "runtime"
        if not metadata.get("error_type"):
            metadata["error_type"] = str(reason or "canceled")
        if not metadata.get("error_message"):
            metadata["error_message"] = str(reason or metadata.get("error_type") or "canceled")
        turn.metadata = metadata

    def _clear_playback_tracking_locked(self) -> None:
        self._playback_req_id = ""
        self._playback_stream_id = ""
        self._playback_item_id = ""
        self._played_output_frames = 0
