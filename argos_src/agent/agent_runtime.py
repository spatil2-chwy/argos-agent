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

from argos_src.agent.orchestrator import EngagementStateMachine, EventCoalescer
from argos_src.agent.preference_segments import _PreferenceSegmentCoordinator
from argos_src.agent.agent_events.dispatch import dispatch_server_event
from argos_src.agent.agent_events.parsing import (
    server_event_item,
    server_event_item_id,
    server_event_response,
    server_event_response_id,
)
from argos_src.agent.agent_audio import (
    VAD_SAMPLE_RATE,
    RealtimeAgentAudioMixin,
)
from argos_src.agent.agent_playback import RealtimeAgentPlaybackMixin
from argos_src.agent.agent_preferences import RealtimeAgentPreferenceMixin
from argos_src.agent.agent_state import RealtimeAgentStateMixin
from argos_src.agent.agent_tools import RealtimeAgentToolsMixin, _debug_log_value
from argos_src.agent.realtime_turns import (
    NO_AUDIO_RESPONSE_RETRY_LIMIT,
    PLAYBACK_STALL_TIMEOUT_SEC,
    RESPONSE_STALL_TIMEOUT_SEC,
    TURN_PHASE_CANCELED,
    TURN_PHASE_FINALIZED,
    TURN_PHASE_PLAYING,
    TURN_PHASE_RESPONSE_REQUESTED,
    TURN_PHASE_SUPERSEDED,
    TURN_PHASE_WAITING_FIRST_AUDIO,
    TURN_PHASE_WAITING_TOOLS,
    WATCHDOG_POLL_SEC,
    FrozenTurnContext,
    PendingToolCall,
    PlaybackBuffer,
    QueuedTurn,
)
from argos_src.agent.runtime_context import (
    format_current_office_location_block,
    format_current_time_block,
    format_people_context,
    format_robot_state_block,
    format_saved_locations,
)
from argos_src.observability.observability import (
    LatencyLogger,
    clear_request_context,
    perf_now,
    set_request_context,
)
from argos_src.observability.pricing import (
    estimate_realtime_response_cost,
    estimate_transcription_cost,
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


def _human_text_from_text_turn(text: str) -> str:
    rendered = str(text or "").strip()
    marker = "[HUMAN INPUT]"
    if marker not in rendered:
        return rendered
    return rendered.split(marker, 1)[1].strip()


class RealtimeRobotAgent(
    RealtimeAgentAudioMixin,
    RealtimeAgentPlaybackMixin,
    RealtimeAgentToolsMixin,
    RealtimeAgentPreferenceMixin,
    RealtimeAgentStateMixin,
):
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
        employee_directory_service: Any = None,
        slack_memory_service: Any = None,
        identity_store: Any = None,
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
        self.employee_directory_service = employee_directory_service
        self.slack_memory_service = slack_memory_service
        self.identity_store = identity_store
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
            getattr(getattr(scenario_profile, "employee_directory", None), "site_code", "")
            or ""
        ).strip()

        self._latency = LatencyLogger("realtime")
        self._tool_latency = LatencyLogger("tool")
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

        self._stop_event = threading.Event()
        self._audio_send_queue: queue.Queue[bytes] = queue.Queue()
        self._turn_queue: queue.Queue[QueuedTurn] = queue.Queue()
        self._tool_queue: queue.Queue[PendingToolCall] = queue.Queue()
        self._playback_buffer = PlaybackBuffer()
        self._input_stream: Optional[Any] = None
        self._output_stream: Optional[Any] = None

        self._recording_lock = threading.RLock()
        self._recording_active = False
        self._recording_started_at = 0.0
        self._last_voice_at = 0.0
        self._current_primary_face_person_id: Optional[str] = None
        self._current_visible_face_person_ids: tuple[str, ...] = ()
        self._current_turn_audio_chunks: list[bytes] = []
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
        self._pending_audio_turn_req_ids: deque[str] = deque()
        self._pending_audio_item_ids: deque[str] = deque()
        self._pending_local_created_items: deque[Any] = deque()
        self._history_item_order: deque[str] = deque()
        self._known_history_item_ids: set[str] = set()
        self._history_item_owner_req_id: dict[str, str] = {}
        self._active_history_owner_key: str = ""
        self._playback_req_id: str = ""
        self._playback_stream_id: str = ""
        self._playback_item_id: str = ""
        self._played_output_frames = 0
        self._ignored_voice_commands: deque[tuple[str, float]] = deque()

        self._tool_registry = {str(getattr(tool, "name", "")).strip(): tool for tool in self.tools}
        self._tool_schemas = [
            self._build_tool_schema(tool)
            for tool in self.tools
            if str(getattr(tool, "name", "")).strip()
        ]
        self._session_id = ""
        self._session_estimated_cost_usd = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect audio, websocket, robot transport, and worker threads."""
        if self._ws is not None:
            return

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
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
        if getattr(self, "employee_directory_service", None) is not None:
            try:
                self.employee_directory_service.shutdown()
            except Exception:
                self.logger.exception("Failed to stop employee directory cleanly")
        if getattr(self, "slack_memory_service", None) is not None:
            try:
                self.slack_memory_service.shutdown()
            except Exception:
                self.logger.exception("Failed to stop Slack memory service cleanly")
        if getattr(self, "battery_cache", None) is not None:
            try:
                shutdown_battery = getattr(self.battery_cache, "shutdown", None)
                if callable(shutdown_battery):
                    shutdown_battery()
            except Exception:
                self.logger.exception("Failed to stop battery cache cleanly")

        self.engagement.shutdown()
        shutdown_robot = getattr(self.robot_client, "shutdown", None)
        if callable(shutdown_robot):
            try:
                shutdown_robot()
            except Exception:
                self.logger.exception("Failed to stop robot client cleanly")
        self._preference_executor.shutdown(wait=False, cancel_futures=False)

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
        self._turn_queue.put(turn)

    def update_face_presence_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Update local mic admission from a face-presence snapshot."""
        self._face_gate.update_from_snapshot(dict(snapshot or {}))

    def flush_preference_segments(self, reason: str = "idle") -> None:
        """Flush any buffered speaker-owned preference segment."""
        if self._preference_segments is None:
            return
        if reason == "idle":
            self._schedule_preference_idle_flush()
            return
        self._cancel_preference_idle_flush()
        self._retry_ready_preference_turns()
        completed_segment = self._preference_segments.flush_active()
        if completed_segment is None:
            return
        self._schedule_preference_segment_extraction(completed_segment, reason=reason)

    def _schedule_preference_idle_flush(self) -> None:
        if self._preference_segments is None:
            return

        def run_flush() -> None:
            with self._preference_idle_flush_lock:
                self._preference_idle_flush_timer = None
            self.flush_preference_segments(reason="idle_timeout")

        with self._preference_idle_flush_lock:
            if self._preference_idle_flush_timer is not None:
                self._preference_idle_flush_timer.cancel()
            timer = threading.Timer(self._preference_idle_flush_delay_sec, run_flush)
            timer.daemon = True
            self._preference_idle_flush_timer = timer
            timer.start()

    def _cancel_preference_idle_flush(self) -> None:
        with self._preference_idle_flush_lock:
            if self._preference_idle_flush_timer is not None:
                self._preference_idle_flush_timer.cancel()
                self._preference_idle_flush_timer = None

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

    def _set_display_mode_async(self, mode: str, *, force: bool = False) -> None:
        if getattr(self, "display_runtime", None) is None:
            return
        rendered = str(mode or "").strip()
        if not rendered:
            return
        with self._display_mode_lock:
            if not force and rendered == self._display_mode:
                return
            self._display_mode = rendered
        self._display_queue.put(("mode", rendered))

    def _clear_passive_alert_display_if_needed(self) -> None:
        if getattr(self, "display_runtime", None) is None:
            return
        with self._display_mode_lock:
            should_clear = self._display_mode == "alert"
        if should_clear:
            self._set_display_mode_async("idle")

    def _show_display_subtitle_async(self, text: str, *, duration_ms: int = 5000) -> None:
        if getattr(self, "display_runtime", None) is None:
            return
        rendered = str(text or "").strip()
        if not rendered:
            return
        self._display_queue.put(
            (
                "subtitle",
                {
                    "text": rendered,
                    "duration_ms": int(duration_ms),
                },
            )
        )

    @staticmethod
    def _display_subtitle_window(text: str, *, max_chars: int = 180) -> str:
        rendered = " ".join(str(text or "").split())
        if len(rendered) <= max_chars:
            return rendered
        trimmed = rendered[-max_chars:]
        if " " in trimmed:
            trimmed = trimmed.split(" ", 1)[1]
        return trimmed.strip()

    def _display_worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                kind, payload = self._display_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                display = getattr(self, "display_runtime", None)
                if display is None:
                    continue
                if kind == "mode":
                    self._apply_display_mode(display, str(payload or ""))
                elif kind == "subtitle" and isinstance(payload, dict):
                    display.show_subtitle(
                        str(payload.get("text", "") or ""),
                        duration_ms=int(payload.get("duration_ms", 5000) or 5000),
                    )
            except Exception:
                self.logger.debug("Display update failed", exc_info=True)
            finally:
                self._display_queue.task_done()

    @staticmethod
    def _apply_display_mode(display: Any, mode: str) -> None:
        if mode == "idle":
            display.show_idle()
        elif mode == "alert":
            display.show_alert()
        elif mode == "recording":
            display.show_recording()
        elif mode == "thinking":
            display.show_thinking()
        elif mode == "speaking":
            display.show_speaking()

    def _configure_session(self) -> None:
        session = realtime_audio_session_payload(
            profile=self.realtime_profile,
            instructions=self.base_system_prompt,
            tools=self._tool_schemas,
        )
        self._send_event({"type": "session.update", "session": session})

    def _build_response_request(self, turn: QueuedTurn) -> dict[str, Any]:
        dynamic_instructions = self._build_turn_instructions(turn)
        static_instructions = str(self.base_system_prompt or "").strip()
        dynamic_instructions = str(dynamic_instructions or "").strip()
        instruction_parts = [
            part for part in (static_instructions, dynamic_instructions) if part
        ]
        instructions = "\n\n".join(instruction_parts)
        if turn.no_audio_retry_count > 0:
            instructions = (
                instructions
                + "\n\n[DELIVERY] Respond with spoken audio for this turn. Do not return a silent text-only reply."
            )
        if turn.incomplete_audio_continuation_count > 0:
            instructions = (
                instructions
                + "\n\n[DELIVERY] Continue the current answer naturally from exactly where you left off. Do not restart, repeat, or summarize what you already said."
            )
        return realtime_response_payload(
            instructions=instructions,
            output_modalities=["audio"],
            max_output_tokens=self.realtime_profile.max_output_tokens,
        )

    def _send_response_create(self, turn: QueuedTurn) -> None:
        if self._is_turn_terminal(turn):
            return
        turn.response_requested_at = time.time()
        turn.pending_response_requests += 1
        self._set_turn_phase(turn, TURN_PHASE_RESPONSE_REQUESTED)
        with self._turn_lock:
            self._pending_response_turn_req_ids.append(turn.req_id)
        self._latency.emit(event="response_create", req_id=turn.req_id)
        self.logger.info(
            "Queueing response.create req_id=%s pending_response_requests=%s",
            turn.req_id,
            turn.pending_response_requests,
        )
        self._send_event(
            {
                "type": "response.create",
                "response": self._build_response_request(turn),
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
            except Exception:
                self.logger.exception("Turn failed req_id=%s", turn.req_id)
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
        while not self._stop_event.wait(WATCHDOG_POLL_SEC):
            now = time.time()
            with self._turn_lock:
                turns = list(self._turns_by_req_id.values())
            for turn in turns:
                if self._is_turn_terminal(turn):
                    continue
                if turn.phase in {TURN_PHASE_RESPONSE_REQUESTED, TURN_PHASE_WAITING_FIRST_AUDIO}:
                    started_at = turn.response_requested_at or turn.phase_updated_at
                    if now - started_at >= RESPONSE_STALL_TIMEOUT_SEC:
                        self.logger.warning(
                            "Realtime response watchdog cancel req_id=%s phase=%s",
                            turn.req_id,
                            turn.phase,
                        )
                        self._terminate_turn(turn, TURN_PHASE_CANCELED, "response_timeout")
                        continue
                if turn.phase == TURN_PHASE_WAITING_TOOLS and turn.pending_tool_calls > 0:
                    started_at = turn.phase_updated_at
                    if now - started_at >= RESPONSE_STALL_TIMEOUT_SEC:
                        self.logger.warning(
                            "Realtime tool watchdog cancel req_id=%s pending_tool_calls=%s",
                            turn.req_id,
                            turn.pending_tool_calls,
                        )
                        self._terminate_turn(turn, TURN_PHASE_CANCELED, "tool_timeout")
                        continue
                if (
                    turn.phase == TURN_PHASE_PLAYING
                    and turn.response_finished.is_set()
                    and not turn.playback_finished.is_set()
                ):
                    progress_at = (
                        turn.last_playback_progress_at
                        or turn.audio_started_at
                        or turn.phase_updated_at
                    )
                    if now - progress_at >= max(
                        PLAYBACK_STALL_TIMEOUT_SEC,
                        float(getattr(self.realtime_profile, "silence_grace_period", 0.0)) + 5.0,
                    ):
                        self.logger.warning(
                            "Realtime playback stall forcing completion req_id=%s response_id=%s",
                            turn.req_id,
                            turn.response_id,
                        )
                        self._force_complete_stalled_playback(turn, reason="stall_timeout")

    # ------------------------------------------------------------------
    # Turn handling
    # ------------------------------------------------------------------

    def _run_turn(self, turn: QueuedTurn) -> None:
        with self._turn_lock:
            self._active_turn = turn
            self._turns_by_req_id[turn.req_id] = turn
            self._clear_playback_tracking_locked()
        self.logger.info("Starting turn req_id=%s kind=%s", turn.req_id, turn.kind)

        set_request_context(
            req_id=turn.req_id,
            speech_end_perf_s=turn.speech_end_perf_s,
            speech_end_unix_s=turn.speech_end_unix_s,
            transcript_perf_s=turn.transcript_perf_s,
        )
        try:
            self._maybe_rotate_history_for_turn(turn)
            if turn.kind == "text":
                self._append_text_message_item(
                    turn,
                    turn.input_text,
                    role="system" if turn.source_is_internal else "user",
                )
            if turn.pending_internal_text:
                self._append_text_message_item(
                    turn,
                    turn.pending_internal_text,
                    role="system",
                )
            self._send_response_create(turn)
            self._wait_for_turn_settled(turn)
            if not self._is_turn_terminal(turn):
                self._complete_turn_success(turn)
            if turn.phase == TURN_PHASE_FINALIZED:
                self._maybe_capture_voice_reference(turn)
                self._maybe_note_preference_turn(turn)
        finally:
            self.logger.info(
                "Finished turn req_id=%s phase=%s finalized_reason=%s audio_started=%s pending_tool_calls=%s pending_response_requests=%s",
                turn.req_id,
                turn.phase,
                turn.finalized_reason,
                turn.audio_started,
                turn.pending_tool_calls,
                turn.pending_response_requests,
            )
            clear_request_context()
            with self._turn_lock:
                if self._active_turn is turn:
                    self._active_turn = None

    def _wait_for_turn_settled(self, turn: QueuedTurn) -> None:
        while not self._stop_event.is_set():
            if turn.response_finished.wait(timeout=0.1) and turn.playback_finished.wait(timeout=0.1):
                return
            if self._is_turn_terminal(turn):
                return

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
        if persons or (
            face_snapshot
            and (
                int(face_snapshot.get("recognized_count", 0) or 0) > 0
                or int(face_snapshot.get("unknown_count", 0) or 0) > 0
            )
        ):
            blocks.append(
                format_people_context(
                    persons,
                    primary_face_person_id=context.primary_face_person_id,
                    face_snapshot=face_snapshot,
                    audio_speaker_id=context.audio_speaker_id,
                    owner_id=context.owner_id,
                    owner_source=context.owner_source,
                    speaker_visible=context.speaker_visible,
                )
            )
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
        dispatch_server_event(self, event)

    def _handle_conversation_item_created(self, event: dict[str, Any]) -> None:
        item = server_event_item(event)
        item_id = server_event_item_id(event, item=item)
        if not item_id:
            return
        if item_id in self._item_id_to_req_id:
            self._register_history_item(item_id)
            return

        item_type = str(item.get("type", "") or "").strip()
        role = str(item.get("role", "") or "").strip()
        response_id = server_event_response_id(event, item=item)
        req_id = ""

        if item_type == "message" and role == "user":
            if self._conversation_item_looks_like_audio_input(item):
                req_id = self._consume_pending_audio_turn_req_id(include_finalized=True)
            else:
                req_id = self._consume_pending_local_created_item("message", "user")
        elif item_type == "message" and role == "system":
            req_id = self._consume_pending_local_created_item("message", "system")
        elif item_type == "function_call_output":
            req_id = self._consume_pending_local_created_item("function_call_output")
        elif item_type == "message" and role == "assistant":
            req_id = self._req_id_for_response_id(response_id)
        elif item_type == "function_call":
            req_id = self._req_id_for_response_id(response_id)
            call_id = str(item.get("call_id") or "").strip()
            if call_id and req_id:
                self._call_id_to_req_id[call_id] = req_id

        self._register_history_item(item_id, owner_req_id=req_id)
        if req_id:
            turn = self._turns_by_req_id.get(req_id)
            if turn is not None:
                self._register_turn_history_item(turn, item_id)
                if item_type == "message" and role == "user" and not turn.user_item_id:
                    turn.user_item_id = item_id
                elif item_type == "message" and role == "assistant":
                    turn.assistant_item_ids.add(item_id)
                elif item_type == "function_call":
                    turn.function_call_item_ids.add(item_id)

    def _handle_input_audio_buffer_committed(self, event: dict[str, Any]) -> None:
        item_id = str(event.get("item_id") or "").strip()
        if not item_id:
            return
        turn = self._resolve_turn_for_item(item_id)
        if turn is None:
            req_id = self._consume_pending_audio_turn_req_id(include_finalized=True)
            if req_id:
                turn = self._turns_by_req_id.get(req_id)
        if turn is None:
            with self._turn_lock:
                self._pending_audio_item_ids.append(item_id)
            self.logger.debug("Queued unbound audio item_id=%s for next audio turn", item_id)
            return
        self._bind_item_id_to_turn(turn, item_id)
        if not turn.user_item_id:
            turn.user_item_id = item_id

    def _handle_response_created(self, event: dict[str, Any]) -> None:
        response = server_event_response(event)
        response_id = server_event_response_id(event, response=response)
        if not response_id:
            return
        turn = self._consume_pending_response_turn(response_id)
        if turn is None:
            return
        turn.pending_response_requests = max(0, turn.pending_response_requests - 1)
        if self._is_turn_terminal(turn):
            try:
                self._send_event({"type": "response.cancel", "response_id": response_id})
            except Exception:
                self.logger.exception("Failed to cancel terminal response_id=%s", response_id)
            return
        self.logger.info("Realtime response created req_id=%s response_id=%s", turn.req_id, response_id)
        self._set_turn_phase(turn, TURN_PHASE_WAITING_FIRST_AUDIO)

    def _handle_input_transcription_completed(self, event: dict[str, Any]) -> None:
        transcript = str(event.get("transcript", "") or "").strip()
        item_id = str(event.get("item_id") or "").strip()
        turn = self._resolve_turn_for_item(item_id) if item_id else None
        if turn is None and item_id:
            req_id = self._consume_pending_audio_turn_req_id(include_finalized=True)
            if req_id:
                turn = self._turns_by_req_id.get(req_id)
                if turn is not None:
                    self._bind_item_id_to_turn(turn, item_id)
        if turn is None:
            return
        if item_id and not turn.user_item_id:
            turn.user_item_id = item_id
        if transcript:
            turn.user_transcript = transcript
            if turn.phase == TURN_PHASE_FINALIZED:
                self._maybe_note_preference_turn(turn)

        usage = event.get("usage", {}) or {}
        if isinstance(usage, dict):
            cost_fields = estimate_transcription_cost(
                usage,
                model_name=self.realtime_profile.transcription_model,
            )
            session_total_cost_usd = self._bump_session_estimated_cost(
                cost_fields.get("estimated_cost_usd")
            )
            self._latency.emit(
                event="transcription_usage",
                req_id=turn.req_id,
                session_id=getattr(self, "_session_id", "") or None,
                item_id=item_id or None,
                model=self.realtime_profile.transcription_model,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
                input_audio_tokens=cost_fields.get("input_audio_tokens"),
                output_text_tokens=cost_fields.get("output_text_tokens"),
                estimated_cost_usd=cost_fields.get("estimated_cost_usd"),
                session_total_cost_usd=session_total_cost_usd,
            )

    def _handle_input_transcription_failed(self, event: dict[str, Any]) -> None:
        item_id = str(event.get("item_id") or "").strip()
        turn = self._resolve_turn_for_item(item_id) if item_id else None
        if turn is None and item_id:
            req_id = self._consume_pending_audio_turn_req_id(include_finalized=True)
            if req_id:
                turn = self._turns_by_req_id.get(req_id)
                if turn is not None:
                    self._bind_item_id_to_turn(turn, item_id)
                    if not turn.user_item_id:
                        turn.user_item_id = item_id
        error = event.get("error", {}) or {}
        if not isinstance(error, dict):
            error = {}
        self.logger.warning(
            "Input transcription failed req_id=%s item_id=%s type=%s code=%s message=%s",
            getattr(turn, "req_id", "<unknown>"),
            item_id or "<unknown>",
            error.get("type", "unknown"),
            error.get("code", "unknown"),
            error.get("message", "unknown"),
        )

    def _handle_output_audio_delta(self, event: dict[str, Any]) -> None:
        response_id = server_event_response_id(event)
        item_id = server_event_item_id(event)
        turn = self._resolve_turn_for_output(response_id=response_id, item_id=item_id)
        if turn is None:
            if response_id or item_id:
                self.logger.warning(
                    "Ignoring output audio for unknown response_id=%s item_id=%s",
                    response_id,
                    item_id,
                )
            return
        if self._is_turn_terminal(turn):
            return
        if response_id:
            self._bind_response_id(turn, response_id)
        if item_id:
            self._bind_item_id_to_turn(turn, item_id)
            turn.assistant_item_id = item_id
            turn.assistant_item_ids.add(item_id)
        audio_bytes = base64.b64decode(str(event.get("delta", "") or ""))
        if not audio_bytes:
            return
        self._playback_buffer.append(audio_bytes)
        if turn.audio_started:
            if response_id and response_id != self._playback_stream_id:
                with self._turn_lock:
                    self._playback_req_id = turn.req_id
                    self._playback_stream_id = response_id
                    self._playback_item_id = turn.assistant_item_id
                    self._played_output_frames = 0
                turn.last_playback_progress_at = time.time()
                self.engagement.on_playback_event(
                    "playback_started",
                    turn.req_id,
                    stream_id=response_id,
                )
            return
        turn.audio_started = True
        turn.audio_started_at = time.time()
        turn.last_playback_progress_at = turn.audio_started_at
        self._set_turn_phase(turn, TURN_PHASE_PLAYING)
        if turn.kind == "audio" and float(turn.speech_end_perf_s) > 0.0:
            first_audio_perf = perf_now()
            self._latency.timing(
                "first_audio_latency_s",
                first_audio_perf - turn.speech_end_perf_s,
                req_id=turn.req_id,
            )
        self.engagement.on_agent_output_started(
            turn.req_id,
            stream_id=turn.response_id,
        )
        self._set_display_mode_async("speaking")
        with self._turn_lock:
            self._playback_req_id = turn.req_id
            self._playback_stream_id = turn.response_id
            self._playback_item_id = turn.assistant_item_id
            self._played_output_frames = 0
        self.engagement.on_playback_event(
            "playback_started",
            turn.req_id,
            stream_id=turn.response_id,
        )

    def _handle_output_transcript_delta(self, event: dict[str, Any]) -> None:
        delta = str(event.get("delta", "") or "")
        if not delta:
            return
        response_id = server_event_response_id(event)
        item_id = server_event_item_id(event)
        turn = self._resolve_turn_for_output(response_id=response_id, item_id=item_id)
        if turn is None:
            return
        if self._is_turn_terminal(turn) and turn.phase != TURN_PHASE_FINALIZED:
            return
        if item_id:
            self._bind_item_id_to_turn(turn, item_id)
            turn.assistant_item_id = item_id or turn.assistant_item_id
            turn.assistant_item_ids.add(item_id)
        turn.assistant_transcript += delta
        self._show_display_subtitle_async(
            self._display_subtitle_window(turn.assistant_transcript),
            duration_ms=5000,
        )
        if turn.phase == TURN_PHASE_FINALIZED:
            self._maybe_note_preference_turn(turn)

    def _handle_output_text_delta(self, event: dict[str, Any]) -> None:
        delta = str(event.get("delta", "") or "")
        if not delta:
            return
        response_id = server_event_response_id(event)
        item_id = server_event_item_id(event)
        turn = self._resolve_turn_for_output(response_id=response_id, item_id=item_id)
        if turn is None:
            return
        if self._is_turn_terminal(turn) and turn.phase != TURN_PHASE_FINALIZED:
            return
        if item_id:
            self._bind_item_id_to_turn(turn, item_id)
            turn.assistant_item_id = item_id or turn.assistant_item_id
            turn.assistant_item_ids.add(item_id)
        turn.assistant_transcript += delta
        self._show_display_subtitle_async(
            self._display_subtitle_window(turn.assistant_transcript),
            duration_ms=5000,
        )
        if turn.phase == TURN_PHASE_FINALIZED:
            self._maybe_note_preference_turn(turn)

    def _arm_playback_completion(self, turn: QueuedTurn) -> None:
        if turn.playback_completion_armed:
            return
        turn.playback_completion_armed = True
        stream_id = str(turn.response_id or self._playback_stream_id or "").strip()
        self.engagement.on_agent_done(has_reply=True, req_id=turn.req_id)
        threading.Thread(
            target=self._wait_for_playback_and_complete,
            args=(turn, stream_id),
            daemon=True,
        ).start()

    def _handle_output_item_done(self, event: dict[str, Any]) -> None:
        item = server_event_item(event)
        item_id = server_event_item_id(event, item=item)
        response_id = server_event_response_id(event, item=item)
        turn = self._resolve_turn_for_output(response_id=response_id, item_id=item_id)
        if turn is None or self._is_turn_terminal(turn):
            return
        if response_id:
            self._bind_response_id(turn, response_id)
        if item_id:
            self._bind_item_id_to_turn(turn, item_id)
            turn.assistant_item_id = item_id or turn.assistant_item_id
            turn.assistant_item_ids.add(item_id)
        item_type = str(item.get("type", "") or "").strip()
        role = str(item.get("role", "") or "").strip()
        status = str(item.get("status", "") or "").strip()
        if item_type != "message" or role != "assistant":
            return
        if status != "completed" or not turn.audio_started:
            return
        self._arm_playback_completion(turn)

    def _handle_function_call_delta(self, event: dict[str, Any]) -> None:
        item_id = str(event.get("item_id", "") or "")
        if not item_id:
            return
        bucket = self._pending_function_args.setdefault(item_id, {})
        if event.get("call_id") is not None:
            bucket["call_id"] = str(event.get("call_id") or "")
        if event.get("name") is not None:
            bucket["name"] = str(event.get("name") or "")
        if event.get("response_id") is not None:
            bucket["response_id"] = str(event.get("response_id") or "")
        bucket["arguments"] = bucket.get("arguments", "") + str(event.get("delta", "") or "")

        response_id = bucket.get("response_id", "")
        turn = self._resolve_turn_for_output(response_id=response_id, item_id=item_id)
        if turn is not None:
            self._bind_item_id_to_turn(turn, item_id)
            turn.function_call_item_ids.add(item_id)

    def _handle_function_call_done(self, event: dict[str, Any]) -> None:
        item_id = str(event.get("item_id", "") or "")
        call_id = str(event.get("call_id", "") or "")
        tool_name = str(event.get("name", "") or "")
        arguments_json = str(event.get("arguments", "") or "")
        cached = self._pending_function_args.pop(item_id, None) if item_id else None
        response_id = str(event.get("response_id", "") or "")
        if cached:
            call_id = call_id or cached.get("call_id", "")
            tool_name = tool_name or cached.get("name", "")
            arguments_json = arguments_json or cached.get("arguments", "")
            response_id = response_id or cached.get("response_id", "")
        turn = self._resolve_turn_for_output(
            response_id=response_id,
            item_id=item_id,
            call_id=call_id,
        )
        if turn is None or self._is_turn_terminal(turn):
            return
        if not call_id or not tool_name:
            self.logger.warning("Ignoring incomplete function call payload")
            return
        self._cancel_owner_turn_for_tool(turn, tool_name)
        self._call_id_to_req_id[call_id] = turn.req_id
        if item_id:
            self._bind_item_id_to_turn(turn, item_id)
            turn.function_call_item_ids.add(item_id)
        turn.pending_tool_calls += 1
        turn.pending_call_ids.add(call_id)
        self._set_turn_phase(turn, TURN_PHASE_WAITING_TOOLS)
        self._tool_queue.put(
            PendingToolCall(
                turn_req_id=turn.req_id,
                call_id=call_id,
                tool_name=tool_name,
                arguments_json=arguments_json or "{}",
                function_item_id=item_id,
            )
        )

    def _handle_response_done(self, event: dict[str, Any]) -> None:
        response = server_event_response(event)
        response_id = server_event_response_id(event, response=response)
        turn = self._resolve_turn_for_output(response_id=response_id)
        if turn is None:
            self.logger.warning("Ignoring response.done for unknown response_id=%s", response_id)
            return
        if self._is_turn_terminal(turn):
            return
        if turn.interrupted:
            turn.response_finished.set()
            turn.playback_finished.set()
            return
        turn.response_done_at = time.time()
        if response_id:
            self._bind_response_id(turn, response_id)
        status = str(response.get("status", "unknown") or "unknown").strip()
        self._emit_response_usage(turn, response)
        if not turn.assistant_transcript:
            turn.assistant_transcript = self._transcript_from_response(response)
        output_items = response.get("output", []) or []
        for output_item in output_items:
            item_id = str(output_item.get("id", "") or "").strip()
            item_type = str(output_item.get("type", "") or "").strip()
            if item_id:
                self._bind_item_id_to_turn(turn, item_id)
                if item_type == "function_call":
                    turn.function_call_item_ids.add(item_id)
                elif item_type == "message":
                    turn.assistant_item_ids.add(item_id)
                    if not turn.assistant_item_id:
                        turn.assistant_item_id = item_id
        has_function_call = any(
            str(item.get("type", "") or "") == "function_call" for item in output_items
        )
        if has_function_call or turn.pending_tool_calls > 0:
            self._set_turn_phase(turn, TURN_PHASE_WAITING_TOOLS)
            return

        completed_tools = tuple(turn.metadata.get("completed_tools", ()) or ())
        if "resolve_employee_identity" in completed_tools:
            self._latency.emit(
                event="assistant_response_after_tool",
                req_id=turn.req_id,
                tool="resolve_employee_identity",
                response_id=response_id,
                status=status,
                transcript=_debug_log_value(turn.assistant_transcript.strip()),
            )

        incomplete_details = response.get("incomplete_details")
        has_audio_reply = turn.audio_started
        if status == "incomplete" and has_audio_reply:
            self.logger.warning(
                "Realtime response finished incomplete after audio started req_id=%s response_id=%s incomplete_details=%s transcript=%r",
                turn.req_id,
                response_id,
                incomplete_details,
                turn.assistant_transcript.strip(),
            )
            if self._should_continue_incomplete_audio_reply(turn):
                self._continue_incomplete_audio_reply(turn)
                return
        elif status != "completed":
            self.logger.warning(
                "Realtime response ended without completion req_id=%s response_id=%s status=%s incomplete_details=%s output_types=%s transcript=%r",
                turn.req_id,
                response_id,
                status,
                incomplete_details,
                self._response_output_types(response),
                turn.assistant_transcript.strip(),
            )
            self._cleanup_silent_response_items(turn, response)
            self._terminate_turn(
                turn,
                TURN_PHASE_CANCELED,
                f"response_status_{status or 'unknown'}",
                send_cancel=False,
            )
            return

        if not has_audio_reply:
            if self._retry_no_audio_response(turn, response):
                return
            self.logger.error(
                "Realtime response completed without audio req_id=%s response_id=%s retries=%s status=%s incomplete_details=%s output_types=%s transcript=%r",
                turn.req_id,
                response_id,
                turn.no_audio_retry_count,
                status,
                incomplete_details,
                self._response_output_types(response),
                turn.assistant_transcript.strip(),
            )
            self._cleanup_silent_response_items(turn, response)
            self._terminate_turn(
                turn,
                TURN_PHASE_CANCELED,
                "response_completed_without_audio",
                send_cancel=False,
            )
            return

        turn.response_finished.set()
        if has_audio_reply:
            self._arm_playback_completion(turn)
        else:
            self.engagement.on_agent_done(has_reply=False, req_id=turn.req_id)

    def _handle_server_error(self, event: dict[str, Any]) -> None:
        error = event.get("error", {})
        self.logger.error(
            "Realtime server error type=%s message=%s",
            error.get("type", "unknown"),
            error.get("message", "unknown"),
        )
        response_id = str(error.get("response_id", "") or "").strip()
        turn = self._resolve_turn_for_output(response_id=response_id)
        if turn is None:
            with self._turn_lock:
                turn = self._active_turn
        if turn is not None:
            self._terminate_turn(turn, TURN_PHASE_CANCELED, "server_error")

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
        self._set_turn_phase(turn, TURN_PHASE_FINALIZED)
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
        self._set_turn_phase(turn, phase)
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

    def _clear_playback_tracking_locked(self) -> None:
        self._playback_req_id = ""
        self._playback_stream_id = ""
        self._playback_item_id = ""
        self._played_output_frames = 0
