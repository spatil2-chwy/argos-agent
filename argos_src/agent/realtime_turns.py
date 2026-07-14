"""Turn-state types and small utilities for the realtime Argos agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Optional

import numpy as np

from argos_src.agent.control.types import TurnState


AUDIO_CHANNELS = 1

TURN_PHASE_COMMITTED = TurnState.COMMITTED.value
TURN_PHASE_QUEUED = TurnState.QUEUED.value
TURN_PHASE_PREPARING_HISTORY = TurnState.PREPARING_HISTORY.value
TURN_PHASE_RESPONSE_REQUESTED = TurnState.RESPONSE_REQUESTED.value
TURN_PHASE_WAITING_FIRST_AUDIO = TurnState.WAITING_FIRST_OUTPUT.value
TURN_PHASE_WAITING_TOOLS = TurnState.WAITING_TOOLS.value
TURN_PHASE_REQUESTING_FOLLOWUP = TurnState.REQUESTING_FOLLOWUP.value
TURN_PHASE_MODEL_DONE = TurnState.MODEL_DONE.value
TURN_PHASE_PLAYING = TurnState.PLAYING.value
TURN_PHASE_FINALIZED = TurnState.FINALIZED.value
TURN_PHASE_CANCELED = TurnState.CANCELED.value
TURN_PHASE_SUPERSEDED = TurnState.SUPERSEDED.value

TERMINAL_TURN_PHASES = {
    TURN_PHASE_FINALIZED,
    TURN_PHASE_CANCELED,
    TURN_PHASE_SUPERSEDED,
}

WATCHDOG_POLL_SEC = 0.2
RESPONSE_STALL_TIMEOUT_SEC = 12.0
PLAYBACK_STALL_TIMEOUT_SEC = 15.0
NO_AUDIO_RESPONSE_RETRY_LIMIT = 1
INCOMPLETE_AUDIO_CONTINUATION_LIMIT = 1


@dataclass
class FrozenTurnContext:
    """Human-scene snapshot frozen when a turn is created."""

    persons: list[Any] = field(default_factory=list)
    face_snapshot: Optional[dict[str, Any]] = None
    primary_face_person_id: Optional[str] = None
    audio_speaker_id: Optional[str] = None
    owner_id: Optional[str] = None
    owner_source: str = "unknown"
    owner_confidence: float = 0.0
    speaker_visible: bool = False
    memory_context_blocks: tuple[str, ...] = ()


@dataclass
class QueuedTurn:
    """Serialized turn request owned by the response worker."""

    kind: str
    req_id: str
    speech_end_perf_s: float
    speech_end_unix_s: float
    transcript_perf_s: float
    primary_face_person_id: Optional[str] = None
    audio_speaker_id: Optional[str] = None
    owner_id: Optional[str] = None
    owner_source: str = "unknown"
    owner_confidence: float = 0.0
    speaker_visible: bool = False
    source_is_internal: bool = False
    exchange_id: str = ""
    exchange_index: int = 0
    input_text: str = ""
    pending_internal_text: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    context_snapshot: FrozenTurnContext = field(default_factory=FrozenTurnContext)
    input_audio_pcm16: bytes = b""
    trimmed_input_audio_pcm16: bytes | None = None
    response_id: str = ""
    assistant_item_id: str = ""
    user_item_id: str = ""
    user_transcript: str = ""
    assistant_transcript: str = ""
    preference_noted: bool = False
    preference_unattributed_flushed: bool = False
    audio_started: bool = False
    interrupted: bool = False
    pending_tool_calls: int = 0
    finalized: bool = False
    finalized_reason: str = ""
    playback_completion_armed: bool = False
    phase: str = TURN_PHASE_COMMITTED
    phase_updated_at: float = field(default_factory=time.time)
    response_requested_at: float = 0.0
    response_done_at: float = 0.0
    audio_started_at: float = 0.0
    last_playback_progress_at: float = 0.0
    pending_response_requests: int = 0
    no_audio_retry_count: int = 0
    incomplete_audio_continuation_count: int = 0
    history_item_ids: set[str] = field(default_factory=set)
    assistant_item_ids: set[str] = field(default_factory=set)
    function_call_item_ids: set[str] = field(default_factory=set)
    pending_call_ids: set[str] = field(default_factory=set)
    pending_tool_names_by_call_id: dict[str, str] = field(default_factory=dict)
    response_finished: threading.Event = field(default_factory=threading.Event)
    playback_finished: threading.Event = field(default_factory=threading.Event)

    def mark_no_audio_reply(self) -> None:
        if not self.audio_started:
            self.playback_finished.set()


@dataclass
class PendingToolCall:
    """Realtime function-call payload awaiting local execution."""

    turn_req_id: str
    call_id: str
    tool_name: str
    arguments_json: str
    function_item_id: str = ""


@dataclass
class PendingCreatedItem:
    """Local conversation.item.create awaiting server item creation."""

    owner_req_id: str
    expected_type: str
    expected_role: str = ""


class PlaybackBuffer:
    """Thread-safe PCM playback buffer."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def append(self, audio_bytes: bytes) -> None:
        if not audio_bytes:
            return
        with self._lock:
            self._buffer.extend(audio_bytes)

    def buffered_frames(self) -> int:
        with self._lock:
            return len(self._buffer) // np.dtype(np.int16).itemsize

    def pop_frames(self, frames: int) -> tuple[np.ndarray, int]:
        bytes_needed = frames * np.dtype(np.int16).itemsize * AUDIO_CHANNELS
        chunk = bytearray(bytes_needed)
        with self._lock:
            available = min(bytes_needed, len(self._buffer))
            chunk[:available] = self._buffer[:available]
            del self._buffer[:available]
        audio = np.frombuffer(bytes(chunk), dtype=np.int16).reshape(-1, AUDIO_CHANNELS)
        actual_frames = available // (np.dtype(np.int16).itemsize * AUDIO_CHANNELS)
        return audio, actual_frames
