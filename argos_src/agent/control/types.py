"""State-axis and transition types for the realtime Argos control plane."""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
from typing import Any


class _StrEnum(str, enum.Enum):
    def __str__(self) -> str:
        return self.value


class StateAxis(_StrEnum):
    SESSION = "session"
    CAPTURE = "capture"
    TRANSCRIPTION = "transcription"
    TURN = "turn"
    PLAYBACK = "playback"
    ENGAGEMENT = "engagement"
    ROBOT_ARBITRATION = "robot_arbitration"
    COALESCER = "coalescer"


class SessionState(_StrEnum):
    STOPPED = "stopped"
    CONNECTING = "connecting"
    CONFIGURING = "configuring"
    READY = "ready"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"


class CaptureState(_StrEnum):
    NOT_READY = "not_ready"
    ADMISSION_CLOSED = "admission_closed"
    ADMISSION_OPEN = "admission_open"
    CANDIDATE_VOICE = "candidate_voice"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    COMMITTING = "committing"
    COMMITTED = "committed"


class TranscriptionState(_StrEnum):
    NONE = "none"
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class TurnState(_StrEnum):
    COMMITTED = "committed"
    QUEUED = "queued"
    PREPARING_HISTORY = "preparing_history"
    RESPONSE_REQUESTED = "response_requested"
    WAITING_FIRST_OUTPUT = "waiting_first_output"
    WAITING_TOOLS = "waiting_tools"
    REQUESTING_FOLLOWUP = "requesting_followup"
    MODEL_DONE = "model_done"
    PLAYING = "playing"
    FINALIZED = "finalized"
    CANCELED = "canceled"
    SUPERSEDED = "superseded"


class PlaybackState(_StrEnum):
    IDLE = "idle"
    BUFFERING = "buffering"
    PLAYING = "playing"
    AWAITING_MODEL_DONE = "awaiting_model_done"
    AWAITING_DRAIN = "awaiting_drain"
    COMPLETED = "completed"
    STOPPED_TRUNCATED = "stopped_truncated"
    FORCE_COMPLETED = "force_completed"


class EngagementMode(_StrEnum):
    IDLE = "idle"
    ALERT = "alert"
    ENGAGED = "engaged"
    SPEAKING = "speaking"
    COOLDOWN = "cooldown"


class RobotArbitrationState(_StrEnum):
    PATROL_ALLOWED = "patrol_allowed"
    PATROL_SUPPRESSED = "patrol_suppressed"
    FACE_ATTENTION_ALLOWED = "face_attention_allowed"
    FACE_ATTENTION_SUPPRESSED = "face_attention_suppressed"
    NAV_INACTIVE = "nav_inactive"
    NAV_INTERRUPTIBLE = "nav_interruptible"
    NAV_FOCUSED = "nav_focused"
    OWNER_TURN_PENDING = "owner_turn_pending"
    OWNER_TURN_ACTIVE = "owner_turn_active"
    BATTERY_OK = "battery_ok"
    BATTERY_LOW_BLOCKING = "battery_low_blocking"


@dataclass(frozen=True)
class StateTransition:
    axis: StateAxis | str
    old_state: str
    new_state: str
    trigger: str
    req_id: str = ""
    stream_id: str = ""
    reason: str = ""
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ControlAction:
    """Declarative side effect emitted by reducers and applied by executors."""

    kind: str
    fields: dict[str, Any] = field(default_factory=dict)
