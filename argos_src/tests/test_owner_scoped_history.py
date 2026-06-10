from __future__ import annotations

from collections import deque
import logging
import sys
import threading
import types


if "websocket" not in sys.modules:
    websocket_stub = types.ModuleType("websocket")
    websocket_stub.WebSocketConnectionClosedException = RuntimeError
    sys.modules["websocket"] = websocket_stub

if "std_msgs" not in sys.modules:
    std_msgs_stub = types.ModuleType("std_msgs")
    std_msgs_msg_stub = types.ModuleType("std_msgs.msg")

    class _String:
        data = ""

    std_msgs_msg_stub.String = _String
    sys.modules["std_msgs"] = std_msgs_stub
    sys.modules["std_msgs.msg"] = std_msgs_msg_stub


from argos_src.agent.agent_preferences import RealtimeAgentPreferenceMixin
from argos_src.agent.agent_state import RealtimeAgentStateMixin
from argos_src.agent.preference_segments import _PreferenceSegmentCoordinator
from argos_src.agent.realtime_turns import (
    FrozenTurnContext,
    QueuedTurn,
    TURN_PHASE_FINALIZED,
)


class FakeRuntime(RealtimeAgentPreferenceMixin, RealtimeAgentStateMixin):
    def __init__(self) -> None:
        self.logger = logging.getLogger("test.owner_scoped_history")
        self._turn_lock = threading.RLock()
        self._turns_by_req_id: dict[str, QueuedTurn] = {}
        self._response_id_to_req_id: dict[str, str] = {}
        self._item_id_to_req_id: dict[str, str] = {}
        self._call_id_to_req_id: dict[str, str] = {}
        self._pending_local_created_items = deque()
        self._pending_response_turn_req_ids = deque()
        self._pending_audio_turn_req_ids = deque()
        self._pending_audio_item_ids = deque()
        self._history_item_order: deque[str] = deque()
        self._known_history_item_ids: set[str] = set()
        self._history_item_owner_req_id: dict[str, str] = {}
        self._active_history_owner_key = ""
        self._playback_item_id = ""
        self._last_tool_name = None
        self._last_tool_summary = None
        self._preference_segments = _PreferenceSegmentCoordinator()
        self.sent_events: list[dict] = []

    def _send_event(self, payload: dict) -> None:
        self.sent_events.append(payload)

    def _cancel_preference_idle_flush(self) -> None:
        return

    def flush_preference_segments(self, reason: str = "idle") -> None:
        self.flushed_reason = reason
        self._preference_segments.flush_active()


def make_turn(
    req_id: str,
    *,
    owner_id: str | None,
    finalized: bool = False,
    source_is_internal: bool = False,
    context_snapshot: FrozenTurnContext | None = None,
) -> QueuedTurn:
    turn = QueuedTurn(
        kind="audio",
        req_id=req_id,
        speech_end_perf_s=0.0,
        speech_end_unix_s=0.0,
        transcript_perf_s=0.0,
        owner_id=owner_id,
        source_is_internal=source_is_internal,
        context_snapshot=context_snapshot or FrozenTurnContext(owner_id=owner_id),
    )
    if finalized:
        turn.phase = TURN_PHASE_FINALIZED
        turn.finalized = True
    return turn


def register_turn_item(runtime: FakeRuntime, turn: QueuedTurn, item_id: str) -> None:
    runtime._turns_by_req_id[turn.req_id] = turn
    runtime._register_turn_history_item(turn, item_id)


def deleted_item_ids(runtime: FakeRuntime) -> list[str]:
    return [
        event["item_id"]
        for event in runtime.sent_events
        if event.get("type") == "conversation.item.delete"
    ]


def test_same_owner_does_not_delete_history_or_clear_tool_summary() -> None:
    runtime = FakeRuntime()
    runtime._active_history_owner_key = "owner:A"
    runtime._last_tool_name = "capture_scene"
    runtime._last_tool_summary = "capture_scene: success"
    old_turn = make_turn("old", owner_id="A", finalized=True)
    register_turn_item(runtime, old_turn, "old-user")

    current_turn = make_turn("current", owner_id="A")
    runtime._turns_by_req_id[current_turn.req_id] = current_turn

    runtime._maybe_rotate_history_for_turn(current_turn)

    assert deleted_item_ids(runtime) == []
    assert list(runtime._history_item_order) == ["old-user"]
    assert runtime._last_tool_name == "capture_scene"
    assert runtime._last_tool_summary == "capture_scene: success"


def test_owner_change_deletes_old_items_and_protects_current_audio_item() -> None:
    runtime = FakeRuntime()
    runtime._active_history_owner_key = "owner:A"
    runtime._last_tool_name = "navigate_to_location"
    runtime._last_tool_summary = "navigate_to_location: success"
    old_turn = make_turn("old", owner_id="A", finalized=True)
    register_turn_item(runtime, old_turn, "old-user")
    register_turn_item(runtime, old_turn, "old-assistant")
    current_turn = make_turn("current", owner_id="B")
    register_turn_item(runtime, current_turn, "current-audio")
    current_turn.user_item_id = "current-audio"

    runtime._maybe_rotate_history_for_turn(current_turn)

    assert deleted_item_ids(runtime) == ["old-user", "old-assistant"]
    assert list(runtime._history_item_order) == ["current-audio"]
    assert "current-audio" in runtime._known_history_item_ids
    assert runtime._active_history_owner_key == "owner:B"
    assert runtime._last_tool_name is None
    assert runtime._last_tool_summary is None


def test_owner_change_protects_newest_unbound_audio_item() -> None:
    runtime = FakeRuntime()
    runtime._active_history_owner_key = "owner:A"
    old_turn = make_turn("old", owner_id="A", finalized=True)
    register_turn_item(runtime, old_turn, "old-user")
    runtime._register_history_item("current-audio-unbound")
    current_turn = make_turn("current", owner_id="B")
    runtime._turns_by_req_id[current_turn.req_id] = current_turn

    runtime._maybe_rotate_history_for_turn(current_turn)

    assert deleted_item_ids(runtime) == ["old-user"]
    assert list(runtime._history_item_order) == ["current-audio-unbound"]
    assert "current-audio-unbound" in runtime._known_history_item_ids


def test_known_owner_to_anonymous_clears_prior_history_and_tool_summary() -> None:
    runtime = FakeRuntime()
    runtime._active_history_owner_key = "owner:A"
    runtime._last_tool_name = "capture_scene"
    runtime._last_tool_summary = "capture_scene: saw a whiteboard"
    old_turn = make_turn("old", owner_id="A", finalized=True)
    register_turn_item(runtime, old_turn, "old-user")
    current_turn = make_turn("anonymous", owner_id=None)
    runtime._turns_by_req_id[current_turn.req_id] = current_turn

    runtime._maybe_rotate_history_for_turn(current_turn)

    assert deleted_item_ids(runtime) == ["old-user"]
    assert runtime._active_history_owner_key == "anonymous"
    assert runtime._last_tool_name is None
    assert runtime._last_tool_summary is None


def test_anonymous_to_owner_clears_anonymous_history_and_keeps_memory_context() -> None:
    runtime = FakeRuntime()
    runtime._active_history_owner_key = "anonymous"
    anonymous_turn = make_turn("anon-old", owner_id=None, finalized=True)
    register_turn_item(runtime, anonymous_turn, "anon-user")
    context = FrozenTurnContext(owner_id="A", memory_context_blocks=("About: likes tea",))
    current_turn = make_turn("owner-a", owner_id="A", context_snapshot=context)
    runtime._turns_by_req_id[current_turn.req_id] = current_turn

    runtime._maybe_rotate_history_for_turn(current_turn)

    assert deleted_item_ids(runtime) == ["anon-user"]
    assert runtime._active_history_owner_key == "owner:A"
    assert current_turn.context_snapshot.memory_context_blocks == ("About: likes tea",)


def test_internal_event_does_not_rotate_current_owner_session() -> None:
    runtime = FakeRuntime()
    runtime._active_history_owner_key = "owner:A"
    old_turn = make_turn("old", owner_id="A", finalized=True)
    register_turn_item(runtime, old_turn, "old-user")
    internal_turn = make_turn("internal", owner_id=None, source_is_internal=True)
    runtime._turns_by_req_id[internal_turn.req_id] = internal_turn

    runtime._maybe_rotate_history_for_turn(internal_turn)

    assert deleted_item_ids(runtime) == []
    assert runtime._active_history_owner_key == "owner:A"
    assert list(runtime._history_item_order) == ["old-user"]


def test_preference_extraction_uses_local_transcripts_after_history_deletion() -> None:
    runtime = FakeRuntime()
    runtime._active_history_owner_key = "owner:A"
    old_turn = make_turn("old", owner_id="A", finalized=True)
    old_turn.user_transcript = "I like jasmine tea."
    old_turn.assistant_transcript = "Got it."
    old_turn.user_item_id = "old-user"
    register_turn_item(runtime, old_turn, "old-user")
    current_turn = make_turn("current", owner_id="B")
    runtime._turns_by_req_id[current_turn.req_id] = current_turn

    runtime._maybe_rotate_history_for_turn(current_turn)
    runtime._maybe_note_preference_turn(old_turn)
    segment = runtime._preference_segments.flush_active()

    assert "old-user" not in runtime._known_history_item_ids
    assert segment is not None
    assert segment.person_id == "A"
    assert segment.turns[0].user_text == "I like jasmine tea."
    assert segment.turns[0].assistant_text == "Got it."

