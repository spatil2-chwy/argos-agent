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


from argos_src.agent.control.history_store import InferenceHistoryIndex
from argos_src.agent.control.preference_runtime import PreferenceRuntime
from argos_src.agent.control.state_runtime import AgentStateRuntime
from argos_src.agent.preference_segments import _PreferenceSegmentCoordinator
from argos_src.agent.realtime_turns import (
    FrozenTurnContext,
    QueuedTurn,
    TURN_PHASE_FINALIZED,
)


class FakeRuntime(AgentStateRuntime):
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
        self._history_items = {}
        self._history_index_store = InferenceHistoryIndex(
            item_order=self._history_item_order,
            known_item_ids=self._known_history_item_ids,
            item_owner_req_id=self._history_item_owner_req_id,
            items=self._history_items,
        )
        self._history_item_snapshots: dict[str, dict[str, str]] = {}
        self._active_inference_owner_key = ""
        self._active_inference_scope_id = ""
        self._pending_anonymous_inference_scope_id = ""
        self._anonymous_inference_patch_index = 0
        self._playback_item_id = ""
        self._last_tool_name = None
        self._last_tool_summary = None
        self._preference_segments = _PreferenceSegmentCoordinator()
        self.preference_extraction_enabled = False
        self.preference_extractor = None
        self._preference_idle_flush_lock = threading.Lock()
        self._preference_idle_flush_timer = None
        self._preference_idle_flush_delay_sec = 0.05
        self._pending_lock = threading.Lock()
        self._pending_preference_segment_ids = set()
        self._preference_runtime = PreferenceRuntime(self)
        self.sent_events: list[dict] = []

    def _send_event(self, payload: dict) -> None:
        self.sent_events.append(payload)

    def _cancel_preference_idle_flush(self) -> None:
        return

    def _maybe_note_preference_turn(self, turn) -> None:
        self._preference_runtime.maybe_note_turn(turn)

    def flush_preference_segments(self, reason: str = "idle") -> None:
        self.flushed_reason = reason
        self._preference_segments.flush_active()


class FakeLatency:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, **fields) -> None:
        self.events.append(fields)


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


def register_done_item(runtime: FakeRuntime, turn: QueuedTurn, item_id: str) -> None:
    runtime._turns_by_req_id[turn.req_id] = turn
    runtime._ensure_inference_scope_for_turn(turn)
    runtime._register_turn_history_item(
        turn,
        item_id,
        item_type="message",
        role="user",
        status="done",
        permitted_for_inference=True,
    )


def sent_delete_events(runtime: FakeRuntime) -> list[dict]:
    return [event for event in runtime.sent_events if event.get("type") == "conversation.item.delete"]


def test_known_owner_history_is_reused_when_owner_returns() -> None:
    runtime = FakeRuntime()
    runtime._latency = FakeLatency()
    owner_a_first = make_turn("a-1", owner_id="A", finalized=True)
    register_done_item(runtime, owner_a_first, "a-user")
    owner_b = make_turn("b-1", owner_id="B", finalized=True)
    register_done_item(runtime, owner_b, "b-user")
    runtime._active_inference_owner_key = "owner:B"
    runtime._active_inference_scope_id = "owner:B"

    owner_a_return = make_turn("a-2", owner_id="A")
    runtime._turns_by_req_id[owner_a_return.req_id] = owner_a_return
    runtime._maybe_rotate_history_for_turn(owner_a_return)

    assert owner_a_return.inference_scope_id == "owner:A"
    assert owner_a_return.selected_inference_history_item_ids == ["a-user"]
    assert sent_delete_events(runtime) == []
    assert runtime._last_tool_name is None


def test_consecutive_unknown_turns_share_one_anonymous_patch() -> None:
    runtime = FakeRuntime()
    first_unknown = make_turn("anon-1", owner_id=None, finalized=True)
    register_done_item(runtime, first_unknown, "anon-1-user")
    runtime._active_inference_owner_key = "anonymous"
    runtime._active_inference_scope_id = first_unknown.inference_scope_id

    second_unknown = make_turn("anon-2", owner_id=None)
    runtime._turns_by_req_id[second_unknown.req_id] = second_unknown
    runtime._maybe_rotate_history_for_turn(second_unknown)

    assert first_unknown.inference_scope_id == "anonymous:1"
    assert second_unknown.inference_scope_id == "anonymous:1"
    assert second_unknown.selected_inference_history_item_ids == ["anon-1-user"]
    assert sent_delete_events(runtime) == []


def test_queued_unknown_turns_bound_before_run_share_patch_without_future_leak() -> None:
    runtime = FakeRuntime()
    runtime._active_inference_owner_key = "owner:A"
    runtime._active_inference_scope_id = "owner:A"
    first_unknown = make_turn("anon-1", owner_id=None)
    second_unknown = make_turn("anon-2", owner_id=None)
    register_done_item(runtime, first_unknown, "anon-1-user")
    register_done_item(runtime, second_unknown, "anon-2-user")

    runtime._maybe_rotate_history_for_turn(first_unknown)

    assert first_unknown.inference_scope_id == "anonymous:1"
    assert second_unknown.inference_scope_id == "anonymous:1"
    assert first_unknown.selected_inference_history_item_ids == []

    first_unknown.phase = TURN_PHASE_FINALIZED
    first_unknown.finalized = True
    runtime._maybe_rotate_history_for_turn(second_unknown)

    assert second_unknown.inference_scope_id == "anonymous:1"
    assert second_unknown.selected_inference_history_item_ids == ["anon-1-user"]
    assert sent_delete_events(runtime) == []


def test_separate_unknown_patches_do_not_share_history_across_known_owner() -> None:
    runtime = FakeRuntime()
    first_unknown = make_turn("anon-1", owner_id=None, finalized=True)
    register_done_item(runtime, first_unknown, "anon-1-user")
    owner_b = make_turn("b-1", owner_id="B", finalized=True)
    register_done_item(runtime, owner_b, "b-user")
    runtime._active_inference_owner_key = "owner:B"
    runtime._active_inference_scope_id = "owner:B"

    second_unknown = make_turn("anon-2", owner_id=None)
    runtime._turns_by_req_id[second_unknown.req_id] = second_unknown
    runtime._maybe_rotate_history_for_turn(second_unknown)

    assert first_unknown.inference_scope_id == "anonymous:1"
    assert second_unknown.inference_scope_id == "anonymous:2"
    assert second_unknown.selected_inference_history_item_ids == []
    assert sent_delete_events(runtime) == []


def test_internal_same_owner_event_keeps_current_owner_scope() -> None:
    runtime = FakeRuntime()
    old_turn = make_turn("old", owner_id="A", finalized=True)
    register_done_item(runtime, old_turn, "old-user")
    runtime._active_inference_owner_key = "owner:A"
    runtime._active_inference_scope_id = "owner:A"
    internal_turn = make_turn("internal", owner_id="A", source_is_internal=True)
    runtime._turns_by_req_id[internal_turn.req_id] = internal_turn

    runtime._maybe_rotate_history_for_turn(internal_turn)

    assert internal_turn.inference_scope_id == "owner:A"
    assert internal_turn.selected_inference_history_item_ids == ["old-user"]
    assert sent_delete_events(runtime) == []


def test_preference_extraction_uses_local_transcripts_after_scope_change() -> None:
    runtime = FakeRuntime()
    old_turn = make_turn("old", owner_id="A", finalized=True)
    old_turn.user_transcript = "I like jasmine tea."
    old_turn.assistant_transcript = "Got it."
    old_turn.user_item_id = "old-user"
    register_done_item(runtime, old_turn, "old-user")
    current_turn = make_turn("current", owner_id="B")
    runtime._turns_by_req_id[current_turn.req_id] = current_turn

    runtime._maybe_rotate_history_for_turn(current_turn)
    runtime._maybe_note_preference_turn(old_turn)
    segment = runtime._preference_segments.flush_active()

    assert "old-user" in runtime._known_history_item_ids
    assert segment is not None
    assert segment.person_id == "A"
    assert segment.turns[0].user_text == "I like jasmine tea."
    assert segment.turns[0].assistant_text == "Got it."


def test_preference_segment_handoff_emits_memory_flush_event() -> None:
    runtime = FakeRuntime()
    runtime._latency = FakeLatency()
    first_turn = make_turn("rt-a", owner_id="A", finalized=True)
    first_turn.user_transcript = "I like green tea."
    first_turn.assistant_transcript = "Noted."
    second_turn = make_turn("rt-b", owner_id="B", finalized=True)
    second_turn.user_transcript = "I prefer coffee."
    second_turn.assistant_transcript = "Got it."

    runtime._maybe_note_preference_turn(first_turn)
    runtime._maybe_note_preference_turn(second_turn)

    assert runtime._latency.events == [
        {
            "event": "memory_segment_flushed",
            "req_id": "rt-a",
            "memory_segment_id": "rt-a",
            "memory_person_id": "A",
            "memory_turn_count": 1,
            "memory_flush_reason": "speaker_handoff",
            "memory_extraction_enabled": False,
            "memory_extraction_scheduled": False,
        }
    ]
