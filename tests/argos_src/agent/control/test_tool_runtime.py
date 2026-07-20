from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from argos_src.agent.control.tool_runtime import ToolRuntime
from argos_src.agent.realtime_turns import PendingToolCall, QueuedTurn, ResponseOutputState
from argos_src.tools.unitree_go2.vision.capture_scene import get_capture_scene_tool


class _Latency:
    def __init__(self) -> None:
        self.events = []

    def emit(self, **fields):
        self.events.append(fields)


class _Tool:
    name = "fake_tool"
    description = "fake"

    def invoke(self, arguments):
        return {"success": True, "arguments": arguments}


class _Host:
    def __init__(self) -> None:
        self.logger = SimpleNamespace(exception=lambda *_args, **_kwargs: None)
        self._turns_by_req_id = {}
        self._tool_registry = {"fake_tool": _Tool()}
        self._tool_latency = _Latency()
        self._last_tool_name = None
        self._last_tool_summary = None
        self._robot_posture = "standing"
        self.sent_events = []
        self.pending_items = []
        self.followups = []
        self.armed_person_ids = []
        self.registered_items = []
        self.snapshots = {}

    def _state_controller(self):
        return SimpleNamespace(_new_local_history_item_id=lambda: f"local-{len(self.registered_items) + 1}")

    def _register_turn_history_item(self, turn, item_id, **kwargs):
        turn.history_item_ids.add(item_id)
        self.registered_items.append((turn.req_id, item_id, kwargs))

    def _update_history_item_snapshot(self, item_id, **kwargs):
        self.snapshots[item_id] = kwargs

    def _is_turn_terminal(self, turn):
        return turn is None or bool(getattr(turn, "finalized", False))

    def _queue_pending_local_created_item(self, req_id, expected_type, expected_role=""):
        self.pending_items.append((req_id, expected_type, expected_role))

    def _send_event(self, payload):
        self.sent_events.append(payload)

    def _stringify_tool_output(self, content):
        return str(content)

    def _send_response_create(self, turn):
        self.followups.append(turn.req_id)

    def _arm_pending_voice_enrollment(self, person_id):
        self.armed_person_ids.append(person_id)


def _turn(req_id: str = "rt-tool") -> QueuedTurn:
    return QueuedTurn(
        kind="audio",
        req_id=req_id,
        speech_end_perf_s=0.0,
        speech_end_unix_s=0.0,
        transcript_perf_s=0.0,
    )


def test_tool_runtime_waits_for_all_tool_results_before_followup() -> None:
    host = _Host()
    runtime = ToolRuntime(host)
    turn = _turn()
    turn.pending_tool_calls = 2
    turn.pending_call_ids = {"call-1", "call-2"}
    host._turns_by_req_id[turn.req_id] = turn

    runtime.execute(
        PendingToolCall(
            turn_req_id=turn.req_id,
            call_id="call-1",
            tool_name="fake_tool",
            arguments_json='{"a": 1}',
        )
    )
    assert turn.pending_tool_calls == 1
    assert host.followups == []

    runtime.execute(
        PendingToolCall(
            turn_req_id=turn.req_id,
            call_id="call-2",
            tool_name="fake_tool",
            arguments_json='{"b": 2}',
        )
    )

    assert turn.pending_tool_calls == 0
    assert host.followups == [turn.req_id]
    assert host.pending_items == []
    assert host.sent_events == []
    assert host.registered_items[-1][2]["item_type"] == "function_call_output"
    assert host.registered_items[-1][2]["permitted_for_inference"] is True


def test_tool_runtime_appends_tool_artifact_message() -> None:
    host = _Host()
    runtime = ToolRuntime(host)
    turn = _turn()

    runtime.maybe_append_artifact_message(
        turn,
        "capture_scene",
        {"images": ["abc123"]},
    )

    assert host.pending_items == []
    assert host.sent_events == []
    item = host.registered_items[0][2]["input_item"]
    assert item["role"] == "user"
    assert item["content"][1]["image_url"] == "data:image/png;base64,abc123"


def test_content_and_artifact_tool_is_invoked_with_tool_call_shape() -> None:
    seen = []

    class _Result:
        content = {"success": True}
        artifact = {"images": ["abc123"]}

    class _ArtifactTool:
        name = "capture_scene"
        response_format = "content_and_artifact"

        def invoke(self, value):
            seen.append(value)
            return _Result()

    result = ToolRuntime.invoke_tool(
        _ArtifactTool(),
        {"camera": "front"},
        call_id="call-capture",
    )

    assert seen == [
        {
            "type": "tool_call",
            "id": "call-capture",
            "name": "capture_scene",
            "args": {"camera": "front"},
        }
    ]
    assert ToolRuntime.split_tool_result(result) == (
        {"success": True},
        {"images": ["abc123"]},
    )


def test_real_capture_scene_langchain_tool_preserves_image_artifact() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    face_service = SimpleNamespace(
        get_cached_latest_frame=lambda **_kwargs: (frame, "head_realsense", 1.0)
    )
    tool = get_capture_scene_tool(face_service)

    result = ToolRuntime.invoke_tool(
        tool,
        {"camera_resource": "head_realsense"},
        call_id="call-real-capture",
    )
    content, artifact = ToolRuntime.split_tool_result(result)

    assert '"success": true' in content
    assert artifact is not None
    assert len(artifact["images"]) == 1
    assert artifact["images"][0]


def test_tool_followup_waits_for_response_done_even_when_tool_finishes_first() -> None:
    host = _Host()
    runtime = ToolRuntime(host)
    turn = _turn()
    turn.pending_tool_calls = 1
    turn.pending_call_ids = {"call-1"}
    turn.response_outputs["resp-1"] = ResponseOutputState(
        response_id="resp-1",
        expected_call_ids={"call-1"},
    )
    host._turns_by_req_id[turn.req_id] = turn

    runtime.execute(
        PendingToolCall(
            turn_req_id=turn.req_id,
            call_id="call-1",
            tool_name="fake_tool",
            arguments_json="{}",
            source_response_id="resp-1",
        )
    )
    assert host.followups == []

    turn.response_outputs["resp-1"].response_done = True
    assert runtime.maybe_request_followup(turn, "resp-1") is True
    assert runtime.maybe_request_followup(turn, "resp-1") is False
    assert host.followups == [turn.req_id]


def test_unknown_tool_returns_error_and_does_not_deadlock_chain() -> None:
    host = _Host()
    runtime = ToolRuntime(host)
    turn = _turn()
    turn.pending_tool_calls = 1
    turn.pending_call_ids = {"call-unknown"}
    state = ResponseOutputState(
        response_id="resp-unknown",
        expected_call_ids={"call-unknown"},
        response_done=True,
    )
    turn.response_outputs[state.response_id] = state
    host._turns_by_req_id[turn.req_id] = turn

    runtime.execute(
        PendingToolCall(
            turn_req_id=turn.req_id,
            call_id="call-unknown",
            tool_name="missing_tool",
            arguments_json="{}",
            source_response_id=state.response_id,
        )
    )

    assert turn.pending_tool_calls == 0
    assert host.followups == [turn.req_id]
    assert "Unknown tool: missing_tool" in host.registered_items[-1][2]["input_item"]["output"]


def test_tool_runtime_builds_schema_without_title() -> None:
    class _ArgsSchema:
        @staticmethod
        def model_json_schema():
            return {"title": "Args", "type": "object", "properties": {"a": {"type": "number"}}}

    tool = SimpleNamespace(name="do_it", description="Does it", args_schema=_ArgsSchema)

    schema = ToolRuntime.build_schema(tool)

    assert schema == {
        "type": "function",
        "name": "do_it",
        "description": "Does it",
        "parameters": {"type": "object", "properties": {"a": {"type": "number"}}},
    }


def test_tool_runtime_handles_enrollment_side_effect() -> None:
    host = _Host()
    runtime = ToolRuntime(host)

    runtime.maybe_handle_side_effects(
        "enroll_visible_person",
        {"success": True, "person_id": "person-1"},
    )

    assert host.armed_person_ids == ["person-1"]
