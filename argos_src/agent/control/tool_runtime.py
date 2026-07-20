"""Tool execution runtime for Realtime function calls."""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from argos_src.agent.realtime_turns import (
    PendingToolCall,
    QueuedTurn,
    TURN_PHASE_REQUESTING_FOLLOWUP,
)
from argos_src.agent.runtime_context import parse_tool_output, summarize_tool_payload
from argos_src.observability.observability import clear_request_context, set_request_context

TOOL_LOG_PREVIEW_LIMIT = 900


def log_preview(value: object, *, limit: int = TOOL_LOG_PREVIEW_LIMIT) -> str:
    """Render a bounded single-line value for pipe-separated latency logs."""
    if isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = json.dumps(value, ensure_ascii=True, sort_keys=True)
        except TypeError:
            rendered = str(value)
    rendered = " ".join(rendered.replace("|", "/").split())
    if len(rendered) <= limit:
        return rendered
    return rendered[: max(limit - 3, 0)] + "..."


def enrollment_tool_log_fields(tool_name: str, content: object) -> dict[str, Any]:
    """Extract bounded enrollment diagnostics for structured dashboard rows."""
    if tool_name != "enroll_visible_person":
        return {}
    payload = parse_tool_output(content) or {}
    diagnostics = payload.get("enrollment_diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    similarities = diagnostics.get("similarities_to_reference")
    if isinstance(similarities, list):
        rendered_similarities = ",".join(str(item) for item in similarities[:8])
    else:
        rendered_similarities = ""
    fields = {
        "tool_enrollment_failure_reason": payload.get("failure_reason"),
        "tool_enrollment_accepted_frames": diagnostics.get("accepted_frame_count"),
        "tool_enrollment_consistent_frames": diagnostics.get("consistent_frame_count"),
        "tool_enrollment_required_frames": diagnostics.get("required_stable_frames"),
        "tool_enrollment_similarity_threshold": diagnostics.get(
            "min_embedding_similarity"
        ),
        "tool_enrollment_best_failed_similarity": diagnostics.get(
            "best_failed_similarity"
        ),
        "tool_enrollment_best_failed_shortfall": diagnostics.get(
            "best_failed_shortfall"
        ),
        "tool_enrollment_similarities": rendered_similarities,
    }
    return {
        key: value
        for key, value in fields.items()
        if value not in (None, "")
    }


class ToolRuntime:
    """Execute model-requested tools and insert their outputs into the session."""

    def __init__(self, host: Any) -> None:
        self._host = host

    def execute(self, pending: PendingToolCall) -> None:
        host = self._host
        turn = host._turns_by_req_id.get(pending.turn_req_id)
        tool = host._tool_registry.get(pending.tool_name)
        if turn is None or host._is_turn_terminal(turn):
            return

        tool_allowed = True
        if tool is None:
            arguments = {}
            result: object = json.dumps(
                {
                    "success": False,
                    "status": "error",
                    "error": f"Unknown tool: {pending.tool_name}",
                }
            )
        else:
            try:
                arguments = json.loads(pending.arguments_json or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("Realtime tool arguments must decode to an object.")
            except Exception as exc:
                arguments = {}
                result = json.dumps({"success": False, "error": str(exc)})
            else:
                set_request_context(
                    req_id=turn.req_id,
                    owner_id=turn.owner_id,
                    owner_source=turn.owner_source,
                    speech_end_perf_s=turn.speech_end_perf_s,
                    speech_end_unix_s=turn.speech_end_unix_s,
                    transcript_perf_s=turn.transcript_perf_s,
                )
                try:
                    allowed_fn = getattr(host, "_is_tool_allowed_for_turn", None)
                    tool_allowed = not callable(allowed_fn) or bool(
                        allowed_fn(turn, pending.tool_name)
                    )
                    if not tool_allowed:
                        result = json.dumps(
                            {
                                "success": False,
                                "status": "blocked",
                                "error": "Tool use is not allowed for this internal event turn.",
                            }
                        )
                    else:
                        result = self.invoke_tool(
                            tool,
                            arguments,
                            call_id=pending.call_id,
                        )
                except Exception as exc:
                    host.logger.exception(
                        "Tool execution failed req_id=%s tool=%s",
                        turn.req_id,
                        pending.tool_name,
                    )
                    result = json.dumps({"success": False, "error": str(exc)})
                finally:
                    clear_request_context()

        content, artifact = self.split_tool_result(result)
        if tool_allowed:
            self.maybe_handle_side_effects(pending.tool_name, content)
        posture, summary = summarize_tool_payload(pending.tool_name, content)
        if tool_allowed:
            host._last_tool_name = pending.tool_name
            if posture:
                host._robot_posture = posture
            if summary:
                host._last_tool_summary = summary

        host._tool_latency.emit(
            event="tool_result",
            req_id=turn.req_id,
            tool=pending.tool_name,
            call_id=pending.call_id,
            tool_success=parse_tool_output(content).get("success"),
            tool_result_preview=log_preview(content),
            **enrollment_tool_log_fields(pending.tool_name, content),
            **(
                host._exchange_log_fields(turn)
                if callable(getattr(host, "_exchange_log_fields", None))
                else {}
            ),
        )
        output = host._stringify_tool_output(content)
        item_id = host._state_controller()._new_local_history_item_id()
        host._register_turn_history_item(
            turn,
            item_id,
            item_type="function_call_output",
            status="done",
            permitted_for_inference=True,
            input_item={
                "id": item_id,
                "type": "function_call_output",
                "call_id": pending.call_id,
                "output": output,
                "status": "completed",
            },
        )
        update_snapshot = getattr(host, "_update_history_item_snapshot", None)
        if callable(update_snapshot):
            update_snapshot(
                item_id,
                text="\n".join(
                    part for part in (f"call_id={pending.call_id}", output) if part
                ),
                item_type="function_call_output",
                status="done",
            )
        self.maybe_append_artifact_message(turn, pending.tool_name, artifact)
        turn.pending_tool_calls = max(0, turn.pending_tool_calls - 1)
        turn.pending_call_ids.discard(pending.call_id)
        turn.pending_tool_names_by_call_id.pop(pending.call_id, None)
        response_state = turn.response_outputs.get(pending.source_response_id)
        if response_state is not None:
            response_state.completed_call_ids.add(pending.call_id)
            response_state.last_progress_at = time.time()
        if pending.function_item_id:
            turn.function_call_item_ids.add(pending.function_item_id)
        if host._is_turn_terminal(turn):
            return
        self.maybe_request_followup(turn, pending.source_response_id)

    def maybe_request_followup(
        self,
        turn: QueuedTurn,
        source_response_id: str,
    ) -> bool:
        """Request one follow-up after response.done and all of its tools settle."""
        host = self._host
        if host._is_turn_terminal(turn) or turn.pending_tool_calls > 0:
            return False
        response_state = turn.response_outputs.get(str(source_response_id or ""))
        if response_state is not None:
            if not response_state.response_done:
                return False
            if not response_state.expected_call_ids.issubset(
                response_state.completed_call_ids
            ):
                return False
            if response_state.followup_requested:
                return False
            response_state.followup_requested = True
        elif source_response_id:
            return False
        if turn.pending_tool_calls == 0:
            set_phase = getattr(host, "_set_turn_phase", None)
            if callable(set_phase):
                set_phase(
                    turn,
                    TURN_PHASE_REQUESTING_FOLLOWUP,
                    trigger="tool_results_complete",
                )
            host._send_response_create(turn)
            return True
        return False

    @staticmethod
    def invoke_tool(
        tool: Any,
        arguments: dict[str, Any],
        *,
        call_id: str = "",
    ) -> object:
        if hasattr(tool, "invoke"):
            if (
                str(getattr(tool, "response_format", "") or "")
                == "content_and_artifact"
                and str(call_id or "").strip()
            ):
                return tool.invoke(
                    {
                        "type": "tool_call",
                        "id": str(call_id),
                        "name": str(getattr(tool, "name", "") or ""),
                        "args": dict(arguments),
                    }
                )
            return tool.invoke(arguments)
        run_fn = getattr(tool, "_run", None)
        if callable(run_fn):
            return run_fn(**arguments)
        func = getattr(tool, "func", None)
        if callable(func):
            return func(**arguments)
        raise RuntimeError(f"Tool '{getattr(tool, 'name', tool)}' is not invokable")

    @staticmethod
    def split_tool_result(result: object) -> tuple[object, Optional[dict[str, Any]]]:
        artifact = getattr(result, "artifact", None)
        if isinstance(artifact, dict) and hasattr(result, "content"):
            return getattr(result, "content"), artifact
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return result[0], result[1]
        return result, None

    def maybe_append_artifact_message(
        self,
        turn: QueuedTurn,
        tool_name: str,
        artifact: Optional[dict[str, Any]],
    ) -> None:
        if not artifact:
            return
        images = list(artifact.get("images", []) or [])
        if not images:
            return
        content: list[dict[str, str]] = [
            {
                "type": "input_text",
                "text": f"[TOOL ARTIFACT] Visual result from {tool_name}. Analyze the attached image(s) directly.",
            }
        ]
        for encoded in images:
            rendered = str(encoded or "").strip()
            if not rendered:
                continue
            if not rendered.startswith("data:"):
                rendered = f"data:image/png;base64,{rendered}"
            content.append({"type": "input_image", "image_url": rendered})
        if len(content) == 1:
            return
        item_id = self._host._state_controller()._new_local_history_item_id()
        self._host._register_turn_history_item(
            turn,
            item_id,
            item_type="message",
            role="user",
            status="done",
            permitted_for_inference=True,
            input_item={
                "id": item_id,
                "type": "message",
                "role": "user",
                "status": "completed",
                "content": content,
            },
        )
        update_snapshot = getattr(self._host, "_update_history_item_snapshot", None)
        if callable(update_snapshot):
            update_snapshot(
                item_id,
                text="[TOOL ARTIFACT] Visual result from " + str(tool_name),
                item_type="message",
                role="user",
                status="done",
            )

    @staticmethod
    def build_schema(tool: Any) -> dict[str, Any]:
        schema_source = getattr(tool, "args_schema", None)
        parameters: dict[str, Any] = {"type": "object", "properties": {}}
        if schema_source is not None:
            try:
                parameters = dict(schema_source.model_json_schema())
            except Exception:
                try:
                    parameters = dict(schema_source.schema())
                except Exception:
                    parameters = {"type": "object", "properties": {}}
        parameters.pop("title", None)
        return {
            "type": "function",
            "name": str(getattr(tool, "name", "") or ""),
            "description": str(getattr(tool, "description", "") or ""),
            "parameters": parameters,
        }

    def maybe_handle_side_effects(self, tool_name: str, content: object) -> None:
        if str(tool_name or "").strip() != "enroll_visible_person":
            return
        payload = parse_tool_output(content)
        if not payload or not bool(payload.get("success", False)):
            return
        person_id = str(payload.get("person_id", "") or "").strip()
        if not person_id:
            return
        arm_fn = getattr(self._host, "_arm_pending_voice_enrollment", None)
        if callable(arm_fn):
            arm_fn(person_id)
