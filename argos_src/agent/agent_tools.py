"""Tool execution helpers for the Argos agent runtime."""

from __future__ import annotations

import json
from typing import Any, Optional

from argos_src.agent.realtime_turns import PendingToolCall, QueuedTurn
from argos_src.agent.runtime_context import parse_tool_output, summarize_tool_payload
from argos_src.observability.observability import (
    clear_request_context,
    set_request_context,
)


def _debug_log_value(value: object, *, limit: int = 1200) -> str:
    rendered = str(value or "").replace("\n", " ").replace("\r", " ").replace("|", "/")
    if len(rendered) > limit:
        return rendered[: max(0, limit - 3)] + "..."
    return rendered


class RealtimeAgentToolsMixin:
    def _execute_tool_call(self, pending: PendingToolCall) -> None:
        turn = self._turns_by_req_id.get(pending.turn_req_id)
        tool = self._tool_registry.get(pending.tool_name)
        if turn is None or tool is None or self._is_turn_terminal(turn):
            return

        try:
            arguments = json.loads(pending.arguments_json or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("Realtime tool arguments must decode to an object.")
        except Exception as exc:
            arguments = {}
            result: object = json.dumps({"success": False, "error": str(exc)})
        else:
            set_request_context(
                req_id=turn.req_id,
                speech_end_perf_s=turn.speech_end_perf_s,
                speech_end_unix_s=turn.speech_end_unix_s,
                transcript_perf_s=turn.transcript_perf_s,
            )
            try:
                result = self._invoke_tool(tool, arguments)
            except Exception as exc:
                self.logger.exception(
                    "Tool execution failed req_id=%s tool=%s",
                    turn.req_id,
                    pending.tool_name,
                )
                result = json.dumps({"success": False, "error": str(exc)})
            finally:
                clear_request_context()

        content, artifact = self._split_tool_result(result)
        self._maybe_handle_tool_side_effects(pending.tool_name, content)
        posture, summary = summarize_tool_payload(pending.tool_name, content)
        self._last_tool_name = pending.tool_name
        if posture:
            self._robot_posture = posture
        if summary:
            self._last_tool_summary = summary

        self._tool_latency.emit(
            event="tool_result",
            req_id=turn.req_id,
            tool=pending.tool_name,
        )
        if pending.tool_name == "resolve_employee_identity":
            payload = parse_tool_output(content) or {}
            data = payload.get("data") if isinstance(payload, dict) else {}
            candidates = data.get("candidates", []) if isinstance(data, dict) else []
            if not isinstance(candidates, list):
                candidates = []
            turn.metadata.setdefault("completed_tools", []).append(pending.tool_name)
            self._tool_latency.emit(
                event="tool_result_payload",
                req_id=turn.req_id,
                tool=pending.tool_name,
                success=payload.get("success") if isinstance(payload, dict) else None,
                status=payload.get("status") if isinstance(payload, dict) else None,
                candidate_count=(
                    payload.get("candidate_count") if isinstance(payload, dict) else None
                ),
                message=_debug_log_value(
                    payload.get("message") if isinstance(payload, dict) else ""
                ),
                shared_first_name=_debug_log_value(arguments.get("shared_first_name")),
                shared_last_name=_debug_log_value(arguments.get("shared_last_name")),
                shared_name=_debug_log_value(arguments.get("shared_name")),
                candidates_json=_debug_log_value(
                    json.dumps(candidates[:3], ensure_ascii=True, separators=(",", ":"))
                ),
            )
        self._queue_pending_local_created_item(turn.req_id, "function_call_output")
        self._send_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": pending.call_id,
                    "output": self._stringify_tool_output(content),
                },
            }
        )
        self._maybe_append_tool_artifact_message(turn, pending.tool_name, artifact)
        turn.pending_tool_calls = max(0, turn.pending_tool_calls - 1)
        turn.pending_call_ids.discard(pending.call_id)
        if pending.function_item_id:
            turn.function_call_item_ids.add(pending.function_item_id)
        if self._is_turn_terminal(turn):
            return
        if turn.pending_tool_calls == 0:
            self._send_response_create(turn)

    def _invoke_tool(self, tool: Any, arguments: dict[str, Any]) -> object:
        if hasattr(tool, "invoke"):
            return tool.invoke(arguments)
        run_fn = getattr(tool, "_run", None)
        if callable(run_fn):
            return run_fn(**arguments)
        func = getattr(tool, "func", None)
        if callable(func):
            return func(**arguments)
        raise RuntimeError(f"Tool '{getattr(tool, 'name', tool)}' is not invokable")

    def _split_tool_result(self, result: object) -> tuple[object, Optional[dict[str, Any]]]:
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return result[0], result[1]
        return result, None

    def _maybe_append_tool_artifact_message(
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
        self._queue_pending_local_created_item(turn.req_id, "message", "user")
        self._send_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": content,
                },
            }
        )

    def _build_tool_schema(self, tool: Any) -> dict[str, Any]:
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

    def _maybe_handle_tool_side_effects(self, tool_name: str, content: object) -> None:
        if str(tool_name or "").strip() != "enroll_visible_person":
            return
        payload = parse_tool_output(content)
        if not payload or not bool(payload.get("success", False)):
            return
        person_id = str(payload.get("person_id", "") or "").strip()
        if not person_id:
            return
        arm_fn = getattr(self, "_arm_pending_voice_enrollment", None)
        if callable(arm_fn):
            arm_fn(person_id)
