"""OpenAI Realtime server-event mutation handlers."""

from __future__ import annotations

import base64
import threading
import time
from typing import Any

from argos_src.agent.agent_events.parsing import (
    server_event_item,
    server_event_item_id,
    server_event_response,
    server_event_response_id,
)
from argos_src.agent.control.types import PlaybackState, TranscriptionState
from argos_src.agent.control.tool_runtime import log_preview
from argos_src.agent.realtime_turns import (
    TURN_PHASE_CANCELED,
    TURN_PHASE_FINALIZED,
    TURN_PHASE_MODEL_DONE,
    TURN_PHASE_PLAYING,
    TURN_PHASE_WAITING_FIRST_AUDIO,
    TURN_PHASE_WAITING_TOOLS,
    PendingToolCall,
    QueuedTurn,
    ResponseOutputState,
)
from argos_src.observability.observability import perf_now
from argos_src.observability.pricing import estimate_transcription_cost


class ServerEventRuntime:
    """Apply server events to turn, playback, tool, and transcription state."""

    def __init__(self, host: Any) -> None:
        object.__setattr__(self, "_host", host)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._host, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_host":
            object.__setattr__(self, name, value)
            return
        setattr(self._host, name, value)

    @staticmethod
    def _response_output_state(
        turn: QueuedTurn,
        response_id: str,
    ) -> ResponseOutputState:
        rendered = str(response_id or turn.response_id or "").strip()
        state = turn.response_outputs.get(rendered)
        if state is None:
            state = ResponseOutputState(response_id=rendered)
            turn.response_outputs[rendered] = state
        return state

    def _release_response_audio(
        self,
        turn: QueuedTurn,
        state: ResponseOutputState,
    ) -> bool:
        """Release one classified audible response into the shared playback path."""
        if state.released or state.discarded or not state.audio:
            return False
        response_id = state.response_id or turn.response_id
        audio_bytes = bytes(state.audio)
        state.audio.clear()
        state.released = True
        self._playback_buffer.append(audio_bytes)

        if state.transcript:
            turn.audible_transcript_parts.append(state.transcript)
            turn.assistant_transcript = "".join(turn.audible_transcript_parts)
        if state.assistant_item_id:
            turn.assistant_item_id = state.assistant_item_id

        if turn.audio_started:
            with self._turn_lock:
                self._playback_req_id = turn.req_id
                self._playback_stream_id = response_id
                self._playback_item_id = turn.assistant_item_id
            turn.last_playback_progress_at = time.time()
            self.engagement.on_playback_event(
                "playback_started",
                turn.req_id,
                stream_id=response_id,
            )
            self._playback_controller().transition(
                PlaybackState.PLAYING,
                trigger="terminal_response_released",
                req_id=turn.req_id,
                stream_id=response_id,
            )
            return True

        self._playback_controller().transition(
            PlaybackState.BUFFERING,
            trigger="terminal_response_released",
            req_id=turn.req_id,
            stream_id=response_id,
        )
        turn.audio_started = True
        turn.audio_started_at = time.time()
        turn.last_playback_progress_at = turn.audio_started_at
        self._set_turn_phase(
            turn,
            TURN_PHASE_PLAYING,
            trigger="terminal_response_released",
        )
        if turn.kind == "audio" and float(turn.speech_end_perf_s) > 0.0:
            self._latency.timing(
                "terminal_audio_release_latency_s",
                perf_now() - turn.speech_end_perf_s,
                req_id=turn.req_id,
                **self._exchange_log_fields(turn),
            )
        self.engagement.on_agent_output_started(turn.req_id, stream_id=response_id)
        self._set_display_mode_async("speaking")
        with self._turn_lock:
            self._playback_req_id = turn.req_id
            self._playback_stream_id = response_id
            self._playback_item_id = turn.assistant_item_id
            self._played_output_frames = 0
        self.engagement.on_playback_event(
            "playback_started",
            turn.req_id,
            stream_id=response_id,
        )
        self._playback_controller().transition(
            PlaybackState.PLAYING,
            trigger="terminal_response_released",
            req_id=turn.req_id,
            stream_id=response_id,
        )
        return True

    def handle_conversation_item_created(self, event: dict[str, Any]) -> None:
        item = server_event_item(event)
        item_id = server_event_item_id(event, item=item)
        if not item_id:
            return
        if item_id in self._item_id_to_req_id:
            item_type = str(item.get("type", "") or "").strip()
            role = str(item.get("role", "") or "").strip()
            looks_like_audio = (
                item_type == "message"
                and role == "user"
                and self._conversation_item_looks_like_audio_input(item)
            )
            self._register_history_item(
                item_id,
                item_type=item_type,
                role=role,
                status="done" if looks_like_audio else "",
                permitted_for_inference=True if looks_like_audio else None,
            )
            record_snapshot = getattr(self, "_record_history_item_snapshot", None)
            if callable(record_snapshot):
                record_snapshot(item_id, item)
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

        self._register_history_item(
            item_id,
            owner_req_id=req_id,
            item_type=item_type,
            role=role,
            status="done" if item_type == "message" and role in {"user", "system"} else "in_progress",
            permitted_for_inference=True if item_type == "message" and role in {"user", "system"} else None,
        )
        record_snapshot = getattr(self, "_record_history_item_snapshot", None)
        if callable(record_snapshot):
            record_snapshot(item_id, item)
        if req_id:
            turn = self._turns_by_req_id.get(req_id)
            if turn is not None:
                self._register_turn_history_item(
                    turn,
                    item_id,
                    item_type=item_type,
                    role=role,
                    status="done" if item_type == "message" and role in {"user", "system"} else "in_progress",
                    permitted_for_inference=True if item_type == "message" and role in {"user", "system"} else None,
                )
                if item_type == "message" and role == "user" and not turn.user_item_id:
                    turn.user_item_id = item_id
                elif item_type == "message" and role == "assistant":
                    turn.assistant_item_ids.add(item_id)
                elif item_type == "function_call":
                    turn.function_call_item_ids.add(item_id)

    def handle_conversation_item_deleted(self, event: dict[str, Any]) -> None:
        item_id = str(event.get("item_id") or "").strip()
        if not item_id:
            item = server_event_item(event)
            item_id = server_event_item_id(event, item=item)
        if not item_id:
            return
        handler = getattr(self, "_forget_deleted_history_item", None)
        if callable(handler):
            handler(item_id)

    def handle_input_audio_buffer_committed(self, event: dict[str, Any]) -> None:
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
        self._history_index().update_item(
            item_id,
            item_type="message",
            role="user",
            status="done",
            permitted_for_inference=True,
        )
        self._set_transcription_state(
            turn,
            TranscriptionState.PENDING,
            trigger="input_audio_buffer.committed",
            item_id=item_id,
        )

    def handle_response_created(self, event: dict[str, Any]) -> None:
        response = server_event_response(event)
        response_id = server_event_response_id(event, response=response)
        if not response_id:
            return
        binding = self._consume_pending_response_binding(response_id)
        if binding is None:
            return
        turn = binding.turn
        turn.pending_response_requests = max(0, turn.pending_response_requests - 1)
        if self._is_turn_terminal(turn):
            if binding.expired_stale:
                self.logger.warning(
                    "Canceling ambiguous expired-stale response_id=%s req_id=%s",
                    response_id,
                    turn.req_id,
                )
            try:
                self._send_event({"type": "response.cancel", "response_id": response_id})
            except Exception:
                self.logger.exception("Failed to cancel terminal response_id=%s", response_id)
            if binding.expired_stale:
                self.recover_pending_response_after_expired_stale(response_id)
            return
        self.logger.info("Realtime response created req_id=%s response_id=%s", turn.req_id, response_id)
        self._response_output_state(turn, response_id)
        self._set_turn_phase(
            turn,
            TURN_PHASE_WAITING_FIRST_AUDIO,
            trigger="response.created",
        )

    def recover_pending_response_after_expired_stale(self, response_id: str) -> None:
        turn = self._next_pending_response_turn()
        if turn is None or self._is_turn_terminal(turn):
            return
        discarded = self._discard_pending_response_turn(turn.req_id)
        if discarded:
            turn.pending_response_requests = max(0, turn.pending_response_requests - discarded)
        self.logger.warning(
            "Reissuing response.create for pending turn after ambiguous stale response response_id=%s req_id=%s discarded_pending_slots=%s",
            response_id,
            turn.req_id,
            discarded,
        )
        self._send_response_create(turn)

    def handle_input_transcription_completed(self, event: dict[str, Any]) -> None:
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
        self._set_transcription_state(
            turn,
            TranscriptionState.COMPLETED,
            trigger="input_transcription.completed",
            item_id=item_id,
        )
        if transcript:
            turn.user_transcript = transcript
            update_snapshot = getattr(self, "_update_history_item_snapshot", None)
            if callable(update_snapshot):
                update_snapshot(
                    item_id,
                    text=transcript,
                    item_type="message",
                    role="user",
                    status="transcribed",
                )
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
                **{
                    key: value
                    for key, value in self._exchange_log_fields(turn).items()
                    if key != "session_id"
                },
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

    def handle_input_transcription_failed(self, event: dict[str, Any]) -> None:
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
        self._set_transcription_state(
            turn,
            TranscriptionState.FAILED,
            trigger="input_transcription.failed",
            item_id=item_id,
            reason=str(error.get("code", "") or error.get("type", "") or "unknown"),
        )
        self.logger.warning(
            "Input transcription failed req_id=%s item_id=%s type=%s code=%s message=%s",
            getattr(turn, "req_id", "<unknown>"),
            item_id or "<unknown>",
            error.get("type", "unknown"),
            error.get("code", "unknown"),
            error.get("message", "unknown"),
        )

    def handle_output_audio_delta(self, event: dict[str, Any]) -> None:
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
        state = self._response_output_state(turn, response_id)
        if state.response_done or state.discarded:
            return
        if item_id:
            self._bind_item_id_to_turn(turn, item_id)
            state.assistant_item_id = item_id
            turn.assistant_item_ids.add(item_id)
            state.assistant_item_ids.add(item_id)
        audio_bytes = base64.b64decode(str(event.get("delta", "") or ""))
        if not audio_bytes:
            return
        if not turn.first_audio_delta_observed:
            turn.first_audio_delta_observed = True
            if turn.kind == "audio" and float(turn.speech_end_perf_s) > 0.0:
                self._latency.timing(
                    "first_audio_latency_s",
                    perf_now() - turn.speech_end_perf_s,
                    req_id=turn.req_id,
                    **self._exchange_log_fields(turn),
                )
        state.audio.extend(audio_bytes)
        state.last_progress_at = time.time()

    def handle_output_transcript_delta(self, event: dict[str, Any]) -> None:
        self._handle_output_text_like_delta(event)

    def handle_output_text_delta(self, event: dict[str, Any]) -> None:
        self._handle_output_text_like_delta(event)

    def _handle_output_text_like_delta(self, event: dict[str, Any]) -> None:
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
        state = self._response_output_state(turn, response_id)
        if state.response_done or state.discarded:
            return
        if item_id:
            self._bind_item_id_to_turn(turn, item_id)
            state.assistant_item_id = item_id or state.assistant_item_id
            state.assistant_item_ids.add(item_id)
            turn.assistant_item_ids.add(item_id)
            self._history_index().update_item(
                item_id,
                item_type="message",
                role="assistant",
                status="in_progress",
                permitted_for_inference=False,
            )
        state.transcript += delta
        state.last_progress_at = time.time()
        if state.assistant_item_id:
            update_snapshot = getattr(self, "_update_history_item_snapshot", None)
            if callable(update_snapshot):
                update_snapshot(
                    state.assistant_item_id,
                    text=state.transcript,
                    item_type="message",
                    role="assistant",
                )

    def arm_playback_completion(self, turn: QueuedTurn) -> None:
        if turn.playback_completion_armed:
            return
        turn.playback_completion_armed = True
        stream_id = str(turn.response_id or self._playback_stream_id or "").strip()
        self.engagement.on_agent_done(has_reply=True, req_id=turn.req_id)
        self._playback_controller().transition(
            PlaybackState.AWAITING_DRAIN
            if turn.response_finished.is_set()
            else PlaybackState.AWAITING_MODEL_DONE,
            trigger="playback_completion_armed",
            req_id=turn.req_id,
            stream_id=stream_id,
        )
        threading.Thread(
            target=self._wait_for_playback_and_complete,
            args=(turn, stream_id),
            daemon=True,
        ).start()

    def handle_output_item_done(self, event: dict[str, Any]) -> None:
        item = server_event_item(event)
        item_id = server_event_item_id(event, item=item)
        response_id = server_event_response_id(event, item=item)
        turn = self._resolve_turn_for_output(response_id=response_id, item_id=item_id)
        if turn is None or self._is_turn_terminal(turn):
            return
        if response_id:
            self._bind_response_id(turn, response_id)
        state = self._response_output_state(turn, response_id)
        item_type = str(item.get("type", "") or "").strip()
        if item_id:
            self._bind_item_id_to_turn(turn, item_id)
            if item_type == "function_call":
                state.function_call_item_ids.add(item_id)
                turn.function_call_item_ids.add(item_id)
            else:
                state.assistant_item_id = item_id or state.assistant_item_id
                state.assistant_item_ids.add(item_id)
                turn.assistant_item_ids.add(item_id)
                self._history_index().update_item(
                    item_id,
                    item_type="message",
                    role="assistant",
                    status="in_progress",
                    permitted_for_inference=False,
                )
        state.last_progress_at = time.time()

    def handle_function_call_delta(self, event: dict[str, Any]) -> None:
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
            state = self._response_output_state(turn, response_id)
            if state.response_done or state.discarded:
                return
            self._bind_item_id_to_turn(turn, item_id)
            turn.function_call_item_ids.add(item_id)
            state.function_call_item_ids.add(item_id)
            state.last_progress_at = time.time()
            self._history_index().update_item(
                item_id,
                item_type="function_call",
                status="in_progress",
                permitted_for_inference=False,
            )

    def handle_function_call_done(self, event: dict[str, Any]) -> None:
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
        state = self._response_output_state(turn, response_id)
        state.expected_call_ids.add(call_id)
        state.last_progress_at = time.time()
        if call_id in turn.pending_call_ids or call_id in state.completed_call_ids:
            return
        self._cancel_owner_turn_for_tool(turn, tool_name)
        self._call_id_to_req_id[call_id] = turn.req_id
        if item_id:
            self._bind_item_id_to_turn(turn, item_id)
            turn.function_call_item_ids.add(item_id)
            state.function_call_item_ids.add(item_id)
            self._history_index().update_item(
                item_id,
                item_type="function_call",
                status="done",
                permitted_for_inference=True,
                input_item={
                    "id": item_id,
                    "type": "function_call",
                    "name": tool_name,
                    "call_id": call_id,
                    "arguments": arguments_json or "{}",
                    "status": "completed",
                },
            )
            update_snapshot = getattr(self, "_update_history_item_snapshot", None)
            if callable(update_snapshot):
                update_snapshot(
                    item_id,
                    text="\n".join(
                        part
                        for part in (
                            f"name={tool_name}",
                            f"call_id={call_id}",
                            f"arguments={arguments_json or '{}'}",
                        )
                        if part
                    ),
                    item_type="function_call",
                )
        turn.pending_tool_calls += 1
        turn.pending_call_ids.add(call_id)
        turn.pending_tool_names_by_call_id[call_id] = tool_name
        self._set_turn_phase(
            turn,
            TURN_PHASE_WAITING_TOOLS,
            trigger="function_call_done",
        )
        self._latency.emit(
            event="tool_call_requested",
            req_id=turn.req_id,
            tool=tool_name,
            call_id=call_id,
            tool_arguments_json=log_preview(arguments_json or "{}"),
            **self._exchange_log_fields(turn),
        )
        self._tool_queue.put(
            PendingToolCall(
                turn_req_id=turn.req_id,
                call_id=call_id,
                tool_name=tool_name,
                arguments_json=arguments_json or "{}",
                function_item_id=item_id,
                source_response_id=state.response_id,
            )
        )

    def handle_response_done(self, event: dict[str, Any]) -> None:
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
        self._set_turn_phase(turn, TURN_PHASE_MODEL_DONE, trigger="response.done")
        self._latency.emit(
            event="response_done",
            req_id=turn.req_id,
            response_status=status,
            response_id=response_id or None,
            **self._exchange_log_fields(turn),
        )
        state = self._response_output_state(turn, response_id)
        state.response_done = True
        state.status = status
        state.last_progress_at = time.time()
        final_transcript = self._transcript_from_response(response)
        if final_transcript and (
            not state.transcript
            or len(final_transcript) >= len(state.transcript.strip())
        ):
            state.transcript = final_transcript
        output_items = response.get("output", []) or []
        for output_item in output_items:
            item_id = str(output_item.get("id", "") or "").strip()
            item_type = str(output_item.get("type", "") or "").strip()
            if item_id:
                self._bind_item_id_to_turn(turn, item_id)
                if item_type == "function_call":
                    turn.function_call_item_ids.add(item_id)
                    state.function_call_item_ids.add(item_id)
                    self._history_index().update_item(
                        item_id,
                        item_type="function_call",
                        status="done",
                        permitted_for_inference=True,
                        input_item={
                            "id": item_id,
                            "type": "function_call",
                            "name": str(output_item.get("name", "") or ""),
                            "call_id": str(output_item.get("call_id", "") or ""),
                            "arguments": str(output_item.get("arguments", "") or "{}"),
                            "status": "completed",
                        },
                    )
                elif item_type == "message":
                    turn.assistant_item_ids.add(item_id)
                    state.assistant_item_id = item_id or state.assistant_item_id
                    state.assistant_item_ids.add(item_id)
                if item_type == "function_call":
                    call_id = str(output_item.get("call_id", "") or "").strip()
                    if call_id:
                        state.expected_call_ids.add(call_id)

        for output_item in output_items:
            if str(output_item.get("type", "") or "").strip() != "function_call":
                continue
            call_id = str(output_item.get("call_id", "") or "").strip()
            if (
                not call_id
                or call_id in turn.pending_call_ids
                or call_id in state.completed_call_ids
            ):
                continue
            self.handle_function_call_done(
                {
                    "response_id": response_id,
                    "item_id": str(output_item.get("id", "") or ""),
                    "call_id": call_id,
                    "name": str(output_item.get("name", "") or ""),
                    "arguments": str(output_item.get("arguments", "") or "{}"),
                }
            )

        has_function_call = bool(state.expected_call_ids or state.function_call_item_ids)
        message_item_ids = set(state.assistant_item_ids)
        if state.assistant_item_id:
            message_item_ids.add(state.assistant_item_id)
        if has_function_call:
            state.discarded = True
            suppressed_audio_bytes = len(state.audio)
            state.audio.clear()
            for item_id in message_item_ids:
                update_snapshot = getattr(self, "_update_history_item_snapshot", None)
                if callable(update_snapshot):
                    update_snapshot(
                        item_id,
                        text=state.transcript,
                        item_type="message",
                        role="assistant",
                        status=status,
                    )
                self._history_index().update_item(
                    item_id,
                    item_type="message",
                    role="assistant",
                    status="done" if status == "completed" else status,
                    permitted_for_inference=False,
                    input_item=None,
                )
            self._latency.emit(
                event="intermediate_response_suppressed",
                req_id=turn.req_id,
                response_id=response_id or None,
                suppressed_audio_bytes=suppressed_audio_bytes,
                suppressed_transcript_chars=len(state.transcript),
                tool_call_count=len(state.expected_call_ids),
                **self._exchange_log_fields(turn),
            )
            self._set_turn_phase(
                turn,
                TURN_PHASE_WAITING_TOOLS,
                trigger="response.done_waiting_tools",
            )
            self._tool_runtime().maybe_request_followup(turn, response_id)
            return

        incomplete_details = response.get("incomplete_details")
        has_audio_reply = bool(state.audio)
        if has_audio_reply:
            self._release_response_audio(turn, state)

        quarantined = False
        quarantine_fn = getattr(self, "_quarantine_anonymous_history_if_needed", None)
        if callable(quarantine_fn):
            quarantined = bool(quarantine_fn(turn))
        for item_id in message_item_ids:
            update_snapshot = getattr(self, "_update_history_item_snapshot", None)
            if callable(update_snapshot):
                update_snapshot(
                    item_id,
                    text=state.transcript,
                    item_type="message",
                    role="assistant",
                    status=status,
                )
            self._history_index().update_item(
                item_id,
                item_type="message",
                role="assistant",
                status="done" if status == "completed" else status,
                permitted_for_inference=status == "completed" and not quarantined,
                input_item={
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": state.transcript.strip(),
                        }
                    ],
                }
                if status == "completed" and not quarantined and state.transcript.strip()
                else None,
            )
        if has_audio_reply and turn.assistant_transcript.strip():
            self._show_display_subtitle_async(
                self._display_subtitle_window(turn.assistant_transcript),
                duration_ms=5000,
            )
        if status == "incomplete" and has_audio_reply:
            self.logger.warning(
                "Realtime response finished incomplete after audio started req_id=%s response_id=%s incomplete_details=%s transcript=%r",
                turn.req_id,
                response_id,
                incomplete_details,
                state.transcript.strip(),
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
                state.transcript.strip(),
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
                state.transcript.strip(),
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
            self.arm_playback_completion(turn)
        else:
            self.engagement.on_agent_done(has_reply=False, req_id=turn.req_id)

    def handle_server_error(self, event: dict[str, Any]) -> None:
        error = event.get("error", {})
        error_type = str(error.get("type", "unknown") or "unknown")
        message = str(error.get("message", "unknown") or "unknown")
        if self._is_no_active_response_cancel_error(error_type=error_type, message=message):
            self.logger.warning(
                "Ignoring stale response.cancel server error type=%s message=%s",
                error_type,
                message,
            )
            return
        self.logger.error(
            "Realtime server error type=%s message=%s",
            error_type,
            message,
        )
        response_id = str(error.get("response_id", "") or "").strip()
        turn = self._resolve_turn_for_output(response_id=response_id)
        if turn is None:
            with self._turn_lock:
                turn = self._active_turn
        if turn is not None:
            metadata = turn.metadata if isinstance(turn.metadata, dict) else {}
            metadata["error_source"] = "openai_realtime"
            metadata["error_type"] = error_type
            metadata["error_code"] = str(error.get("code", "") or "")
            metadata["error_message"] = message
            metadata["server_error_type"] = error_type
            metadata["server_error_code"] = str(error.get("code", "") or "")
            metadata["server_error_message"] = message
            turn.metadata = metadata
            self._terminate_turn(turn, TURN_PHASE_CANCELED, "server_error")

    @staticmethod
    def _is_no_active_response_cancel_error(*, error_type: str, message: str) -> bool:
        return (
            error_type == "invalid_request_error"
            and message.lower().startswith("cancellation failed: no active response found")
        )
