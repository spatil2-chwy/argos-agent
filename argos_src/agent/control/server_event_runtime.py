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
            if looks_like_audio:
                self.replay_pending_input_transcription(item_id)
            return

        item_type = str(item.get("type", "") or "").strip()
        role = str(item.get("role", "") or "").strip()
        response_id = server_event_response_id(event, item=item)
        req_id = ""

        if item_type == "message" and role == "user":
            if self._conversation_item_looks_like_audio_input(item):
                req_id = self._consume_pending_audio_turn_req_id(include_terminal=True)
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
        if req_id and item_type == "message" and role == "user":
            self.replay_pending_input_transcription(item_id)

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
            req_id = self._consume_pending_audio_turn_req_id(include_terminal=True)
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
        transcription_state = str(
            (turn.metadata or {}).get("_transcription_state", TranscriptionState.NONE.value)
        )
        if transcription_state not in {
            TranscriptionState.COMPLETED.value,
            TranscriptionState.FAILED.value,
        }:
            self._set_transcription_state(
                turn,
                TranscriptionState.PENDING,
                trigger="input_audio_buffer.committed",
                item_id=item_id,
            )
        self.replay_pending_input_transcription(item_id)

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

    def _defer_input_transcription(self, event: dict[str, Any], item_id: str) -> None:
        if not item_id:
            return
        with self._turn_lock:
            self._pending_input_transcription_events[item_id] = dict(event)
        self._latency.emit(
            event="input_transcription_deferred",
            item_id=item_id,
        )
        self.logger.debug(
            "Deferred input transcription until audio turn binding item_id=%s",
            item_id,
        )

    def replay_pending_input_transcription(self, item_id: str) -> None:
        rendered_item_id = str(item_id or "").strip()
        if not rendered_item_id:
            return
        with self._turn_lock:
            event = self._pending_input_transcription_events.pop(rendered_item_id, None)
        if event is None:
            return
        self._latency.emit(
            event="input_transcription_replayed",
            item_id=rendered_item_id,
        )
        event_type = str(event.get("type") or "").strip()
        if event_type.endswith(".completed"):
            self.handle_input_transcription_completed(event)
        elif event_type.endswith(".failed"):
            self.handle_input_transcription_failed(event)

    def handle_input_transcription_completed(self, event: dict[str, Any]) -> None:
        transcript = str(event.get("transcript", "") or "").strip()
        item_id = str(event.get("item_id") or "").strip()
        turn = self._resolve_turn_for_item(item_id) if item_id else None
        if turn is None and item_id:
            req_id = self._consume_pending_audio_turn_req_id(include_terminal=True)
            if req_id:
                turn = self._turns_by_req_id.get(req_id)
                if turn is not None:
                    self._bind_item_id_to_turn(turn, item_id)
        if turn is None:
            self._defer_input_transcription(event, item_id)
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
            if self._is_turn_terminal(turn):
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
            req_id = self._consume_pending_audio_turn_req_id(include_terminal=True)
            if req_id:
                turn = self._turns_by_req_id.get(req_id)
                if turn is not None:
                    self._bind_item_id_to_turn(turn, item_id)
                    if not turn.user_item_id:
                        turn.user_item_id = item_id
        if turn is None:
            self._defer_input_transcription(event, item_id)
            return
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
        if item_id:
            self._bind_item_id_to_turn(turn, item_id)
            turn.assistant_item_id = item_id
            turn.assistant_item_ids.add(item_id)
        audio_bytes = base64.b64decode(str(event.get("delta", "") or ""))
        if not audio_bytes:
            return
        if not turn.audio_started:
            self._playback_controller().transition(
                PlaybackState.BUFFERING,
                trigger="output_audio.delta",
                req_id=turn.req_id,
                stream_id=response_id,
            )
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
                self._playback_controller().transition(
                    PlaybackState.PLAYING,
                    trigger="output_audio.delta",
                    req_id=turn.req_id,
                    stream_id=response_id,
                )
            return
        turn.audio_started = True
        turn.audio_started_at = time.time()
        turn.last_playback_progress_at = turn.audio_started_at
        self._set_turn_phase(
            turn,
            TURN_PHASE_PLAYING,
            trigger="output_audio.delta",
        )
        if turn.kind == "audio" and float(turn.speech_end_perf_s) > 0.0:
            first_audio_perf = perf_now()
            self._latency.timing(
                "first_audio_latency_s",
                first_audio_perf - turn.speech_end_perf_s,
                req_id=turn.req_id,
                **self._exchange_log_fields(turn),
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
        self._playback_controller().transition(
            PlaybackState.PLAYING,
            trigger="output_audio.delta",
            req_id=turn.req_id,
            stream_id=turn.response_id,
        )

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
        if item_id:
            self._bind_item_id_to_turn(turn, item_id)
            turn.assistant_item_id = item_id or turn.assistant_item_id
            turn.assistant_item_ids.add(item_id)
            self._history_index().update_item(
                item_id,
                item_type="message",
                role="assistant",
                status="in_progress",
                permitted_for_inference=False,
            )
        turn.assistant_transcript += delta
        if turn.assistant_item_id:
            update_snapshot = getattr(self, "_update_history_item_snapshot", None)
            if callable(update_snapshot):
                update_snapshot(
                    turn.assistant_item_id,
                    text=turn.assistant_transcript,
                    item_type="message",
                    role="assistant",
                )
        self._show_display_subtitle_async(
            self._display_subtitle_window(turn.assistant_transcript),
            duration_ms=5000,
        )
        if turn.phase == TURN_PHASE_FINALIZED:
            self._maybe_note_preference_turn(turn)

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
        if item_id:
            self._bind_item_id_to_turn(turn, item_id)
            turn.assistant_item_id = item_id or turn.assistant_item_id
            turn.assistant_item_ids.add(item_id)
            self._history_index().update_item(
                item_id,
                item_type="message",
                role="assistant",
                status="in_progress",
                permitted_for_inference=False,
            )
        item_type = str(item.get("type", "") or "").strip()
        role = str(item.get("role", "") or "").strip()
        status = str(item.get("status", "") or "").strip()
        if item_type != "message" or role != "assistant":
            return
        if status != "completed" or not turn.audio_started:
            return
        self.arm_playback_completion(turn)

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
            self._bind_item_id_to_turn(turn, item_id)
            turn.function_call_item_ids.add(item_id)
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
        self._cancel_owner_turn_for_tool(turn, tool_name)
        self._call_id_to_req_id[call_id] = turn.req_id
        if item_id:
            self._bind_item_id_to_turn(turn, item_id)
            turn.function_call_item_ids.add(item_id)
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
        final_transcript = self._transcript_from_response(response)
        if final_transcript and (
            not turn.assistant_transcript
            or len(final_transcript) >= len(turn.assistant_transcript.strip())
        ):
            turn.assistant_transcript = final_transcript
        quarantined = False
        quarantine_fn = getattr(self, "_quarantine_anonymous_history_if_needed", None)
        if callable(quarantine_fn):
            quarantined = bool(quarantine_fn(turn))
        if turn.assistant_item_id and turn.assistant_transcript.strip():
            update_snapshot = getattr(self, "_update_history_item_snapshot", None)
            if callable(update_snapshot):
                update_snapshot(
                    turn.assistant_item_id,
                    text=turn.assistant_transcript,
                    item_type="message",
                    role="assistant",
                    status=status,
                )
            self._history_index().update_item(
                turn.assistant_item_id,
                item_type="message",
                role="assistant",
                status="done" if status == "completed" else status,
                permitted_for_inference=status == "completed" and not quarantined,
                input_item={
                    "id": turn.assistant_item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": turn.assistant_transcript.strip(),
                        }
                    ],
                }
                if status == "completed" and not quarantined
                else None,
            )
        if turn.audio_started and turn.assistant_transcript.strip():
            self._show_display_subtitle_async(
                self._display_subtitle_window(turn.assistant_transcript),
                duration_ms=5000,
            )
        output_items = response.get("output", []) or []
        for output_item in output_items:
            item_id = str(output_item.get("id", "") or "").strip()
            item_type = str(output_item.get("type", "") or "").strip()
            if item_id:
                self._bind_item_id_to_turn(turn, item_id)
                if item_type == "function_call":
                    turn.function_call_item_ids.add(item_id)
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
                    if not turn.assistant_item_id:
                        turn.assistant_item_id = item_id
                    if status == "completed" and turn.assistant_transcript.strip():
                        self._history_index().update_item(
                            item_id,
                            item_type="message",
                            role="assistant",
                            status="done",
                            permitted_for_inference=not quarantined,
                            input_item={
                                "id": item_id,
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": turn.assistant_transcript.strip(),
                                    }
                                ],
                            }
                            if not quarantined
                            else None,
                        )
        has_function_call = any(
            str(item.get("type", "") or "") == "function_call" for item in output_items
        )
        if has_function_call or turn.pending_tool_calls > 0:
            self._set_turn_phase(
                turn,
                TURN_PHASE_WAITING_TOOLS,
                trigger="response.done_waiting_tools",
            )
            return

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
