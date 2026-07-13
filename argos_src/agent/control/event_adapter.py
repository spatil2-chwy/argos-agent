"""Realtime server-event adapter for the agent runtime."""

from __future__ import annotations

from typing import Any

from argos_src.agent.agent_events.parsing import server_event_response, server_event_type
from argos_src.agent.control.types import SessionState

_HANDLER_BY_EVENT_TYPE = {
    "conversation.item.created": "_handle_conversation_item_created",
    "conversation.item.deleted": "_handle_conversation_item_deleted",
    "response.created": "_handle_response_created",
    "conversation.item.input_audio_transcription.completed": (
        "_handle_input_transcription_completed"
    ),
    "conversation.item.input_audio_transcription.failed": (
        "_handle_input_transcription_failed"
    ),
    "input_audio_buffer.committed": "_handle_input_audio_buffer_committed",
    "response.output_text.delta": "_handle_output_text_delta",
    "response.output_item.done": "_handle_output_item_done",
    "response.function_call_arguments.delta": "_handle_function_call_delta",
    "response.function_call_arguments.done": "_handle_function_call_done",
    "response.done": "_handle_response_done",
    "error": "_handle_server_error",
}

_AUDIO_DELTA_EVENT_TYPES = frozenset({"response.output_audio.delta"})
_TRANSCRIPT_DELTA_EVENT_TYPES = frozenset({"response.output_audio_transcript.delta"})


class RealtimeEventAdapter:
    """Route OpenAI Realtime server events into runtime handlers."""

    def __init__(self, host: Any) -> None:
        self._host = host

    def handle(self, event: dict[str, Any]) -> bool:
        host = self._host
        event_type = server_event_type(event)
        if event_type == "session.created":
            self._handle_session_created(event)
            return True
        if event_type == "session.updated":
            host._session_ready.set()
            setter = getattr(host, "_set_session_state", None)
            if callable(setter):
                setter(SessionState.READY, trigger="session.updated")
            host.logger.info("Realtime session updated")
            return True
        if event_type in _AUDIO_DELTA_EVENT_TYPES:
            host._handle_output_audio_delta(event)
            return True
        if event_type in _TRANSCRIPT_DELTA_EVENT_TYPES:
            host._handle_output_transcript_delta(event)
            return True
        handler_name = _HANDLER_BY_EVENT_TYPE.get(event_type)
        if not handler_name:
            return False
        getattr(host, handler_name)(event)
        return True

    def _handle_session_created(self, event: dict[str, Any]) -> None:
        host = self._host
        session = event.get("session", {}) or {}
        response = server_event_response(event)
        if not session and response:
            session = response
        host._session_id = str(session.get("id", "") or "").strip()
        host._session_estimated_cost_usd = 0.0
        setter = getattr(host, "_set_session_state", None)
        if callable(setter):
            setter(SessionState.CONFIGURING, trigger="session.created")
        host.logger.info(
            "Realtime session created session_id=%s model=%s voice=%s",
            host._session_id or "<unknown>",
            session.get("model", host.realtime_profile.model),
            host.realtime_profile.voice,
        )
