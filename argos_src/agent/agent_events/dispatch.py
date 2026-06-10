"""Routing helpers for OpenAI Realtime server events."""

from __future__ import annotations

from typing import Any

from .parsing import server_event_response, server_event_type

_HANDLER_BY_EVENT_TYPE = {
    "conversation.item.created": "_handle_conversation_item_created",
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


def dispatch_server_event(agent: Any, event: dict[str, Any]) -> bool:
    """Route one server event to the matching agent handler."""
    event_type = server_event_type(event)
    if event_type == "session.created":
        _handle_session_created(agent, event)
        return True
    if event_type == "session.updated":
        agent._session_ready.set()
        agent.logger.info("Realtime session updated")
        return True
    if event_type in _AUDIO_DELTA_EVENT_TYPES:
        agent._handle_output_audio_delta(event)
        return True
    if event_type in _TRANSCRIPT_DELTA_EVENT_TYPES:
        agent._handle_output_transcript_delta(event)
        return True
    handler_name = _HANDLER_BY_EVENT_TYPE.get(event_type)
    if not handler_name:
        return False
    getattr(agent, handler_name)(event)
    return True


def _handle_session_created(agent: Any, event: dict[str, Any]) -> None:
    session = event.get("session", {}) or {}
    response = server_event_response(event)
    if not session and response:
        session = response
    agent._session_id = str(session.get("id", "") or "").strip()
    agent._session_estimated_cost_usd = 0.0
    agent.logger.info(
        "Realtime session created session_id=%s model=%s voice=%s",
        agent._session_id or "<unknown>",
        session.get("model", agent.realtime_profile.model),
        agent.realtime_profile.voice,
    )
