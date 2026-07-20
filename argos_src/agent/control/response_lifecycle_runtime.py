"""Response retry, cleanup, and continuation helpers for realtime turns."""

from __future__ import annotations

from typing import Any

from argos_src.agent.realtime_turns import (
    INCOMPLETE_AUDIO_CONTINUATION_LIMIT,
    NO_AUDIO_RESPONSE_RETRY_LIMIT,
    TURN_PHASE_REQUESTING_FOLLOWUP,
    QueuedTurn,
)


class ResponseLifecycleRuntime:
    """Own recovery decisions after a Realtime response is created or done."""

    def __init__(self, host: Any) -> None:
        self._host = host

    def forget_response_id(self, response_id: str) -> None:
        rendered = str(response_id or "").strip()
        if not rendered:
            return
        with self._host._turn_lock:
            self._host._response_id_to_req_id.pop(rendered, None)

    def discard_pending_response_turn(self, req_id: str) -> int:
        host = self._host
        with host._turn_lock:
            store = host._response_bindings()
            discarded = store.discard(req_id)
            host._pending_response_turn_req_ids = store.pending_req_ids
            return discarded

    def response_output_types(self, response: dict[str, Any]) -> list[str]:
        output_types: list[str] = []
        for output_item in response.get("output", []) or []:
            rendered = str(output_item.get("type", "") or "").strip()
            if rendered:
                output_types.append(rendered)
        return output_types

    def cleanup_silent_response_items(
        self,
        turn: QueuedTurn,
        response: dict[str, Any],
    ) -> None:
        host = self._host
        assistant_item_ids: list[str] = []
        for output_item in response.get("output", []) or []:
            if str(output_item.get("type", "") or "").strip() != "message":
                continue
            item_id = str(output_item.get("id", "") or "").strip()
            if item_id:
                assistant_item_ids.append(item_id)
        for item_id in assistant_item_ids:
            try:
                transport_host = (
                    host._transport_host()
                    if callable(getattr(host, "_transport_host", None))
                    else host
                )
                transport_host._send_event(
                    {"type": "conversation.item.delete", "item_id": item_id}
                )
            except Exception:
                host.logger.exception(
                    "Failed to delete silent assistant item req_id=%s item_id=%s",
                    turn.req_id,
                    item_id,
                )
                continue
            host._forget_history_item(turn, item_id)

    def retry_no_audio_response(
        self,
        turn: QueuedTurn,
        response: dict[str, Any],
    ) -> bool:
        host = self._host
        if turn.no_audio_retry_count >= NO_AUDIO_RESPONSE_RETRY_LIMIT:
            return False
        turn.no_audio_retry_count += 1
        response_id = str(response.get("id", "") or turn.response_id).strip()
        response_state = turn.response_outputs.get(response_id)
        host.logger.warning(
            "Realtime response completed without audio; retrying req_id=%s response_id=%s retry=%s output_types=%s transcript=%r",
            turn.req_id,
            response_id,
            turn.no_audio_retry_count,
            self.response_output_types(response),
            str(getattr(response_state, "transcript", "") or "").strip(),
        )
        self.cleanup_silent_response_items(turn, response)
        self.forget_response_id(response_id)
        if response_state is not None:
            turn.response_outputs.pop(response_id, None)
        turn.response_id = ""
        if response_state is not None:
            for item_id in response_state.assistant_item_ids:
                turn.assistant_item_ids.discard(item_id)
            if turn.assistant_item_id in response_state.assistant_item_ids:
                turn.assistant_item_id = ""
        if not turn.audible_transcript_parts:
            turn.assistant_transcript = ""
        turn.response_done_at = 0.0
        host._set_turn_phase(
            turn,
            TURN_PHASE_REQUESTING_FOLLOWUP,
            trigger="no_audio_retry",
        )
        host._send_response_create(turn)
        return True

    @staticmethod
    def transcript_looks_truncated(transcript: str) -> bool:
        rendered = str(transcript or "").rstrip()
        if not rendered:
            return False
        return rendered[-1] not in ".!?)]}\"'"

    def should_continue_incomplete_audio_reply(self, turn: QueuedTurn) -> bool:
        if turn.incomplete_audio_continuation_count >= INCOMPLETE_AUDIO_CONTINUATION_LIMIT:
            return False
        if turn.pending_tool_calls > 0:
            return False
        return self.transcript_looks_truncated(turn.assistant_transcript)

    def continue_incomplete_audio_reply(self, turn: QueuedTurn) -> None:
        host = self._host
        response_id = turn.response_id
        turn.incomplete_audio_continuation_count += 1
        host.logger.info(
            "Continuing incomplete audio reply req_id=%s previous_response_id=%s continuation_count=%s",
            turn.req_id,
            response_id,
            turn.incomplete_audio_continuation_count,
        )
        self.forget_response_id(response_id)
        turn.response_id = ""
        turn.response_done_at = 0.0
        host._set_turn_phase(
            turn,
            TURN_PHASE_REQUESTING_FOLLOWUP,
            trigger="incomplete_audio_continuation",
        )
        host._send_response_create(turn)
