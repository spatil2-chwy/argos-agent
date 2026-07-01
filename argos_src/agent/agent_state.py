"""State, history, and transport helpers for the Argos agent runtime."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
import os
import time
from typing import Any, Optional

import websocket

from argos_src.agent.realtime_turns import (
    INCOMPLETE_AUDIO_CONTINUATION_LIMIT,
    NO_AUDIO_RESPONSE_RETRY_LIMIT,
    TERMINAL_TURN_PHASES,
    TURN_PHASE_FINALIZED,
    FrozenTurnContext,
    PendingCreatedItem,
    QueuedTurn,
)
from argos_src.face_recognition.models import PersonContext
from argos_src.identity.prompting import format_identity_profile_lines


class RealtimeAgentStateMixin:
    def _enrich_person_context_with_memory(self, person: PersonContext) -> PersonContext:
        compiler = getattr(self, "memory_context_compiler", None)
        if compiler is None:
            return person
        person_id = str(getattr(person, "person_id", "") or "").strip()
        if not person_id:
            return person
        try:
            context = compiler.person_context(
                person_id,
                fallback_profile_lines=tuple(getattr(person, "memory_profile_lines", ()) or ()),
                fallback_followup_lines=tuple(getattr(person, "potential_followups", ()) or ()),
            )
        except Exception:
            self.logger.exception("Failed to compile memory context for %s", person_id)
            return person
        person.memory_profile_lines = tuple(context.profile_lines or ())
        person.potential_followups = tuple(context.followup_lines or ())
        if context.preferred_language:
            person.preferred_language = context.preferred_language
        return person

    def _compile_memory_context_blocks(self, current_person_id: Optional[str]) -> tuple[str, ...]:
        compiler = getattr(self, "memory_context_compiler", None)
        if compiler is None:
            return ()
        site_code = str(getattr(self, "_current_office_location", "") or "").strip()
        if not site_code:
            return ()
        try:
            return tuple(
                compiler.site_blocks(
                    site_code,
                    current_person_id=str(current_person_id or "").strip() or None,
                )
            )
        except Exception:
            self.logger.exception("Failed to compile site memory context")
            return ()

    def _append_text_message_item(
        self,
        turn: QueuedTurn,
        text: str,
        *,
        role: str,
    ) -> None:
        rendered = str(text or "").strip()
        if not rendered:
            return
        rendered_role = str(role or "").strip().lower()
        if rendered_role not in {"user", "system"}:
            raise ValueError(f"Unsupported realtime text message role: {role!r}")
        self._queue_pending_local_created_item(turn.req_id, "message", rendered_role)
        self._send_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": rendered_role,
                    "content": [{"type": "input_text", "text": rendered}],
                },
            }
        )

    def _queue_pending_local_created_item(
        self,
        owner_req_id: str,
        expected_type: str,
        expected_role: str = "",
    ) -> None:
        with self._turn_lock:
            self._pending_local_created_items.append(
                PendingCreatedItem(
                    owner_req_id=owner_req_id,
                    expected_type=expected_type,
                    expected_role=expected_role,
                )
            )

    def _consume_pending_local_created_item(self, expected_type: str, expected_role: str = "") -> str:
        with self._turn_lock:
            pending = deque()
            owner_req_id = ""
            while self._pending_local_created_items:
                candidate = self._pending_local_created_items.popleft()
                if (
                    candidate.expected_type == expected_type
                    and (not expected_role or candidate.expected_role == expected_role)
                ):
                    owner_req_id = candidate.owner_req_id
                    break
                pending.append(candidate)
            while pending:
                self._pending_local_created_items.appendleft(pending.pop())
            return owner_req_id

    def _register_pending_audio_turn(self, turn: QueuedTurn) -> None:
        with self._turn_lock:
            bound_audio_item = False
            if not self._pending_audio_item_ids:
                self._pending_audio_turn_req_ids.append(turn.req_id)
                return
            while self._pending_audio_item_ids:
                item_id = self._pending_audio_item_ids.popleft()
                if not item_id or item_id in self._item_id_to_req_id:
                    continue
                self._bind_item_id_to_turn(turn, item_id)
                if not turn.user_item_id:
                    turn.user_item_id = item_id
                bound_audio_item = True
                break
            if not bound_audio_item:
                self._pending_audio_turn_req_ids.append(turn.req_id)

    def _consume_pending_audio_turn_req_id(self, *, include_finalized: bool = False) -> str:
        with self._turn_lock:
            while self._pending_audio_turn_req_ids:
                req_id = self._pending_audio_turn_req_ids.popleft()
                turn = self._turns_by_req_id.get(req_id)
                if turn is None:
                    continue
                if self._is_turn_terminal(turn) and not (
                    include_finalized and turn.phase == TURN_PHASE_FINALIZED
                ):
                    continue
                return req_id
        return ""

    def _capture_turn_context(
        self,
        *,
        primary_face_person_id: Optional[str] = None,
        audio_speaker_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        owner_source: str = "unknown",
        owner_confidence: float = 0.0,
        speaker_visible: bool = False,
        allow_live_primary_lookup: bool = True,
    ) -> FrozenTurnContext:
        def identity_person(person_id: Optional[str]) -> PersonContext | None:
            rendered = str(person_id or "").strip()
            identity_store = getattr(self, "identity_store", None)
            if not rendered or identity_store is None:
                return None
            try:
                record = identity_store.get_person(rendered)
                if record is None:
                    return None
                metadata = dict(record.get("metadata") or {})
                person = PersonContext(
                    person_id=rendered,
                    name=str(record.get("name") or metadata.get("name") or rendered),
                    interaction_count=int(metadata.get("interaction_count", 0) or 0),
                    confidence=1.0,
                    bbox_area=0,
                    timestamp=time.time(),
                    directory_profile_lines=format_identity_profile_lines(metadata),
                    memory_profile_lines=(),
                    preferred_language="",
                    potential_followups=(),
                    visible=False,
                )
                return self._enrich_person_context_with_memory(person)
            except Exception:
                self.logger.exception("Failed to build identity context for %s", rendered)
                return None

        current_person_id = owner_id
        memory_context_blocks = self._compile_memory_context_blocks(current_person_id)
        if self.face_service is None:
            person = identity_person(owner_id)
            return FrozenTurnContext(
                persons=[person] if person is not None else [],
                primary_face_person_id=primary_face_person_id,
                audio_speaker_id=audio_speaker_id,
                owner_id=owner_id,
                owner_source=owner_source,
                owner_confidence=owner_confidence,
                speaker_visible=speaker_visible,
                memory_context_blocks=memory_context_blocks,
            )
        try:
            persons = deepcopy(self.face_service.get_cached_persons())
            owner_context_id = str(owner_id or "").strip()
            has_turn_identity_anchor = bool(
                str(primary_face_person_id or "").strip()
                or str(audio_speaker_id or "").strip()
                or owner_context_id
            )
            if owner_context_id:
                persons = [
                    self._enrich_person_context_with_memory(person)
                    if str(getattr(person, "person_id", "") or "").strip() == owner_context_id
                    else person
                    for person in persons
                ]
            face_snapshot = deepcopy(self.face_service.get_presence_snapshot())
            if owner_context_id and not any(
                str(getattr(person, "person_id", "") or "").strip() == owner_context_id
                for person in persons
            ):
                person = identity_person(owner_context_id)
                if person is not None:
                    persons.append(person)
            if not has_turn_identity_anchor and not allow_live_primary_lookup:
                persons = []
                face_snapshot = None
            if primary_face_person_id is None and allow_live_primary_lookup:
                attention_getter = getattr(
                    self.face_service,
                    "get_primary_attention_person_id",
                    None,
                )
                if callable(attention_getter):
                    primary_face_person_id = attention_getter()
                if primary_face_person_id is None:
                    getter = getattr(self.face_service, "get_primary_face_person_id", None)
                    if callable(getter):
                        primary_face_person_id = getter()
                    else:
                        primary_face_person_id = self.face_service.get_attention_target_person_id()
            return FrozenTurnContext(
                persons=persons,
                face_snapshot=face_snapshot,
                primary_face_person_id=primary_face_person_id,
                audio_speaker_id=audio_speaker_id,
                owner_id=owner_id,
                owner_source=owner_source,
                owner_confidence=owner_confidence,
                speaker_visible=speaker_visible,
                memory_context_blocks=memory_context_blocks,
            )
        except Exception:
            self.logger.exception("Failed to capture turn context snapshot")
            return FrozenTurnContext(
                primary_face_person_id=primary_face_person_id,
                audio_speaker_id=audio_speaker_id,
                owner_id=owner_id,
                owner_source=owner_source,
                owner_confidence=owner_confidence,
                speaker_visible=speaker_visible,
                memory_context_blocks=memory_context_blocks,
            )

    def _set_turn_phase(self, turn: QueuedTurn, phase: str) -> None:
        if turn.phase == phase:
            return
        turn.phase = phase
        turn.phase_updated_at = time.time()

    def _is_turn_terminal(self, turn: Optional[QueuedTurn]) -> bool:
        return turn is None or turn.phase in TERMINAL_TURN_PHASES or turn.finalized

    def _bind_response_id(self, turn: QueuedTurn, response_id: str) -> None:
        rendered = str(response_id or "").strip()
        if not rendered:
            return
        turn.response_id = rendered
        with self._turn_lock:
            self._response_id_to_req_id[rendered] = turn.req_id

    def _bind_item_id_to_turn(self, turn: QueuedTurn, item_id: str) -> None:
        rendered = str(item_id or "").strip()
        if not rendered:
            return
        with self._turn_lock:
            self._item_id_to_req_id[rendered] = turn.req_id
        self._register_turn_history_item(turn, rendered)

    def _req_id_for_response_id(self, response_id: str) -> str:
        rendered = str(response_id or "").strip()
        if not rendered:
            return ""
        with self._turn_lock:
            return self._response_id_to_req_id.get(rendered, "")

    def _resolve_turn_for_item(self, item_id: str) -> Optional[QueuedTurn]:
        rendered = str(item_id or "").strip()
        if not rendered:
            return None
        with self._turn_lock:
            req_id = self._item_id_to_req_id.get(rendered, "")
        if not req_id:
            return None
        return self._turns_by_req_id.get(req_id)

    def _resolve_turn_for_output(
        self,
        *,
        response_id: str = "",
        item_id: str = "",
        call_id: str = "",
    ) -> Optional[QueuedTurn]:
        turn = self._resolve_turn_for_item(item_id)
        if turn is not None:
            return turn
        rendered_response_id = str(response_id or "").strip()
        if rendered_response_id:
            req_id = self._req_id_for_response_id(rendered_response_id)
            if req_id:
                return self._turns_by_req_id.get(req_id)
        rendered_call_id = str(call_id or "").strip()
        if rendered_call_id:
            with self._turn_lock:
                req_id = self._call_id_to_req_id.get(rendered_call_id, "")
            if req_id:
                return self._turns_by_req_id.get(req_id)
        return None

    def _consume_pending_response_turn(
        self,
        response_id: str,
        *,
        consume_only_if_missing: bool = True,
    ) -> Optional[QueuedTurn]:
        rendered = str(response_id or "").strip()
        if not rendered:
            return None
        with self._turn_lock:
            existing_req_id = self._response_id_to_req_id.get(rendered, "")
            if existing_req_id and consume_only_if_missing:
                return self._turns_by_req_id.get(existing_req_id)
            while self._pending_response_turn_req_ids:
                req_id = self._pending_response_turn_req_ids.popleft()
                turn = self._turns_by_req_id.get(req_id)
                if turn is None or self._is_turn_terminal(turn):
                    continue
                self._response_id_to_req_id[rendered] = req_id
                turn.response_id = rendered
                return turn
        return None

    def _conversation_item_looks_like_audio_input(self, item: dict[str, Any]) -> bool:
        for content in item.get("content", []) or []:
            rendered_type = str(content.get("type", "") or "").strip()
            if rendered_type in {"input_audio", "audio"}:
                return True
        return False

    def _register_history_item(self, item_id: str, *, owner_req_id: str = "") -> None:
        rendered = str(item_id or "").strip()
        if not rendered:
            return
        with self._turn_lock:
            if rendered not in self._known_history_item_ids:
                self._known_history_item_ids.add(rendered)
                self._history_item_order.append(rendered)
            if owner_req_id:
                self._history_item_owner_req_id[rendered] = owner_req_id

    def _register_turn_history_item(self, turn: QueuedTurn, item_id: str) -> None:
        rendered = str(item_id or "").strip()
        if not rendered:
            return
        turn.history_item_ids.add(rendered)
        self._register_history_item(rendered, owner_req_id=turn.req_id)

    def _forget_history_item(self, turn: Optional[QueuedTurn], item_id: str) -> None:
        rendered = str(item_id or "").strip()
        if not rendered:
            return
        if turn is not None:
            turn.history_item_ids.discard(rendered)
            if turn.user_item_id == rendered:
                turn.user_item_id = ""
            if turn.assistant_item_id == rendered:
                turn.assistant_item_id = ""
            turn.assistant_item_ids.discard(rendered)
            turn.function_call_item_ids.discard(rendered)
        with self._turn_lock:
            self._item_id_to_req_id.pop(rendered, None)
            self._history_item_owner_req_id.pop(rendered, None)
            self._known_history_item_ids.discard(rendered)
            try:
                self._history_item_order.remove(rendered)
            except ValueError:
                pass

    def _history_owner_key_for_turn(self, turn: QueuedTurn) -> str:
        owner_id = str(getattr(turn, "owner_id", "") or "").strip()
        if owner_id:
            return f"owner:{owner_id}"
        return "anonymous"

    def _history_protected_item_ids(self, current_turn: Optional[QueuedTurn]) -> set[str]:
        protected_item_ids: set[str] = set()
        with self._turn_lock:
            for turn in self._turns_by_req_id.values():
                if not self._is_turn_terminal(turn):
                    protected_item_ids.update(turn.history_item_ids)
                    if turn.user_item_id:
                        protected_item_ids.add(turn.user_item_id)
                    if turn.assistant_item_id:
                        protected_item_ids.add(turn.assistant_item_id)
                    protected_item_ids.update(turn.assistant_item_ids)
                    protected_item_ids.update(turn.function_call_item_ids)
            if current_turn is not None:
                protected_item_ids.update(current_turn.history_item_ids)
                if current_turn.user_item_id:
                    protected_item_ids.add(current_turn.user_item_id)
                if current_turn.assistant_item_id:
                    protected_item_ids.add(current_turn.assistant_item_id)
                protected_item_ids.update(current_turn.assistant_item_ids)
                protected_item_ids.update(current_turn.function_call_item_ids)
                if (
                    current_turn.kind == "audio"
                    and not current_turn.user_item_id
                    and self._history_item_order
                ):
                    for item_id in reversed(self._history_item_order):
                        if (
                            item_id not in self._history_item_owner_req_id
                            and item_id not in self._item_id_to_req_id
                        ):
                            protected_item_ids.add(item_id)
                            break
                protected_item_ids.update(self._pending_audio_item_ids)
            if self._playback_item_id:
                protected_item_ids.add(self._playback_item_id)
        return protected_item_ids

    def _forget_deleted_history_item(self, item_id: str) -> None:
        rendered = str(item_id or "").strip()
        if not rendered:
            return
        with self._turn_lock:
            req_id = self._history_item_owner_req_id.get(
                rendered,
                self._item_id_to_req_id.get(rendered, ""),
            )
            turn = self._turns_by_req_id.get(req_id) if req_id else None
        self._forget_history_item(turn, rendered)

    def _maybe_rotate_history_for_turn(self, turn: QueuedTurn) -> None:
        new_owner_key = self._history_owner_key_for_turn(turn)
        if getattr(turn, "source_is_internal", False):
            if not getattr(self, "_active_history_owner_key", ""):
                self._active_history_owner_key = new_owner_key
            return

        old_owner_key = str(getattr(self, "_active_history_owner_key", "") or "")
        if old_owner_key == new_owner_key:
            return

        protected_item_ids = self._history_protected_item_ids(turn)
        with self._turn_lock:
            history_snapshot = list(self._history_item_order)

        deleted_count = 0
        for item_id in history_snapshot:
            if item_id in protected_item_ids:
                continue
            try:
                self._send_event({"type": "conversation.item.delete", "item_id": item_id})
            except Exception:
                self.logger.exception(
                    "Failed to delete owner-scoped conversation item_id=%s",
                    item_id,
                )
                continue
            self._forget_deleted_history_item(item_id)
            deleted_count += 1

        self._active_history_owner_key = new_owner_key
        self._last_tool_name = None
        self._last_tool_summary = None
        self.logger.info(
            "Rotated realtime history old_owner_key=%s new_owner_key=%s "
            "deleted_items=%s protected_items=%s",
            old_owner_key or "<none>",
            new_owner_key,
            deleted_count,
            len(protected_item_ids),
        )

    def _forget_response_id(self, response_id: str) -> None:
        rendered = str(response_id or "").strip()
        if not rendered:
            return
        with self._turn_lock:
            self._response_id_to_req_id.pop(rendered, None)

    def _discard_pending_response_turn(self, req_id: str) -> None:
        rendered = str(req_id or "").strip()
        if not rendered:
            return
        with self._turn_lock:
            self._pending_response_turn_req_ids = deque(
                candidate
                for candidate in self._pending_response_turn_req_ids
                if candidate != rendered
            )

    def _response_output_types(self, response: dict[str, Any]) -> list[str]:
        output_types: list[str] = []
        for output_item in response.get("output", []) or []:
            rendered = str(output_item.get("type", "") or "").strip()
            if rendered:
                output_types.append(rendered)
        return output_types

    def _cleanup_silent_response_items(
        self,
        turn: QueuedTurn,
        response: dict[str, Any],
    ) -> None:
        assistant_item_ids: list[str] = []
        for output_item in response.get("output", []) or []:
            if str(output_item.get("type", "") or "").strip() != "message":
                continue
            item_id = str(output_item.get("id", "") or "").strip()
            if item_id:
                assistant_item_ids.append(item_id)
        for item_id in assistant_item_ids:
            try:
                self._send_event({"type": "conversation.item.delete", "item_id": item_id})
            except Exception:
                self.logger.exception(
                    "Failed to delete silent assistant item req_id=%s item_id=%s",
                    turn.req_id,
                    item_id,
                )
                continue
            self._forget_history_item(turn, item_id)

    def _retry_no_audio_response(
        self,
        turn: QueuedTurn,
        response: dict[str, Any],
    ) -> bool:
        if turn.no_audio_retry_count >= NO_AUDIO_RESPONSE_RETRY_LIMIT:
            return False
        turn.no_audio_retry_count += 1
        response_id = turn.response_id
        self.logger.warning(
            "Realtime response completed without audio; retrying req_id=%s response_id=%s retry=%s output_types=%s transcript=%r",
            turn.req_id,
            response_id,
            turn.no_audio_retry_count,
            self._response_output_types(response),
            turn.assistant_transcript.strip(),
        )
        self._cleanup_silent_response_items(turn, response)
        self._forget_response_id(response_id)
        turn.response_id = ""
        turn.assistant_item_id = ""
        turn.assistant_item_ids.clear()
        turn.assistant_transcript = ""
        turn.response_done_at = 0.0
        self._send_response_create(turn)
        return True

    def _transcript_looks_truncated(self, transcript: str) -> bool:
        rendered = str(transcript or "").rstrip()
        if not rendered:
            return False
        return rendered[-1] not in ".!?)]}\"'"

    def _should_continue_incomplete_audio_reply(self, turn: QueuedTurn) -> bool:
        if turn.incomplete_audio_continuation_count >= INCOMPLETE_AUDIO_CONTINUATION_LIMIT:
            return False
        if turn.pending_tool_calls > 0:
            return False
        return self._transcript_looks_truncated(turn.assistant_transcript)

    def _continue_incomplete_audio_reply(self, turn: QueuedTurn) -> None:
        response_id = turn.response_id
        turn.incomplete_audio_continuation_count += 1
        self.logger.info(
            "Continuing incomplete audio reply req_id=%s previous_response_id=%s continuation_count=%s",
            turn.req_id,
            response_id,
            turn.incomplete_audio_continuation_count,
        )
        self._forget_response_id(response_id)
        turn.response_id = ""
        turn.response_done_at = 0.0
        self._send_response_create(turn)

    def _stringify_tool_output(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, (dict, list, tuple)):
            return json.dumps(content, ensure_ascii=True)
        return str(content)

    def _send_event(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            if self._stop_event.is_set():
                return
            raise RuntimeError("Realtime websocket is not connected.")
        with self._ws_lock:
            try:
                self._ws.send(json.dumps(payload))
            except websocket.WebSocketConnectionClosedException:
                if not self._stop_event.is_set():
                    self.logger.warning("Realtime websocket closed during send; stopping runtime")
                    self._stop_event.set()
                self._ws = None
                return

    def _transcript_from_response(self, response: dict[str, Any]) -> str:
        parts: list[str] = []
        for output_item in response.get("output", []) or []:
            for content in output_item.get("content", []) or []:
                transcript = str(content.get("transcript", "") or "").strip()
                text = str(content.get("text", "") or "").strip()
                if transcript:
                    parts.append(transcript)
                elif text:
                    parts.append(text)
        return " ".join(part for part in parts if part).strip()

    def _log_wakeword_debug(self, *, wake_detected: bool, wake_output: dict[str, Any]) -> None:
        now_s = time.time()
        wake_data = wake_output.get("open_wake_word", {}) if isinstance(wake_output, dict) else {}
        predictions = wake_data.get("predictions", {}) if isinstance(wake_data, dict) else {}
        if not isinstance(predictions, dict) or not predictions:
            return
        debug_enabled = os.getenv("ARGOS_WAKEWORD_DEBUG", "0").strip().lower() not in {
            "",
            "0",
            "false",
            "no",
        }
        top_label, top_score = max(
            ((str(label), float(score)) for label, score in predictions.items()),
            key=lambda item: item[1],
        )
        if not wake_detected and not debug_enabled:
            return
        should_log = wake_detected or (now_s - self._last_wake_debug_log_s >= 5.0)
        if not should_log:
            return
        self._last_wake_debug_log_s = now_s
        self.logger.info(
            "Wake word debug top_label=%s score=%.3f threshold=%.3f detected=%s",
            top_label,
            top_score,
            self._wake_word.threshold,
            wake_detected,
        )

    def _get_current_primary_face_person_id(self) -> Optional[str]:
        if self.face_service is None:
            return None
        try:
            attention_getter = getattr(
                self.face_service,
                "get_primary_attention_person_id",
                None,
            )
            if callable(attention_getter):
                attention_person_id = attention_getter()
                if attention_person_id:
                    return attention_person_id
            getter = getattr(self.face_service, "get_primary_face_person_id", None)
            if callable(getter):
                return getter()
            return self.face_service.get_attention_target_person_id()
        except Exception:
            return None

    def _get_current_visible_face_person_ids(self) -> tuple[str, ...]:
        if self.face_service is None:
            return ()
        try:
            persons = list(self.face_service.get_cached_persons() or [])
        except Exception:
            return ()
        visible_ids: list[str] = []
        for person in persons:
            if not bool(getattr(person, "visible", True)):
                continue
            rendered = str(getattr(person, "person_id", "") or "").strip()
            if rendered and rendered not in visible_ids:
                visible_ids.append(rendered)
        return tuple(visible_ids)

    def note_local_voice_command(self, command: str, *, ttl_sec: float = 1.5) -> None:
        rendered = str(command or "").strip().lower()
        if not rendered:
            return
        expires_at = time.time() + max(0.1, ttl_sec)
        with self._turn_lock:
            self._prune_ignored_voice_commands_locked(now_s=time.time())
            self._ignored_voice_commands.append((rendered, expires_at))

    def _prune_ignored_voice_commands_locked(self, *, now_s: float) -> None:
        while self._ignored_voice_commands and self._ignored_voice_commands[0][1] <= now_s:
            self._ignored_voice_commands.popleft()

    def _should_ignore_voice_command(self, command: str) -> bool:
        rendered = str(command or "").strip().lower()
        if not rendered:
            return False
        now_s = time.time()
        with self._turn_lock:
            self._prune_ignored_voice_commands_locked(now_s=now_s)
            pending = deque()
            ignored = False
            while self._ignored_voice_commands:
                candidate, expires_at = self._ignored_voice_commands.popleft()
                if not ignored and candidate == rendered and expires_at > now_s:
                    ignored = True
                    break
                pending.append((candidate, expires_at))
            while pending:
                self._ignored_voice_commands.appendleft(pending.pop())
        return ignored

    def _on_voice_command(self, msg: Any) -> None:
        raw_command = getattr(msg, "data", msg)
        command = str(raw_command or "").strip().lower()
        if not command:
            return
        if self._should_ignore_voice_command(command):
            self.logger.debug("Ignoring self-published voice command=%s", command)
            return
        if command == "stop":
            self.interrupt_current_response(reason="voice_command")

    def handle_voice_command(self, command: str) -> None:
        """Handle a local or bridge-provided voice command."""
        self._on_voice_command(command)
