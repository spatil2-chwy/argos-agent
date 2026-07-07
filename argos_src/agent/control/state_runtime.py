"""State, history, and transport helpers for the realtime control runtime."""

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
    RESPONSE_STALL_TIMEOUT_SEC,
    TERMINAL_TURN_PHASES,
    TURN_PHASE_FINALIZED,
    FrozenTurnContext,
    PendingCreatedItem,
    QueuedTurn,
)
from argos_src.agent.control.observers import safe_transition
from argos_src.agent.control.history_store import OwnerScopedHistoryIndex
from argos_src.agent.control.turn_store import PendingResponseBindingStore, ResponseBinding
from argos_src.agent.control.types import StateAxis, StateTransition
from argos_src.face_recognition.models import PersonContext
from argos_src.identity.prompting import format_identity_profile_lines


class AgentStateRuntime:
    """State/history/transport helper surface for the realtime agent.

    The class can still be subclassed by focused tests. In production it is
    composed by `RealtimeRobotAgent` and proxies field access to that host.
    """

    def __init__(self, host: Any | None = None) -> None:
        if host is not None:
            object.__setattr__(self, "_host", host)

    def __getattr__(self, name: str) -> Any:
        host = object.__getattribute__(self, "__dict__").get("_host")
        if host is None:
            raise AttributeError(name)
        return getattr(host, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_host":
            object.__setattr__(self, name, value)
            return
        host = object.__getattribute__(self, "__dict__").get("_host")
        if host is None:
            object.__setattr__(self, name, value)
            return
        setattr(host, name, value)

    def _transport_host(self) -> Any:
        return object.__getattribute__(self, "__dict__").get("_host") or self

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
        self._transport_host()._send_event(
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
            if primary_face_person_id is None:
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
        old_phase = turn.phase
        turn.phase = phase
        turn.phase_updated_at = time.time()
        exchange_fields = {}
        fields_fn = getattr(self, "_exchange_log_fields", None)
        if callable(fields_fn):
            exchange_fields = dict(fields_fn(turn))
        safe_transition(
            getattr(self, "_state_observer", None),
            StateTransition(
                axis=StateAxis.TURN,
                old_state=old_phase,
                new_state=phase,
                trigger="set_turn_phase",
                req_id=turn.req_id,
                fields={
                    "audio_started": bool(getattr(turn, "audio_started", False)),
                    "pending_tool_calls": int(getattr(turn, "pending_tool_calls", 0) or 0),
                    "pending_response_requests": int(
                        getattr(turn, "pending_response_requests", 0) or 0
                    ),
                    **exchange_fields,
                },
            ),
        )

    def _is_turn_terminal(self, turn: Optional[QueuedTurn]) -> bool:
        return turn is None or turn.phase in TERMINAL_TURN_PHASES or turn.finalized

    def _response_bindings(self) -> PendingResponseBindingStore:
        store = getattr(self, "_response_binding_store", None)
        pending_req_ids = getattr(self, "_pending_response_turn_req_ids", None)
        if pending_req_ids is None:
            pending_req_ids = deque()
            self._pending_response_turn_req_ids = pending_req_ids
        stale_deadlines = getattr(self, "_stale_response_deadlines_by_req_id", None)
        if stale_deadlines is None:
            stale_deadlines = {}
            self._stale_response_deadlines_by_req_id = stale_deadlines
        expired_stale_req_ids = getattr(self, "_expired_stale_response_turn_req_ids", None)
        if expired_stale_req_ids is None:
            expired_stale_req_ids = deque()
            self._expired_stale_response_turn_req_ids = expired_stale_req_ids
        response_id_to_req_id = getattr(self, "_response_id_to_req_id", None)
        if response_id_to_req_id is None:
            response_id_to_req_id = {}
            self._response_id_to_req_id = response_id_to_req_id
        turns_by_req_id = getattr(self, "_turns_by_req_id", None)
        if turns_by_req_id is None:
            turns_by_req_id = {}
            self._turns_by_req_id = turns_by_req_id
        if (
            store is None
            or store.pending_req_ids is not pending_req_ids
            or store.expired_stale_req_ids is not expired_stale_req_ids
            or store.stale_deadlines_by_req_id is not stale_deadlines
            or store.response_id_to_req_id is not response_id_to_req_id
            or store.turns_by_req_id is not turns_by_req_id
        ):
            store = PendingResponseBindingStore(
                turns_by_req_id=turns_by_req_id,
                is_terminal=self._is_turn_terminal,
                pending_req_ids=pending_req_ids,
                expired_stale_req_ids=expired_stale_req_ids,
                stale_deadlines_by_req_id=stale_deadlines,
                response_id_to_req_id=response_id_to_req_id,
                now=time.time,
            )
            self._response_binding_store = store
        return store

    def _history_index(self) -> OwnerScopedHistoryIndex:
        store = getattr(self, "_history_index_store", None)
        item_order = getattr(self, "_history_item_order", None)
        if item_order is None:
            item_order = deque()
            self._history_item_order = item_order
        known_item_ids = getattr(self, "_known_history_item_ids", None)
        if known_item_ids is None:
            known_item_ids = set()
            self._known_history_item_ids = known_item_ids
        item_owner_req_id = getattr(self, "_history_item_owner_req_id", None)
        if item_owner_req_id is None:
            item_owner_req_id = {}
            self._history_item_owner_req_id = item_owner_req_id
        if (
            store is None
            or store.item_order is not item_order
            or store.known_item_ids is not known_item_ids
            or store.item_owner_req_id is not item_owner_req_id
        ):
            store = OwnerScopedHistoryIndex(
                item_order=item_order,
                known_item_ids=known_item_ids,
                item_owner_req_id=item_owner_req_id,
            )
            self._history_index_store = store
        return store

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
        binding = self._consume_pending_response_binding(
            response_id,
            consume_only_if_missing=consume_only_if_missing,
        )
        return binding.turn if binding is not None else None

    def _consume_pending_response_binding(
        self,
        response_id: str,
        *,
        consume_only_if_missing: bool = True,
    ) -> ResponseBinding | None:
        with self._turn_lock:
            return self._response_bindings().consume_binding(
                response_id,
                consume_only_if_missing=consume_only_if_missing,
            )

    def _queue_pending_response_turn(self, req_id: str) -> None:
        with self._turn_lock:
            self._response_bindings().queue(req_id)

    def _mark_pending_response_turn_stale(self, req_id: str) -> bool:
        with self._turn_lock:
            return self._response_bindings().mark_stale(
                req_id,
                timeout_s=RESPONSE_STALL_TIMEOUT_SEC,
            )

    def _pending_stale_response_deadline(self) -> float | None:
        with self._turn_lock:
            return self._response_bindings().next_stale_deadline()

    def _next_pending_response_turn(self) -> Optional[QueuedTurn]:
        with self._turn_lock:
            while self._pending_response_turn_req_ids:
                req_id = self._pending_response_turn_req_ids[0]
                turn = self._turns_by_req_id.get(req_id)
                if turn is None or self._is_turn_terminal(turn):
                    self._discard_pending_response_turn(req_id)
                    continue
                return turn
        return None

    def _wait_for_stale_response_slot(self) -> bool:
        while not self._stop_event.is_set():
            deadline = self._pending_stale_response_deadline()
            if deadline is None:
                return True
            wait_s = max(0.0, min(0.05, float(deadline) - time.time()))
            self._stop_event.wait(wait_s)
        return False

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
            self._history_index().register(rendered, owner_req_id=owner_req_id)

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
            self._history_index().forget(rendered)

    def _history_owner_key_for_turn(self, turn: QueuedTurn) -> str:
        return self._history_index().owner_key(getattr(turn, "owner_id", None))

    def _history_protected_item_ids(self, current_turn: Optional[QueuedTurn]) -> set[str]:
        with self._turn_lock:
            return self._history_index().protected_item_ids(
                turns=self._turns_by_req_id.values(),
                is_terminal=self._is_turn_terminal,
                current_turn=current_turn,
                pending_audio_item_ids=self._pending_audio_item_ids,
                playback_item_id=self._playback_item_id,
                bound_item_ids=self._item_id_to_req_id,
            )

    def _forget_deleted_history_item(self, item_id: str) -> None:
        rendered = str(item_id or "").strip()
        if not rendered:
            return
        with self._turn_lock:
            req_id = self._history_index().owner_req_id_for(
                rendered,
                fallback=self._item_id_to_req_id.get(rendered, ""),
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
            history_snapshot = self._history_index().delete_candidates(
                protected_item_ids=protected_item_ids
            )

        deleted_count = 0
        for item_id in history_snapshot:
            try:
                self._transport_host()._send_event(
                    {"type": "conversation.item.delete", "item_id": item_id}
                )
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

    def _discard_pending_response_turn(self, req_id: str) -> int:
        with self._turn_lock:
            store = self._response_bindings()
            discarded = store.discard(req_id)
            self._pending_response_turn_req_ids = store.pending_req_ids
            return discarded

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
                self._transport_host()._send_event(
                    {"type": "conversation.item.delete", "item_id": item_id}
                )
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
