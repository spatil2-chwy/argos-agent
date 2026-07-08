"""Conversation history indexes for the realtime control plane."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Iterable, MutableMapping, MutableSet


class OwnerScopedHistoryIndex:
    """Track Realtime conversation item order and local turn ownership."""

    def __init__(
        self,
        *,
        item_order: deque[str] | None = None,
        known_item_ids: MutableSet[str] | None = None,
        item_owner_req_id: MutableMapping[str, str] | None = None,
    ) -> None:
        self.item_order: deque[str] = item_order if item_order is not None else deque()
        self.known_item_ids = known_item_ids if known_item_ids is not None else set()
        self.item_owner_req_id = item_owner_req_id if item_owner_req_id is not None else {}

    def register(self, item_id: str, *, owner_req_id: str = "") -> bool:
        rendered = str(item_id or "").strip()
        if not rendered:
            return False
        was_new = rendered not in self.known_item_ids
        if was_new:
            self.known_item_ids.add(rendered)
            self.item_order.append(rendered)
        owner = str(owner_req_id or "").strip()
        if owner:
            self.item_owner_req_id[rendered] = owner
        return was_new

    def forget(self, item_id: str) -> bool:
        rendered = str(item_id or "").strip()
        if not rendered:
            return False
        existed = rendered in self.known_item_ids or rendered in self.item_owner_req_id
        self.item_owner_req_id.pop(rendered, None)
        self.known_item_ids.discard(rendered)
        try:
            self.item_order.remove(rendered)
            existed = True
        except ValueError:
            pass
        return existed

    def snapshot(self) -> list[str]:
        return list(self.item_order)

    def owner_req_id_for(self, item_id: str, *, fallback: str = "") -> str:
        rendered = str(item_id or "").strip()
        if not rendered:
            return ""
        return self.item_owner_req_id.get(rendered, fallback)

    def newest_unbound_item(
        self,
        *,
        bound_item_ids: MutableMapping[str, str],
    ) -> str:
        for item_id in reversed(self.item_order):
            if item_id not in self.item_owner_req_id and item_id not in bound_item_ids:
                return item_id
        return ""

    @staticmethod
    def owner_key(owner_id: str | None) -> str:
        rendered = str(owner_id or "").strip()
        if rendered:
            return f"owner:{rendered}"
        return "anonymous"

    def protected_item_ids(
        self,
        *,
        turns: Iterable[Any],
        is_terminal: Callable[[Any], bool],
        current_turn: Any | None = None,
        pending_audio_item_ids: Iterable[str] = (),
        playback_item_id: str = "",
        bound_item_ids: MutableMapping[str, str] | None = None,
    ) -> set[str]:
        protected: set[str] = set()
        for turn in turns:
            if not is_terminal(turn):
                protected.update(self._turn_item_ids(turn))

        if current_turn is not None:
            protected.update(self._turn_item_ids(current_turn))
            if (
                getattr(current_turn, "kind", "") == "audio"
                and not str(getattr(current_turn, "user_item_id", "") or "").strip()
                and self.item_order
            ):
                item_id = self.newest_unbound_item(bound_item_ids=bound_item_ids or {})
                if item_id:
                    protected.add(item_id)
            protected.update(str(item_id or "").strip() for item_id in pending_audio_item_ids)

        rendered_playback_item_id = str(playback_item_id or "").strip()
        if rendered_playback_item_id:
            protected.add(rendered_playback_item_id)
        protected.discard("")
        return protected

    def delete_candidates(self, *, protected_item_ids: set[str]) -> list[str]:
        return [item_id for item_id in self.item_order if item_id not in protected_item_ids]

    @staticmethod
    def _turn_item_ids(turn: Any) -> set[str]:
        item_ids = set(getattr(turn, "history_item_ids", set()) or set())
        user_item_id = str(getattr(turn, "user_item_id", "") or "").strip()
        if user_item_id:
            item_ids.add(user_item_id)
        assistant_item_id = str(getattr(turn, "assistant_item_id", "") or "").strip()
        if assistant_item_id:
            item_ids.add(assistant_item_id)
        item_ids.update(getattr(turn, "assistant_item_ids", set()) or set())
        item_ids.update(getattr(turn, "function_call_item_ids", set()) or set())
        item_ids.discard("")
        return item_ids
