"""Inference-scope history indexes for the realtime control plane."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, MutableMapping, MutableSet


@dataclass
class InferenceHistoryItem:
    """Local metadata for one Realtime item that may be selected for inference."""

    item_id: str
    owner_req_id: str = ""
    owner_key: str = "anonymous"
    scope_id: str = ""
    item_type: str = ""
    role: str = ""
    status: str = "in_progress"
    permitted_for_inference: bool = False
    input_item: dict[str, Any] | None = None


class InferenceHistoryIndex:
    """Track Realtime item order and explicit model-input eligibility."""

    def __init__(
        self,
        *,
        item_order: deque[str] | None = None,
        known_item_ids: MutableSet[str] | None = None,
        item_owner_req_id: MutableMapping[str, str] | None = None,
        items: MutableMapping[str, InferenceHistoryItem] | None = None,
    ) -> None:
        self.item_order: deque[str] = item_order if item_order is not None else deque()
        self.known_item_ids = known_item_ids if known_item_ids is not None else set()
        self.item_owner_req_id = item_owner_req_id if item_owner_req_id is not None else {}
        self.items = items if items is not None else {}

    @staticmethod
    def owner_key(owner_id: str | None) -> str:
        rendered = str(owner_id or "").strip()
        if rendered:
            return f"owner:{rendered}"
        return "anonymous"

    @staticmethod
    def scope_kind(scope_id: str) -> str:
        rendered = str(scope_id or "").strip()
        if rendered.startswith("owner:"):
            return "known_owner"
        if rendered.startswith("anonymous:"):
            return "anonymous_patch"
        return "unknown"

    def register(
        self,
        item_id: str,
        *,
        owner_req_id: str = "",
        owner_key: str = "",
        scope_id: str = "",
        item_type: str = "",
        role: str = "",
        status: str = "",
        permitted_for_inference: bool | None = None,
        input_item: dict[str, Any] | None = None,
    ) -> bool:
        rendered = str(item_id or "").strip()
        if not rendered:
            return False
        was_new = rendered not in self.known_item_ids
        if was_new:
            self.known_item_ids.add(rendered)
            self.item_order.append(rendered)
            self.items[rendered] = InferenceHistoryItem(item_id=rendered)

        item = self.items.setdefault(rendered, InferenceHistoryItem(item_id=rendered))
        owner = str(owner_req_id or "").strip()
        if owner:
            item.owner_req_id = owner
            self.item_owner_req_id[rendered] = owner
        rendered_owner_key = str(owner_key or "").strip()
        if rendered_owner_key:
            item.owner_key = rendered_owner_key
        rendered_scope_id = str(scope_id or "").strip()
        if rendered_scope_id:
            item.scope_id = rendered_scope_id
        if item_type:
            item.item_type = str(item_type)
        if role:
            item.role = str(role)
        if status:
            item.status = str(status)
        if permitted_for_inference is not None:
            item.permitted_for_inference = bool(permitted_for_inference)
        if input_item is not None:
            item.input_item = dict(input_item)
        return was_new

    def forget(self, item_id: str) -> bool:
        rendered = str(item_id or "").strip()
        if not rendered:
            return False
        existed = (
            rendered in self.known_item_ids
            or rendered in self.item_owner_req_id
            or rendered in self.items
        )
        self.item_owner_req_id.pop(rendered, None)
        self.items.pop(rendered, None)
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
        item = self.items.get(rendered)
        if item is not None and item.owner_req_id:
            return item.owner_req_id
        return self.item_owner_req_id.get(rendered, fallback)

    def newest_unbound_item(
        self,
        *,
        bound_item_ids: MutableMapping[str, str],
    ) -> str:
        for item_id in reversed(self.item_order):
            item = self.items.get(item_id)
            if (
                item_id not in self.item_owner_req_id
                and item_id not in bound_item_ids
                and (item is None or not item.owner_req_id)
            ):
                return item_id
        return ""

    def update_item(
        self,
        item_id: str,
        *,
        item_type: str = "",
        role: str = "",
        status: str = "",
        permitted_for_inference: bool | None = None,
        input_item: dict[str, Any] | None = None,
    ) -> None:
        rendered = str(item_id or "").strip()
        if not rendered:
            return
        item = self.items.setdefault(rendered, InferenceHistoryItem(item_id=rendered))
        if item_type:
            item.item_type = str(item_type)
        if role:
            item.role = str(role)
        if status:
            item.status = str(status)
        if permitted_for_inference is not None:
            item.permitted_for_inference = bool(permitted_for_inference)
        if input_item is not None:
            item.input_item = dict(input_item)

    def mark_scope(
        self,
        item_id: str,
        *,
        owner_key: str,
        scope_id: str,
    ) -> None:
        rendered = str(item_id or "").strip()
        if not rendered:
            return
        item = self.items.setdefault(rendered, InferenceHistoryItem(item_id=rendered))
        item.owner_key = str(owner_key or "").strip() or item.owner_key
        item.scope_id = str(scope_id or "").strip() or item.scope_id

    def selected_item_ids(
        self,
        *,
        scope_id: str,
        exclude_req_id: str = "",
    ) -> list[str]:
        rendered_scope_id = str(scope_id or "").strip()
        rendered_exclude_req_id = str(exclude_req_id or "").strip()
        selected: list[str] = []
        for item_id in self.item_order:
            item = self.items.get(item_id)
            if item is None:
                continue
            if item.scope_id != rendered_scope_id:
                continue
            if rendered_exclude_req_id and item.owner_req_id == rendered_exclude_req_id:
                continue
            if item.status != "done" or not item.permitted_for_inference:
                continue
            selected.append(item_id)
        return selected

    def input_entry_for(self, item_id: str) -> dict[str, Any]:
        rendered = str(item_id or "").strip()
        item = self.items.get(rendered)
        if item is not None and item.input_item is not None:
            return dict(item.input_item)
        return {"type": "item_reference", "id": rendered}

    def input_entries_for(self, item_ids: list[str]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item_id in item_ids:
            rendered = str(item_id or "").strip()
            if not rendered or rendered in seen:
                continue
            seen.add(rendered)
            entries.append(self.input_entry_for(rendered))
        return entries
