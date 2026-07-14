from __future__ import annotations

from collections import deque

from argos_src.agent.control.history_store import InferenceHistoryIndex


def test_history_index_registers_once_and_tracks_owner_scope() -> None:
    index = InferenceHistoryIndex()

    assert index.register(
        "item-1",
        owner_req_id="rt-1",
        owner_key="owner:person-1",
        scope_id="owner:person-1",
        status="done",
        permitted_for_inference=True,
    )
    assert not index.register("item-1", owner_req_id="rt-1")

    assert index.snapshot() == ["item-1"]
    assert index.owner_req_id_for("item-1") == "rt-1"
    assert index.selected_item_ids(scope_id="owner:person-1") == ["item-1"]


def test_history_index_forgets_item_from_all_indexes() -> None:
    index = InferenceHistoryIndex()
    index.register("item-1", owner_req_id="rt-1")
    index.register("item-2")

    assert index.forget("item-1")

    assert index.snapshot() == ["item-2"]
    assert index.owner_req_id_for("item-1") == ""
    assert "item-1" not in index.known_item_ids
    assert "item-1" not in index.items


def test_history_index_finds_newest_unbound_item() -> None:
    index = InferenceHistoryIndex()
    index.register("old-owned", owner_req_id="rt-old")
    index.register("bound-but-unowned")
    index.register("new-unbound")

    assert index.newest_unbound_item(
        bound_item_ids={"bound-but-unowned": "rt-current"}
    ) == "new-unbound"


def test_history_index_uses_supplied_runtime_containers() -> None:
    item_order = deque(["existing"])
    known_item_ids = {"existing"}
    item_owner_req_id = {"existing": "rt-existing"}
    items = {}
    index = InferenceHistoryIndex(
        item_order=item_order,
        known_item_ids=known_item_ids,
        item_owner_req_id=item_owner_req_id,
        items=items,
    )

    index.register("next", owner_req_id="rt-next")

    assert item_order is index.item_order
    assert known_item_ids is index.known_item_ids
    assert item_owner_req_id is index.item_owner_req_id
    assert items is index.items
    assert list(item_order) == ["existing", "next"]


def test_history_index_selects_only_done_permitted_items_in_scope() -> None:
    index = InferenceHistoryIndex()
    index.register(
        "owner-a-user",
        scope_id="owner:A",
        status="done",
        permitted_for_inference=True,
    )
    index.register(
        "owner-b-user",
        scope_id="owner:B",
        status="done",
        permitted_for_inference=True,
    )
    index.register(
        "owner-a-draft",
        scope_id="owner:A",
        status="in_progress",
        permitted_for_inference=True,
    )
    index.register(
        "owner-a-quarantined",
        scope_id="owner:A",
        status="done",
        permitted_for_inference=False,
    )

    assert index.selected_item_ids(scope_id="owner:A") == ["owner-a-user"]


def test_history_index_input_entries_prefer_local_raw_items() -> None:
    index = InferenceHistoryIndex()
    raw_item = {
        "id": "local-1",
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": "hello"}],
    }
    index.register("local-1", input_item=raw_item)
    index.register("server-1")

    assert index.input_entries_for(["local-1", "server-1"]) == [
        raw_item,
        {"type": "item_reference", "id": "server-1"},
    ]


def test_history_index_owner_key_uses_anonymous_for_unresolved_owner() -> None:
    assert InferenceHistoryIndex.owner_key("person-1") == "owner:person-1"
    assert InferenceHistoryIndex.owner_key(None) == "anonymous"
