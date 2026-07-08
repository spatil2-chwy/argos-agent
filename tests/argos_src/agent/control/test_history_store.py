from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from argos_src.agent.control.history_store import OwnerScopedHistoryIndex


def test_history_index_registers_once_and_tracks_owner() -> None:
    index = OwnerScopedHistoryIndex()

    assert index.register("item-1", owner_req_id="rt-1")
    assert not index.register("item-1", owner_req_id="rt-1")

    assert index.snapshot() == ["item-1"]
    assert index.owner_req_id_for("item-1") == "rt-1"


def test_history_index_forgets_item_from_all_indexes() -> None:
    index = OwnerScopedHistoryIndex()
    index.register("item-1", owner_req_id="rt-1")
    index.register("item-2")

    assert index.forget("item-1")

    assert index.snapshot() == ["item-2"]
    assert index.owner_req_id_for("item-1") == ""
    assert "item-1" not in index.known_item_ids


def test_history_index_finds_newest_unbound_item() -> None:
    index = OwnerScopedHistoryIndex()
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
    index = OwnerScopedHistoryIndex(
        item_order=item_order,
        known_item_ids=known_item_ids,
        item_owner_req_id=item_owner_req_id,
    )

    index.register("next", owner_req_id="rt-next")

    assert item_order is index.item_order
    assert known_item_ids is index.known_item_ids
    assert item_owner_req_id is index.item_owner_req_id
    assert list(item_order) == ["existing", "next"]


def test_history_index_plans_protected_items_and_delete_candidates() -> None:
    index = OwnerScopedHistoryIndex()
    index.register("old-user", owner_req_id="old")
    index.register("active-function", owner_req_id="active")
    index.register("current-audio")
    active_turn = SimpleNamespace(
        history_item_ids=set(),
        user_item_id="",
        assistant_item_id="",
        assistant_item_ids=set(),
        function_call_item_ids={"active-function"},
        finalized=False,
        kind="audio",
    )
    current_turn = SimpleNamespace(
        history_item_ids=set(),
        user_item_id="",
        assistant_item_id="",
        assistant_item_ids=set(),
        function_call_item_ids=set(),
        finalized=False,
        kind="audio",
    )

    protected = index.protected_item_ids(
        turns=[active_turn],
        is_terminal=lambda turn: bool(getattr(turn, "finalized", False)),
        current_turn=current_turn,
        bound_item_ids={},
    )

    assert protected == {"active-function", "current-audio"}
    assert index.delete_candidates(protected_item_ids=protected) == ["old-user"]


def test_history_index_owner_key_uses_anonymous_for_unresolved_owner() -> None:
    assert OwnerScopedHistoryIndex.owner_key("person-1") == "owner:person-1"
    assert OwnerScopedHistoryIndex.owner_key(None) == "anonymous"
