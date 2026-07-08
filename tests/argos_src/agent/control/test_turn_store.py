from __future__ import annotations

from dataclasses import dataclass

from argos_src.agent.control.turn_store import PendingResponseBindingStore


@dataclass
class _Turn:
    req_id: str
    phase: str = "committed"
    finalized: bool = False
    response_id: str = ""


def _is_terminal(turn: _Turn | None) -> bool:
    return turn is None or turn.finalized or turn.phase in {"finalized", "canceled", "superseded"}


def test_stale_terminal_response_consumes_queue_head_without_binding_next_turn():
    old_turn = _Turn("rt-old", phase="canceled", finalized=True)
    new_turn = _Turn("rt-new")
    store = PendingResponseBindingStore(
        turns_by_req_id={old_turn.req_id: old_turn, new_turn.req_id: new_turn},
        is_terminal=_is_terminal,
        now=lambda: 10.0,
    )
    store.queue(old_turn.req_id)
    store.queue(new_turn.req_id)

    assert store.mark_stale(old_turn.req_id, timeout_s=12.0)

    consumed = store.consume("resp-stale-old")

    assert consumed is old_turn
    assert old_turn.response_id == "resp-stale-old"
    assert new_turn.response_id == ""
    assert list(store.pending_req_ids) == [new_turn.req_id]
    assert store.response_id_to_req_id["resp-stale-old"] == old_turn.req_id


def test_expired_stale_response_slot_consumes_next_ambiguous_response():
    now = 10.0
    old_turn = _Turn("rt-old", phase="canceled", finalized=True)
    new_turn = _Turn("rt-new")
    store = PendingResponseBindingStore(
        turns_by_req_id={old_turn.req_id: old_turn, new_turn.req_id: new_turn},
        is_terminal=_is_terminal,
        now=lambda: now,
    )
    store.queue(old_turn.req_id)
    store.queue(new_turn.req_id)
    assert store.mark_stale(old_turn.req_id, timeout_s=2.0)

    now = 13.0
    consumed = store.consume_binding("resp-late")

    assert consumed is not None
    assert consumed.turn is old_turn
    assert consumed.expired_stale is True
    assert old_turn.response_id == "resp-late"
    assert new_turn.response_id == ""
    assert list(store.pending_req_ids) == [new_turn.req_id]


def test_next_stale_deadline_blocks_behind_live_stale_queue_head_until_expired():
    now = 10.0
    old_turn = _Turn("rt-old", phase="canceled", finalized=True)
    new_turn = _Turn("rt-new")
    store = PendingResponseBindingStore(
        turns_by_req_id={old_turn.req_id: old_turn, new_turn.req_id: new_turn},
        is_terminal=_is_terminal,
        now=lambda: now,
    )
    store.queue(old_turn.req_id)
    store.queue(new_turn.req_id)
    assert store.mark_stale(old_turn.req_id, timeout_s=2.0)

    assert store.next_stale_deadline() == 12.0

    now = 13.0
    assert store.next_stale_deadline() is None
    assert list(store.pending_req_ids) == [new_turn.req_id]
    assert list(store.expired_stale_req_ids) == [old_turn.req_id]


def test_discard_pending_response_turn_removes_all_matching_requests():
    store = PendingResponseBindingStore(turns_by_req_id={}, is_terminal=_is_terminal)
    store.queue("rt-a")
    store.queue("rt-b")
    store.queue("rt-a")
    store.stale_deadlines_by_req_id["rt-a"] = 99.0

    assert store.discard("rt-a") == 2

    assert list(store.pending_req_ids) == ["rt-b"]
    assert "rt-a" not in store.stale_deadlines_by_req_id
