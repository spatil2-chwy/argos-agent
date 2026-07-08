"""Small state stores for realtime turn/response binding."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, MutableMapping, Protocol
import time


class ResponseTurn(Protocol):
    req_id: str
    response_id: str


IsTerminalTurn = Callable[[ResponseTurn | None], bool]
NowFn = Callable[[], float]


@dataclass(frozen=True)
class ResponseBinding:
    turn: ResponseTurn
    expired_stale: bool = False


class PendingResponseBindingStore:
    """Queue that binds Realtime `response.created` ids back to local turns.

    The queue preserves order because the Realtime API returns response ids
    asynchronously after `response.create`. When a turn is canceled before its
    `response.created` arrives, the stale queue head is retained briefly so the
    late response can be canceled instead of accidentally binding to the next
    turn.
    """

    def __init__(
        self,
        *,
        turns_by_req_id: MutableMapping[str, ResponseTurn],
        is_terminal: IsTerminalTurn,
        pending_req_ids: deque[str] | None = None,
        expired_stale_req_ids: deque[str] | None = None,
        stale_deadlines_by_req_id: MutableMapping[str, float] | None = None,
        response_id_to_req_id: MutableMapping[str, str] | None = None,
        now: NowFn | None = None,
    ) -> None:
        self.turns_by_req_id = turns_by_req_id
        self.is_terminal = is_terminal
        self.pending_req_ids: deque[str] = pending_req_ids if pending_req_ids is not None else deque()
        self.expired_stale_req_ids: deque[str] = (
            expired_stale_req_ids if expired_stale_req_ids is not None else deque()
        )
        self.stale_deadlines_by_req_id = (
            stale_deadlines_by_req_id if stale_deadlines_by_req_id is not None else {}
        )
        self.response_id_to_req_id = (
            response_id_to_req_id if response_id_to_req_id is not None else {}
        )
        self.now = now or time.time

    def queue(self, req_id: str) -> None:
        rendered = str(req_id or "").strip()
        if rendered:
            self.pending_req_ids.append(rendered)

    def consume(
        self,
        response_id: str,
        *,
        consume_only_if_missing: bool = True,
    ) -> ResponseTurn | None:
        binding = self.consume_binding(
            response_id,
            consume_only_if_missing=consume_only_if_missing,
        )
        return binding.turn if binding is not None else None

    def consume_binding(
        self,
        response_id: str,
        *,
        consume_only_if_missing: bool = True,
    ) -> ResponseBinding | None:
        rendered = str(response_id or "").strip()
        if not rendered:
            return None

        existing_req_id = self.response_id_to_req_id.get(rendered, "")
        if existing_req_id and consume_only_if_missing:
            turn = self.turns_by_req_id.get(existing_req_id)
            return ResponseBinding(turn) if turn is not None else None

        while self.expired_stale_req_ids:
            req_id = self.expired_stale_req_ids.popleft()
            turn = self.turns_by_req_id.get(req_id)
            if turn is None:
                continue
            self.response_id_to_req_id[rendered] = req_id
            turn.response_id = rendered
            return ResponseBinding(turn, expired_stale=True)

        while self.pending_req_ids:
            req_id = self.pending_req_ids.popleft()
            turn = self.turns_by_req_id.get(req_id)
            if turn is None:
                self.stale_deadlines_by_req_id.pop(req_id, None)
                continue

            stale_deadline = self.stale_deadlines_by_req_id.get(req_id)
            if self.is_terminal(turn):
                if stale_deadline is None:
                    continue
                if self.now() > float(stale_deadline):
                    self.stale_deadlines_by_req_id.pop(req_id, None)
                    self.response_id_to_req_id[rendered] = req_id
                    turn.response_id = rendered
                    return ResponseBinding(turn, expired_stale=True)
                self.stale_deadlines_by_req_id.pop(req_id, None)
                self.response_id_to_req_id[rendered] = req_id
                turn.response_id = rendered
                return ResponseBinding(turn)

            self.response_id_to_req_id[rendered] = req_id
            turn.response_id = rendered
            return ResponseBinding(turn)

        return None

    def mark_stale(self, req_id: str, *, timeout_s: float) -> bool:
        rendered = str(req_id or "").strip()
        if not rendered or rendered not in self.pending_req_ids:
            return False
        self.stale_deadlines_by_req_id[rendered] = self.now() + float(timeout_s)
        return True

    def next_stale_deadline(self) -> float | None:
        while self.pending_req_ids:
            req_id = self.pending_req_ids[0]
            deadline = self.stale_deadlines_by_req_id.get(req_id)
            if deadline is None:
                return None
            if self.now() > float(deadline):
                self.pending_req_ids.popleft()
                self.stale_deadlines_by_req_id.pop(req_id, None)
                self.expired_stale_req_ids.append(req_id)
                continue
            return float(deadline)
        return None

    def discard(self, req_id: str) -> int:
        rendered = str(req_id or "").strip()
        if not rendered:
            return 0
        before = len(self.pending_req_ids)
        self.pending_req_ids = deque(
            candidate for candidate in self.pending_req_ids if candidate != rendered
        )
        discarded = before - len(self.pending_req_ids)
        if discarded:
            self.stale_deadlines_by_req_id.pop(rendered, None)
        self.expired_stale_req_ids = deque(
            candidate for candidate in self.expired_stale_req_ids if candidate != rendered
        )
        return discarded
