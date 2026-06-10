"""Pure payload readers for OpenAI Realtime server events."""

from __future__ import annotations

from typing import Any


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def server_event_type(event: dict[str, Any]) -> str:
    return _clean_text(event.get("type"))


def server_event_response(event: dict[str, Any]) -> dict[str, Any]:
    response = event.get("response", {}) or {}
    return response if isinstance(response, dict) else {}


def server_event_item(event: dict[str, Any]) -> dict[str, Any]:
    item = event.get("item", {}) or {}
    return item if isinstance(item, dict) else {}


def server_event_response_id(
    event: dict[str, Any],
    *,
    response: dict[str, Any] | None = None,
    item: dict[str, Any] | None = None,
) -> str:
    response_payload = response if response is not None else server_event_response(event)
    item_payload = item if item is not None else server_event_item(event)
    return (
        _clean_text(event.get("response_id"))
        or _clean_text(response_payload.get("id"))
        or _clean_text(item_payload.get("response_id"))
    )


def server_event_item_id(
    event: dict[str, Any],
    *,
    item: dict[str, Any] | None = None,
) -> str:
    item_payload = item if item is not None else server_event_item(event)
    return _clean_text(item_payload.get("id")) or _clean_text(event.get("item_id"))
