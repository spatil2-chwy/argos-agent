"""Realtime websocket transport helpers."""

from __future__ import annotations

import json
from typing import Any

import websocket


class TransportRuntime:
    """Own low-level websocket sends and transport-safe payload rendering."""

    def __init__(self, host: Any) -> None:
        self._host = host

    @staticmethod
    def stringify_tool_output(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, (dict, list, tuple)):
            return json.dumps(content, ensure_ascii=True)
        return str(content)

    def send_event(self, payload: dict[str, Any]) -> None:
        host = self._host
        if host._ws is None:
            if host._stop_event.is_set():
                return
            raise RuntimeError("Realtime websocket is not connected.")
        with host._ws_lock:
            try:
                host._ws.send(json.dumps(payload))
            except websocket.WebSocketConnectionClosedException:
                if not host._stop_event.is_set():
                    host.logger.warning(
                        "Realtime websocket closed during send; stopping runtime"
                    )
                    host._stop_event.set()
                host._ws = None
                return
            except RuntimeError as exc:
                if "closed" not in str(exc).lower():
                    raise
                if not host._stop_event.is_set():
                    host.logger.warning(
                        "Realtime websocket closed during send; stopping runtime"
                    )
                    host._stop_event.set()
                host._ws = None
                return
