"""Asynchronous display updates for the realtime control runtime."""

from __future__ import annotations

import queue
from typing import Any


class DisplayController:
    """Serialize display mode/subtitle updates onto the display runtime."""

    def __init__(self, host: Any) -> None:
        self._host = host

    def set_mode_async(self, mode: str, *, force: bool = False) -> None:
        host = self._host
        if getattr(host, "display_runtime", None) is None:
            return
        rendered = str(mode or "").strip()
        if not rendered:
            return
        with host._display_mode_lock:
            if not force and rendered == host._display_mode:
                return
            host._display_mode = rendered
        host._display_queue.put(("mode", rendered))

    def clear_passive_alert_if_needed(self) -> None:
        host = self._host
        if getattr(host, "display_runtime", None) is None:
            return
        with host._display_mode_lock:
            should_clear = host._display_mode == "alert"
        if should_clear:
            self.set_mode_async("idle")

    def show_subtitle_async(self, text: str, *, duration_ms: int = 5000) -> None:
        host = self._host
        if getattr(host, "display_runtime", None) is None:
            return
        rendered = str(text or "").strip()
        if not rendered:
            return
        host._display_queue.put(
            (
                "subtitle",
                {
                    "text": rendered,
                    "duration_ms": int(duration_ms),
                },
            )
        )

    def worker_loop(self) -> None:
        host = self._host
        while not host._stop_event.is_set():
            try:
                kind, payload = host._display_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                display = getattr(host, "display_runtime", None)
                if display is None:
                    continue
                if kind == "mode":
                    self.apply_mode(display, str(payload or ""))
                elif kind == "subtitle" and isinstance(payload, dict):
                    display.show_subtitle(
                        str(payload.get("text", "") or ""),
                        duration_ms=int(payload.get("duration_ms", 5000) or 5000),
                    )
            except Exception:
                host.logger.debug("Display update failed", exc_info=True)
            finally:
                host._display_queue.task_done()

    @staticmethod
    def subtitle_window(text: str) -> str:
        return " ".join(str(text or "").split())

    @staticmethod
    def apply_mode(display: Any, mode: str) -> None:
        if mode == "idle":
            display.show_idle()
        elif mode == "alert":
            display.show_alert()
        elif mode == "recording":
            display.show_recording()
        elif mode == "thinking":
            display.show_thinking()
        elif mode == "speaking":
            display.show_speaking()
