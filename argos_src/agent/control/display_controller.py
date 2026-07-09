"""Asynchronous display updates for the realtime control runtime."""

from __future__ import annotations

import queue
import re
import threading
from typing import Any


DEFAULT_SUBTITLE_MAX_CHARS = 350
DEFAULT_SUBTITLE_RECENT_SENTENCES = 3
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


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
        payload = {
            "text": rendered,
            "duration_ms": int(duration_ms),
        }
        with self._subtitle_lock():
            host._display_pending_subtitle = payload
            if bool(getattr(host, "_display_subtitle_queued", False)):
                return
            host._display_subtitle_queued = True
        host._display_queue.put(("subtitle_latest", None))

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
                    self._apply_subtitle(display, payload)
                elif kind == "subtitle_latest":
                    payload = self._take_latest_subtitle()
                    if payload is not None:
                        self._apply_subtitle(display, payload)
            except Exception:
                host.logger.debug("Display update failed", exc_info=True)
            finally:
                host._display_queue.task_done()

    @staticmethod
    def subtitle_window(
        text: str,
        *,
        max_chars: int = DEFAULT_SUBTITLE_MAX_CHARS,
        recent_sentences: int = DEFAULT_SUBTITLE_RECENT_SENTENCES,
    ) -> str:
        rendered = " ".join(str(text or "").split())
        if len(rendered) <= max_chars:
            return rendered
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_END_RE.split(rendered)
            if sentence.strip()
        ]
        if len(sentences) > 1:
            window = sentences[-max(1, int(recent_sentences)) :]
            while len(window) > 1:
                candidate = " ".join(window)
                if len(candidate) <= max_chars:
                    return candidate
                window = window[1:]
            rendered = window[0]
        trimmed = rendered[-max_chars:].strip()
        if " " in trimmed:
            trimmed = trimmed.split(" ", 1)[1]
        return trimmed.strip()

    def _subtitle_lock(self) -> threading.Lock:
        host = self._host
        lock = getattr(host, "_display_subtitle_lock", None)
        if lock is None:
            lock = threading.Lock()
            host._display_subtitle_lock = lock
            host._display_pending_subtitle = None
            host._display_subtitle_queued = False
        return lock

    def _take_latest_subtitle(self) -> dict[str, Any] | None:
        host = self._host
        with self._subtitle_lock():
            payload = getattr(host, "_display_pending_subtitle", None)
            host._display_pending_subtitle = None
            host._display_subtitle_queued = False
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _apply_subtitle(display: Any, payload: dict[str, Any]) -> None:
        display.show_subtitle(
            str(payload.get("text", "") or ""),
            duration_ms=int(payload.get("duration_ms", 5000) or 5000),
        )

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
