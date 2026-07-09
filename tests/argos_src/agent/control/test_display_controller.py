from __future__ import annotations

import logging
import queue
import threading

from argos_src.agent.control.display_controller import DisplayController


class _Display:
    def __init__(self) -> None:
        self.modes: list[str] = []
        self.subtitles: list[tuple[str, int]] = []

    def show_alert(self) -> None:
        self.modes.append("alert")

    def show_subtitle(self, text: str, *, duration_ms: int = 5000) -> None:
        self.subtitles.append((text, duration_ms))


class _Host:
    def __init__(self) -> None:
        self.display_runtime = _Display()
        self._display_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._display_mode_lock = threading.Lock()
        self._display_mode = ""
        self._display_subtitle_lock = threading.Lock()
        self._display_pending_subtitle = None
        self._display_subtitle_queued = False
        self._stop_event = threading.Event()
        self.logger = logging.getLogger("test.display_controller")


def test_subtitle_window_keeps_complete_short_turn() -> None:
    text = (
        "First sentence stays visible. Second sentence stays visible too. "
        "Third sentence also stays visible."
    )

    assert DisplayController.subtitle_window(text) == text


def test_subtitle_window_falls_back_to_recent_complete_sentences() -> None:
    text = (
        "Intro sentence is no longer useful. "
        "Middle sentence still gives context. "
        "Final sentence should stay readable."
    )

    assert DisplayController.subtitle_window(text, max_chars=75) == (
        "Middle sentence still gives context. Final sentence should stay readable."
    )


def test_subtitle_updates_coalesce_to_latest_pending_text() -> None:
    host = _Host()
    controller = DisplayController(host)

    controller.show_subtitle_async("The", duration_ms=1000)
    controller.show_subtitle_async("The best", duration_ms=2000)
    controller.show_subtitle_async("The best way is this.", duration_ms=3000)

    worker = threading.Thread(target=controller.worker_loop)
    worker.start()
    host._display_queue.join()
    host._stop_event.set()
    worker.join(timeout=1.0)

    assert host.display_runtime.subtitles == [("The best way is this.", 3000)]


def test_subtitle_coalescing_preserves_mode_order() -> None:
    host = _Host()
    controller = DisplayController(host)

    controller.set_mode_async("alert")
    controller.show_subtitle_async("old")
    controller.show_subtitle_async("new")

    worker = threading.Thread(target=controller.worker_loop)
    worker.start()
    host._display_queue.join()
    host._stop_event.set()
    worker.join(timeout=1.0)

    assert host.display_runtime.modes == ["alert"]
    assert host.display_runtime.subtitles == [("new", 5000)]
