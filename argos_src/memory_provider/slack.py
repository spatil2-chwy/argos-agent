"""Argos scheduler wrapper for Tailwag Slack ingestion."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
from typing import Any


logger = logging.getLogger(__name__)


class TailwagSlackMemoryService:
    """Poll configured Slack channels through Tailwag's ingestion path."""

    def __init__(self, *, profile: Any, episode_recorder: Any) -> None:
        self.profile = profile
        self.episode_recorder = episode_recorder
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._poller: Any | None = None

    def start_background(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="tailwag-slack-memory",
            daemon=True,
        )
        self._thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None

    def poll_once(self) -> None:
        poller = self._get_poller()
        for channel in getattr(self.profile, "channels", ()) or ():
            channel_id = str(getattr(channel, "channel_id", "") or "").strip()
            if not channel_id:
                logger.warning(
                    "Skipping Tailwag Slack memory channel without channel_id name=%s",
                    str(getattr(channel, "name", "") or "").strip() or "<unnamed>",
                )
                continue
            backfill_hours = getattr(channel, "backfill_hours", None)
            if backfill_hours is None:
                backfill_hours = getattr(self.profile, "backfill_hours", None)
            result = poller.poll_once(
                channel_id,
                backfill_hours=backfill_hours,
                force_backfill=bool(getattr(self.profile, "force_backfill", False)),
                history_limit=int(getattr(self.profile, "history_limit", 200) or 200),
                reply_limit=int(getattr(self.profile, "reply_limit", 200) or 200),
                extract_memory=bool(getattr(self.profile, "extract_memory", True)),
            )
            logger.info(
                "Tailwag Slack memory poll channel=%s checked_threads=%s "
                "ingested_threads=%s armed_without_backfill=%s",
                channel_id,
                result.checked_threads,
                result.ingested_threads,
                result.armed_without_backfill,
            )

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception:
                logger.exception("Tailwag Slack memory polling pass failed")
            interval = float(getattr(self.profile, "poll_interval_sec", 1800.0) or 1800.0)
            self._stop_event.wait(max(1.0, interval))

    def _get_poller(self) -> Any:
        if self._poller is not None:
            return self._poller

        token_env = str(getattr(self.profile, "bot_token_env", "") or "SLACK_BOT_TOKEN")
        token = os.getenv(token_env, "").strip()
        if not token:
            raise RuntimeError(f"{token_env} is required for Tailwag Slack memory polling.")

        from tailwag_memory.slack_ingestion import SlackMemoryPoller, SlackWebApiClient

        slack_client = SlackWebApiClient(
            token,
            include_email=bool(getattr(self.profile, "include_email", True)),
        )
        self._poller = SlackMemoryPoller(
            slack_client,
            self.episode_recorder,
            Path(str(getattr(self.profile, "state_path", "") or ".tailwag/slack-state.json")),
            retention_class=str(getattr(self.episode_recorder, "retention_class", "") or "standard"),
            active_thread_hours=float(
                getattr(self.profile, "active_thread_hours", 24.0) or 24.0
            ),
        )
        return self._poller
