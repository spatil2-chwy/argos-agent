"""Background Slack memory polling service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
import sqlite3
import sys
import threading
from typing import Any

from argos_src.memory.slack.client import SlackApiError, SlackWebApiClient
from argos_src.memory.slack.extract import SlackMemoryExtractor
from argos_src.memory.slack.identity import SlackIdentityResolver
from argos_src.memory.slack.models import (
    SlackChannelWindow,
    SlackMessage,
    SlackUserProfile,
)
from argos_src.memory.slack.normalize import mentioned_user_ids, normalize_message
from argos_src.memory.slack.pending import (
    init_pending_slack_memory_schema,
    promote_resolved_pending_slack_memory,
)


logger = logging.getLogger(__name__)


class SlackMemoryService:
    def __init__(
        self,
        *,
        profile: Any,
        memory_store: Any,
        identity_store: Any | None = None,
        default_site_code: str = "",
        client: SlackWebApiClient | None = None,
        debug_llm_prompt: bool = False,
        debug_llm_output: bool = False,
    ) -> None:
        self.profile = profile
        self.memory_store = memory_store
        self.identity_store = identity_store
        self.default_site_code = str(default_site_code or "").strip()
        self.client = client
        self.debug_llm_prompt = bool(debug_llm_prompt)
        self.debug_llm_output = bool(debug_llm_output)
        self.identity_resolver = SlackIdentityResolver(identity_store)
        self.extractor: SlackMemoryExtractor | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._init_checkpoint_schema()

    def start_background(self) -> None:
        if not getattr(self.profile, "enabled", False):
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run_forever,
            name="argos-slack-memory",
            daemon=True,
        )
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def run_once(self) -> None:
        if not getattr(self.profile, "enabled", False):
            return
        channels = tuple(getattr(self.profile, "channels", ()) or ())
        if not channels:
            logger.info("[SlackMemory] enabled with no configured channels; skipping")
            return
        self._promote_resolved_pending_memory()
        client = self._client()
        if client is None:
            logger.info("[SlackMemory] missing Slack token; set %s", self.profile.bot_token_env)
            return
        for channel in channels:
            channel_id = self._channel_id_for_config(client, channel)
            if not channel_id:
                logger.info(
                    "[SlackMemory] channel %s could not be resolved",
                    getattr(channel, "name", ""),
                )
                continue
            fallback_oldest = datetime.now(timezone.utc) - timedelta(
                minutes=int(getattr(self.profile, "lookback_minutes", 30) or 30)
            )
            oldest = self._checkpoint_datetime(channel_id) or fallback_oldest
            latest = datetime.now(timezone.utc)
            try:
                window = self._fetch_channel_window(
                    client,
                    channel=channel,
                    channel_id=channel_id,
                    oldest=oldest,
                    latest=latest,
                )
            except SlackApiError:
                logger.exception(
                    "[SlackMemory] failed to fetch channel=%s",
                    getattr(channel, "name", channel_id),
                )
                continue
            if not window.messages:
                if self.debug_llm_prompt or self.debug_llm_output:
                    print(
                        (
                            "[SlackMemory] no messages to extract "
                            f"channel={getattr(channel, 'name', channel_id)} "
                            f"oldest={oldest.isoformat()} latest={latest.isoformat()}"
                        ),
                        file=sys.stderr,
                    )
                self._set_checkpoint_ts(channel_id, str(latest.timestamp()))
                continue
            try:
                self._extractor().extract_and_store_window(
                    window=window,
                    person_memory_enabled=bool(
                        getattr(channel, "person_memory_enabled", True)
                    ),
                    site_memory_enabled=bool(getattr(channel, "site_memory_enabled", True)),
                    debug_llm_prompt=self.debug_llm_prompt,
                    debug_llm_output=self.debug_llm_output,
                )
                self._set_checkpoint_ts(channel_id, window.end_ts)
                self._promote_resolved_pending_memory()
            except Exception:
                logger.exception(
                    "[SlackMemory] failed to extract channel=%s",
                    getattr(channel, "name", channel_id),
                )

    def run_forever(self) -> None:
        interval = float(getattr(self.profile, "poll_interval_sec", 1800.0) or 1800.0)
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("[SlackMemory] polling cycle failed")
            self._stop.wait(interval)

    def _client(self) -> SlackWebApiClient | None:
        if self.client is not None:
            return self.client
        token = os.environ.get(str(getattr(self.profile, "bot_token_env", "") or ""))
        if not token:
            return None
        self.client = SlackWebApiClient(token)
        return self.client

    def _checkpoint_db_path(self) -> str:
        return str(getattr(self.memory_store, "db_path", "") or "").strip()

    def _init_checkpoint_schema(self) -> None:
        db_path = self._checkpoint_db_path()
        if not db_path:
            return
        init_pending_slack_memory_schema(self.memory_store)
        with sqlite3.connect(db_path, timeout=30.0) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS slack_channel_checkpoints (
                    channel_id TEXT PRIMARY KEY,
                    last_ts TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _checkpoint_datetime(self, channel_id: str) -> datetime | None:
        db_path = self._checkpoint_db_path()
        if not db_path:
            return None
        with sqlite3.connect(db_path, timeout=30.0) as connection:
            row = connection.execute(
                "SELECT last_ts FROM slack_channel_checkpoints WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            return datetime.fromtimestamp(float(row[0]), tz=timezone.utc)
        except Exception:
            return None

    def _set_checkpoint_ts(self, channel_id: str, ts: str) -> None:
        db_path = self._checkpoint_db_path()
        if not db_path:
            return
        rendered_channel = str(channel_id or "").strip()
        rendered_ts = str(ts or "").strip()
        if not rendered_channel or not rendered_ts:
            return
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with sqlite3.connect(db_path, timeout=30.0) as connection:
            connection.execute(
                """
                INSERT INTO slack_channel_checkpoints (channel_id, last_ts, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    last_ts = excluded.last_ts,
                    updated_at = excluded.updated_at
                """,
                (rendered_channel, rendered_ts, now),
            )

    def _extractor(self) -> SlackMemoryExtractor:
        if self.extractor is None:
            self.extractor = SlackMemoryExtractor(memory_store=self.memory_store)
        return self.extractor

    def _promote_resolved_pending_memory(self) -> None:
        if self.identity_store is None:
            return
        affected = promote_resolved_pending_slack_memory(
            self.memory_store,
            identity_resolver=self.identity_resolver,
        )
        if affected:
            logger.info("[SlackMemory] promoted pending memories=%s", len(affected))

    def _channel_id_for_config(self, client: SlackWebApiClient, channel: Any) -> str:
        channel_id = str(getattr(channel, "channel_id", "") or "").strip()
        if channel_id:
            return channel_id
        wanted = str(getattr(channel, "name", "") or "").strip().lstrip("#")
        if not wanted:
            return ""
        cursor = ""
        while True:
            response = client.conversations_list(cursor=cursor)
            for item in response.get("channels", []) or []:
                if str(item.get("name") or "").strip() == wanted:
                    return str(item.get("id") or "").strip()
            cursor = str(
                (response.get("response_metadata") or {}).get("next_cursor") or ""
            ).strip()
            if not cursor:
                return ""

    def _fetch_channel_window(
        self,
        client: SlackWebApiClient,
        *,
        channel: Any,
        channel_id: str,
        oldest: datetime,
        latest: datetime,
    ) -> SlackChannelWindow:
        raw_messages: list[dict[str, Any]] = []
        cursor = ""
        max_messages = int(getattr(channel, "max_messages_per_window", 200) or 200)
        while len(raw_messages) < max_messages:
            response = client.conversations_history(
                channel=channel_id,
                oldest=str(oldest.timestamp()),
                latest=str(latest.timestamp()),
                limit=min(200, max_messages - len(raw_messages)),
                cursor=cursor,
            )
            raw_messages.extend(
                self._message_without_reactions(item)
                for item in response.get("messages", []) or []
            )
            cursor = str(
                (response.get("response_metadata") or {}).get("next_cursor") or ""
            ).strip()
            if not cursor:
                break
        replies_by_parent: dict[str, list[dict[str, Any]]] = {}
        if bool(getattr(channel, "include_threads", True)):
            for raw in raw_messages:
                if int(raw.get("reply_count", 0) or 0) <= 0:
                    continue
                parent_ts = str(raw.get("ts") or "").strip()
                if not parent_ts:
                    continue
                try:
                    replies = self._fetch_thread_replies(
                        client,
                        channel_id=channel_id,
                        parent_ts=parent_ts,
                    )
                except SlackApiError:
                    logger.exception(
                        "[SlackMemory] failed to fetch thread channel=%s ts=%s",
                        getattr(channel, "name", channel_id),
                        parent_ts,
                    )
                    replies = []
                replies_by_parent[parent_ts] = [
                    self._message_without_reactions(item)
                    for item in replies or []
                    if str(item.get("ts") or "") != parent_ts
                ]

        profiles = self._profiles_for_raw_messages(client, raw_messages, replies_by_parent)
        channel_name = str(getattr(channel, "name", "") or "").strip().lstrip("#")
        messages: list[SlackMessage] = []
        for raw in raw_messages:
            parent_ts = str(raw.get("ts") or "").strip()
            replies = tuple(
                reply
                for reply in (
                    normalize_message(
                        reply_raw,
                        channel_id=channel_id,
                        channel_name=channel_name,
                        user_profiles=profiles,
                        parent_ts=parent_ts,
                    )
                    for reply_raw in replies_by_parent.get(parent_ts, [])
                )
                if reply is not None
            )
            message = normalize_message(
                raw,
                channel_id=channel_id,
                channel_name=channel_name,
                user_profiles=profiles,
                replies=replies,
            )
            if message is not None:
                messages.append(message)
        return SlackChannelWindow(
            channel_name=channel_name,
            channel_id=channel_id,
            site_code=(
                str(getattr(channel, "site_code", "") or "").strip()
                or self.default_site_code
            ),
            start_ts=str(oldest.timestamp()),
            end_ts=str(latest.timestamp()),
            messages=tuple(messages),
            user_profiles=tuple(profiles.values()),
        )

    @staticmethod
    def _message_without_reactions(raw: Any) -> dict[str, Any]:
        message = dict(raw) if isinstance(raw, dict) else {}
        message.pop("reactions", None)
        return message

    def _fetch_thread_replies(
        self,
        client: SlackWebApiClient,
        *,
        channel_id: str,
        parent_ts: str,
    ) -> list[dict[str, Any]]:
        replies: list[dict[str, Any]] = []
        cursor = ""
        while True:
            response = client.conversations_replies(
                channel=channel_id,
                ts=parent_ts,
                cursor=cursor,
            )
            replies.extend(dict(item) for item in response.get("messages", []) or [])
            cursor = str(
                (response.get("response_metadata") or {}).get("next_cursor") or ""
            ).strip()
            if not cursor:
                return replies

    def _profiles_for_raw_messages(
        self,
        client: SlackWebApiClient,
        raw_messages: list[dict[str, Any]],
        replies_by_parent: dict[str, list[dict[str, Any]]],
    ) -> dict[str, SlackUserProfile]:
        user_ids: set[str] = set()

        def collect(raw: dict[str, Any]) -> None:
            user_id = str(raw.get("user") or "").strip()
            if user_id:
                user_ids.add(user_id)
            for mentioned_id in mentioned_user_ids(str(raw.get("text") or "")):
                user_ids.add(mentioned_id)

        for raw in raw_messages:
            collect(raw)
        for replies in replies_by_parent.values():
            for raw in replies:
                collect(raw)
        profiles: dict[str, SlackUserProfile] = {}
        for user_id in sorted(user_ids):
            try:
                user_payload = client.users_info(user=user_id).get("user", {}) or {}
            except SlackApiError:
                logger.exception("[SlackMemory] failed to fetch user=%s", user_id)
                user_payload = {}
            profile_payload = dict(user_payload.get("profile") or {})
            profile = SlackUserProfile(
                slack_user_id=user_id,
                username=str(user_payload.get("name") or "").strip(),
                display_name=str(profile_payload.get("display_name") or "").strip(),
                real_name=str(user_payload.get("real_name") or "").strip(),
                email=str(profile_payload.get("email") or "").strip(),
            )
            profiles[user_id] = self.identity_resolver.resolve_user(profile)
        return profiles

    @staticmethod
    def _profiles_for_window(window: SlackChannelWindow) -> dict[str, SlackUserProfile]:
        profiles: dict[str, SlackUserProfile] = {
            profile.slack_user_id: profile
            for profile in window.user_profiles
            if profile.slack_user_id
        }

        def add(message: SlackMessage) -> None:
            if message.user_id:
                existing = profiles.get(message.user_id)
                profiles[message.user_id] = SlackUserProfile(
                    slack_user_id=message.user_id,
                    username=existing.username if existing is not None else "",
                    display_name=(
                        existing.display_name if existing is not None else message.user_label
                    ),
                    real_name=existing.real_name if existing is not None else "",
                    email=existing.email if existing is not None else "",
                    person_id=message.person_id
                    or (existing.person_id if existing is not None else ""),
                )
            for mention in message.mentions:
                existing = profiles.get(mention.slack_user_id)
                profiles[mention.slack_user_id] = SlackUserProfile(
                    slack_user_id=mention.slack_user_id,
                    username=existing.username if existing is not None else "",
                    display_name=(
                        existing.display_name if existing is not None else mention.label
                    ),
                    real_name=existing.real_name if existing is not None else "",
                    email=existing.email if existing is not None else "",
                    person_id=mention.person_id
                    or (existing.person_id if existing is not None else ""),
                )
            for reply in message.replies:
                add(reply)

        for message in window.messages:
            add(message)
        return profiles
