"""Typed Slack message shapes used before memory extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SlackUserProfile:
    slack_user_id: str
    username: str = ""
    display_name: str = ""
    real_name: str = ""
    email: str = ""
    person_id: str = ""

    @property
    def label(self) -> str:
        return (
            self.display_name.strip()
            or self.real_name.strip()
            or self.username.strip()
            or self.email.strip()
            or self.slack_user_id
        )

    @property
    def handle(self) -> str:
        username = self.username.strip()
        return f"@{username}" if username else ""

    @property
    def prompt_label(self) -> str:
        label = self.label
        handle = self.handle
        username = self.username.strip()
        if handle and label.strip().lstrip("@").casefold() != username.casefold():
            return f"{label} ({handle})"
        return label


@dataclass(frozen=True)
class SlackMention:
    slack_user_id: str
    label: str = ""
    person_id: str = ""


@dataclass(frozen=True)
class SlackMessage:
    channel_id: str
    channel_name: str
    ts: str
    text: str
    user_id: str = ""
    user_label: str = ""
    person_id: str = ""
    thread_ts: str = ""
    parent_ts: str = ""
    subtype: str = ""
    permalink: str = ""
    mentions: tuple[SlackMention, ...] = ()
    replies: tuple["SlackMessage", ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_ref(self) -> str:
        return f"{self.channel_id}:{self.ts}"

    @property
    def is_thread_reply(self) -> bool:
        return bool(self.thread_ts and self.thread_ts != self.ts)


@dataclass(frozen=True)
class SlackChannelWindow:
    channel_name: str
    channel_id: str
    site_code: str
    start_ts: str
    end_ts: str
    messages: tuple[SlackMessage, ...] = ()
    user_profiles: tuple[SlackUserProfile, ...] = ()

    @property
    def source_ref(self) -> str:
        return f"{self.channel_id}:{self.start_ts}-{self.end_ts}"
