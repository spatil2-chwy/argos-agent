"""Normalize Slack API payloads into prompt-safe window text."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from argos_src.memory.slack.models import (
    SlackChannelWindow,
    SlackMention,
    SlackMessage,
    SlackUserProfile,
)

MENTION_RE = re.compile(r"<@([UW][A-Z0-9]+)(?:\|[^>]+)?>")


def slack_ts_to_datetime(ts: str) -> datetime | None:
    text = str(ts or "").strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except Exception:
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(microsecond=0)


def slack_ts_to_iso(ts: str) -> str:
    parsed = slack_ts_to_datetime(ts)
    return parsed.isoformat() if parsed is not None else ""


def mentioned_user_ids(text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in MENTION_RE.finditer(str(text or "")):
        user_id = match.group(1)
        if user_id in seen:
            continue
        seen.add(user_id)
        ordered.append(user_id)
    return tuple(ordered)


def render_slack_text(text: str, profiles: dict[str, SlackUserProfile] | None) -> str:
    lookup = profiles or {}

    def replace(match: re.Match[str]) -> str:
        user_id = match.group(1)
        profile = lookup.get(user_id)
        if profile is None:
            return "@unknown_user"
        return f"@{profile.prompt_label}"

    return MENTION_RE.sub(replace, str(text or ""))


def normalize_message(
    raw: dict[str, Any],
    *,
    channel_id: str,
    channel_name: str,
    user_profiles: dict[str, SlackUserProfile] | None = None,
    parent_ts: str = "",
    replies: tuple[SlackMessage, ...] = (),
) -> SlackMessage | None:
    ts = str(raw.get("ts") or "").strip()
    if not ts:
        return None
    subtype = str(raw.get("subtype") or "").strip()
    if subtype in {"message_deleted", "channel_join", "channel_leave"}:
        return None
    profiles = user_profiles or {}
    user_id = str(raw.get("user") or raw.get("bot_id") or "").strip()
    profile = profiles.get(user_id)
    raw_text = str(raw.get("text") or "").strip()
    mentions = tuple(
        SlackMention(
            slack_user_id=mentioned_id,
            label=(
                profiles.get(mentioned_id).prompt_label
                if profiles.get(mentioned_id)
                else "unknown_user"
            ),
            person_id=(
                profiles.get(mentioned_id).person_id if profiles.get(mentioned_id) else ""
            ),
        )
        for mentioned_id in mentioned_user_ids(raw_text)
    )
    return SlackMessage(
        channel_id=channel_id,
        channel_name=channel_name,
        ts=ts,
        text=render_slack_text(raw_text, profiles),
        user_id=user_id,
        user_label=profile.prompt_label if profile is not None else "unknown",
        person_id=profile.person_id if profile is not None else "",
        thread_ts=str(raw.get("thread_ts") or "").strip(),
        parent_ts=parent_ts,
        subtype=subtype,
        permalink=str(raw.get("permalink") or "").strip(),
        mentions=mentions,
        replies=replies,
        metadata={
            "reply_count": raw.get("reply_count", 0) or 0,
            "raw_type": raw.get("type", ""),
        },
    )


def render_window_for_prompt(window: SlackChannelWindow) -> str:
    lines: list[str] = []
    for message in sorted(window.messages, key=lambda item: item.ts):
        _append_message_lines(lines, message, indent="")
    return "\n".join(lines)


def _append_message_lines(lines: list[str], message: SlackMessage, *, indent: str) -> None:
    author = message.user_label or "unknown"
    observed = slack_ts_to_iso(message.ts) or message.ts
    text = " ".join(str(message.text or "").split())
    lines.append(f"{indent}[{observed}] {author}: {text}")
    if message.mentions:
        rendered_mentions = ", ".join(
            mention.label or "unknown_user"
            for mention in message.mentions
        )
        lines.append(f"{indent}  mentioned_users: {rendered_mentions}")
    for reply in sorted(message.replies, key=lambda item: item.ts):
        _append_message_lines(lines, reply, indent=f"{indent}  ")
