"""Minimal dependency-free Slack Web API client."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any


class SlackApiError(RuntimeError):
    pass


class SlackWebApiClient:
    def __init__(self, token: str, *, timeout_sec: float = 20.0) -> None:
        self.token = str(token or "").strip()
        self.timeout_sec = float(timeout_sec)
        if not self.token:
            raise SlackApiError("Slack bot token is required.")

    def api_call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"https://slack.com/api/{method}"
        query = urllib.parse.urlencode(
            {
                key: value
                for key, value in dict(params or {}).items()
                if value is not None and value != ""
            }
        )
        if query:
            url = f"{url}?{query}"
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict) or not payload.get("ok"):
            error = payload.get("error") if isinstance(payload, dict) else "invalid_response"
            raise SlackApiError(f"Slack API {method} failed: {error}")
        return payload

    def conversations_list(self, *, cursor: str = "") -> dict[str, Any]:
        return self.api_call(
            "conversations.list",
            {
                "types": "public_channel,private_channel",
                "exclude_archived": "true",
                "limit": 200,
                "cursor": cursor,
            },
        )

    def conversations_history(
        self,
        *,
        channel: str,
        oldest: str,
        latest: str = "",
        limit: int = 200,
        cursor: str = "",
    ) -> dict[str, Any]:
        return self.api_call(
            "conversations.history",
            {
                "channel": channel,
                "oldest": oldest,
                "latest": latest,
                "inclusive": "false",
                "limit": limit,
                "cursor": cursor,
            },
        )

    def conversations_replies(
        self,
        *,
        channel: str,
        ts: str,
        cursor: str = "",
    ) -> dict[str, Any]:
        return self.api_call(
            "conversations.replies",
            {
                "channel": channel,
                "ts": ts,
                "limit": 200,
                "cursor": cursor,
            },
        )

    def users_info(self, *, user: str) -> dict[str, Any]:
        return self.api_call("users.info", {"user": user})
