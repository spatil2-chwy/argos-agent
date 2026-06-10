"""Slack-backed memory ingestion for Argos."""

from __future__ import annotations

from argos_src.memory.slack.extract import (
    SlackExtractionOutput,
    SlackMemoryExtractor,
    SlackMemoryOperationOutput,
    build_slack_extraction_prompt,
)
from argos_src.memory.slack.models import (
    SlackChannelWindow,
    SlackMessage,
    SlackUserProfile,
)
from argos_src.memory.slack.pending import (
    list_pending_slack_memory,
    promote_pending_slack_memory,
)
from argos_src.memory.slack.service import SlackMemoryService
from argos_src.memory.slack.writer import write_slack_memory_operations

__all__ = [
    "SlackChannelWindow",
    "SlackExtractionOutput",
    "SlackMemoryExtractor",
    "SlackMemoryOperationOutput",
    "SlackMemoryService",
    "SlackMessage",
    "SlackUserProfile",
    "build_slack_extraction_prompt",
    "list_pending_slack_memory",
    "promote_pending_slack_memory",
    "write_slack_memory_operations",
]
