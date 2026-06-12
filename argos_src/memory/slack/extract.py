"""Structured LLM contract for extracting memory operations from Slack windows."""

from __future__ import annotations

from datetime import datetime
import json
import logging
import sys
from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - pydantic v1 compatibility
    ConfigDict = None

from argos_src.memory.slack.models import SlackChannelWindow
from argos_src.memory.slack.normalize import render_window_for_prompt
from argos_src.memory.slack.writer import write_slack_memory_operations


logger = logging.getLogger(__name__)
SLACK_MEMORY_EXTRACTION_MODEL = "gpt-4.1"


class _StrictBaseModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover - pydantic v1 compatibility
        class Config:
            extra = "forbid"


class SlackMemoryOperationOutput(_StrictBaseModel):
    op: Literal["create", "update", "archive", "noop"]
    scope_type: Literal["person", "site"] = "site"
    scope_id: str = ""
    target_users: list[str] = Field(default_factory=list)
    memory_id: str = ""
    kind: str = ""
    key: str = ""
    summary: str = ""
    observed_at: str = ""
    due_at: str = ""
    expires_at: str = ""


class SlackExtractionOutput(_StrictBaseModel):
    update: bool = False
    ops: list[SlackMemoryOperationOutput] = Field(default_factory=list)


SLACK_EXTRACTION_PROMPT = """You are the memory layer for a workplace AI agent that has casual, ongoing conversations with coworkers — small talk, weekend plans, pet updates, work stress, preferences.

You will receive a conversation transcript from a Slack channel. It will contain messages from multiple users around their job and work, personal preferences, general small talk, team updates, office events, etc.
Your job is to extract person level or site level memories that would help the agent talk like a good colleague later.


Current date: {current_date}
Current time: {current_time}
Channel: #{channel_name}
Site: {site_code}
For site-scoped memory, use `scope_type="site"`; the system will set the site.
For person-scoped creates, leave `scope_id` empty and put the people the memory
is about in `target_users`. Prefer the exact `@username` visible in the
transcript. If no username is visible, use the exact person's name from the
message.

Existing relevant Slack site memories:
{candidate_memories}

## The core question

Before storing anything, ask: would a good colleague naturally remember this and
bring it up later?

If yes, store it. If it is filler, a one-off opinion, a random update, or a
message with no future conversational value, skip it.

## Attribution

Slack messages have an author and may mention other users. Attribute memory to
the person the message is about, not automatically to the author.

- If the author says something about themself, target the author's visible `@username`
- If the author congratulates, announces, or describes mentioned people, target
  the mentioned people's visible `@username`s
- If a message mentions several people and the same fact applies to each, return all relevant people in one operation's `target_users`.

Example: if person X posts "Happy 2 year Chewy-versary @person_y and @person_z",
create milestone/fact memory for person Y and person Z. Do not store that milestone on person X.

## What to store

Person-scoped memory (`scope_type="person"`) is for useful context about a
specific person:
- durable preferences, boundaries, pets, facts, or concise notes
- short-lived followups with `due_at` and `expires_at`
- work/life milestones when directly stated, such as a Chewy-versary

Write person memory as prompt-ready context for the future agent, not as a Slack event recap. Do not say "posted", "said in Slack", "was congratulated by", or "Person X announced". Store the useful fact itself. 

For example
1. scope_type="person", target_users=["@person_x"],
"Working on project so-so" 
How it can be useful: While talking to person X, agent can bring up the project and ask how it's going, what are the challenges, etc.

2. scope_type="site"
"Person Y printed 3D printed name tags for team. Can be grabbed from Person Y's desk."
How it can be useful: While talking to anyone from BOS3, agent can bring up the name tags and ask how they like it, where to get them from etc.

3. scope_type="person", target_users=["@person_x", "@person_y"],
"milestone: completed 2 years at Chewy on 2026-06-01."
The above might have been authored by Person Z, but the memory is about the milestone completed by person X and person Y.

Bad person summaries:
- `Person X congratulated Person Y on Slack.`
- `Person Y was mentioned in a Slack thread.`

Site-scoped memory (`scope_type="site"`) is for context relevant to people at
the site:
- events, freebies, office changes, demos, outages, room closures, reminders
- use `kind="office_event"`
- every site item should usually have `expires_at`

## Time and expiry

You know today's date. The agent will not see this transcript — only the memory. So resolve all relative time references now.

Use short expiry windows unless Slack context clearly says otherwise:
- free food: same day, usually a few hours
- event today: after the event/day ends
- future event: after the event ends
- sick/WFH/personal short-lived context: due on the natural next check-in day,
  expires in a few days

## What not to store

- raw transcript snippets
- jokes or one-off chatter with no future value
- inferred personality traits
- org chart facts such as title, manager, team, cost center
- anything that would feel creepy to mention later

NOTE: Most of slack messages are going to be operational or logistic. SO it's not required to make a memory out of everything. You are supposed to act as a colleague and understand what they would naturally remember and bring up later.

Prefer updating an existing memory over creating duplicates.

Allowed person kinds: `preference` `boundary` `pet` `fact` `note` `followup`
Allowed site kind: `office_event`
Datetimes must be ISO 8601 with timezone offset.

Return exactly one JSON object. If nothing should change:
`{{"update": false, "ops": []}}`

Example person milestone from a mention:
```json
{{
  "update": true,
  "ops": [
    {{
      "op": "create",
      "scope_type": "person",
      "target_users": ["@person_x", "@person_y"],
      "kind": "fact",
      "key": "chewy_versary_2_year_2026_06_01",
      "summary": "milestone: completed 2 years at Chewy on 2026-06-01."
    }}
  ]
}}
```

Example site update:
```json
{{
  "update": true,
  "ops": [
    {{
      "op": "create",
      "scope_type": "site",
      "kind": "office_event",
      "key": "free_donuts_kitchen_2026_06_01",
      "summary": "2026-06-01: Free donuts are available in the third-floor kitchen until about 2 PM.",
      "expires_at": "2026-06-01T16:00:00-04:00"
    }}
  ]
}}
```

Slack window:
{conversation}
"""


def build_slack_extraction_prompt(
    *,
    window: SlackChannelWindow,
    current_date: str,
    current_time: str,
    candidate_memories: list[dict[str, Any]] | None = None,
) -> str:
    return SLACK_EXTRACTION_PROMPT.format(
        current_date=current_date,
        current_time=current_time,
        channel_name=window.channel_name,
        site_code=window.site_code or "unspecified",
        candidate_memories=json.dumps(candidate_memories or [], ensure_ascii=True),
        conversation=render_window_for_prompt(window),
    )


def _structured_response_to_payload(response: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = response
    raw = response
    parsing_error = None
    if isinstance(response, dict) and (
        "parsed" in response or "raw" in response or "parsing_error" in response
    ):
        parsed = response.get("parsed")
        raw = response.get("raw")
        parsing_error = response.get("parsing_error")
    if parsing_error is not None:
        raise ValueError(f"structured output parsing failed: {parsing_error}")
    usage_metadata = getattr(raw, "usage_metadata", {}) or {}
    if isinstance(parsed, BaseModel):
        return parsed.model_dump(mode="json"), usage_metadata
    if isinstance(parsed, dict):
        return parsed, usage_metadata
    if isinstance(response, BaseModel):
        return response.model_dump(mode="json"), usage_metadata
    return {}, usage_metadata


def _invoke_structured_llm(structured_llm: Any, prompt: str) -> Any:
    from langchain_core.messages import HumanMessage

    return structured_llm.invoke([HumanMessage(content=prompt)])


def _memory_payload(item: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "memory_id": item.memory_id,
        "scope_type": item.scope_type,
        "scope_id": item.scope_id,
        "kind": item.kind,
        "key": item.key,
        "summary": item.summary,
        "source": item.source,
    }
    if item.due_at:
        payload["due_at"] = item.due_at
    if item.expires_at:
        payload["expires_at"] = item.expires_at
    return payload


def candidate_memory_payload(
    memory_store: Any,
    *,
    site_code: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    if site_code:
        site_items = memory_store.list_active_items(
            scope_type="site",
            scope_id=site_code,
            kinds=("office_event",),
            source="slack",
            limit=limit,
        )
        payload.extend(_memory_payload(item) for item in site_items)
    return payload


def _filter_payload_by_channel_policy(
    data: dict[str, Any],
    *,
    site_code: str,
    person_memory_enabled: bool,
    site_memory_enabled: bool,
) -> dict[str, Any]:
    ops: list[dict[str, Any]] = []
    for raw in data.get("ops", []) or []:
        if not isinstance(raw, dict):
            continue
        op = dict(raw)
        scope_type = str(op.get("scope_type") or "").strip()
        if scope_type == "person" and not person_memory_enabled:
            continue
        if scope_type == "site" and not site_memory_enabled:
            continue
        if scope_type == "site":
            if not site_code:
                continue
            op["scope_id"] = site_code
        ops.append(op)
    filtered = dict(data)
    filtered["ops"] = ops
    filtered["update"] = bool(data.get("update")) and bool(ops)
    return filtered


class SlackMemoryExtractor:
    """Extract source-aware memory operations from normalized Slack windows."""

    def __init__(self, *, memory_store: Any) -> None:
        from argos_src.integrations.openai_models import get_llm_model_direct
        from argos_src.profile_config import FRAMEWORK_CONFIG_PATH

        self.memory_store = memory_store
        self.llm = get_llm_model_direct(
            SLACK_MEMORY_EXTRACTION_MODEL,
            vendor="openai",
            config_path=str(FRAMEWORK_CONFIG_PATH),
            temperature=0,
        )
        try:
            self.structured_llm = self.llm.with_structured_output(
                SlackExtractionOutput,
                method="json_schema",
                strict=True,
                include_raw=True,
            )
        except TypeError:
            self.structured_llm = self.llm.with_structured_output(
                SlackExtractionOutput,
                include_raw=True,
            )

    def extract_and_store_window(
        self,
        *,
        window: SlackChannelWindow,
        person_memory_enabled: bool,
        site_memory_enabled: bool,
        debug_llm_prompt: bool = False,
        debug_llm_output: bool = False,
    ) -> list[str]:
        if not (person_memory_enabled or site_memory_enabled):
            return []
        conversation = render_window_for_prompt(window)
        if not conversation.strip():
            return []
        candidate_memories = candidate_memory_payload(
            self.memory_store,
            site_code=window.site_code if site_memory_enabled else "",
        )
        now = datetime.now().astimezone()
        prompt = build_slack_extraction_prompt(
            window=window,
            current_date=now.strftime("%Y-%m-%d"),
            current_time=now.strftime("%H:%M %Z"),
            candidate_memories=candidate_memories,
        )
        if debug_llm_prompt:
            print(prompt, file=sys.stderr)
        response = _invoke_structured_llm(self.structured_llm, prompt)
        data, _usage_metadata = _structured_response_to_payload(response)
        if debug_llm_output:
            print(
                json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True),
                file=sys.stderr,
            )
        data = _filter_payload_by_channel_policy(
            data,
            site_code=window.site_code,
            person_memory_enabled=person_memory_enabled,
            site_memory_enabled=site_memory_enabled,
        )
        affected = write_slack_memory_operations(
            self.memory_store,
            operations=data,
            source_ref=window.source_ref,
            default_site_code=window.site_code,
            slack_user_profiles={
                profile.slack_user_id: profile
                for profile in window.user_profiles
                if profile.slack_user_id
            },
        )
        logger.info(
            "[SlackMemory] extracted affected=%s channel=%s",
            len(affected),
            window.channel_name,
        )
        return affected
