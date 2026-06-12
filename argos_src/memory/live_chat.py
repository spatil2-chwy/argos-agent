"""
Live-chat memory extraction for Argos.

This module owns the source-aware memory write path for speaker-owned realtime
chat segments.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
from typing import Any, Literal
import warnings

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - pydantic v1 compatibility
    ConfigDict = None

from argos_src.profile_config import FRAMEWORK_CONFIG_PATH

from argos_src.agent.preference_types import PreferenceSegment
from argos_src.memory.models import MemoryItem, normalize_key, parse_iso_datetime
from argos_src.memory.store import MemoryStore


logger = logging.getLogger(__name__)
PREFERENCE_EXTRACTION_MODEL = "gpt-4.1"
CANDIDATE_MEMORY_LIMIT = 12

MEMORY_KIND_SPECS: dict[str, dict[str, Any]] = {
    "preference": {
        "profile_fields": ("preferred_name", "preferred_language", "nickname_for_robot"),
    },
    "boundary": {},
    "pet": {"pin_candidate": True},
    "fact": {
        "profile_fields": ("birthday",),
    },
    "note": {},
    "followup": {"pin_candidate": True, "requires_expires_at": True},
}
LIVE_CHAT_MEMORY_KINDS = tuple(MEMORY_KIND_SPECS)
PROFILE_MEMORY_FIELD_KINDS = {
    field: kind
    for kind, spec in MEMORY_KIND_SPECS.items()
    for field in spec.get("profile_fields", ())
}
PINNED_MEMORY_KEYS = frozenset(PROFILE_MEMORY_FIELD_KINDS)
PINNED_MEMORY_KINDS = frozenset(
    kind for kind, spec in MEMORY_KIND_SPECS.items() if spec.get("pin_candidate")
)


class _StrictBaseModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover - pydantic v1 compatibility
        class Config:
            extra = "forbid"


class MemoryOperationValue(_StrictBaseModel):
    field: str = ""
    value: str = ""
    name: str = ""
    kind: str = ""
    age: str = ""
    notes: str = ""


class MemoryOperationOutput(_StrictBaseModel):
    op: Literal["create", "update", "archive", "noop"]
    memory_id: str = ""
    kind: str = ""
    key: str = ""
    summary: str = ""
    value: MemoryOperationValue = Field(default_factory=MemoryOperationValue)
    due_at: str = ""
    expires_at: str = ""


class PreferenceExtractionOutput(_StrictBaseModel):
    update: bool = False
    ops: list[MemoryOperationOutput] = Field(default_factory=list)


EXTRACTION_PROMPT = """You are the memory layer for a workplace AI agent that has casual, ongoing conversations with coworkers — small talk, weekend plans, pet updates, work stress, preferences. Your job: read a conversation transcript and extract only what would help the agent talk like a good colleague later.

Current date: {current_date}
Current time: {current_time}

Existing relevant memories:
{candidate_memories}

---

## The core question

Before storing anything, ask: would a good colleague naturally remember this and bring it up later?

If yes — store it. If it's filler, a one-off opinion, or something that would feel odd to reference weeks from now — skip it.

---

## Two categories

**Durable** — stays true and useful over time. Examples: preferred name, pet names, life facts, dislikes, what they're working on. Shown in the agent's background context about the person.

**Short-lived** — only interesting for a brief window, with a natural follow-up moment. Examples: a weekend trip, a visiting friend, a deadline, a medical appointment. Stored as followup context with `due_at` and `expires_at`; the realtime agent decides whether and how to mention it.

Never store a short-lived thing as a durable note "just in case."

---

## Resolving time

You know today's date. The agent will not see this transcript — only the memory. So resolve all relative time references now.

If someone mentions a weekend plan: figure out when that weekend is, set `due_at` for after it ends, and write the summary as self-contained context — not a suggested sentence for the agent to say.

| What they said | What to store |
|---|---|
| "going to Cape Cod this weekend" | `"Cape Cod trip with their parents planned for the weekend of 2026-05-16."` |
| "parents visiting next week" | `"Parents are visiting during the week of 2026-05-18."` |
| "big presentation on Thursday" | `"Has a presentation on 2026-05-21."` |

`due_at` = when the memory first becomes eligible to use. `expires_at` = when it is too stale to show.

---

## What not to store

- Inferred traits, personality, or communication style
- One-off opinions or reactions
- Work metadata (team, title, org, manager)
- Anything that would feel creepy or clinical to reference later
- A followup with no natural check-in moment

---

## Preferred name

Only store if the person explicitly says how they want to be addressed. Never infer from spelling or how the agent addressed them.

---

## Existing memories

Prefer updating an existing memory over creating a duplicate.

For followup memories, `expires_at` is the latest date when the followup may still be useful. It is not a signal to keep asking until that date.

Treat followups as pending threads. When the conversation shows a followup has been acted on, do not leave that same followup pending:
- archive it if the moment has passed, the agent asked about it and the user answered, or the user reports how it went;
- update it only if there is still a future check-in to make, such as a postponed event or an unresolved situation;
- create or update a durable memory separately if the answer reveals something stable and useful long term.

Before returning, inspect every existing `followup` memory in the candidate list:
- If the conversation answers or resolves that pending thread, include an `archive` op for that exact `memory_id`.
- If the same answer also reveals a durable preference/fact/pet/note, include both operations in the same `ops` array: archive the followup, then create or update the durable memory.
- If the followup was not asked about, leave it unchanged.
- If the followup is unrelated to this conversation, leave it unchanged.

Do not leave an already-used followup in the same pending state unless the user explicitly asks to be reminded again. Do not rely on `expires_at` to clean this up later.

---

## Output

Return one JSON object with exactly this top-level shape: `"update"` plus `"ops"`.
If nothing should change: `"update": false, "ops": []`.
If anything should change: `"update": true`, and put every operation in the same `"ops"` list.
The examples below are complete responses, not standalone operation fragments.

**Create durable memory:**
```json
{{
  "update": true,
  "ops": [
    {{
      "op": "create",
      "kind": "pet",
      "key": "pet_luna",
      "summary": "pet: Luna (dog): had a benign lump removed in May 2026 and recovered well",
      "value": {{ "name": "Luna", "kind": "dog", "notes": "had a benign lump removed in May 2026 and recovered well" }}
    }}
  ]
}}
```

**Create followup:**
```json
{{
  "update": true,
  "ops": [
    {{
      "op": "create",
      "kind": "followup",
      "key": "cape_cod_trip",
      "summary": "Cape Cod trip with their parents planned for the weekend of 2026-05-16.",
      "due_at": "2026-05-18T09:00:00-04:00",
      "expires_at": "2026-05-22T23:59:00-04:00"
    }}
  ]
}}
```

**Update existing:**
```json
{{
  "update": true,
  "ops": [
    {{
      "op": "update",
      "memory_id": "mem_123",
      "summary": "Luna's surgery was postponed to 2026-05-24.",
      "due_at": "2026-05-25T09:00:00-04:00",
      "expires_at": "2026-05-31T23:59:00-04:00"
    }}
  ]
}}
```

**Archive resolved followup:**
```json
{{
  "update": true,
  "ops": [
    {{
      "op": "archive",
      "memory_id": "mem_456"
    }}
  ]
}}
```

**Archive resolved followup. Like the assistant may have followed up on the Cape Cod weekend trip and in that conversation, user couldve mentioned lunch at Crisp Pizza which turned out be a favorite. So since agent already checked in on the followup with no new useful followup, it can be archived, and the pizza place can be stored as a preference:**
```json
{{
  "update": true,
  "ops": [
    {{
      "op": "archive",
      "memory_id": "mem_456"
    }},
    {{
      "op": "create",
      "kind": "preference",
      "key": "fav_pizza_place",
      "summary": "One of favorite pizza places was Crisp Pizza in Cape Cod."
    }}
  ]
}}
```

Allowed kinds: `preference` `boundary` `pet` `fact` `note` `followup`

The realtime agent decides whether and how to mention a stored memory. The memory layer stores context only.
Datetimes in ISO 8601 with timezone offset.

---

Conversation:
{conversation}
"""


IDENTITY_OWNED_NOTE_PREFIXES = (
    "team:",
    "title:",
    "business title:",
    "tenure:",
    "manager:",
    "manager name:",
    "cost center:",
    "business function:",
    "leadership org:",
    "senior leadership team:",
    "job family:",
    "job level:",
    "c level:",
)


def _segment_to_conversation(segment: PreferenceSegment) -> str:
    """Format a buffered speaker-owned segment into a plain conversation transcript."""
    lines: list[str] = []
    for turn in segment.turns:
        if turn.person_id != segment.person_id:
            continue
        user_text = str(turn.user_text or "").strip()
        assistant_text = str(turn.assistant_text or "").strip()
        if user_text:
            lines.append(f"User: {user_text}")
        if assistant_text:
            lines.append(f"Assistant: {assistant_text}")
    return "\n".join(lines)


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
    """Invoke structured output while suppressing a known ParsedChatCompletion warning."""
    from langchain_core.messages import HumanMessage

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Pydantic serializer warnings:.*",
            category=UserWarning,
        )
        return structured_llm.invoke([HumanMessage(content=prompt)])


def _profile_field_for_operation(kind: str, key: str, value: dict[str, Any]) -> str:
    field = normalize_key(value.get("field") or "")
    if field in PROFILE_MEMORY_FIELD_KINDS:
        return field
    if key in PROFILE_MEMORY_FIELD_KINDS:
        return key
    return ""


def _operation_violates_kind_contract(
    *,
    op: str,
    kind: str,
    key: str,
    summary: str,
    value: dict[str, Any],
    due_at: str,
    expires_at: str,
) -> bool:
    del op
    field = _profile_field_for_operation(kind, key, value)
    expected_kind = PROFILE_MEMORY_FIELD_KINDS.get(field)
    if field and expected_kind and kind and kind != expected_kind:
        logger.info(
            "[PrefExtract] rejected profile update with wrong kind field=%s key=%s kind=%s",
            field,
            key,
            kind,
        )
        return True

    if due_at and parse_iso_datetime(due_at) is None:
        logger.info("[PrefExtract] rejected memory with invalid due_at key=%s", key)
        return True
    if expires_at and parse_iso_datetime(expires_at) is None:
        logger.info("[PrefExtract] rejected memory with invalid expires_at key=%s", key)
        return True

    kind_spec = MEMORY_KIND_SPECS.get(kind, {})
    if kind == "followup":
        if kind_spec.get("requires_expires_at") and not expires_at:
            logger.info("[PrefExtract] rejected followup without expires_at key=%s", key)
            return True

    return False


def _parse_operations_payload(
    data: Any,
    *,
    fallback_turn_ids: list[str],
    conversation: str = "",
) -> tuple[bool, dict[str, list[dict[str, Any]]]]:
    del fallback_turn_ids, conversation
    if not isinstance(data, dict):
        return False, {"ops": []}

    ops: list[dict[str, Any]] = []
    for raw in data.get("ops", []) or []:
        if not isinstance(raw, dict):
            continue
        op = str(raw.get("op") or "").strip().casefold()
        if op not in {"create", "update", "archive", "noop"}:
            continue
        if op == "noop":
            continue

        memory_id = str(raw.get("memory_id") or "").strip()
        kind = str(raw.get("kind") or "").strip()
        key = normalize_key(raw.get("key") or "")
        summary = str(raw.get("summary") or "").strip()
        value = _compact_memory_value(raw.get("value"))
        now = datetime.now().astimezone().replace(microsecond=0).isoformat()
        due_at = str(raw.get("due_at") or "").strip()
        expires_at = str(raw.get("expires_at") or "").strip()

        if op == "create":
            if kind not in LIVE_CHAT_MEMORY_KINDS or not summary:
                continue
            key = key or _fallback_memory_key(kind=kind, summary=summary, value=value)
            if not key:
                continue
        elif op == "update":
            if not memory_id or not summary:
                continue
        elif op == "archive" and not memory_id:
            continue

        if op in {"create", "update"} and _operation_violates_kind_contract(
            op=op,
            kind=kind,
            key=key,
            summary=summary,
            value=value,
            due_at=due_at,
            expires_at=expires_at,
        ):
            continue

        ops.append(
            {
                "op": op,
                "memory_id": memory_id,
                "kind": kind,
                "key": key,
                "summary": summary,
                "value": value,
                "due_at": due_at,
                "expires_at": expires_at,
                "updated_at": now,
            }
        )

    should_update = bool(data.get("update")) and bool(ops)
    return should_update, {"ops": ops}


def _memory_key(field: str, value: Any) -> str:
    return f"{field}_{normalize_key(value)}"


def _fallback_memory_key(*, kind: str, summary: str, value: dict[str, Any]) -> str:
    if kind == "pet":
        key = _pet_memory_key(value, summary)
        if key:
            return key
    for profile_key in PINNED_MEMORY_KEYS:
        if profile_key in normalize_key(summary):
            return profile_key
    return f"{kind}_{normalize_key(summary)[:60]}"


def _stringify(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def normalize_pet_entry(entry: Any) -> dict[str, str] | None:
    if isinstance(entry, str):
        name = _stringify(entry)
        return {"name": name, "kind": "", "age": "", "notes": ""} if name else None
    if hasattr(entry, "model_dump"):
        try:
            entry = entry.model_dump(mode="json")
        except Exception:
            return None
    if not isinstance(entry, dict):
        return None
    name = _stringify(entry.get("name"))
    kind = _stringify(entry.get("kind") or entry.get("type") or entry.get("species"))
    age = _stringify(entry.get("age"))
    notes = _stringify(entry.get("notes"))
    if not any((name, kind, age, notes)):
        return None
    return {
        "name": name,
        "kind": kind,
        "age": age,
        "notes": notes,
    }


def summarize_pet(entry: dict[str, str]) -> str:
    name = _stringify(entry.get("name"))
    kind = _stringify(entry.get("kind"))
    age = _stringify(entry.get("age"))
    notes = _stringify(entry.get("notes"))
    if name:
        extras = [value for value in (kind, age) if value]
        base = f"{name} ({', '.join(extras)})" if extras else name
    else:
        base = ", ".join(value for value in (kind, age) if value)
    if notes:
        base = f"{base}: {notes}" if base else notes
    return base


def _json_safe_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _compact_memory_value(value: Any) -> dict[str, Any]:
    safe = _json_safe_value(value)
    if not isinstance(safe, dict):
        return {}
    compact: dict[str, Any] = {}
    for key, item in safe.items():
        if item is None:
            continue
        if isinstance(item, str):
            rendered = item.strip()
            if rendered:
                compact[str(key)] = rendered
            continue
        if item:
            compact[str(key)] = item
    return compact


def _pet_summary(value: Any) -> str:
    normalized = normalize_pet_entry(value)
    if normalized is None:
        return ""
    rendered = summarize_pet(normalized)
    return f"pet: {rendered}" if rendered else ""


def _pet_memory_key(value: Any, rendered: str) -> str:
    normalized = normalize_pet_entry(value)
    if normalized is not None:
        name = normalize_key(normalized.get("name"))
        if name:
            return f"pet_{name}"
    return _memory_key("pets", rendered)


def _should_skip_live_chat_note(*, field: str, rendered: str, summary: str) -> bool:
    text = str(rendered or summary or "").strip()
    lowered = text.casefold()
    if field == "notes" and lowered.startswith(IDENTITY_OWNED_NOTE_PREFIXES):
        return True
    return False


def _operation_metadata(
    update: dict[str, Any],
    *,
    existing: MemoryItem | None = None,
) -> dict[str, Any]:
    metadata = dict(existing.metadata) if existing is not None else {}
    value = _compact_memory_value(update.get("value"))
    if value:
        metadata["value"] = _json_safe_value(value)
    metadata["last_memory_op"] = str(update.get("op") or "").strip()
    return metadata


def write_live_chat_memory_items(
    memory_store: MemoryStore,
    *,
    person_id: str,
    operations: dict[str, list[dict[str, Any]]],
    source_ref: str,
) -> None:
    """Persist parsed live-chat operations into the source-aware MemoryStore."""
    rendered_person_id = str(person_id or "").strip()
    if not rendered_person_id:
        return

    for update in operations.get("ops", []) or []:
        op = str(update.get("op") or "").casefold()
        if op == "archive":
            item = memory_store.get_item(str(update.get("memory_id") or ""))
            if (
                item is not None
                and item.scope_type == "person"
                and item.scope_id == rendered_person_id
            ):
                memory_store.archive_item(item.memory_id)
            continue

        if op == "update":
            item = memory_store.get_item(str(update.get("memory_id") or ""))
            if (
                item is None
                or item.scope_type != "person"
                or item.scope_id != rendered_person_id
            ):
                continue
            summary = str(update.get("summary") or "").strip()
            if not summary:
                continue
            due_at = str(update.get("due_at") or item.due_at or "")
            expires_at = str(update.get("expires_at") or item.expires_at or "")
            if _operation_violates_kind_contract(
                op=op,
                kind=item.kind,
                key=item.key,
                summary=summary,
                value=_compact_memory_value(update.get("value")),
                due_at=due_at,
                expires_at=expires_at,
            ):
                continue
            if item.kind == "note" and _should_skip_live_chat_note(
                field="notes",
                rendered=summary,
                summary=summary,
            ):
                continue
            memory_store.update_item(
                item.memory_id,
                summary=summary,
                source_ref=source_ref,
                status="active",
                observed_at=str(update.get("updated_at") or ""),
                due_at=due_at,
                expires_at=expires_at,
                metadata=_operation_metadata(update, existing=item),
            )
            continue

        if op != "create":
            continue
        kind = str(update.get("kind") or "").strip()
        key = normalize_key(update.get("key") or "")
        summary = str(update.get("summary") or "").strip()
        value = _compact_memory_value(update.get("value"))
        if kind not in LIVE_CHAT_MEMORY_KINDS or not summary:
            continue
        if kind == "note" and _should_skip_live_chat_note(
            field="notes",
            rendered=summary,
            summary=summary,
        ):
            continue
        key = key or _fallback_memory_key(kind=kind, summary=summary, value=value)
        if not key:
            continue
        due_at = str(update.get("due_at") or "")
        expires_at = str(update.get("expires_at") or "")
        if _operation_violates_kind_contract(
            op=op,
            kind=kind,
            key=key,
            summary=summary,
            value=value,
            due_at=due_at,
            expires_at=expires_at,
        ):
            continue
        memory_store.upsert_item(
            scope_type="person",
            scope_id=rendered_person_id,
            kind=kind,
            key=key,
            summary=summary,
            source="live_chat",
            source_ref=source_ref,
            observed_at=str(update.get("updated_at") or ""),
            due_at=due_at,
            expires_at=expires_at,
            metadata=_operation_metadata(update),
        )


def _tokenize_memory_text(text: str) -> set[str]:
    rendered = str(text or "").casefold()
    normalized = "".join(char if char.isalnum() else " " for char in rendered)
    return {part for part in normalized.split() if len(part) >= 3}


def _memory_payload(item: MemoryItem) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "memory_id": item.memory_id,
        "kind": item.kind,
        "key": item.key,
        "summary": item.summary,
    }
    value = item.metadata.get("value")
    if isinstance(value, dict) and value:
        payload["value"] = value
    if item.kind == "followup":
        if item.due_at:
            payload["due_at"] = item.due_at
        if item.expires_at:
            payload["expires_at"] = item.expires_at
    return payload


def _candidate_pin_rank(item: MemoryItem) -> tuple[int, str]:
    if item.key in PINNED_MEMORY_KEYS:
        return (0, item.key)
    if item.kind == "pet":
        return (1, item.key)
    if item.kind == "followup":
        return (2, item.key)
    return (3, item.key)


def _candidate_memory_payload(
    memory_store: MemoryStore,
    person_id: str,
    conversation: str,
    *,
    limit: int = CANDIDATE_MEMORY_LIMIT,
) -> list[dict[str, Any]]:
    items = memory_store.list_active_items(
        scope_type="person",
        scope_id=person_id,
        kinds=LIVE_CHAT_MEMORY_KINDS,
        limit=100,
    )
    conversation_tokens = _tokenize_memory_text(conversation)
    pinned: list[MemoryItem] = []
    scored: list[tuple[int, str, MemoryItem]] = []

    for item in items:
        if item.key in PINNED_MEMORY_KEYS or item.kind in PINNED_MEMORY_KINDS:
            pinned.append(item)
            continue
        text = " ".join(
            [
                item.kind,
                item.key,
                item.summary,
                json.dumps(item.metadata.get("value", {}), ensure_ascii=True),
            ]
        )
        overlap = len(conversation_tokens & _tokenize_memory_text(text))
        if overlap <= 0:
            continue
        score = overlap * 10
        if item.kind in {"pet", "followup"}:
            score += 5
        scored.append((score, item.observed_at, item))

    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    pinned.sort(key=_candidate_pin_rank)
    selected: list[MemoryItem] = []
    seen_ids: set[str] = set()
    for item in [*pinned, *(row[2] for row in scored)]:
        if item.memory_id in seen_ids:
            continue
        selected.append(item)
        seen_ids.add(item.memory_id)
        if len(selected) >= max(1, limit):
            break
    return [_memory_payload(item) for item in selected]


class PreferenceExtractor:
    """Extracts meaningful social memory operations from conversation history."""

    def __init__(
        self,
        *,
        memory_store: MemoryStore,
    ):
        from argos_src.integrations.openai_models import get_llm_model_direct
        from argos_src.observability.observability import LatencyLogger, perf_now
        from argos_src.observability.pricing import estimate_text_generation_cost

        self._perf_now = perf_now
        self._estimate_text_generation_cost = estimate_text_generation_cost
        self.llm = get_llm_model_direct(
            PREFERENCE_EXTRACTION_MODEL,
            vendor="openai",
            config_path=str(FRAMEWORK_CONFIG_PATH),
            temperature=0,
        )
        try:
            self.structured_llm = self.llm.with_structured_output(
                PreferenceExtractionOutput,
                method="json_schema",
                strict=True,
                include_raw=True,
            )
        except TypeError:
            logger.warning(
                "[PrefExtract] structured output wrapper does not accept json_schema args; "
                "falling back to default structured output"
            )
            self.structured_llm = self.llm.with_structured_output(
                PreferenceExtractionOutput,
                include_raw=True,
            )
        self.memory_store = memory_store
        self.latency = LatencyLogger("pref_extract")

    def extract_and_store_segment(self, segment: PreferenceSegment) -> None:
        perf_now = self._perf_now
        estimate_text_generation_cost = self._estimate_text_generation_cost
        t_total = perf_now()
        person_id = str(segment.person_id or "").strip()
        if not person_id:
            logger.debug("[PrefExtract] empty person id, skipping")
            return
        try:
            conversation = _segment_to_conversation(segment)
            if not conversation.strip():
                logger.debug("[PrefExtract] empty conversation, skipping")
                return
            t0 = perf_now()
            candidate_memories = _candidate_memory_payload(
                self.memory_store,
                person_id,
                conversation,
            )
            self.latency.timing("db_read", perf_now() - t0, person_id=person_id)

            local_now = datetime.now().astimezone()
            prompt = EXTRACTION_PROMPT.format(
                current_date=local_now.strftime("%Y-%m-%d"),
                current_time=local_now.strftime("%H:%M %Z"),
                candidate_memories=json.dumps(candidate_memories, ensure_ascii=True),
                conversation=conversation,
            )
            t1 = perf_now()
            response = _invoke_structured_llm(self.structured_llm, prompt)
            self.latency.timing("llm_extract", perf_now() - t1, person_id=person_id)
            data, usage_metadata = _structured_response_to_payload(response)
            usage_fields = estimate_text_generation_cost(
                usage_metadata,
                model_name=PREFERENCE_EXTRACTION_MODEL,
            )
            if usage_fields.get("estimated_cost_usd") is not None:
                self.latency.emit(
                    event="llm_usage",
                    person_id=person_id,
                    model=PREFERENCE_EXTRACTION_MODEL,
                    input_tokens=usage_fields.get("input_tokens"),
                    output_tokens=usage_fields.get("output_tokens"),
                    total_tokens=usage_fields.get("total_tokens"),
                    cached_tokens=usage_fields.get("cached_tokens"),
                    uncached_input_tokens=usage_fields.get("uncached_input_tokens"),
                    estimated_cost_usd=usage_fields.get("estimated_cost_usd"),
                    estimated_cached_savings_usd=usage_fields.get(
                        "estimated_cached_savings_usd"
                    ),
                )
            should_update, operations = _parse_operations_payload(
                data,
                fallback_turn_ids=[turn.turn_id for turn in segment.turns],
                conversation=conversation,
            )

            if should_update:
                t2 = perf_now()
                write_live_chat_memory_items(
                    self.memory_store,
                    person_id=person_id,
                    operations=operations,
                    source_ref=segment.segment_id,
                )
                self.latency.timing("db_write", perf_now() - t2, person_id=person_id)
                logger.info("[PrefExtract] person=%s changed=True", person_id)
            else:
                logger.info("[PrefExtract] person=%s changed=False", person_id)
            self.latency.timing("total", perf_now() - t_total, person_id=person_id)
        except Exception as e:
            logger.exception("[PrefExtract] person=%s failed: %s", person_id, e)
