"""Persist Slack extraction operations into the source-aware MemoryStore."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from argos_src.memory.models import VALID_KINDS, normalize_key, parse_iso_datetime
from argos_src.memory.store import MemoryStore
from argos_src.memory.slack.models import SlackUserProfile
from argos_src.memory.slack.pending import upsert_pending_slack_memory


PERSON_MEMORY_KINDS = {"preference", "boundary", "pet", "fact", "note", "followup"}
SITE_MEMORY_KINDS = {"office_event"}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _operation_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, BaseModel):
        return raw.model_dump(mode="json")
    return dict(raw) if isinstance(raw, dict) else {}


def _valid_datetime_or_empty(value: Any) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        return ""
    return rendered if parse_iso_datetime(rendered) is not None else ""


def _fallback_key(*, scope_type: str, kind: str, summary: str) -> str:
    prefix = "site" if scope_type == "site" else kind
    return f"{prefix}_{normalize_key(summary)[:60]}"


def _metadata_from_operation(
    operation: dict[str, Any],
    *,
    existing_metadata: dict[str, Any] | None = None,
    target_profile: SlackUserProfile | None = None,
) -> dict[str, Any]:
    metadata = dict(existing_metadata or {})
    source_refs = [
        str(ref).strip()
        for ref in operation.get("source_refs", []) or []
        if str(ref or "").strip()
    ]
    if source_refs:
        metadata["slack_source_refs"] = source_refs[:10]
    confidence = operation.get("confidence")
    if confidence not in (None, ""):
        metadata["confidence"] = _json_safe_value(confidence)
    if target_profile is not None:
        metadata["slack_user_id"] = target_profile.slack_user_id
        if target_profile.username:
            metadata["slack_username"] = target_profile.username
        if target_profile.email:
            metadata["slack_email"] = target_profile.email
        if target_profile.label:
            metadata["slack_user_label"] = target_profile.label
    metadata["last_memory_op"] = str(operation.get("op") or "").strip()
    metadata["source_feed"] = "slack"
    return metadata


def _source_ref_for_operation(operation: dict[str, Any], fallback: str) -> str:
    for ref in operation.get("source_refs", []) or []:
        rendered = str(ref or "").strip()
        if rendered:
            return rendered
    return fallback


def _operation_is_allowed(*, scope_type: str, kind: str, expires_at: str) -> bool:
    if kind not in VALID_KINDS:
        return False
    if scope_type == "person":
        if kind not in PERSON_MEMORY_KINDS:
            return False
        return kind != "followup" or bool(expires_at)
    if scope_type == "site":
        return kind in SITE_MEMORY_KINDS
    return False


def _target_users(operation: dict[str, Any]) -> tuple[str, ...]:
    raw_targets = operation.get("target_users", []) or []
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    targets: list[str] = []
    seen: set[str] = set()
    for raw in raw_targets:
        rendered = str(raw or "").strip()
        if not rendered or rendered in seen:
            continue
        seen.add(rendered)
        targets.append(rendered)
    return tuple(targets)


def _normalize_target(value: Any) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        return ""
    if rendered.startswith("@"):
        rendered = rendered[1:].strip()
    return rendered.casefold()


def _profile_lookup(
    slack_user_profiles: dict[str, SlackUserProfile],
) -> dict[str, SlackUserProfile | None]:
    lookup: dict[str, SlackUserProfile | None] = {}

    def add(key: str, profile: SlackUserProfile) -> None:
        normalized = _normalize_target(key)
        if not normalized:
            return
        existing = lookup.get(normalized)
        if existing is not None and existing != profile:
            lookup[normalized] = None
            return
        if normalized not in lookup:
            lookup[normalized] = profile

    for profile in slack_user_profiles.values():
        add(profile.slack_user_id, profile)
        add(profile.username, profile)
        add(profile.handle, profile)
        add(profile.label, profile)
        add(profile.prompt_label, profile)
        add(profile.display_name, profile)
        add(profile.real_name, profile)
        add(profile.email, profile)
        if profile.email:
            add(profile.email.split("@", 1)[0], profile)
    return lookup


def _profile_for_target(
    target_user: str,
    profile_lookup: dict[str, SlackUserProfile | None],
) -> SlackUserProfile:
    rendered = str(target_user or "").strip()
    normalized = _normalize_target(rendered)
    profile = profile_lookup.get(normalized)
    if profile is not None:
        return profile
    username = rendered[1:].strip() if rendered.startswith("@") else ""
    return SlackUserProfile(
        slack_user_id=username or rendered,
        username=username,
        display_name=rendered,
    )


def _write_person_create(
    memory_store: MemoryStore,
    *,
    operation: dict[str, Any],
    source_ref: str,
    slack_user_profiles: dict[str, SlackUserProfile],
) -> list[str]:
    kind = str(operation.get("kind") or "").strip()
    summary = str(operation.get("summary") or "").strip()
    if not kind or not summary:
        return []
    due_at = _valid_datetime_or_empty(operation.get("due_at"))
    expires_at = _valid_datetime_or_empty(operation.get("expires_at"))
    if not _operation_is_allowed(scope_type="person", kind=kind, expires_at=expires_at):
        return []
    base_key = normalize_key(operation.get("key") or "") or _fallback_key(
        scope_type="person",
        kind=kind,
        summary=summary,
    )
    if not base_key:
        return []

    affected: list[str] = []
    targets = _target_users(operation)
    if targets:
        profile_lookup = _profile_lookup(slack_user_profiles)
        for target_user in targets:
            profile = _profile_for_target(target_user, profile_lookup)
            if profile.person_id:
                affected.append(
                    memory_store.upsert_item(
                        scope_type="person",
                        scope_id=profile.person_id,
                        kind=kind,
                        key=base_key,
                        summary=summary,
                        source="slack",
                        source_ref=_source_ref_for_operation(operation, source_ref),
                        observed_at=(
                            _valid_datetime_or_empty(operation.get("observed_at"))
                            or datetime.now().astimezone().replace(microsecond=0).isoformat()
                        ),
                        due_at=due_at,
                        expires_at=expires_at,
                        metadata=_metadata_from_operation(
                            operation,
                            target_profile=profile,
                        ),
                    )
                )
                continue
            affected.append(
                upsert_pending_slack_memory(
                    memory_store,
                    profile=profile,
                    kind=kind,
                    key=base_key,
                    summary=summary,
                    source_ref=_source_ref_for_operation(operation, source_ref),
                    observed_at=(
                        _valid_datetime_or_empty(operation.get("observed_at"))
                        or datetime.now().astimezone().replace(microsecond=0).isoformat()
                    ),
                    due_at=due_at,
                    expires_at=expires_at,
                    metadata=_metadata_from_operation(
                        operation,
                        target_profile=profile,
                    ),
                )
            )
        return affected

    scope_id = str(operation.get("scope_id") or "").strip()
    if not scope_id:
        return []
    affected.append(
        memory_store.upsert_item(
            scope_type="person",
            scope_id=scope_id,
            kind=kind,
            key=base_key,
            summary=summary,
            source="slack",
            source_ref=_source_ref_for_operation(operation, source_ref),
            observed_at=(
                _valid_datetime_or_empty(operation.get("observed_at"))
                or datetime.now().astimezone().replace(microsecond=0).isoformat()
            ),
            due_at=due_at,
            expires_at=expires_at,
            metadata=_metadata_from_operation(operation),
        )
    )
    return affected


def write_slack_memory_operations(
    memory_store: MemoryStore,
    *,
    operations: Any,
    source_ref: str = "",
    default_site_code: str = "",
    slack_user_profiles: dict[str, SlackUserProfile] | None = None,
) -> list[str]:
    if isinstance(operations, BaseModel):
        operations = operations.model_dump(mode="json")
    if isinstance(operations, dict):
        raw_ops = operations.get("ops", []) or []
        if not (bool(operations.get("update")) and raw_ops):
            return []
    else:
        raw_ops = operations or []

    affected: list[str] = []
    for raw in raw_ops:
        operation = _operation_dict(raw)
        op = str(operation.get("op") or "").strip().casefold()
        if op in {"", "noop"}:
            continue
        scope_type = str(operation.get("scope_type") or "").strip()
        scope_id = str(operation.get("scope_id") or "").strip()
        if scope_type == "site":
            scope_id = str(default_site_code or scope_id or "").strip()
        memory_id = str(operation.get("memory_id") or "").strip()

        if op == "archive":
            item = memory_store.get_item(memory_id)
            if item is None:
                continue
            if item.source != "slack":
                continue
            if scope_type and item.scope_type != scope_type:
                continue
            if scope_id and item.scope_id != scope_id:
                continue
            if memory_store.archive_item(item.memory_id):
                affected.append(item.memory_id)
            continue

        if op == "update":
            item = memory_store.get_item(memory_id)
            if item is None:
                continue
            if item.source != "slack":
                continue
            if scope_type and item.scope_type != scope_type:
                continue
            if scope_id and item.scope_id != scope_id:
                continue
            summary = str(operation.get("summary") or "").strip()
            if not summary:
                continue
            due_at = _valid_datetime_or_empty(operation.get("due_at")) or item.due_at
            expires_at = (
                _valid_datetime_or_empty(operation.get("expires_at")) or item.expires_at
            )
            if not _operation_is_allowed(
                scope_type=item.scope_type,
                kind=item.kind,
                expires_at=expires_at,
            ):
                continue
            if memory_store.update_item(
                item.memory_id,
                summary=summary,
                source_ref=_source_ref_for_operation(operation, source_ref),
                status="active",
                observed_at=(
                    _valid_datetime_or_empty(operation.get("observed_at"))
                    or datetime.now().astimezone().replace(microsecond=0).isoformat()
                ),
                due_at=due_at,
                expires_at=expires_at,
                metadata=_metadata_from_operation(
                    operation,
                    existing_metadata=item.metadata,
                ),
            ):
                affected.append(item.memory_id)
            continue

        if op != "create":
            continue
        if scope_type == "person":
            affected.extend(
                _write_person_create(
                    memory_store,
                    operation=operation,
                    source_ref=source_ref,
                    slack_user_profiles=slack_user_profiles or {},
                )
            )
            continue
        if scope_type != "site" or not scope_id:
            continue
        kind = str(operation.get("kind") or "").strip()
        summary = str(operation.get("summary") or "").strip()
        if not kind or not summary:
            continue
        due_at = _valid_datetime_or_empty(operation.get("due_at"))
        expires_at = _valid_datetime_or_empty(operation.get("expires_at"))
        if not _operation_is_allowed(scope_type=scope_type, kind=kind, expires_at=expires_at):
            continue
        key = normalize_key(operation.get("key") or "") or _fallback_key(
            scope_type=scope_type,
            kind=kind,
            summary=summary,
        )
        if not key:
            continue
        affected.append(
            memory_store.upsert_item(
                scope_type=scope_type,
                scope_id=scope_id,
                kind=kind,
                key=key,
                summary=summary,
                source="slack",
                source_ref=_source_ref_for_operation(operation, source_ref),
                observed_at=(
                    _valid_datetime_or_empty(operation.get("observed_at"))
                    or datetime.now().astimezone().replace(microsecond=0).isoformat()
                ),
                due_at=due_at,
                expires_at=expires_at,
                metadata=_metadata_from_operation(operation),
            )
        )
    return affected
