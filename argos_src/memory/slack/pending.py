"""Pending Slack memory for users not yet linked to Argos people."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from argos_src.memory.models import VALID_KINDS, normalize_key, parse_iso_datetime
from argos_src.memory.store import MemoryStore
from argos_src.memory.slack.models import SlackUserProfile


PENDING_STATUSES = {"pending", "promoted", "archived"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return str(value)


def _json_dumps(value: Any) -> str:
    safe = _json_safe(value)
    return json.dumps(safe if isinstance(safe, dict) else {}, ensure_ascii=True, sort_keys=True)


def _json_loads(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        loaded = json.loads(value)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _connect(memory_store: MemoryStore) -> sqlite3.Connection:
    db_path = str(getattr(memory_store, "db_path", "") or "").strip()
    if not db_path:
        raise ValueError("memory_store.db_path is required for Slack pending memory")
    connection = sqlite3.connect(db_path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def init_pending_slack_memory_schema(memory_store: MemoryStore) -> None:
    with _connect(memory_store) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS slack_pending_memory (
                pending_id TEXT PRIMARY KEY,
                slack_user_id TEXT NOT NULL,
                slack_display_name TEXT NOT NULL DEFAULT '',
                slack_real_name TEXT NOT NULL DEFAULT '',
                slack_email TEXT NOT NULL DEFAULT '',
                kind TEXT NOT NULL,
                key TEXT NOT NULL,
                summary TEXT NOT NULL,
                source_ref TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                due_at TEXT NOT NULL DEFAULT '',
                expires_at TEXT NOT NULL DEFAULT '',
                promoted_person_id TEXT NOT NULL DEFAULT '',
                promoted_at TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                UNIQUE (slack_user_id, kind, key)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_slack_pending_user_status
                ON slack_pending_memory(slack_user_id, status)
            """
        )


def upsert_pending_slack_memory(
    memory_store: MemoryStore,
    *,
    profile: SlackUserProfile,
    kind: str,
    key: str,
    summary: str,
    source_ref: str = "",
    observed_at: str = "",
    due_at: str = "",
    expires_at: str = "",
    metadata: dict[str, Any] | None = None,
) -> str:
    init_pending_slack_memory_schema(memory_store)
    rendered_kind = str(kind or "").strip()
    if rendered_kind not in VALID_KINDS:
        raise ValueError(f"invalid kind: {kind!r}")
    rendered_key = normalize_key(key)
    rendered_summary = str(summary or "").strip()
    slack_user_id = str(profile.slack_user_id or "").strip()
    if not slack_user_id:
        raise ValueError("slack_user_id is required")
    if not rendered_key:
        raise ValueError("key is required")
    if not rendered_summary:
        raise ValueError("summary is required")
    now = _utc_now_iso()
    observed = str(observed_at or "").strip()
    if observed and parse_iso_datetime(observed) is None:
        observed = ""
    due = str(due_at or "").strip()
    if due and parse_iso_datetime(due) is None:
        due = ""
    expires = str(expires_at or "").strip()
    if expires and parse_iso_datetime(expires) is None:
        expires = ""

    pending_id = f"slack_pending_{uuid4().hex}"
    with _connect(memory_store) as connection:
        row = connection.execute(
            """
            SELECT pending_id, created_at
            FROM slack_pending_memory
            WHERE slack_user_id = ? AND kind = ? AND key = ?
            """,
            (slack_user_id, rendered_kind, rendered_key),
        ).fetchone()
        if row is not None:
            pending_id = str(row["pending_id"])
            created_at = str(row["created_at"])
        else:
            created_at = now
        connection.execute(
            """
            INSERT INTO slack_pending_memory (
                pending_id, slack_user_id, slack_display_name, slack_real_name,
                slack_email, kind, key, summary, source_ref, status, created_at,
                observed_at, updated_at, due_at, expires_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slack_user_id, kind, key) DO UPDATE SET
                slack_display_name = excluded.slack_display_name,
                slack_real_name = excluded.slack_real_name,
                slack_email = excluded.slack_email,
                summary = excluded.summary,
                source_ref = excluded.source_ref,
                status = 'pending',
                observed_at = excluded.observed_at,
                updated_at = excluded.updated_at,
                due_at = excluded.due_at,
                expires_at = excluded.expires_at,
                metadata_json = excluded.metadata_json
            """,
            (
                pending_id,
                slack_user_id,
                str(profile.display_name or "").strip(),
                str(profile.real_name or "").strip(),
                str(profile.email or "").strip(),
                rendered_kind,
                rendered_key,
                rendered_summary,
                str(source_ref or "").strip(),
                created_at,
                observed or now,
                now,
                due,
                expires,
                _json_dumps(metadata or {}),
            ),
        )
    return pending_id


def list_pending_slack_memory(
    memory_store: MemoryStore,
    *,
    slack_user_id: str = "",
    status: str = "pending",
) -> list[dict[str, Any]]:
    init_pending_slack_memory_schema(memory_store)
    rendered_user = str(slack_user_id or "").strip()
    rendered_status = str(status or "").strip()
    if rendered_status and rendered_status not in PENDING_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    clauses: list[str] = []
    params: list[str] = []
    if rendered_user:
        clauses.append("slack_user_id = ?")
        params.append(rendered_user)
    if rendered_status:
        clauses.append("status = ?")
        params.append(rendered_status)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    with _connect(memory_store) as connection:
        rows = connection.execute(
            f"""
            SELECT *
            FROM slack_pending_memory
            {where}
            ORDER BY observed_at DESC, pending_id
            """,
            tuple(params),
        ).fetchall()
    return [
        {
            "pending_id": str(row["pending_id"]),
            "slack_user_id": str(row["slack_user_id"]),
            "slack_display_name": str(row["slack_display_name"] or ""),
            "slack_real_name": str(row["slack_real_name"] or ""),
            "slack_email": str(row["slack_email"] or ""),
            "kind": str(row["kind"]),
            "key": str(row["key"]),
            "summary": str(row["summary"]),
            "source_ref": str(row["source_ref"] or ""),
            "status": str(row["status"]),
            "created_at": str(row["created_at"]),
            "observed_at": str(row["observed_at"]),
            "updated_at": str(row["updated_at"]),
            "due_at": str(row["due_at"] or ""),
            "expires_at": str(row["expires_at"] or ""),
            "promoted_person_id": str(row["promoted_person_id"] or ""),
            "promoted_at": str(row["promoted_at"] or ""),
            "metadata": _json_loads(row["metadata_json"]),
        }
        for row in rows
    ]


def promote_pending_slack_memory(
    memory_store: MemoryStore,
    *,
    slack_user_id: str,
    person_id: str,
) -> list[str]:
    rendered_user = str(slack_user_id or "").strip()
    rendered_person = str(person_id or "").strip()
    if not rendered_user or not rendered_person:
        return []
    rows = list_pending_slack_memory(
        memory_store,
        slack_user_id=rendered_user,
        status="pending",
    )
    affected: list[str] = []
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        metadata["source_feed"] = "slack"
        metadata["promoted_from_slack_pending_id"] = row["pending_id"]
        metadata["slack_user_id"] = rendered_user
        if row.get("slack_email"):
            metadata["slack_email"] = row["slack_email"]
        memory_id = memory_store.upsert_item(
            scope_type="person",
            scope_id=rendered_person,
            kind=row["kind"],
            key=row["key"],
            summary=row["summary"],
            source="slack",
            source_ref=row["source_ref"],
            observed_at=row["observed_at"],
            due_at=row["due_at"],
            expires_at=row["expires_at"],
            metadata=metadata,
        )
        affected.append(memory_id)
        _mark_pending_promoted(
            memory_store,
            pending_id=row["pending_id"],
            person_id=rendered_person,
        )
    return affected


def promote_resolved_pending_slack_memory(
    memory_store: MemoryStore,
    *,
    identity_resolver: Any,
) -> list[str]:
    affected: list[str] = []
    for row in list_pending_slack_memory(memory_store, status="pending"):
        metadata = dict(row.get("metadata") or {})
        profile = SlackUserProfile(
            slack_user_id=str(row.get("slack_user_id") or "").strip(),
            username=str(metadata.get("slack_username") or "").strip(),
            display_name=(
                str(row.get("slack_display_name") or "").strip()
                or str(metadata.get("slack_user_label") or "").strip()
            ),
            real_name=str(row.get("slack_real_name") or "").strip(),
            email=str(row.get("slack_email") or metadata.get("slack_email") or "").strip(),
        )
        try:
            resolved = identity_resolver.resolve_user(profile)
        except Exception:
            continue
        if not resolved.person_id:
            continue
        affected.extend(
            promote_pending_slack_memory(
                memory_store,
                slack_user_id=profile.slack_user_id,
                person_id=resolved.person_id,
            )
        )
    return affected


def _mark_pending_promoted(
    memory_store: MemoryStore,
    *,
    pending_id: str,
    person_id: str,
) -> None:
    now = _utc_now_iso()
    with _connect(memory_store) as connection:
        connection.execute(
            """
            UPDATE slack_pending_memory
            SET status = 'promoted',
                promoted_person_id = ?,
                promoted_at = ?,
                updated_at = ?
            WHERE pending_id = ?
            """,
            (person_id, now, now, pending_id),
        )
