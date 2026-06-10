"""SQLite-backed source-aware memory store for Argos."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from argos_src.memory.constants import DEFAULT_MEMORY_DB_PATH
from argos_src.memory.models import (
    MemoryItem,
    MemoryKind,
    MemoryScope,
    MemorySource,
    MemoryStatus,
    VALID_KINDS,
    VALID_SCOPES,
    VALID_SOURCES,
    VALID_STATUSES,
    is_expired,
    normalize_key,
    parse_iso_datetime,
    require_valid,
    utc_now_iso,
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value if isinstance(value, dict) else {}, ensure_ascii=True, sort_keys=True)


def _json_loads(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        loaded = json.loads(value)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


class MemoryStore:
    """Owns source-aware person and site memory items."""

    def __init__(self, db_path: str | Path = DEFAULT_MEMORY_DB_PATH) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    memory_id TEXT PRIMARY KEY,
                    scope_type TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    key TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_ref TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    due_at TEXT NOT NULL DEFAULT '',
                    expires_at TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    UNIQUE (scope_type, scope_id, kind, key, source)
                );

                CREATE INDEX IF NOT EXISTS idx_memory_scope_status_kind
                    ON memory_items(scope_type, scope_id, status, kind);
                CREATE INDEX IF NOT EXISTS idx_memory_source_ref
                    ON memory_items(source, source_ref);
                CREATE INDEX IF NOT EXISTS idx_memory_kind_expires
                    ON memory_items(kind, expires_at);
                """
            )

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            memory_id=str(row["memory_id"]),
            scope_type=str(row["scope_type"]),  # type: ignore[arg-type]
            scope_id=str(row["scope_id"]),
            kind=str(row["kind"]),  # type: ignore[arg-type]
            key=str(row["key"]),
            summary=str(row["summary"]),
            source=str(row["source"]),  # type: ignore[arg-type]
            source_ref=str(row["source_ref"] or ""),
            status=str(row["status"]),  # type: ignore[arg-type]
            created_at=str(row["created_at"]),
            observed_at=str(row["observed_at"]),
            updated_at=str(row["updated_at"]),
            due_at=str(row["due_at"] or ""),
            expires_at=str(row["expires_at"] or ""),
            metadata=_json_loads(row["metadata_json"]),
        )

    def upsert_item(
        self,
        *,
        scope_type: MemoryScope | str,
        scope_id: str,
        kind: MemoryKind | str,
        key: str,
        summary: str,
        source: MemorySource | str,
        source_ref: str = "",
        status: MemoryStatus | str = "active",
        observed_at: str = "",
        due_at: str = "",
        expires_at: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        rendered_scope = require_valid(str(scope_type), VALID_SCOPES, "scope_type")
        rendered_kind = require_valid(str(kind), VALID_KINDS, "kind")
        rendered_source = require_valid(str(source), VALID_SOURCES, "source")
        rendered_status = require_valid(str(status), VALID_STATUSES, "status")
        rendered_scope_id = str(scope_id or "").strip()
        rendered_summary = str(summary or "").strip()
        rendered_key = normalize_key(key)
        if not rendered_scope_id:
            raise ValueError("scope_id is required")
        if not rendered_key:
            raise ValueError("key is required")
        if not rendered_summary:
            raise ValueError("summary is required")

        now = utc_now_iso()
        memory_id = f"mem_{uuid4().hex}"
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT memory_id, created_at
                FROM memory_items
                WHERE scope_type = ? AND scope_id = ? AND kind = ? AND key = ? AND source = ?
                """,
                (
                    rendered_scope,
                    rendered_scope_id,
                    rendered_kind,
                    rendered_key,
                    rendered_source,
                ),
            ).fetchone()
            if row is not None:
                memory_id = str(row["memory_id"])
                created_at = str(row["created_at"])
            else:
                created_at = now
            connection.execute(
                """
                INSERT INTO memory_items (
                    memory_id, scope_type, scope_id, kind, key, summary, source,
                    source_ref, status, created_at, observed_at, updated_at,
                    due_at, expires_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_type, scope_id, kind, key, source) DO UPDATE SET
                    summary = excluded.summary,
                    source_ref = excluded.source_ref,
                    status = excluded.status,
                    observed_at = excluded.observed_at,
                    updated_at = excluded.updated_at,
                    due_at = excluded.due_at,
                    expires_at = excluded.expires_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    memory_id,
                    rendered_scope,
                    rendered_scope_id,
                    rendered_kind,
                    rendered_key,
                    rendered_summary,
                    rendered_source,
                    str(source_ref or "").strip(),
                    rendered_status,
                    created_at,
                    str(observed_at or "").strip() or now,
                    now,
                    str(due_at or "").strip(),
                    str(expires_at or "").strip(),
                    _json_dumps(metadata or {}),
                ),
            )
        return memory_id

    def archive_item(self, memory_id: str) -> bool:
        rendered = str(memory_id or "").strip()
        if not rendered:
            return False
        now = utc_now_iso()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE memory_items
                SET status = 'archived', updated_at = ?
                WHERE memory_id = ?
                """,
                (now, rendered),
            )
        return cursor.rowcount > 0

    def get_item(self, memory_id: str) -> MemoryItem | None:
        rendered = str(memory_id or "").strip()
        if not rendered:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM memory_items
                WHERE memory_id = ?
                """,
                (rendered,),
            ).fetchone()
        return self._row_to_item(row) if row is not None else None

    def update_item(
        self,
        memory_id: str,
        *,
        summary: str,
        source_ref: str = "",
        status: MemoryStatus | str = "active",
        observed_at: str = "",
        due_at: str = "",
        expires_at: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        rendered = str(memory_id or "").strip()
        rendered_summary = str(summary or "").strip()
        rendered_status = require_valid(str(status), VALID_STATUSES, "status")
        if not rendered or not rendered_summary:
            return False
        now = utc_now_iso()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE memory_items
                SET summary = ?,
                    source_ref = ?,
                    status = ?,
                    observed_at = ?,
                    updated_at = ?,
                    due_at = ?,
                    expires_at = ?,
                    metadata_json = ?
                WHERE memory_id = ?
                """,
                (
                    rendered_summary,
                    str(source_ref or "").strip(),
                    rendered_status,
                    str(observed_at or "").strip() or now,
                    now,
                    str(due_at or "").strip(),
                    str(expires_at or "").strip(),
                    _json_dumps(metadata or {}),
                    rendered,
                ),
            )
        return cursor.rowcount > 0

    def list_active_items(
        self,
        *,
        scope_type: MemoryScope | str,
        scope_id: str,
        kinds: Iterable[MemoryKind | str] | None = None,
        source: MemorySource | str = "",
        now: datetime | None = None,
        limit: int = 20,
    ) -> list[MemoryItem]:
        rendered_scope = require_valid(str(scope_type), VALID_SCOPES, "scope_type")
        rendered_scope_id = str(scope_id or "").strip()
        if not rendered_scope_id:
            return []
        params: list[Any] = [rendered_scope, rendered_scope_id]
        kind_filter = ""
        rendered_kinds = [str(kind) for kind in (kinds or []) if str(kind or "").strip()]
        if rendered_kinds:
            for kind in rendered_kinds:
                require_valid(kind, VALID_KINDS, "kind")
            kind_filter = "AND kind IN ({})".format(",".join("?" for _ in rendered_kinds))
            params.extend(rendered_kinds)
        source_filter = ""
        rendered_source = str(source or "").strip()
        if rendered_source:
            require_valid(rendered_source, VALID_SOURCES, "source")
            source_filter = "AND source = ?"
            params.append(rendered_source)
        params.append(max(1, int(limit or 20)))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM memory_items
                WHERE scope_type = ?
                  AND scope_id = ?
                  AND status = 'active'
                  {kind_filter}
                  {source_filter}
                ORDER BY
                  CASE WHEN due_at = '' THEN 1 ELSE 0 END,
                  due_at,
                  observed_at DESC,
                  updated_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        ref_now = now or datetime.now(timezone.utc)
        return [
            item
            for item in (self._row_to_item(row) for row in rows)
            if not is_expired(item.expires_at, now=ref_now)
        ]

    def list_items(
        self,
        *,
        scope_type: MemoryScope | str,
        scope_id: str,
        kinds: Iterable[MemoryKind | str] | None = None,
        statuses: Iterable[MemoryStatus | str] | None = None,
        source: MemorySource | str = "",
        include_expired: bool = True,
        now: datetime | None = None,
        limit: int = 100,
    ) -> list[MemoryItem]:
        rendered_scope = require_valid(str(scope_type), VALID_SCOPES, "scope_type")
        rendered_scope_id = str(scope_id or "").strip()
        if not rendered_scope_id:
            return []
        params: list[Any] = [rendered_scope, rendered_scope_id]
        kind_filter = ""
        rendered_kinds = [str(kind) for kind in (kinds or []) if str(kind or "").strip()]
        if rendered_kinds:
            for kind in rendered_kinds:
                require_valid(kind, VALID_KINDS, "kind")
            kind_filter = "AND kind IN ({})".format(",".join("?" for _ in rendered_kinds))
            params.extend(rendered_kinds)
        status_filter = ""
        rendered_statuses = [
            str(status) for status in (statuses or []) if str(status or "").strip()
        ]
        if rendered_statuses:
            for status in rendered_statuses:
                require_valid(status, VALID_STATUSES, "status")
            status_filter = "AND status IN ({})".format(
                ",".join("?" for _ in rendered_statuses)
            )
            params.extend(rendered_statuses)
        source_filter = ""
        rendered_source = str(source or "").strip()
        if rendered_source:
            require_valid(rendered_source, VALID_SOURCES, "source")
            source_filter = "AND source = ?"
            params.append(rendered_source)
        params.append(max(1, int(limit or 100)))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM memory_items
                WHERE scope_type = ?
                  AND scope_id = ?
                  {kind_filter}
                  {status_filter}
                  {source_filter}
                ORDER BY observed_at DESC, updated_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        items = [self._row_to_item(row) for row in rows]
        if include_expired:
            return items
        ref_now = now or datetime.now(timezone.utc)
        return [item for item in items if not is_expired(item.expires_at, now=ref_now)]

    def record_encounter(
        self,
        *,
        person_id: str,
        name: str,
        site_code: str,
        metadata: dict[str, Any] | None = None,
        observed_at: str | None = None,
    ) -> str:
        rendered_person_id = str(person_id or "").strip()
        rendered_name = str(name or "").strip() or rendered_person_id
        rendered_site = str(site_code or "").strip()
        if not rendered_person_id:
            raise ValueError("person_id is required")
        observed = observed_at or utc_now_iso()
        observed_dt = parse_iso_datetime(observed) or datetime.now(timezone.utc)
        expires_at = (observed_dt.replace(microsecond=0) + timedelta(hours=2)).isoformat()
        return self.upsert_item(
            scope_type="person",
            scope_id=rendered_person_id,
            kind="encounter",
            key=f"latest_encounter_{rendered_person_id}",
            summary=f"Met {rendered_name}"
            + (f" at {rendered_site}." if rendered_site else "."),
            source="robot",
            status="active",
            observed_at=observed,
            expires_at=expires_at,
            metadata={
                "person_id": rendered_person_id,
                "name": rendered_name,
                "site_code": rendered_site,
                **dict(metadata or {}),
            },
        )

    def list_recent_encounters(
        self,
        *,
        site_code: str,
        since: datetime,
        exclude_person_id: str | None = None,
        limit: int = 5,
    ) -> list[MemoryItem]:
        site = str(site_code or "").strip()
        if not site:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM memory_items
                WHERE kind = 'encounter'
                  AND source = 'robot'
                  AND status = 'active'
                  AND observed_at >= ?
                ORDER BY observed_at DESC
                """,
                (since.isoformat(),),
            ).fetchall()
        now = datetime.now(timezone.utc)
        excluded = str(exclude_person_id or "").strip()
        max_items = max(1, int(limit or 5))
        items: list[MemoryItem] = []
        for item in (self._row_to_item(row) for row in rows):
            if is_expired(item.expires_at, now=now):
                continue
            if str(item.metadata.get("site_code") or "").strip() != site:
                continue
            if excluded and str(item.metadata.get("person_id") or item.scope_id) == excluded:
                continue
            items.append(item)
            if len(items) >= max_items:
                break
        return items
