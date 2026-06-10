"""SQLite-backed identity store for Argos."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from argos_src.identity.constants import DEFAULT_IDENTITY_DB_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_alias(value: str) -> str:
    """Return a stable lookup key for human-entered names and aliases."""
    normalized = "".join(
        character.casefold() if character.isalnum() else " "
        for character in str(value or "")
    )
    return " ".join(normalized.split())


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def stringify(value: Any) -> str:
    return str(value).strip() if value is not None else ""


class IdentityStore:
    """Owns person identity, aliases, and directory metadata."""

    def __init__(self, db_path: str | Path = DEFAULT_IDENTITY_DB_PATH) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS people (
                    person_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    interaction_count INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS aliases (
                    person_id TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    normalized_alias TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'name',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (person_id, normalized_alias),
                    FOREIGN KEY (person_id) REFERENCES people(person_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_aliases_normalized_alias
                    ON aliases(normalized_alias);
                CREATE INDEX IF NOT EXISTS idx_people_name
                    ON people(name);
                """
            )
            self._assert_supported_people_schema(connection)

    @staticmethod
    def _assert_supported_people_schema(connection: sqlite3.Connection) -> None:
        expected = {
            "person_id",
            "name",
            "first_seen",
            "last_seen",
            "interaction_count",
            "metadata_json",
            "created_at",
            "updated_at",
        }
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(people)").fetchall()
        }
        if columns != expected:
            raise RuntimeError(
                "Unsupported identity database schema. Delete or recreate the identity "
                "SQLite database so Argos can create the current identity-only schema."
            )

    @staticmethod
    def make_person_id(name: str) -> str:
        slug = "_".join(part for part in normalize_alias(name).split() if part)
        slug = slug or "unknown"
        return f"person_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def create_person(
        self,
        *,
        name: str,
        person_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        cleaned_name = stringify(name) or "Unknown"
        rendered_person_id = stringify(person_id) or self.make_person_id(cleaned_name)
        now = _utc_now()
        meta = dict(metadata or {})
        first_seen = stringify(meta.get("first_seen")) or now
        last_seen = stringify(meta.get("last_seen")) or first_seen
        try:
            interaction_count = int(meta.get("interaction_count", 0) or 0)
        except Exception:
            interaction_count = 0
        extra_metadata = {
            key: value
            for key, value in meta.items()
            if key
            not in {
                "name",
                "first_seen",
                "last_seen",
                "interaction_count",
            }
        }

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO people (
                    person_id, name, first_seen, last_seen, interaction_count,
                    metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rendered_person_id,
                    cleaned_name,
                    first_seen,
                    last_seen,
                    interaction_count,
                    _json_dumps(extra_metadata),
                    now,
                    now,
                ),
            )
            self._upsert_aliases(
                connection,
                rendered_person_id,
                self._aliases_from_metadata(cleaned_name, extra_metadata),
            )
        return rendered_person_id

    def ensure_person(
        self,
        *,
        person_id: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        rendered_person_id = stringify(person_id)
        if not rendered_person_id:
            raise ValueError("person_id is required")
        if self.get_person(rendered_person_id) is not None:
            self._merge_aliases_from_metadata(rendered_person_id, name=name, metadata=metadata)
            return rendered_person_id

        return self.create_person(
            name=stringify(name) or stringify((metadata or {}).get("name")) or rendered_person_id,
            person_id=rendered_person_id,
            metadata=metadata,
        )

    def _aliases_from_metadata(
        self,
        name: str,
        metadata: dict[str, Any],
    ) -> list[tuple[str, str]]:
        aliases: list[tuple[str, str]] = []
        for alias, kind in (
            (name, "name"),
            (metadata.get("official_name"), "official_name"),
            (metadata.get("employee_name"), "employee_name"),
            (metadata.get("username"), "username"),
        ):
            rendered = stringify(alias)
            if rendered:
                aliases.append((rendered, kind))
        return aliases

    @staticmethod
    def _upsert_aliases(
        connection: sqlite3.Connection,
        person_id: str,
        aliases: list[tuple[str, str]],
    ) -> None:
        now = _utc_now()
        for alias, kind in aliases:
            normalized = normalize_alias(alias)
            if not normalized:
                continue
            connection.execute(
                """
                INSERT OR IGNORE INTO aliases (
                    person_id, alias, normalized_alias, kind, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (person_id, alias, normalized, kind or "alias", now),
            )

    def _merge_aliases_from_metadata(
        self,
        person_id: str,
        *,
        name: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        current = self.get_person(person_id)
        if current is None:
            return
        aliases = self._aliases_from_metadata(name, dict(metadata or {}))
        with self._connect() as connection:
            self._upsert_aliases(connection, person_id, aliases)

    def _row_to_metadata(self, row: sqlite3.Row) -> dict[str, Any]:
        try:
            extra = json.loads(row["metadata_json"] or "{}")
        except Exception:
            extra = {}
        metadata = dict(extra if isinstance(extra, dict) else {})
        metadata.update(
            {
                "name": row["name"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "interaction_count": int(row["interaction_count"] or 0),
            }
        )
        return metadata

    def get_person(self, person_id: str) -> dict[str, Any] | None:
        rendered = stringify(person_id)
        if not rendered:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM people WHERE person_id = ?",
                (rendered,),
            ).fetchone()
        if row is None:
            return None
        metadata = self._row_to_metadata(row)
        return {
            "person_id": row["person_id"],
            "name": row["name"] or "Unknown",
            "metadata": metadata,
        }

    def list_people(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM people ORDER BY name COLLATE NOCASE, person_id"
            ).fetchall()
        people: list[dict[str, Any]] = []
        for row in rows:
            metadata = self._row_to_metadata(row)
            people.append(
                {
                    "person_id": row["person_id"],
                    "name": row["name"] or "Unknown",
                    "metadata": metadata,
                }
            )
        return people

    def resolve_person_id(self, name_or_id: str) -> str | None:
        rendered = stringify(name_or_id)
        if not rendered:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT person_id FROM people WHERE person_id = ?",
                (rendered,),
            ).fetchone()
            if row is not None:
                return str(row["person_id"])
            normalized = normalize_alias(rendered)
            row = connection.execute(
                "SELECT person_id FROM aliases WHERE normalized_alias = ? ORDER BY kind LIMIT 1",
                (normalized,),
            ).fetchone()
            if row is not None:
                return str(row["person_id"])
            rows = connection.execute(
                """
                SELECT person_id
                FROM aliases
                WHERE normalized_alias LIKE ?
                ORDER BY length(normalized_alias), kind
                LIMIT 2
                """,
                (f"%{normalized}%",),
            ).fetchall()
        if len(rows) == 1:
            return str(rows[0]["person_id"])
        return None

    def list_aliases(self, person_id: str) -> list[dict[str, Any]]:
        rendered = stringify(person_id)
        if not rendered:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT alias, normalized_alias, kind, created_at
                FROM aliases
                WHERE person_id = ?
                ORDER BY kind, alias COLLATE NOCASE
                """,
                (rendered,),
            ).fetchall()
        return [
            {
                "alias": row["alias"],
                "normalized_alias": row["normalized_alias"],
                "kind": row["kind"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def delete_person(self, person_id: str) -> bool:
        rendered = stringify(person_id)
        if not rendered:
            return False
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM people WHERE person_id = ?",
                (rendered,),
            )
        return cursor.rowcount > 0

    def update_interaction(self, person_id: str) -> dict[str, Any] | None:
        rendered = stringify(person_id)
        if not rendered:
            return None
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE people
                SET last_seen = ?,
                    interaction_count = interaction_count + 1,
                    updated_at = ?
                WHERE person_id = ?
                """,
                (now, now, rendered),
            )
        person = self.get_person(rendered)
        return dict(person["metadata"]) if person is not None else None
