#!/usr/bin/env python3
from __future__ import annotations

"""Inspect Argos source-aware memory items."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from argos_src.identity.constants import DEFAULT_IDENTITY_DB_PATH
from argos_src.identity.prompting import format_identity_profile_lines
from argos_src.identity.store import IdentityStore
from argos_src.memory.context import MemoryContextCompiler
from argos_src.memory.constants import DEFAULT_MEMORY_DB_PATH
from argos_src.memory.store import MemoryStore


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _item_payload(item) -> dict:
    return {
        "memory_id": item.memory_id,
        "scope_type": item.scope_type,
        "scope_id": item.scope_id,
        "kind": item.kind,
        "key": item.key,
        "summary": item.summary,
        "source": item.source,
        "source_ref": item.source_ref,
        "status": item.status,
        "created_at": item.created_at,
        "observed_at": item.observed_at,
        "updated_at": item.updated_at,
        "due_at": item.due_at,
        "expires_at": item.expires_at,
        "metadata": item.metadata,
    }


def _resolve_person(identity_store: IdentityStore, name_or_id: str) -> str | None:
    rendered = str(name_or_id or "").strip()
    if not rendered:
        return None
    return identity_store.resolve_person_id(rendered) or rendered


def _print_items(items, *, include_all: bool) -> None:
    if not items:
        print("  (none)")
        return
    for item in items:
        print(f"- [{item.kind}] {item.summary}")
        print(f"    id={item.memory_id} source={item.source} status={item.status}")
        print(f"    key={item.key}")
        if item.due_at:
            print(f"    due_at={item.due_at}")
        if item.expires_at:
            print(f"    expires_at={item.expires_at}")
        if include_all and item.metadata:
            metadata = json.dumps(
                _json_ready(item.metadata),
                ensure_ascii=True,
                sort_keys=True,
            )
            print(f"    metadata={metadata}")


def _print_recent_site_encounters(memory_store: MemoryStore, site_code: str) -> None:
    encounters = memory_store.list_recent_encounters(
        site_code=site_code,
        since=datetime.now(timezone.utc) - timedelta(hours=2),
        limit=10,
    )
    print("\nRecent Encounters At Site")
    print("=========================")
    if not encounters:
        print("  (none)")
        return
    for item in encounters:
        name = str(item.metadata.get("name") or item.summary).strip()
        person_id = str(item.metadata.get("person_id") or item.scope_id).strip()
        print(f"- {name}")
        print(f"    person_id={person_id} id={item.memory_id}")
        print(f"    observed_at={item.observed_at}")
        if item.expires_at:
            print(f"    expires_at={item.expires_at}")


def show_prompt_context(
    *,
    person: str,
    site: str = "",
    memory_db_path: str = str(DEFAULT_MEMORY_DB_PATH),
    identity_db_path: str = str(DEFAULT_IDENTITY_DB_PATH),
    as_json: bool = False,
) -> int:
    if not str(person or "").strip():
        print("--prompt requires --person NAME_OR_ID.")
        return 2

    memory_store = MemoryStore(memory_db_path)
    identity_store = IdentityStore(identity_db_path)
    person_id = _resolve_person(identity_store, person) or person
    identity_record = identity_store.get_person(person_id)
    directory_lines = format_identity_profile_lines(
        dict((identity_record or {}).get("metadata") or {})
    )
    compiler = MemoryContextCompiler(memory_store, identity_store=identity_store)
    person_context = compiler.person_context(person_id)
    site_code = str(site or "").strip()
    site_blocks = (
        compiler.site_blocks(site_code, current_person_id=person_id)
        if site_code
        else ()
    )

    if as_json:
        print(
            json.dumps(
                {
                    "person_id": person_id,
                    "site_code": site_code,
                    "directory": list(directory_lines),
                    "about": list(person_context.profile_lines),
                    "potential_followups": list(person_context.followup_lines),
                    "memory_context_blocks": list(site_blocks),
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        return 0

    title = f"Prompt Person Context: {person_id}"
    print(f"\n{title}")
    print("=" * len(title))
    print("\nDirectory")
    print("---------")
    if directory_lines:
        for line in directory_lines:
            print(f"- {line}")
    else:
        print("  (none)")

    print("\nAbout")
    print("-----")
    if person_context.profile_lines:
        for line in person_context.profile_lines:
            print(f"- {line}")
    else:
        print("  (none)")

    print("\nPotential Followups")
    print("-------------------")
    if person_context.followup_lines:
        for line in person_context.followup_lines:
            print(f"- {line}")
    else:
        print("  (none)")

    if site_code:
        print("\nSite/Encounter Blocks")
        print("---------------------")
        if site_blocks:
            for block in site_blocks:
                print(block)
        else:
            print("  (none)")
    return 0


def show_memory(
    *,
    person: str = "",
    site: str = "",
    memory_db_path: str = str(DEFAULT_MEMORY_DB_PATH),
    identity_db_path: str = str(DEFAULT_IDENTITY_DB_PATH),
    include_all: bool = False,
    source: str = "",
    as_json: bool = False,
    prompt_context: bool = False,
) -> int:
    if prompt_context:
        return show_prompt_context(
            person=person,
            site=site,
            memory_db_path=memory_db_path,
            identity_db_path=identity_db_path,
            as_json=as_json,
        )
    if bool(person) == bool(site):
        print("Provide exactly one of --person or --site.")
        return 2

    memory_store = MemoryStore(memory_db_path)
    if person:
        identity_store = IdentityStore(identity_db_path)
        scope_type = "person"
        scope_id = _resolve_person(identity_store, person) or person
        title = f"Person Memory: {scope_id}"
    else:
        scope_type = "site"
        scope_id = str(site or "").strip()
        title = f"Site Memory: {scope_id}"

    rendered_source = str(source or "").strip()
    try:
        if include_all:
            items = memory_store.list_items(
                scope_type=scope_type,
                scope_id=scope_id,
                source=rendered_source,
            )
        else:
            items = memory_store.list_active_items(
                scope_type=scope_type,
                scope_id=scope_id,
                source=rendered_source,
                limit=100,
            )
    except ValueError as exc:
        print(str(exc))
        return 2

    if as_json:
        print(json.dumps([_item_payload(item) for item in items], ensure_ascii=True, indent=2))
        return 0

    print(f"\n{title}")
    print("=" * len(title))
    _print_items(items, include_all=include_all)
    if site:
        _print_recent_site_encounters(memory_store, scope_id)
    return 0


def archive_memory_item(
    *,
    memory_id: str,
    memory_db_path: str = str(DEFAULT_MEMORY_DB_PATH),
) -> int:
    rendered = str(memory_id or "").strip()
    if not rendered:
        print("memory_id is required for --archive.")
        return 2
    memory_store = MemoryStore(memory_db_path)
    archived = memory_store.archive_item(rendered)
    if archived:
        print(f"Archived memory item {rendered}.")
        return 0
    print(f"No memory item found for {rendered}.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Argos source-aware memory items",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python memory/manage_memory.py --person "Sakshee Patil"
  python memory/manage_memory.py --site BOS3
  python memory/manage_memory.py --person "Sakshee Patil" --source slack
  python memory/manage_memory.py --person "Sakshee Patil" --site BOS3 --prompt
  python memory/manage_memory.py --person person_sakshee_patil_20260513_150604 --all --json
  python memory/manage_memory.py --archive mem_abc123
        """,
    )
    parser.add_argument("--person", metavar="NAME_OR_ID", default="", help="Show person memory")
    parser.add_argument("--site", metavar="SITE_CODE", default="", help="Show site memory")
    parser.add_argument("--source", default="", help="Filter memory by source, e.g. slack")
    parser.add_argument("--archive", metavar="MEMORY_ID", default="", help="Archive one memory item")
    parser.add_argument("--all", action="store_true", help="Include archived and expired items")
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Show prompt-ready identity, memory, and optional site context",
    )
    parser.add_argument("--json", action="store_true", help="Print memory items as JSON")
    parser.add_argument(
        "--memory-db-path",
        default=str(DEFAULT_MEMORY_DB_PATH),
        help="Path to the memory SQLite database",
    )
    parser.add_argument(
        "--identity-db-path",
        default=str(DEFAULT_IDENTITY_DB_PATH),
        help="Path to the identity SQLite database used for name resolution",
    )
    args = parser.parse_args()
    if args.archive:
        return archive_memory_item(
            memory_id=args.archive,
            memory_db_path=args.memory_db_path,
        )
    return show_memory(
        person=args.person,
        site=args.site,
        memory_db_path=args.memory_db_path,
        identity_db_path=args.identity_db_path,
        include_all=args.all,
        source=args.source,
        as_json=args.json,
        prompt_context=args.prompt,
    )


if __name__ == "__main__":
    raise SystemExit(main())
