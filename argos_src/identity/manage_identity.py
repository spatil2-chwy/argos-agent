#!/usr/bin/env python3
from __future__ import annotations

"""Manage Argos identities and linked modality embeddings."""

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from argos_src.face_recognition.constants import DEFAULT_FACE_DB_PATH
from argos_src.identity.constants import DEFAULT_IDENTITY_DB_PATH
from argos_src.speaker_recognition.constants import DEFAULT_SPEAKER_DB_PATH


_IDENTITY_METADATA_KEYS = {
    "name",
    "first_seen",
    "last_seen",
    "interaction_count",
}


def _humanize_person_id(person_id: str) -> str:
    rendered = str(person_id or "").strip()
    if not rendered:
        return ""
    match = re.match(r"^person_(.+?)_(\d{8})_(\d{6})$", rendered)
    slug = match.group(1) if match else rendered.removeprefix("person_")
    cleaned = " ".join(part for part in slug.replace("_", " ").split() if part)
    return cleaned.title() if cleaned else rendered


def _open_stores(
    *,
    identity_db_path: str,
    face_db_path: str,
    speaker_db_path: str,
):
    from argos_src.identity.embeddings.speaker_store import SpeakerEmbeddingStore
    from argos_src.face_recognition.store import FaceRecognitionStore
    from argos_src.identity import IdentityStore

    identity_store = IdentityStore(db_path=identity_db_path)
    face_db = FaceRecognitionStore(
        db_path=face_db_path,
        identity_store=identity_store,
    )
    speaker_db = SpeakerEmbeddingStore(db_path=speaker_db_path)
    return identity_store, face_db, speaker_db


def _display_name(identity_store, person_id: str) -> str:
    record = identity_store.get_person(person_id)
    if record is not None:
        return str(record.get("name") or person_id)
    return _humanize_person_id(person_id) or person_id


def _embedding_shape(record: dict | None) -> str | None:
    if not record:
        return None
    embedding = record.get("embedding")
    shape = getattr(embedding, "shape", None)
    return "x".join(str(part) for part in shape) if shape is not None else None


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"shape": list(shape)}
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return value


def _print_key_value_lines(
    title: str,
    values: dict,
    *,
    ordered_keys: tuple[str, ...] | None = None,
    indent: str = "  ",
) -> None:
    print(f"\n{title}:")
    keys = list(ordered_keys or ())
    keys.extend(sorted(key for key in values if key not in keys))
    printed = False
    for key in keys:
        value = values.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            print(f"{indent}{key}:")
            for item in value:
                if isinstance(item, dict):
                    print(f"{indent}  - {json.dumps(_json_ready(item), ensure_ascii=True, sort_keys=True)}")
                else:
                    print(f"{indent}  - {item}")
        elif isinstance(value, dict):
            print(f"{indent}{key}: {json.dumps(_json_ready(value), ensure_ascii=True, sort_keys=True)}")
        else:
            print(f"{indent}{key}: {value}")
        printed = True
    if not printed:
        print(f"{indent}(none)")


def _resolve_target(identity_store, face_db, speaker_db, name_or_id: str) -> str | None:
    rendered = str(name_or_id or "").strip()
    if not rendered:
        return None
    person_id = identity_store.resolve_person_id(rendered)
    if person_id:
        return person_id
    if face_db.embedding_store.get_embedding(rendered) is not None:
        return rendered
    if speaker_db.has_reference(rendered):
        return rendered
    needle = rendered.casefold()
    for ref in speaker_db.list_all_references():
        ref_person_id = str(ref.get("person_id") or "").strip()
        if not ref_person_id:
            continue
        if _humanize_person_id(ref_person_id).casefold() == needle:
            return ref_person_id
    return None


def list_identities(
    *,
    identity_db_path: str = DEFAULT_IDENTITY_DB_PATH,
    face_db_path: str = DEFAULT_FACE_DB_PATH,
    speaker_db_path: str = DEFAULT_SPEAKER_DB_PATH,
) -> None:
    identity_store, face_db, speaker_db = _open_stores(
        identity_db_path=identity_db_path,
        face_db_path=face_db_path,
        speaker_db_path=speaker_db_path,
    )
    face_ids = {record["person_id"] for record in face_db.embedding_store.list_embeddings()}
    speaker_ids = {ref["person_id"] for ref in speaker_db.list_all_references()}
    people = identity_store.list_people()
    if not people and not face_ids and not speaker_ids:
        print("No identities saved.")
        return

    print(f"\n{'=' * 60}")
    print(f"Argos Identities ({len(people)} total)")
    print(f"{'=' * 60}")
    for person in people:
        person_id = str(person["person_id"])
        meta = dict(person.get("metadata") or {})
        modalities = []
        if person_id in face_ids:
            modalities.append("face")
        if person_id in speaker_ids:
            modalities.append("speaker")
        print(f"\nName: {person['name']}")
        print(f"  ID:           {person_id}")
        print(f"  Modalities:   {', '.join(modalities) if modalities else '(identity only)'}")
        print(f"  First seen:   {meta.get('first_seen', 'unknown')}")
        print(f"  Last seen:    {meta.get('last_seen', 'unknown')}")
        print(f"  Interactions: {meta.get('interaction_count', 0)}")

    orphan_speaker_ids = sorted(speaker_ids - {person["person_id"] for person in people})
    orphan_face_ids = sorted(face_ids - {person["person_id"] for person in people})
    if orphan_face_ids:
        print("\nFace embeddings without identity rows:")
        for person_id in orphan_face_ids:
            print(f"  - {_humanize_person_id(person_id)} ({person_id})")
    if orphan_speaker_ids:
        print("\nSpeaker references without identity rows:")
        for person_id in orphan_speaker_ids:
            print(f"  - {_humanize_person_id(person_id)} ({person_id})")


def show_identity(
    name_or_id: str,
    *,
    identity_db_path: str = DEFAULT_IDENTITY_DB_PATH,
    face_db_path: str = DEFAULT_FACE_DB_PATH,
    speaker_db_path: str = DEFAULT_SPEAKER_DB_PATH,
    as_json: bool = False,
) -> int:
    identity_store, face_db, speaker_db = _open_stores(
        identity_db_path=identity_db_path,
        face_db_path=face_db_path,
        speaker_db_path=speaker_db_path,
    )
    person_id = _resolve_target(identity_store, face_db, speaker_db, name_or_id)
    if not person_id:
        print(f"No identity found for '{name_or_id}'.")
        print("Use --list to see saved identities.")
        return 1

    record = identity_store.get_person(person_id)
    face_record = face_db.embedding_store.get_embedding(person_id)
    speaker_record = speaker_db.get_reference(person_id)
    meta = dict((record or {}).get("metadata") or {})
    aliases = identity_store.list_aliases(person_id)
    extra_metadata = {
        key: value
        for key, value in meta.items()
        if key not in _IDENTITY_METADATA_KEYS
    }
    face_metadata = dict((face_record or {}).get("metadata") or {})
    speaker_metadata = dict((speaker_record or {}).get("metadata") or {})

    if as_json:
        payload = {
            "person_id": person_id,
            "name": _display_name(identity_store, person_id),
            "aliases": aliases,
            "identity_metadata": extra_metadata,
            "modalities": {
                "face": {
                    "present": bool(face_record),
                    "embedding_shape": _embedding_shape(face_record),
                    "metadata": face_metadata,
                },
                "speaker": {
                    "present": bool(speaker_record),
                    "embedding_shape": _embedding_shape(speaker_record),
                    "metadata": speaker_metadata,
                },
            },
            "system": {
                "first_seen": meta.get("first_seen"),
                "last_seen": meta.get("last_seen"),
                "interaction_count": meta.get("interaction_count", 0),
            },
        }
        print(json.dumps(_json_ready(payload), ensure_ascii=True, indent=2, sort_keys=True))
        return 0

    print(f"\n{'=' * 60}")
    print(f"Identity: {_display_name(identity_store, person_id)}")
    print(f"{'=' * 60}")
    print(f"  person_id:        {person_id}")
    print(
        f"  face embedding:   "
        f"{'yes' if face_record else 'no'}"
        f"{f' ({_embedding_shape(face_record)})' if _embedding_shape(face_record) else ''}"
    )
    print(
        f"  speaker embedding:"
        f"{'yes' if speaker_record else 'no'}"
        f"{f' ({_embedding_shape(speaker_record)})' if _embedding_shape(speaker_record) else ''}"
    )
    print(f"  first_seen:       {meta.get('first_seen', '—')}")
    print(f"  last_seen:        {meta.get('last_seen', '—')}")
    print(f"  interaction_count:{meta.get('interaction_count', 0)}")

    print("\nAliases:")
    if aliases:
        for alias in aliases:
            print(
                f"  - {alias['alias']} "
                f"[kind={alias['kind']} normalized={alias['normalized_alias']}]"
            )
    else:
        print("  (none)")

    _print_key_value_lines("Identity Metadata", extra_metadata)
    _print_key_value_lines("Face Embedding Metadata", face_metadata)
    _print_key_value_lines("Speaker Reference Metadata", speaker_metadata)
    return 0


def delete_identity(
    name_or_id: str,
    *,
    identity_db_path: str = DEFAULT_IDENTITY_DB_PATH,
    face_db_path: str = DEFAULT_FACE_DB_PATH,
    speaker_db_path: str = DEFAULT_SPEAKER_DB_PATH,
    yes: bool = False,
) -> int:
    identity_store, face_db, speaker_db = _open_stores(
        identity_db_path=identity_db_path,
        face_db_path=face_db_path,
        speaker_db_path=speaker_db_path,
    )
    person_id = _resolve_target(identity_store, face_db, speaker_db, name_or_id)
    if not person_id:
        print(f"No identity found for '{name_or_id}'.")
        print("Use --list to see saved identities.")
        return 1

    display_name = _display_name(identity_store, person_id)
    if not yes:
        confirm = input(
            f"Delete identity, face embedding, and speaker embedding for "
            f"'{display_name}' ({person_id})? [y/N]: "
        )
        if confirm.strip().lower() != "y":
            print("Cancelled.")
            return 1

    face_deleted = face_db.embedding_store.delete_embedding(person_id)
    speaker_deleted = speaker_db.delete_reference(person_id)
    identity_deleted = identity_store.delete_person(person_id)
    print(f"Deleted {display_name} ({person_id}).")
    print(f"  identity:          {'deleted' if identity_deleted else 'not found'}")
    print(f"  face embedding:    {'deleted' if face_deleted else 'not found'}")
    print(f"  speaker embedding: {'deleted' if speaker_deleted else 'not found'}")
    return 0 if any((face_deleted, speaker_deleted, identity_deleted)) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage Argos identities and linked face/speaker embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m argos_src.identity.manage_identity --list
  python -m argos_src.identity.manage_identity --show "Sakshee Patil"
  python -m argos_src.identity.manage_identity --delete "Sakshee Patil"
  python -m argos_src.identity.manage_identity --delete person_sakshee_patil_20260504_152002 -y
        """,
    )
    parser.add_argument("--list", action="store_true", help="List saved identities")
    parser.add_argument("--show", metavar="NAME_OR_ID", help="Show one identity")
    parser.add_argument(
        "--json",
        action="store_true",
        help="With --show, print the identity and modality metadata as JSON",
    )
    parser.add_argument(
        "--delete",
        metavar="NAME_OR_ID",
        help="Delete identity plus linked face and speaker embeddings",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation for --delete")
    parser.add_argument(
        "--identity-db-path",
        default=str(DEFAULT_IDENTITY_DB_PATH),
        help="Path to the identity SQLite database",
    )
    parser.add_argument(
        "--face-db-path",
        default=str(DEFAULT_FACE_DB_PATH),
        help="Path to the face embedding database",
    )
    parser.add_argument(
        "--speaker-db-path",
        default=str(DEFAULT_SPEAKER_DB_PATH),
        help="Path to the speaker embedding database",
    )
    args = parser.parse_args()

    if args.list:
        list_identities(
            identity_db_path=args.identity_db_path,
            face_db_path=args.face_db_path,
            speaker_db_path=args.speaker_db_path,
        )
        return 0
    if args.show:
        return show_identity(
            args.show,
            identity_db_path=args.identity_db_path,
            face_db_path=args.face_db_path,
            speaker_db_path=args.speaker_db_path,
            as_json=args.json,
        )
    if args.delete:
        return delete_identity(
            args.delete,
            identity_db_path=args.identity_db_path,
            face_db_path=args.face_db_path,
            speaker_db_path=args.speaker_db_path,
            yes=args.yes,
        )

    parser.error("use --list, --show, or --delete")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
