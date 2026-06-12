#!/usr/bin/env python3
from __future__ import annotations

"""Manage saved speaker-reference embeddings for Argos."""

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from argos_src.identity.constants import DEFAULT_IDENTITY_DB_PATH
from argos_src.speaker_recognition.constants import DEFAULT_SPEAKER_DB_PATH


def _load_identity_name_map(
    *,
    identity_db_path: str = DEFAULT_IDENTITY_DB_PATH,
) -> dict[str, str]:
    from argos_src.identity import IdentityStore

    identity_store = IdentityStore(identity_db_path)
    return {
        str(person["person_id"]): str(person.get("name") or "").strip()
        for person in identity_store.list_people()
    }


def _humanize_person_id(person_id: str) -> str:
    rendered = str(person_id or "").strip()
    if not rendered:
        return ""
    match = re.match(r"^person_(.+?)_(\d{8})_(\d{6})$", rendered)
    slug = match.group(1) if match else rendered.removeprefix("person_")
    cleaned = " ".join(part for part in slug.replace("_", " ").split() if part)
    return cleaned.title() if cleaned else rendered


def _display_name_for_reference(
    person_id: str,
    *,
    metadata: dict[str, object] | None = None,
    identity_name_map: dict[str, str] | None = None,
) -> str:
    rendered_person_id = str(person_id or "").strip()
    meta = dict(metadata or {})
    identity_name = str((identity_name_map or {}).get(rendered_person_id) or "").strip()
    stored_name = str(meta.get("display_name") or meta.get("name") or "").strip()
    return identity_name or stored_name or _humanize_person_id(rendered_person_id) or rendered_person_id


def _resolve_reference_target(
    name_or_id: str,
    *,
    speaker_db,
    identity_db_path: str = DEFAULT_IDENTITY_DB_PATH,
) -> str | None:
    rendered = str(name_or_id or "").strip()
    if not rendered:
        return None
    if rendered.startswith("person_") and speaker_db.has_reference(rendered):
        return rendered

    identity_name_map = _load_identity_name_map(identity_db_path=identity_db_path)
    needle = rendered.casefold()
    for ref in speaker_db.list_all_references():
        person_id = str(ref.get("person_id") or "").strip()
        if not person_id:
            continue
        display_name = _display_name_for_reference(
            person_id,
            metadata=ref.get("metadata"),
            identity_name_map=identity_name_map,
        )
        lowered = display_name.casefold()
        if lowered == needle or needle in lowered:
            return person_id
    return None


def list_voice_references(
    *,
    speaker_db_path: str = DEFAULT_SPEAKER_DB_PATH,
    identity_db_path: str = DEFAULT_IDENTITY_DB_PATH,
) -> None:
    from argos_src.identity.embeddings.speaker_store import SpeakerEmbeddingStore

    speaker_db = SpeakerEmbeddingStore(db_path=speaker_db_path)
    name_map = _load_identity_name_map(identity_db_path=identity_db_path)
    refs = speaker_db.list_all_references()
    if not refs:
        print("No voice references saved.")
        return

    print(f"\n{'=' * 60}")
    print(f"Voice References ({len(refs)} total)")
    print(f"{'=' * 60}")
    for ref in refs:
        person_id = str(ref["person_id"])
        meta = dict(ref.get("metadata") or {})
        display_name = _display_name_for_reference(
            person_id,
            metadata=meta,
            identity_name_map=name_map,
        )
        print(f"\nName: {display_name}")
        print(f"  ID:           {person_id}")
        print(f"  Created at:   {meta.get('created_at', 'unknown')}")
        print(f"  Updated at:   {meta.get('last_updated_at', 'unknown')}")
        print(f"  Model:        {meta.get('model_name', 'unknown')}")
        print(f"  Clips:        {meta.get('clip_count', 'unknown')}")
        print(f"  Duration (s): {meta.get('query_duration_s', 'unknown')}")
        print(f"  Total voiced: {meta.get('total_voiced_sec', 'unknown')}")
        print(f"  RMS level:    {meta.get('rms_level', 'unknown')}")
        print(f"  Mean RMS:     {meta.get('mean_rms_level', 'unknown')}")


def show_voice_reference(
    name_or_id: str,
    *,
    speaker_db_path: str = DEFAULT_SPEAKER_DB_PATH,
    identity_db_path: str = DEFAULT_IDENTITY_DB_PATH,
) -> int:
    from argos_src.identity.embeddings.speaker_store import SpeakerEmbeddingStore

    speaker_db = SpeakerEmbeddingStore(db_path=speaker_db_path)
    person_id = _resolve_reference_target(
        name_or_id,
        speaker_db=speaker_db,
        identity_db_path=identity_db_path,
    )
    if not person_id:
        print(f"No voice reference found for '{name_or_id}'.")
        print("Use --list to see all saved voice references.")
        return 1

    record = speaker_db.get_reference(person_id)
    if not record:
        print(f"Voice reference {person_id} not found.")
        return 1

    name_map = _load_identity_name_map(identity_db_path=identity_db_path)
    meta = dict(record.get("metadata") or {})
    embedding = record.get("embedding")
    print(f"\n{'=' * 60}")
    print(
        "Voice Reference: "
        f"{_display_name_for_reference(person_id, metadata=meta, identity_name_map=name_map)}"
    )
    print(f"{'=' * 60}")
    print(f"  person_id:        {person_id}")
    print(f"  created_at:       {meta.get('created_at', '—')}")
    print(f"  last_updated_at:  {meta.get('last_updated_at', '—')}")
    print(f"  model_name:       {meta.get('model_name', '—')}")
    print(f"  clip_count:       {meta.get('clip_count', '—')}")
    print(f"  query_duration_s: {meta.get('query_duration_s', '—')}")
    print(f"  total_voiced_sec: {meta.get('total_voiced_sec', '—')}")
    print(f"  rms_level:        {meta.get('rms_level', '—')}")
    print(f"  mean_rms_level:   {meta.get('mean_rms_level', '—')}")
    print(
        f"  embedding shape:  "
        f"{embedding.shape if getattr(embedding, 'shape', None) is not None else '—'}"
    )
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage Argos speaker-reference embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m argos_src.speaker_recognition.manage_voice --list
  python -m argos_src.speaker_recognition.manage_voice --show "Sakshee Patil"
        """,
    )
    parser.add_argument("--list", action="store_true", help="List all saved voice references")
    parser.add_argument("--show", metavar="NAME_OR_ID", help="Show one saved voice reference")
    parser.add_argument(
        "--speaker-db-path",
        default=str(DEFAULT_SPEAKER_DB_PATH),
        help="Path to the speaker reference database",
    )
    parser.add_argument(
        "--identity-db-path",
        default=str(DEFAULT_IDENTITY_DB_PATH),
        help="Path to the identity database used for name resolution",
    )
    args = parser.parse_args()

    if args.list:
        list_voice_references(
            speaker_db_path=args.speaker_db_path,
            identity_db_path=args.identity_db_path,
        )
        return 0
    if args.show:
        return show_voice_reference(
            args.show,
            speaker_db_path=args.speaker_db_path,
            identity_db_path=args.identity_db_path,
        )

    parser.error("use --list or --show")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
