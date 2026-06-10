# Argos Identity and Embedding Stores

Argos separates identity, embeddings, and social memory.

```text
identity store
    -> person_id, canonical name, aliases, interaction count, directory metadata

face embedding store
    -> person_id -> FaceNet embedding

speaker embedding store
    -> person_id -> ECAPA speaker embedding

memory store
    -> person_id/site_code -> source-aware social/context memory
```

The identity store is plain SQLite and uses only the Python standard library.
Face and speaker embeddings use ChromaDB. Social memory lives in
`argos_src/memory/store.py`, not in the identity row.

## Main Components

| File | Responsibility |
|---|---|
| `argos_src/identity/store.py` | SQLite-backed identity, alias, interaction, and directory metadata store. |
| `argos_src/identity/manage_identity.py` | Operator CLI for listing, showing, and deleting identities plus linked embeddings. |
| `argos_src/memory/store.py` | SQLite-backed source-aware memory items for people and sites. |
| `argos_src/memory/manage_memory.py` | Operator CLI for inspecting person/site memory items. |
| `argos_src/embedding_stores/face_store.py` | Face embedding collection keyed by `person_id`. |
| `argos_src/embedding_stores/speaker_store.py` | Speaker reference collection keyed by `person_id`. |
| `argos_src/face_recognition/store.py` | Face recognition store that wires face embeddings to identity rows. |
| `argos_src/speaker_recognition/manage_voice.py` | Operator CLI for listing and showing speaker references. |

## Operator Commands

From `argos_src`:

```bash
python3 identity/manage_identity.py --list
python3 identity/manage_identity.py --show "Your Name"
python3 identity/manage_identity.py --delete "Your Name"
python3 identity/manage_identity.py --delete person_your_name_20260505_123456 -y
```

Inspect memory separately:

```bash
python3 memory/manage_memory.py --person "Your Name"
python3 memory/manage_memory.py --site BOS3
python3 memory/manage_memory.py --person "Your Name" --site BOS3 --prompt
python3 memory/manage_memory.py --person person_your_name_20260505_123456 --all --json
```

## Fresh Local Reset

After the identity/memory split, old local identity databases that still contain
social-memory columns are not supported. For a clean local reset from `argos_src`,
remove the identity DB, both embedding DB directories, and the memory DB:

```bash
rm -rf identity/db/identity.sqlite3 face_recognition/db speaker_recognition/db memory/db
```

The next runtime/enrollment run recreates the current schemas:

- `identity/db/identity.sqlite3`: identity-only SQLite
- `face_recognition/db`: ChromaDB face embeddings
- `speaker_recognition/db`: ChromaDB speaker embeddings
- `memory/db/memory.sqlite3`: source-aware memory SQLite

`identity/manage_identity.py --delete` removes:

- the identity row and aliases
- the face embedding, if present
- the speaker embedding, if present

It does not delete memory items. Memory has its own store so operators can audit
what was learned from live chat, robot encounters, and future Slack imports.

## Runtime Behavior

Face recognition and speaker recognition both resolve to `person_id`. The
runtime builds a recognized-person context from:

- identity store: name, aliases, employee/directory metadata, interaction count
- memory store: prompt-safe person preferences, notes, follow-ups, recent encounters

In the dynamic prompt, identity metadata appears as `Directory` lines inside
`[PEOPLE IN VIEW]`; social memory appears as `About` and `Potential Followups`.
For example, title, manager, org, and tenure come from IdentityStore/Snowflake,
not MemoryStore.

That means speaker recognition can still provide the person's name when face
recognition is disabled or the speaker is not visible. The social/context memory
still comes from `MemoryStore`.
