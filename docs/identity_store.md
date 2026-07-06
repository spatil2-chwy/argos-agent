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
| `argos_src/identity/embeddings/face_store.py` | Face embedding collection keyed by `person_id`. |
| `argos_src/identity/embeddings/speaker_store.py` | Speaker reference collection keyed by `person_id`. |
| `argos_src/face_recognition/store.py` | Face recognition store that wires face embeddings to identity rows. |
| `argos_src/speaker_recognition/manage_voice.py` | Operator CLI for listing and showing speaker references. |

## Operator Commands

From the repo root:

```bash
python3 -m argos_src.identity.manage_identity --list
python3 -m argos_src.identity.manage_identity --show "Your Name"
python3 -m argos_src.identity.manage_identity --delete "Your Name"
python3 -m argos_src.identity.manage_identity --delete person_your_name_20260505_123456 -y
```

Inspect memory separately:

```bash
python3 -m argos_src.memory.manage_memory --person "Your Name"
python3 -m argos_src.memory.manage_memory --site BOS3
python3 -m argos_src.memory.manage_memory --person "Your Name" --site BOS3 --prompt
python3 -m argos_src.memory.manage_memory --person person_your_name_20260505_123456 --all --json
```

## Fresh Local Reset

After the identity/memory split, old local identity databases that still contain
social-memory columns are not supported. The reset below is destructive: it
deletes local identity rows, face embeddings, speaker embeddings, and memory
items. Use it only for disposable local smoke-test state, after exporting or
backing up anything you need to keep:

```bash
rm -rf var/identity/identity.sqlite3 var/face_recognition var/speaker_recognition var/memory
```

The next runtime/enrollment run recreates the current schemas:

- `var/identity/identity.sqlite3`: identity-only SQLite
- `var/face_recognition`: ChromaDB face embeddings
- `var/speaker_recognition`: ChromaDB speaker embeddings
- `var/memory/memory.sqlite3`: source-aware memory SQLite

`argos_src.identity.manage_identity --delete` removes:

- the identity row and aliases
- the face embedding, if present
- the speaker embedding, if present

It does not delete memory items. Memory has its own store so operators can audit
what was learned from live chat, robot encounters, and Slack imports.

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
