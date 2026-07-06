# Argos Identity and Embedding Stores

Argos separates identity, embeddings, and social memory.

```text
identity store
    -> person_id, canonical name, aliases, interaction count, directory metadata

face embedding store
    -> person_id -> FaceNet embedding

speaker embedding store
    -> person_id -> ECAPA speaker embedding

Tailwag memory
    -> Argos person_id -> social/context memory and Slack episodes
```

The identity store is plain SQLite and uses only the Python standard library.
Face and speaker embeddings use ChromaDB. Social/context memory lives in
Tailwag through `argos_src/memory_provider`, not in the identity row.

## Main Components

| File | Responsibility |
|---|---|
| `argos_src/identity/store.py` | SQLite-backed identity, alias, interaction, and directory metadata store. |
| `argos_src/identity/manage_identity.py` | Operator CLI for listing, showing, and deleting identities plus linked embeddings. |
| `argos_src/memory_provider/` | Tailwag-backed person context, realtime episodes, encounters, and Slack memory integration. |
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

Inspect social/context memory with Tailwag tooling.

## Fresh Local Reset

The reset below is destructive: it deletes local identity rows, face embeddings,
and speaker embeddings. Use it only for disposable local smoke-test state, after
exporting or backing up anything you need to keep:

```bash
rm -rf var/identity/identity.sqlite3 var/face_recognition var/speaker_recognition
```

The next runtime/enrollment run recreates the current schemas:

- `var/identity/identity.sqlite3`: identity-only SQLite
- `var/face_recognition`: ChromaDB face embeddings
- `var/speaker_recognition`: ChromaDB speaker embeddings

`argos_src.identity.manage_identity --delete` removes:

- the identity row and aliases
- the face embedding, if present
- the speaker embedding, if present

It does not delete Tailwag memory. Tailwag is external to Argos local runtime
state and must be inspected or repaired with Tailwag tooling.

## Runtime Behavior

Face recognition and speaker recognition both resolve to `person_id`. The
runtime builds a recognized-person context from:

- identity store: name, aliases, employee/directory metadata, interaction count
- Tailwag memory provider: prompt-safe person preferences, notes, follow-ups,
  and recent social/context memory

In the dynamic prompt, identity metadata appears as `Directory` lines inside
`[PERSON SPEAKING TO YOU]` only when the turn has a resolved `owner_id`;
social memory appears as `About` and `Potential Followups` for that owner.
For example, title, manager, org, and tenure come from IdentityStore/Snowflake,
not Tailwag memory.

That means speaker recognition can still provide the person's name when face
recognition is disabled or the speaker is not visible. The social/context memory
comes from Tailwag when `memory.enabled` is true.

## Snowflake Username And Slack Identity

Snowflake username is the unique local-part of the employee email address, the
text before `@`. Argos can use that rule to converge Slack and robot-seen
identity:

```text
Slack profile email -> email local-part -> Snowflake username -> Argos person_id
```

For example, `arushi@example.com` maps to Snowflake username `arushi`. When
Argos later sees the person locally, Tailwag can rekey the temporary
`slack:<user_id>` person to the Argos canonical `person_id`.
