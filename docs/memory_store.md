# Argos Memory Store

`argos_src/memory` is the canonical social/context memory layer.

It stores prompt-useful facts as source-aware SQLite rows. Identity remains the
owner of who someone is. Memory owns what the robot learned from interactions or
context feeds.

## Table

`argos_src/memory/store.py` creates `memory_items`:

| Column | Meaning |
|---|---|
| `memory_id` | Stable item id. |
| `scope_type` | `person` or `site`. |
| `scope_id` | `person_id` or site code. |
| `kind` | `preference`, `boundary`, `pet`, `fact`, `note`, `followup`, `encounter`, `office_event`. |
| `key` | Stable dedupe/update key within scope/kind/source. |
| `summary` | Prompt-safe text. |
| `source` | `live_chat`, `robot`, or `slack`. |
| `source_ref` | Source event/message/segment id when available. |
| `status` | `active`, `archived`, or `superseded`. |
| `created_at` | First insert time. |
| `observed_at` | When the memory was observed. |
| `updated_at` | Last write time. |
| `due_at` | Optional follow-up priority time. |
| `expires_at` | Optional prompt-visibility expiry. |
| `metadata_json` | Small source/context payload for debugging and future retrieval. |

Expired rows stay in SQLite for audit/debug visibility, but
`list_active_items()` filters them out before prompt compilation.

## Memory Kind Contracts

The `kind` column is a behavioral contract, not just a category label.

| Kind | Stores | Should Still Make Sense Later? | Prompt Surface |
|---|---|---:|---|
| `preference` | Stable user-stated choices: preferred name/language, likes, dislikes, what they call the robot. Requires explicit user evidence. | Yes | Person `About` |
| `boundary` | Stable user-stated comfort or behavior constraints: give more space, do not greet loudly, avoid a topic. | Yes | Person `About`, behavior-shaping |
| `pet` | Named pet entities and durable pet updates. | Yes | Person `About` |
| `fact` | Durable biographical facts: birthday, family context, hobbies, language ability. Not short-term plans. | Yes | Person `About` |
| `note` | Durable ongoing context that does not fit another kind, such as a long-running project. | Yes | Person `About`, capped |
| `followup` | Short-lived check-in opportunity: trip, visit, recovery, deadline, pending event. | No, it must expire | `Potential Followups` only |
| `encounter` | Robot-observed recent presence at a site. | No, short-lived | Recent encounter block |
| `office_event` | Site-level context from robot/system feeds. | Depends on expiry | Office context block |

Temporal plans should not become `note` or `fact`. For example, “my parents
are visiting and we are going to Cape Cod this weekend” should become a
`followup` with self-contained context like “Cape Cod trip with their parents
planned for the weekend of 2026-05-16,” plus concrete `due_at` and
`expires_at` datetimes. The memory does not script what the agent should say;
the realtime agent decides whether and how to bring it up.

That classification is an LLM responsibility, not a regex post-filter. The LLM
chooses the memory `kind`; Argos derives the prompt surface from that kind.
Deterministic validation only enforces structural contracts, such as valid
kinds, parseable datetime fields, and follow-ups having an expiry.

## Current Sources

### `live_chat`

`argos_src/memory/live_chat.py` extracts memory from recognized speaker-owned
conversation segments and writes `preference`, `boundary`, `pet`, `fact`,
`note`, and `followup` rows.

The extractor is best understood as a future prompt-context compiler. The
future robot will not see the original conversation; it may only see the memory
summary, due/expiry times, and small typed metadata. Because of that, the LLM is
asked to write each memory as if it may be inserted directly into a future turn
prompt.

Live-chat extraction does not receive the person's full memory history. Before
the LLM call, Argos selects a small candidate set: pinned profile memories such as
preferred name/language, active pets and follow-ups, plus memories with lexical
overlap with the current conversation segment. The extractor can update or
archive only those candidate `memory_id`s, or create new memories for clearly
new topics.

Live-chat notes are keyed topic records, not transcript snippets. Stable keys,
such as `parents_visit_boston`, let later details merge into one prompt-ready
summary. This keeps memory from growing as repeated near-copy sentences.

#### Live-Chat Extraction Contract

The LLM returns a single operation list:

| Field | Meaning |
|---|---|
| `op` | `create`, `update`, `archive`, or `noop`. |
| `memory_id` | Required for updates/archives; must refer to a candidate memory. |
| `kind` | Memory kind for new rows. |
| `key` | Stable snake_case dedupe key within scope/kind/source. |
| `summary` | Prompt-ready future context. |
| `value` | Optional small typed payload when structure helps, such as `{name, kind, notes}` for a pet or `{field, value}` for a profile field. Omit it when the summary is enough. |
| `due_at` | Optional ISO 8601 datetime for when a follow-up first becomes useful. |
| `expires_at` | ISO 8601 datetime for follow-ups; optional expiry for other memory. |

There is no separate `About` or `Potential Followups` output field. Those are
prompt views derived from `kind`:

- `preference`, `boundary`, `pet`, `fact`, and `note` become `About`.
- `followup` becomes `Potential Followups`.

Example weekend-plan operation:

```json
{
  "op": "create",
  "kind": "followup",
  "key": "parents_cape_cod_trip",
  "summary": "Cape Cod trip with their parents planned for the weekend of 2026-05-16.",
  "due_at": "2026-05-18T09:00:00-04:00",
  "expires_at": "2026-05-22T23:59:00-04:00"
}
```

This is intentionally a semantic contract with the LLM. Argos does not try to
decide from phrase matching whether “this weekend” is good or bad memory. The
prompt tells the LLM what future context should look like, and the validator
only checks that the returned operation can safely be stored and retrieved.

#### Structural Validation

`argos_src/memory/live_chat.py` still validates operations before writing them:

- `create` operations must have an allowed kind and non-empty summary.
- `update` and `archive` operations must target an existing memory item for the
  same person.
- profile keys such as `preferred_name`, `preferred_language`,
  `nickname_for_robot`, and `birthday` must use the expected memory kind.
- `due_at` and `expires_at`, when present, must parse as ISO datetimes.
- `followup` rows must include `expires_at`.
- live-chat memory still refuses identity-owned directory summaries such as
  `team:`, `title:`, `manager:`, or `cost center:`.

These are mechanical storage checks. They are not open-ended semantic quality
filters.

### `robot`

`argos_src/memory/store.py::record_encounter()` writes short-lived `encounter`
rows when face recognition records a new presence episode.

### `slack`

Optional Slack ingestion is implemented under `argos_src/memory/slack/` and
documented in `docs/slack_memory.md`. It polls approved Slack channels, extracts
prompt-ready person or site memories, writes normal `memory_items` rows with
`source='slack'`, and stages unresolved person memories in `slack_pending_memory`
until a local identity row can be matched.

Prompt compilation does not know Slack-specific parsing details. Once promoted
to active `memory_items`, Slack-derived rows flow through the same
`MemoryContextCompiler` path as live chat and robot encounter memory.

## Prompt Flow

```text
recognized person/site
    -> MemoryContextCompiler
    -> MemoryStore SQL queries by scope/kind/status/time
    -> compact prompt blocks
    -> RealtimeRobotAgent dynamic turn instructions
```

The live-chat extractor also reads a small relevant subset of active
MemoryStore rows before calling the extraction LLM. Its prompt uses:

- `Relevant existing memories JSON`: capped candidate `preference`, `boundary`,
  `pet`, `fact`, `note`, and `followup` items for the speaker, including
  `memory_id`, `kind`, `key`, `summary`, and typed `value` when available

The extractor returns one operation list:

- `create`: add a new memory item
- `update`: replace the summary/value of an existing candidate `memory_id`
- `archive`: hide an existing candidate `memory_id` when corrected, resolved,
  or no longer useful
- `noop`: no change

Those operations are validated and normalized into `memory_items`. The store
persists `summary`, `due_at`, and `expires_at` as first-class columns. The
operation's `value` is stored in `metadata_json` when present so typed details
can support future retrieval, dedupe, and formatting experiments.

Person memory currently feeds:

- structured lines from `preference`, `boundary`, `pet`, and `fact`
- capped durable context from `note`
- follow-up lines from `followup`

Person prompt projection is intentionally curated, not unbounded. Structured
person memory is capped before prompt projection, notes are capped at 10, and
follow-ups are capped separately. Follow-ups with a future `due_at` are hidden
until due, and expired follow-ups are filtered before prompt projection. This
keeps one person's long memory history from crowding out the current turn while
preserving high-signal typed facts like pets, birthdays, explicit boundaries,
and preferences.

Site memory currently feeds:

- office/site context from `office_event`
- recent same-site encounters from `encounter`, only when identity metadata
  shows a useful org relationship with the current person, such as same manager,
  cost center, business function, or leadership org

Encounter prompt lines include approximate age, for example:

```text
[RECENT ENCOUNTERS]
- You met Alex Kim 10 minutes ago; they are in the current person's org context (same manager).
```

## Inspection

From the repo root:

```bash
python3 -m argos_src.memory.manage_memory --person "Sakshee Patil"
python3 -m argos_src.memory.manage_memory --site BOS3
python3 -m argos_src.memory.manage_memory --person "Sakshee Patil" --site BOS3 --prompt
python3 -m argos_src.memory.manage_memory --person person_sakshee_patil_20260513_150604 --all --json
python3 -m argos_src.memory.manage_memory --archive mem_abc123
```

Use `--all` when you want to inspect archived or expired rows.
Use `--archive` to hide a bad or stale item from prompt retrieval without
deleting the row.
Use `--prompt` with `--person` and optional `--site` to inspect the compiled
prompt projection: identity `Directory` lines, memory `About`,
`Potential Followups`, office context, and relation-filtered recent encounters.
Plain `--person` shows raw person rows; plain `--site` shows site rows plus
recent same-site encounters for debugging.

If `python3 -m argos_src.memory.manage_memory --person ...` reports an
unsupported identity database schema, the local identity DB is still from the old
identity-owned memory design. The destructive local reset below deletes identity,
face, speaker, and memory state. Use it only for disposable local smoke-test
state, after exporting or backing up anything you need to keep:

```bash
rm -rf var/identity/identity.sqlite3 var/face_recognition var/speaker_recognition var/memory
```

Then re-enroll faces/voices so the identity and embedding stores point at the
same fresh `person_id` rows.
