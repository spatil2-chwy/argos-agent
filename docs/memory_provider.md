# Argos Memory Provider

Argos social/context memory is Tailwag-backed through `argos_src/memory_provider`.

Identity, face recognition, and speaker recognition remain local to Argos:

- `IdentityStore` owns Argos `person_id`, names, aliases, and employee metadata.
- Face and speaker embedding stores stay inside Argos.
- Tailwag owns social memory, Slack memory, episode storage, person context
  synthesis, archival, and durable follow-up extraction.

## Runtime Switches

Profiles use a `memory` section to decide whether Argos connects to Tailwag:

```yaml
memory:
  enabled: true
  retention_class: standard
  place_room_id: realtime
  extract_live_turn_memory: true
```

When `memory.enabled: false`, Argos should run without Tailwag context or memory
ingestion. Face, voice, identity, tools, and realtime turns still run.

When `memory.enabled: true`, Argos creates `TailwagMemoryProvider` and uses it
for:

- prompt-time person context through `person_context(...)`
- recognized-person encounter updates through `upsert_person(...)`
- realtime conversation episodes through `record_episode(...)`
- person archival through Tailwag when exposed by operator tooling

Tailwag is loaded lazily. Production memory use requires the `tailwag-memory`
package and Tailwag environment variables, including Neo4j and OpenAI settings.

## Realtime Conversation Episodes

Argos buffers completed speaker-owned realtime turns locally. Each active
conversation becomes one Tailwag episode:

```text
completed recognized turns
    -> Argos preference segment buffer
    -> Tailwag EpisodeInput(id="argos:conversation:<uuid>")
    -> Tailwag record_episode(..., extract_memory=memory.extract_live_turn_memory)
```

Speaker handoff does not start a new Tailwag episode. It only flushes the
current speaker-owned text into the same active conversation episode. The
episode ends on idle timeout or runtime shutdown.

Episode participants are Argos `person_id` values. Argos sends transcript text,
summary, place, role/source metadata, and retention class. Argos does not send
face embeddings, audio embeddings, raw images, or raw audio to Tailwag.

## Person Context

At prompt time, `RealtimeRobotAgent` asks the memory provider for Tailwag person
context for the current Argos `person_id`.

The provider maps Tailwag-rendered context into the existing prompt projection:

- `About`: durable person facts/preferences/boundaries/notes
- `Potential Followups`: due short-lived check-ins
- preferred language defaults to English unless Tailwag context says otherwise

Site context is not currently pulled from Tailwag by Argos. The provider returns
no site blocks until Tailwag exposes a first-class prompt-ready site-context
contract.

## Encounters And `last_seen`

Face recognition still runs locally. When Argos recognizes a person and records
an interaction, it calls `TailwagMemoryProvider.record_encounter(...)`.

Tailwag receives a person upsert with non-biometric metadata. Tailwag owns the
`last_seen` update semantics for that person.

## Slack Identity Convergence

Slack ingestion creates temporary Tailwag people such as `slack:U0123456789`.
Slack profile email should be captured whenever Slack scopes allow it.

Argos canonical identity uses Snowflake username. The Snowflake username is the
email local-part, before `@`, and these usernames are unique. Therefore:

```text
Slack email:        arushi@example.com
Snowflake username: arushi
Argos person id:    generated from the Argos/Snowflake identity for arushi
```

When Argos later sees a person and has Snowflake username/email metadata, the
Tailwag provider should rekey the email-matched Slack person to the Argos
canonical `person_id`. If email or username evidence is missing, the Slack
person remains under the temporary `slack:<user_id>` id until it can be resolved.

## Inspection

Use Tailwag operator tooling to inspect, archive, or repair social memory. Argos
operator docs treat Tailwag as the source of truth for social/context memory,
while local Argos tools remain responsible for identity and biometric stores.

## Local Reset

Reset local Argos identity and biometric stores with:

```bash
rm -rf var/identity/identity.sqlite3 var/face_recognition var/speaker_recognition
```

Tailwag/Neo4j data is external to Argos and must be reset with Tailwag tooling
or database operations.
