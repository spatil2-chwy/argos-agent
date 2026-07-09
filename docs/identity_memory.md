# Identity Memory Boundary

Argos no longer owns durable identity or memory state.

Argos keeps realtime sensing and runtime behavior:

- face detection and FaceNet embedding generation
- audio capture and ECAPA voice embedding generation
- enrollment quality checks
- attention, depth, and turn-owner runtime gates
- dashboard log fields and prompt assembly

Tailwag owns durable identity and memory:

- Neo4j person profiles
- Snowflake-backed directory records
- face and voice biometric references
- owner resolution from face and voice evidence
- Slack ingestion
- live episode ingestion and memory extraction
- semantic memory search and prompt context

Argos imports Tailwag only through `argos_src.identity_memory.tailwag_package`.
Everything else uses the `IdentityMemoryClient` protocol or the noop fallback.

The shipped profile section keeps only the normal operational knobs:

```yaml
identity_memory:
  enabled: true
  site_code: BOS3
```

The remaining identity-memory profile keys are advanced switches with defaults:

- `backend`: defaults to `tailwag_package`; set to `noop` only for disabled
  identity-memory test or fallback runs.
- `place_room_id`: defaults to `realtime`; attached to Tailwag live episodes as
  room metadata.
- `retention_class`: defaults to `standard`; attached to Tailwag live episodes
  for Tailwag-owned retention policy.
- `record_live_episodes`: defaults to `true`; controls whether resolved live
  turn segments are sent to Tailwag as episodes.
- `extract_live_turn_memory`: defaults to `true`; controls whether Tailwag
  extracts semantic memory from recorded live episodes.

Removed Argos sections include `identity_store`, `employee_directory`, `memory`,
and `slack_memory`. Face and speaker recognition no longer accept local database
paths for biometric storage.
