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

The shipped profile section is:

```yaml
identity_memory:
  enabled: true
  backend: tailwag_package
  site_code: BOS3
  place_room_id: realtime
  retention_class: standard
  record_live_episodes: true
  extract_live_turn_memory: true
  timeout_ms: 750
```

Removed Argos sections include `identity_store`, `employee_directory`, `memory`,
and `slack_memory`. Face and speaker recognition no longer accept local database
paths for biometric storage.
