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
- face and voice biometric references, including adaptive reference sample counts
- owner resolution from face and voice evidence
- Slack ingestion
- live episode ingestion and memory extraction
- semantic memory search and prompt context

Argos reaches Tailwag through the generic HTTP provider transport and the
`argos_src.identity_memory.tailwag_http` adapter. Everything else uses the
`IdentityMemoryClient` protocol or the noop fallback.

The shipped profile section keeps only the normal operational knobs:

```yaml
identity_memory:
  enabled: true
  site_code: BOS3
```

The remaining identity-memory profile keys are advanced switches with defaults:

- `backend`: defaults to `tailwag_http`; set to `noop` only for disabled
  identity-memory test or fallback runs.
- `place_room_id`: defaults to `realtime`; attached to Tailwag live episodes as
  room metadata.
- `retention_class`: defaults to `standard`; attached to Tailwag live episodes
  for Tailwag-owned retention policy.
- `record_live_episodes`: defaults to `true`; controls whether resolved live
  turn segments are sent to Tailwag as episodes.
- `extract_live_turn_memory`: defaults to `false`; controls whether Tailwag
  extracts semantic memory from recorded live episodes. Leave this off for
  lower live-turn latency, and opt in only when episode sends should also run
  Tailwag memory extraction.

Removed Argos sections include `identity_store`, `employee_directory`, `memory`,
and `slack_memory`. Face and speaker recognition no longer accept local database
paths for biometric storage.

The selected manifest must include an HTTP `memory` provider and a `memory`
resource with `memory.identity`. Argos sends bearer auth from
`TAILWAG_API_BEARER_TOKEN` when the provider declares:

```yaml
auth:
  type: bearer
  token_env: TAILWAG_API_BEARER_TOKEN
```

## Runtime Memory Search

Profiles may expose Tailwag semantic search through the public tool ID
`memory.search_semantic`, which resolves to the runtime tool
`search_memory_semantic`. The tool schema exposes only:

- `query`
- `limit`

The LLM does not pass a `person_id`. Argos scopes the search to the current
resolved turn owner from request context, and the tool returns an error when no
recognized owner is available. Search itself is read-only from Argos' point of
view; episode ingestion, extraction, archival, and repair remain Tailwag-owned.

## Adaptive Biometric Updates

Initial face and voice enrollment still work the same way: Argos produces one
FaceNet or ECAPA embedding and Tailwag stores the first durable reference sample.
After that, Argos may offer additional embeddings only when ownership evidence is
cross-modal safe:

- voice observations can be offered only when face and voice agree
- face observations can be offered only when face and voice agree
- voice-only ownership does not self-train voice
- face-only ownership does not self-train face or voice

Argos does not store durable sample counts or update centroids. It only keeps a
session-local completion cache in
`argos_src/identity_memory/biometric_updates.py` so it can offer every
cross-modal-safe turn until Tailwag reports that the reference is complete.

Tailwag returns update result fields such as `accepted`, `status`, `reason`,
`sample_count`, `target_sample_count`, and `similarity`. Argos logs those as the
structured dashboard event `adaptive_biometric_update`. These fields are for
operators and debugging only; prompt assembly does not include biometric update
details.
