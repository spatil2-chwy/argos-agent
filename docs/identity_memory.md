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

The normal memory backend is the AWS-hosted Tailwag service selected by the
shipped runtime manifests. The Argos host needs outbound HTTPS connectivity to
the Tailwag API Gateway and the bearer token from AWS Secrets Manager secret
`aaggarwal1-tailwag/api-bearer-token` exposed as
`TAILWAG_API_BEARER_TOKEN`. The token must remain outside the repository.
Tailwag's Neo4j, OpenAI, Slack, Snowflake, embedding, and model
configuration are owned by the hosted service and are not configured in Argos.
Argos's own Realtime API credential remains a separate runtime prerequisite.

The shipped profile section keeps only the normal operational knobs:

```yaml
identity_memory:
  enabled: true
  site_code: BOS3
  place_room_id: __site__
```

The remaining identity-memory profile keys are advanced switches with defaults:

- `backend`: defaults to `tailwag_http`; set to `noop` only for disabled
  identity-memory test or fallback runs.
- `place_room_id`: defaults to `realtime` for custom profiles. The shipped BOS3
  profiles explicitly use `__site__`, Tailwag's canonical building-level Place,
  so live episodes and employee home-base provenance reuse the same site node.
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
resource with `memory.identity`, `memory.person_context`, and `memory.episodes`.
Argos sends bearer auth from `TAILWAG_API_BEARER_TOKEN` when the provider
declares:

```yaml
auth:
  type: bearer
  token_env: TAILWAG_API_BEARER_TOKEN
```

Argos calls Tailwag's `memory.person_context` operation for the resolved turn
owner and includes the active manifest robot's stable `robot_id`. The HTTP transport posts that operation to
`/argos/providers/memory/resources/memory/request/person_context`. Tailwag
returns `context_markdown`, and Argos pastes that prompt-ready markdown into the
`[PERSON SPEAKING TO YOU]` block after any Argos-owned `Directory` lines. Argos
does not parse or rebuild the Tailwag memory section locally. Tailwag uses the
robot ID to include memories backed by robot-free sources such as Slack and by
episodes involving this robot, while excluding memories backed only by other
robots' interactions.

## Live Episode Robot Attribution

The runtime derives stable robot identity from the selected manifest, not from
an identity-memory profile override. `TailwagHttpIdentityMemoryClient` requires
that manifest's robot `id` and `display_name` when the runtime is assembled.
Every live conversation episode includes exactly one host robot:

```json
{
  "robots": [
    {
      "id": "cody",
      "display_name": "Cody",
      "role": "host",
      "source": "argos"
    }
  ]
}
```

Puffle, Cody, and Navigation therefore send their own manifest IDs even when
display names collide or later change. Attribution rides on the existing
`memory.episodes_record` provider request and the existing `memory.episodes`
resource capability; it does not add a second realtime request, provider
operation, or robot capability. Tailwag owns the durable Robot node, the
episode-time display-name snapshot, and retrieval by stable robot ID. Argos does
not use Robot provenance for person identity, biometric matching, or memory
extraction targets.

## Runtime Memory Search

Profiles may expose Tailwag semantic search through the public tool ID
`memory.search_semantic`, which resolves to the runtime tool
`search_memory_semantic`. The tool schema exposes only:

- `query`
- `limit`

The LLM does not pass a `person_id`. Argos scopes the search to the current
resolved turn owner from request context and the active manifest robot's stable
ID, and the tool returns an error when no recognized owner is available. Search
itself is read-only from Argos' point of view; episode ingestion, extraction,
archival, and repair remain Tailwag-owned.

## Local Biometric Capture Bundles

The operator lab in `scripts/labs/biometric_enrollment_lab.py` is a narrow
exception to the normal no-local-biometric-store rule: it stages explicitly
captured enrollment evidence under
`data_collection/.biometric_enrollment_bundles/<bundle_uuid>/` until a
separate cleanup command deletes it. Push does not delete the bundle. These
bundles are not read by the realtime runtime, identity resolution, prompt
construction, or memory retrieval.

Capture makes no Tailwag/identity-memory request. It stores accepted raw
photos/WAVs, checksums, model provenance, accepted sample counts, and one
normalized aggregate vector per modality. Configured camera and display
provider transports can still use their configured endpoints. The bundle files
are unencrypted; host access and retention are operator responsibilities.
Interrupted/incomplete, corrupt, or tampered bundles are retained for
administrator review and are not eligible for normal cleanup.

Push requires a verified canonical directory identity; when a Person node
already exists, it must explicitly be active. The bundle binds all retries to
that identity and calls
`memory.biometrics_face_references_exists` and
`memory.biometrics_voice_references_exists`, and uploads only missing
references. Before any embedding leaves the bundle, the operator must verify
that the subject consented and type the canonical name; enrollment is then sent
as `consented`. Existing references are never updated, and this workflow does
not call adaptive observation operations. Deploy the matching Tailwag
face-existence endpoint before enabling push; unavailable or malformed
existence responses fail closed.

All missing modalities are conflict-searched globally across sites before the
first enrollment.
Enrollment and local journaling remain per-modality, so one modality can be
durable if a later enrollment fails; retry checks only unfinished modalities.
The existence check and enrollment request are also separate client operations.
A concurrent writer can therefore create a reference between them; this small
race is an explicit operator-lab tradeoff.

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
