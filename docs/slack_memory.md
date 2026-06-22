# Slack Memory With Tailwag

Slack memory is Tailwag-backed. Argos configures Slack polling and identity
convergence inputs; Tailwag owns Slack episode construction, transcript
formatting, memory extraction, and persistence.

## Modes

Slack memory is controlled by profile config:

```yaml
slack_memory:
  enabled: true
  start_with_agent: false
```

- `enabled: false`: Argos creates no Slack memory service.
- `enabled: true`, `start_with_agent: false`: profile is configured, but Argos
  does not poll Slack during normal agent launch.
- `enabled: true`, `start_with_agent: true`: Argos starts a background scheduler
  that calls Tailwag's Slack poller until agent shutdown.

If Slack ingestion should run outside the robot process, use Tailwag's own CLI
or service runner.

## Configuration

Do not put the Slack token in YAML. Store it in the environment and point the
profile at the environment variable:

```bash
export SLACK_BOT_TOKEN='xoxb-...'
```

```yaml
slack_memory:
  enabled: true
  start_with_agent: false
  bot_token_env: SLACK_BOT_TOKEN
  poll_interval_sec: 1800.0
  state_path: .tailwag/slack-state.json
  backfill_hours: 2.0
  force_backfill: false
  active_thread_hours: 24.0
  history_limit: 50
  reply_limit: 200
  extract_memory: true
  include_email: true
  channels:
    - name: robotics_internal_team
      channel_id: C0896C8CE83
```

Channel IDs belong in `slack_memory.channels[*].channel_id`. The `name` field is
an operator label; Tailwag polling uses the channel ID when present.

`include_email: true` is required for Slack-to-Argos identity convergence. The
Slack app needs `users:read.email` in addition to the Tailwag-required channel
history/profile scopes.

## Polling And State

Argos scheduling is intentionally thin:

```text
TailwagSlackMemoryService
    -> SlackWebApiClient(token, include_email=true)
    -> SlackMemoryPoller.poll_once(channel_id, ...)
    -> TailwagMemoryProvider.record_episode(...)
```

Tailwag converts Slack threads into conversation episodes. Stable episode IDs
look like:

```text
slack:<channel_id>:<thread_ts>
```

Polling state is a JSON cursor file at `slack_memory.state_path`, defaulting to
`.tailwag/slack-state.json`.

Avoid polling the same channel from more than one process at the same time. The
state file is not a cross-process lease.

## Slack Identity Resolution

Tailwag initially stores Slack participants as temporary people:

```text
Person.id = slack:<slack_user_id>
Person.email = <Slack profile email, only when include_email=true and Slack returns it>
```

Temporary `slack:<user_id>` people are not Argos canonical people. They can
remain in Tailwag as Slack-only participants until Argos can prove the canonical
identity.

Argos resolves canonical identity through Snowflake username. The Snowflake
username is the email local-part, before `@`, and these usernames are unique.
That makes the mapping deterministic:

```text
Slack profile email
  -> lowercased email local-part before "@"
  -> Snowflake username
  -> Argos canonical person_id
```

Example:

```text
Slack email:        arushi@example.com
Snowflake username: arushi
Argos person id:    person_arushi_...
```

When Argos later sees that person locally, it should upsert the Argos canonical
person into Tailwag and ask Tailwag to rekey the email-matched `slack:<id>`
person to the Argos `person_id`.

Tailwag changes the matched `Person.id` in place, so existing Slack episodes and
relationships stay attached to the same graph node. Existing memory item IDs are
not renamed; callers should use person-scoped APIs/relationships after rekey.

If Slack email is missing, `include_email` is false, the local-part does not
match a Snowflake username, multiple Argos identities claim the same username,
or Tailwag rekey returns false, the Slack person remains unresolved as
`slack:<user_id>`. Treat this as identity-review work, not an automatic merge.
Unresolved Slack-only people must not be surfaced as recognized Argos identities
in runtime prompt context.

## Tailwag-Owned Responsibilities

Tailwag owns:

- Slack episode construction
- transcript formatting
- memory extraction and persistence
- Slack participant storage

Argos keeps biometric data local and does not send face vectors, audio vectors,
raw images, or raw audio to Tailwag.
