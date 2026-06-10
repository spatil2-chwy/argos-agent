# Slack Memory Ingestion

Slack memory ingestion lets Argos read approved Slack channels and turn useful workplace context into normal source-aware `memory_items`.

The implementation lives in `argos_src/memory/slack/`.

## Big Picture

Slack memory is optional. The realtime agent can run exactly as before without Slack.

There are three useful modes:

```text
Slack disabled
  slack_memory.enabled: false
  -> no Slack service is created
  -> no Slack polling happens
  -> the agent still uses live chat, face, voice, and existing memory normally

Slack enabled but not started with the agent
  slack_memory.enabled: true
  slack_memory.start_with_agent: false
  -> the agent constructs the Slack service but does not poll Slack
  -> run Slack ingestion separately with python3 -m argos_src.memory.slack.run

Slack enabled and started with the agent
  slack_memory.enabled: true
  slack_memory.start_with_agent: true
  -> the agent starts a background Slack polling thread
  -> the thread is stopped when the agent runtime shuts down
```

Use `start_with_agent: false` when you want Slack memory to keep running as an independent ingestion process, or when you want explicit control over when Slack is polled.

Use `start_with_agent: true` when you only want Slack ingestion while the robot agent process is alive.

## Configuration

Do not paste the Slack bot token into source code or YAML. Store it in an environment variable.

You need:

- a Slack bot token in an environment variable
- the bot invited to each channel
- a Slack channel ID, or a channel name that can be resolved with Slack API
- a site code for site-scoped memories, usually `BOS3`
- local Argos identity rows if you want Slack person memory to become prompt-visible immediately

Example:

```bash
export SLACK_BOT_TOKEN='xoxb-...'
```

```yaml
slack_memory:
  enabled: true
  start_with_agent: false
  bot_token_env: SLACK_BOT_TOKEN
  poll_interval_sec: 1800.0
  lookback_minutes: 30
  channels:
    - name: robotics_internal_team
      channel_id: C0896C8CE83
      site_code: BOS3
      person_memory_enabled: true
      site_memory_enabled: true
      include_threads: true
      max_messages_per_window: 200
```

`channel.site_code` is the site scope for site memory. If it is empty, the service falls back to `employee_directory.site_code`. The writer enforces this site for `office_event` rows.

`person_memory_enabled` and `site_memory_enabled` are enforced in code. They are not sent to the LLM. If both are false, Slack ingestion skips the LLM call for that window.

## Running

Run one Slack polling cycle and exit:

```bash
python3 -m argos_src.memory.slack.run --profile static_interaction --once
```

Run Slack ingestion continuously without starting the realtime robot agent:

```bash
python3 -m argos_src.memory.slack.run --profile static_interaction
```

The standalone command opens the identity DB, memory DB, Slack client, and memory extraction LLM. It does not start ROS, realtime audio, face recognition, wake
word, or the robot agent.

If `slack_memory.enabled: false`, the standalone command exits cleanly and does
nothing.

Avoid running two Slack ingestion loops for the same profile/channel at the same
time, for example one standalone loop and one agent-started background loop.
They share the same checkpoint table, but there is no cross-process lease. You
may get duplicate LLM work or overlapping windows. Memory writes are mostly
upserts, but the extra polling is still wasteful and harder to reason about.

## Agent Lifecycle

When the agent starts, `argos_src/agent/factory.py` checks
`scenario_profile.slack_memory.enabled`.

If Slack memory is disabled, no Slack memory service is created.

If Slack memory is enabled, the factory creates:

- `IdentityStore`
- `MemoryStore`
- `SlackMemoryService`

If `start_with_agent` is true, the service starts a daemon background thread
that calls `run_forever()`. If `start_with_agent` is false, the service exists
but does not poll Slack.

When the agent runtime shuts down, it calls `slack_memory_service.shutdown()`.
That stops the background loop if one was started.

## Polling And Checkpoints

Slack ingestion is polling-based. Slack does not push events to Argos in this
design.

Each configured channel has a checkpoint stored in the memory SQLite DB:

```text
slack_channel_checkpoints(channel_id, last_ts, updated_at)
```

For each channel, one cycle chooses:

```text
oldest = checkpoint timestamp if present
       = now - lookback_minutes if no checkpoint exists

latest = now
```

Then it asks Slack for messages in `[oldest, latest]`.

After a successful extraction/write attempt, the checkpoint advances to
`latest`. If Slack returns no messages, the checkpoint still advances to
`latest`. If Slack fetch or LLM extraction fails, the checkpoint does not
advance, so the same window can be retried.

Example:

```text
2:15 PM first agent run, no checkpoint, lookback_minutes=30
  reads roughly 1:45 PM -> 2:15 PM
  stores checkpoint at 2:15 PM

2:30 PM next agent run using the same memory DB
  reads roughly 2:15 PM -> 2:30 PM
  does not intentionally re-read 1:45 PM -> 2:15 PM
```

So starting the agent at 2:15 and again at 2:30 does not create a 15-minute
overlap once the first run successfully wrote its checkpoint. If the first run
failed before checkpointing, the second run will fall back to the previous
checkpoint or `lookback_minutes`.

`max_messages_per_window` caps how many top-level channel messages are fetched
for one polling window. Keep it high enough for the busiest configured channel;
it is a safety cap, not a summarization strategy.

## What Slack Data Is Fetched

For each configured channel, one cycle uses:

- `conversations.history` for top-level channel messages
- `conversations.replies` for thread replies, only when `include_threads: true`
- `conversations.list`, only when `channel_id` is empty and the service must
  resolve a channel name
- `users.info` for users who authored messages or were mentioned in message text

No reaction-specific API calls are made. If Slack includes a `reactions` field
inside a message payload, Argos strips that field immediately. Reactions are not
rendered into the prompt, and reaction-only users do not trigger `users.info`
lookups.

Deleted messages, channel joins, and channel leaves are ignored.

## Normalization

Slack API payloads are normalized before the LLM sees them.

The normalized prompt window includes:

- message timestamp
- channel name
- author display label and Slack username when available
- message text
- mentioned user labels and Slack usernames when available
- thread replies as indented messages when `include_threads: true`

The normalized prompt window does not include:

- Slack user IDs
- Argos `person_id`s
- channel IDs
- raw Slack source refs such as `C123:1780339980.000100`
- reaction names, counts, or users

Example prompt transcript shape:

```text
[2026-06-02T18:11:16+00:00] Jakub Kowalewski (@jkowalewski): For those who have co-ops/interns joining, please send me their names so I can 3D print their name tags!
  [2026-06-02T18:12:07+00:00] Sakshee Patil (@spatil): Arushi Aggarwal
```

Slack IDs and source refs still exist internally so post-processing can write
source-aware memory and resolve people. They are just not prompt tokens. The
stored Slack `source_ref` is the channel/window ref unless an internal caller
provides a more specific ref.

## What The LLM Gets

The extraction prompt contains:

- current date and time
- channel name
- site code
- compact existing relevant memories
- the normalized Slack transcript
- instructions for what is worth remembering
- the structured JSON schema

The prompt does not contain:

- channel policy booleans
- Slack user to person mappings
- Slack user IDs
- Argos `person_id`s
- raw Slack source refs
- reactions

Existing relevant memories are included so the model can prefer updates over
duplicates. The candidate memory payload is compact: memory identity, scope,
kind, key, summary, source, and any relevant due/expiry/value metadata.

## What The LLM Returns

The LLM returns one JSON object:

```json
{
  "update": true,
  "ops": [
    {
      "op": "create",
      "scope_type": "person",
      "target_users": ["@person_x", "@person_y"],
      "kind": "fact",
      "key": "chewy_versary_2_year_2026_06_01",
      "summary": "milestone: completed 2 years at Chewy on 2026-06-01."
    }
  ]
}
```

If nothing should change:

```json
{"update": false, "ops": []}
```

Allowed person kinds:

```text
preference, boundary, pet, fact, note, followup
```

Allowed site kind:

```text
office_event
```

For person creates, the LLM should leave `scope_id` empty and return the people
in `target_users`. It should prefer exact visible `@username`s. If no username is
visible, it can use the exact name from the message.

For site creates, the LLM returns `scope_type="site"`. The writer sets the final
site `scope_id`; the model does not need to know implementation IDs.

## What Gets Stored

After the LLM returns JSON, code applies guardrails before writing:

- drops person ops when `person_memory_enabled` is false
- drops site ops when `site_memory_enabled` is false
- forces site memories to the configured site code
- rejects unsupported kinds
- requires `followup` memories to have an expiry
- normalizes keys

For person creates, the writer resolves each `target_users` entry against the
Slack profiles fetched for that window. Resolution can match:

- Slack username, with or without `@`
- display name
- real name
- email
- email prefix
- Slack user ID as a defensive internal fallback

If a target resolves to an Argos `person_id`, the writer upserts a normal
person-scoped memory:

```text
memory_items(scope_type="person", scope_id="<person_id>", source="slack")
```

If a target does not resolve to an Argos `person_id`, the writer stages it:

```text
slack_pending_memory
```

Pending rows are not included in agent context. They become visible only after
they are promoted to normal `memory_items`.

For site memories, the writer upserts:

```text
memory_items(scope_type="site", scope_id="<site_code>", source="slack")
```

## Attribution

Slack messages have an author and may mention other people. Memory should be
stored on whoever the fact is about, not automatically on the author.

Example:

```text
Olivia Ordonez (@olivia): Happy 2 year Chewy-versary @Joseph Papagno (@joseph) and @Thomas Walewski (@thomas)!
```

The desired operation is about Joseph and Thomas:

```text
scope_type = person
target_users = ["@joseph", "@thomas"]
kind = fact
summary = milestone: completed 2 years at Chewy on 2026-06-01.
```

It should not store that milestone on Olivia just because Olivia authored the
message.

Person memory summaries should be prompt-ready facts for the future agent, not
Slack event recaps. Prefer:

```text
milestone: completed 2 years at Chewy on 2026-06-01.
```

Avoid:

```text
Olivia congratulated Joseph in Slack.
```

## Identity Mapping And Backfill

Slack does not directly map to Snowflake rows inside the memory writer.
Snowflake feeds the employee directory; enrollment stores verified employee
metadata in the local `IdentityStore`; Slack then resolves against that local
identity metadata.

The resolver compares Slack profile data with local identity aliases and these
identity metadata fields:

```text
slack_user_id
slack_id
slack_username
slack_email
email
work_email
username
employee_username
```

It tries Slack display name, real name, username, and email prefix as local
identity aliases. The most useful path is usually Slack username or email prefix
to `username` or `employee_username`, because employee directory records already
provide employee usernames during registration.

If a Slack user is mapped:

```text
slack profile/name/email/username -> IdentityStore person_id -> memory_items(scope_type=person)
```

If a Slack user is not mapped:

```text
slack profile/name/email/username -> slack_pending_memory
```

Slack ingestion automatically scans pending rows and promotes any row that the
identity resolver can now map. This happens near the start of `run_once()` and
again after a successful Slack extraction/write attempt.

Manual helpers are also available:

```python
from argos_src.memory.slack.pending import (
    promote_pending_slack_memory,
    promote_resolved_pending_slack_memory,
)

promote_pending_slack_memory(
    memory_store,
    slack_user_id="U123",
    person_id="person_example",
)

promote_resolved_pending_slack_memory(
    memory_store,
    identity_resolver=resolver,
)
```

## Agent Visibility

Slack memories are stored in the same `MemoryStore` as live chat and encounter
memory.

The agent does not directly read Slack. It sees Slack-derived memories only
after they are stored as active `memory_items` and selected by the normal memory
context compiler for a future interaction.

Person-scoped pending Slack memory is not prompt-visible until it is promoted to
a real Argos `person_id`.

## Slack Scopes

For public channels:

```text
channels:read
channels:history
users:read
```

For private channels:

```text
groups:read
groups:history
users:read
```

Optional, only if you want email-based identity matching:

```text
users:read.email
```
