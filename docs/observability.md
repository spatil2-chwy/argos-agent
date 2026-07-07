# Observability

The realtime Argos runtime writes latency logs to:

`logs/latency.log`

You can override that with:

- `GO2_LATENCY_LOG_PATH`
- `GO2_LATENCY_CONSOLE=0` to suppress stdout mirroring

## Current Logging Model

The primary structured-log components are:

- `realtime`
- `tool`
- `action`
- `state`

Cost visibility is folded into those same structured events rather than written to a
separate billing log.

## Log Format

Each line is pipe-separated `key=value` fields:

```text
ts=<timestamp> | component=<name> | event=<name> | ...
ts=<timestamp> | component=<name> | metric=<name> | duration_s=<seconds> | ...
```

`event=` lines are point-in-time markers.

`metric=` lines are duration measurements.

## Core Realtime Turn Markers

Typical human turn flow:

| component | event / metric | meaning |
|---|---|---|
| `realtime` | `event=recording_started` | local admission opened, an `exchange_id` was assigned, and the runtime started buffering speech |
| `realtime` | `event=speech_end` | local end-of-speech detection fired |
| `realtime` | `event=audio_commit` | mic audio was committed into the Realtime session |
| `realtime` | `event=exchange_context` | face, speaker, owner, and pending internal-event context captured for the exchange |
| `realtime` | `event=response_create` | the model response was explicitly requested |
| `realtime` | `metric=first_audio_latency_s` | speech end to first playback audio delta |
| `realtime` | `event=tool_call_requested` | the model requested a tool call |
| `tool` | `event=tool_result` | a tool call finished and returned a result to the session |
| `realtime` | `event=response_done` | the Realtime response stream reached `response.done` |
| `realtime` | `event=transcription_usage` | input transcription token usage and estimated transcription cost for the turn |
| `realtime` | `event=response_usage` | final token/caching usage from `response.done`, including modality token counts and estimated response cost |
| `realtime` | `event=playback_completed` | local speaker playback drained for the exchange |
| `realtime` | `event=exchange_complete` | the exchange reached a clean terminal state |

This is the main felt-latency path now.

Operator-facing dashboard views treat a human exchange as:

```text
recording_started -> speech_end -> audio_commit -> response_create
-> first_audio_latency_s -> optional tools -> response_done
-> playback_completed -> exchange_complete
```

The stable join keys are:

- `run_id`: local Argos process run shown as an operator session.
- `exchange_id`: one human speech input through one robot/model response.
- `exchange_index`: exchange number within the local run.
- `req_id`: Realtime request/turn id, kept for diagnostics.
- `openai_session_id`: raw OpenAI Realtime session id, kept for diagnostics.

Identity and context fields such as `primary_face_person_id`,
`audio_speaker_id`, `owner_id`, `owner_source`, `owner_confidence`,
`speaker_visible`, `trigger`, and `admission_reason` are logged with the
exchange so the dashboard can show why the mic opened and who the model was
responding to.

## State Transition Markers

The realtime control plane also emits structured state events:

| component | event | meaning |
|---|---|---|
| `state` | `event=transition` | One control-plane state axis moved from `old_state` to `new_state`. |
| `state` | `event=ignored` | A trigger was intentionally ignored, with `ignored_reason`. |

Typical fields are:

- `axis`
- `old_state`
- `new_state`
- `trigger`
- `req_id`
- `stream_id`
- `ignored_reason`

These records are meant for debugging and future evaluation dashboards. They
complement the latency markers rather than replacing them.

Robot arbitration transitions currently cover patrol resume suppression/allow
decisions and proactive face-attention suppression/allow decisions. Common
states include `patrol_allowed`, `patrol_suppressed`,
`battery_low_blocking`, `face_attention_allowed`, and
`face_attention_suppressed`.

## Tool Timing

Tool-related signals still matter:

| component | event / metric | meaning |
|---|---|---|
| `action` | `metric=tool_dispatch_s` | speech end to first robot command dispatch for action-style tools |
| `realtime` | `event=tool_call_requested` | model requested a tool call, including the tool name and call id |
| `tool` | `event=tool_result` | a tool call finished and returned a result to the session |
| `tool` | `event=memory_query_start` | a Tailwag memory query tool started; logs tool name, query kind, and person id but not memory text |
| `tool` | `metric=memory_query_s` | Tailwag memory query duration and result count |

## Memory Ingestion

Tailwag memory ingestion runs asynchronously from completed attributed turns and
should not block speech output. Argos logs scheduling and provider failures
through the normal runtime logger. The Tailwag `record_episode(...,
extract_memory=...)` result is treated as Tailwag-owned; Argos does not emit
structured latency fields for Tailwag memory extraction outcomes.

## Example

```text
ts=2026-04-24 13:23:45.100 | component=realtime | event=recording_started
ts=2026-04-24 13:23:47.100 | component=realtime | event=speech_end
ts=2026-04-24 13:23:47.149 | component=realtime | event=audio_commit | req_id=rt-abc123
ts=2026-04-24 13:23:47.150 | component=realtime | event=response_create | req_id=rt-abc123
ts=2026-04-24 13:23:47.220 | component=realtime | event=transcription_usage | req_id=rt-abc123 | model=gpt-4o-mini-transcribe | estimated_cost_usd=0.00025000 | session_total_cost_usd=0.00025000
ts=2026-04-24 13:23:47.492 | component=realtime | metric=first_audio_latency_s | duration_s=0.392 | req_id=rt-abc123
ts=2026-04-24 13:23:48.002 | component=realtime | event=response_usage | req_id=rt-abc123 | input_tokens=1800 | cached_tokens=1320 | uncached_input_tokens=480 | cache_hit_ratio=0.733 | estimated_cost_usd=0.01492000 | session_total_cost_usd=0.01517000
ts=2026-04-24 13:23:48.010 | component=tool | event=tool_result | tool=capture_scene | req_id=rt-abc123
```

## Provider Events

The runtime receives face-presence updates through the configured robot provider
and mirrors the latest snapshot into local mic-admission state. Inspect provider
events or Argos logs when debugging face-triggered interaction.

## CLI Helpers

Live tail:

```bash
python3 -m argos_src.observability.latency_tail --follow
python3 -m argos_src.observability.latency_tail --follow --component realtime
```

Aggregate summary:

```bash
python3 -m argos_src.observability.latency_report
python3 -m argos_src.observability.state_report
```

## Dashboard

The repo now includes a long-term dashboard shell under `dashboard/`:

- FastAPI API and static host:
  `argos_src.observability.dashboard_server:app`
- log indexing and session/interaction aggregation:
  `argos_src.observability.dashboard_data`
- Vite React frontend:
  `dashboard/`

Run the API locally:

```bash
source setup_shell.sh
uvicorn argos_src.observability.dashboard_server:app --host 127.0.0.1 --port 8765 --reload
```

Run the frontend locally:

```bash
cd dashboard
npm install
npm run dev
```

Open `http://127.0.0.1:5173` during development. After `npm run build`, the
FastAPI app serves the built dashboard from `http://127.0.0.1:8765`.

The API reads `logs/latency.log` by default. Override with
`ARGOS_DASHBOARD_LOG_PATH=/path/to/latency.log`.

### Viewing A Jetson Dashboard From Your PC

Preferred: use an SSH tunnel so the dashboard is not exposed to the whole
network.

On the Jetson:

```bash
cd ~/argos-agent
source setup_shell.sh
cd dashboard
npm install
npm run build
cd ..
uvicorn argos_src.observability.dashboard_server:app --host 127.0.0.1 --port 8765
```

On your PC:

```bash
ssh -L 8765:127.0.0.1:8765 USER@JETSON_HOST
```

Then open `http://127.0.0.1:8765` on your PC.

Less preferred: bind the dashboard to the Jetson network interface and open it
directly from the PC:

```bash
uvicorn argos_src.observability.dashboard_server:app --host 0.0.0.0 --port 8765
```

Then open `http://JETSON_HOST:8765`. Use this only on a trusted network because
the dashboard is an operator tool and does not currently include authentication.

The main endpoint is:

```text
GET /api/snapshot
```

It returns:

- session summaries keyed by local `run_id` when available
- per-exchange lifecycle timelines keyed by `exchange_id`
- context for trigger, mic admission, primary face, speaker, and resolved owner
- state transitions, ignored state events, and raw rows as diagnostics
- latency metrics such as `first_audio_latency_s`
- tool usage, cost fields, component counts, and detected error markers

## What To Look For

If the robot feels slow, the main questions are:

1. Is `first_audio_latency_s` high?
2. Are tool results coming back late?
3. Is the runtime not reaching `response_create` quickly after `audio_commit`?
4. Is playback not starting even though the model turn completed?
5. Is `cached_tokens` staying near zero even after several similar turns in one session?
6. Is `session_total_cost_usd` climbing faster than expected because transcription or background extraction is dominating?

Those usually separate:

- audio gating issues
- network/model latency
- tool latency
- playback/device issues
