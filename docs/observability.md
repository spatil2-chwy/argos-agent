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
| `realtime` | `event=recording_started` | local admission opened and the runtime started buffering speech |
| `realtime` | `event=speech_end` | local end-of-speech detection fired |
| `realtime` | `event=audio_commit` | mic audio was committed into the Realtime session |
| `realtime` | `event=response_create` | the model response was explicitly requested |
| `realtime` | `metric=first_audio_latency_s` | speech end to first playback audio delta |
| `realtime` | `event=transcription_usage` | input transcription token usage and estimated transcription cost for the turn |
| `realtime` | `event=response_usage` | final token/caching usage from `response.done`, including modality token counts and estimated response cost |

This is the main felt-latency path now.

## Tool Timing

Tool-related signals still matter:

| component | event / metric | meaning |
|---|---|---|
| `action` | `metric=tool_dispatch_s` | speech end to first robot command dispatch for action-style tools |
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
```

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
