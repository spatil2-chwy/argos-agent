# Realtime State Model

Argos should be modeled as several orthogonal state axes, not one giant runtime
state. This document names those axes for tests, logs, and future dashboards.

## Instrumentation Status

The axes below are implemented as typed runtime states and structured
`component=state` rows. Session, capture, transcription, turn, playback,
engagement, robot arbitration, and coalescer transitions now have explicit
emitters. The only expected growth is additional trigger coverage as new robot
behaviors are added.

## Axes

| Axis | Meaning |
|---|---|
| `session` | Websocket/session readiness and shutdown lifecycle. |
| `capture` | Local ASR admission, VAD confirmation, recording, and audio commit. |
| `transcription` | OpenAI input transcription side channel for committed audio. |
| `turn` | Model response lifecycle for a single audio or text turn. |
| `playback` | Local TTS/audio buffering, output, drain, stop, and force-complete. |
| `engagement` | Social interaction mode used for patrol suppression and passive listening. |
| `robot_arbitration` | Navigation, patrol, battery, owner-turn, and motion safety policy. |
| `coalescer` | Internal event batching and deduplication. |

## Expected Flow

Typical human audio turn:

```text
capture: admission_closed -> admission_open -> candidate_voice -> recording
capture: recording -> finalizing -> committing -> committed
turn: committed -> queued -> preparing_history -> response_requested
turn: response_requested -> waiting_first_output -> playing
playback: idle -> buffering -> playing -> awaiting_drain -> completed
turn: playing -> finalized
engagement: idle/alert/cooldown -> engaged -> speaking -> cooldown -> idle
```

Tool turn branch:

```text
turn: waiting_first_output -> waiting_tools -> requesting_followup
turn: requesting_followup -> response_requested
first preamble: playback playing -> idle; engagement speaking -> engaged
terminal answer: playback buffering -> playing -> awaiting_drain -> completed
```

No-audio recovery branch:

```text
turn: waiting_first_output -> requesting_followup
turn: requesting_followup -> response_requested
```

The runtime must still terminate every turn as `finalized`, `canceled`, or
`superseded`.

## Arbitration Priority

Robot-facing policy should be resolved from state snapshots in this order:

1. Battery safety.
2. Explicit human or wake interaction.
3. Non-interruptible or focused navigation.
4. Active recording or playback.
5. Patrol resume.
6. Proactive face greeting.
7. Idle gestures and display.

## Observability Contract

State transitions should emit structured rows with:

```text
component=state event=transition axis=<axis> old_state=<old> new_state=<new>
trigger=<trigger> req_id=<req_id> stream_id=<stream_id>
```

Ignored events should emit:

```text
component=state event=ignored axis=<axis> trigger=<trigger> ignored_reason=<reason>
```

These logs intentionally complement, rather than replace, the latency markers in
`docs/observability.md`.
