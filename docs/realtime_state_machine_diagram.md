# Realtime State Machine Diagram

This is a visual companion to `realtime_state_model.md` and
`realtime_turn_flow.md`. It is intentionally Mermaid-first so the diagram can be
reviewed, patched, and diffed alongside code changes.

## Why Argos Uses State Axes

Argos is not one monolithic finite-state machine. It runs one persistent
Realtime session while local controllers track several mostly independent
state axes:

- `session`: websocket readiness and shutdown.
- `capture`: local mic admission, VAD, recording, and audio commit.
- `transcription`: committed-audio transcription side channel.
- `turn`: model response lifecycle for one audio or text turn.
- `playback`: local audio buffering, output, drain, and truncation.
- `engagement`: social interaction mode for patrol suppression and passive
  listening.
- `robot_arbitration`: navigation, patrol, battery, owner-turn, and motion
  policy.
- `coalescer`: robot/internal-event batching and deduplication.

This structure keeps the runtime honest. A user can be in `engagement=speaking`
while `turn=model_done` and `playback=awaiting_drain`; forcing that into one
giant enum would either lose information or create an explosion of combined
states.

Primary source files:

- `argos_src/agent/control/types.py` defines the stable axis and state names.
- `argos_src/agent/control/reducers/engagement.py` defines pure engagement
  transition decisions and declarative actions.
- `argos_src/agent/control/reducers/coalescing.py` defines pure internal-event
  dedup/render rules.
- `argos_src/agent/control/engagement_runtime.py`,
  `audio_runtime.py`, `state_runtime.py`, `server_event_runtime.py`,
  `playback_runtime.py`, `event_adapter.py`, and `robot_arbitration.py` apply
  transitions, side effects, and structured logs.
- `argos_src/observability/state_observer.py` writes `component=state`
  transition and ignored-event rows.

One nuance: `coalescer` is an observable state axis for transition/ignored
events, but it is not currently a rich enum like `CaptureState`, `TurnState`,
or `PlaybackState`.

## Runtime Map

```mermaid
flowchart TD
    start["RealtimeRobotAgent.start"] --> connect["websocket connect"]
    connect --> update["session.update"]
    update --> ready["session.updated / session ready"]
    ready --> source{"input source"}

    source --> mic["microphone chunks"]
    source --> internal["face/nav/battery/patrol events"]

    mic --> capture["capture axis"]
    internal --> coalescer["coalescer axis"]
    coalescer -.-> capture

    capture --> turn["turn axis"]
    coalescer --> turn
    turn --> realtime["response.create / Realtime stream"]
    realtime --> tools{"function calls?"}
    tools -->|"yes"| localTools["local tool execution"]
    localTools --> realtime
    realtime --> playback["playback axis"]
    playback --> engagement["engagement axis"]
    engagement --> arbitration["robot_arbitration axis"]
```

## Human Audio Turn

```mermaid
flowchart TD
    mic["microphone chunks"] --> gate["local VAD + wake word + face/attention admission"]
    gate -->|"allowed + voice"| rec["capture: recording"]
    gate -->|"blocked"| closed["capture: admission_closed"]
    rec -->|"silence grace"| finalizing["capture: finalizing"]
    finalizing --> commit["input_audio_buffer.commit"]
    commit --> committed["capture: committed"]
    committed --> queue["turn: committed -> queued"]
    queue --> history["turn: preparing_history"]
    history --> pendingEvents{"pending internal events?"}
    pendingEvents -->|"yes"| attach["append [PENDING EVENTS] system item"]
    pendingEvents -->|"no"| responseCreate["response.create"]
    attach --> responseCreate
    responseCreate --> waiting["turn: response_requested -> waiting_first_output"]
    waiting --> firstAudio{"first output audio?"}
    firstAudio -->|"yes"| playing["turn: playing"]
    firstAudio -->|"no, no-audio retry"| followup["turn: requesting_followup"]
    followup --> responseCreate
    playing --> pb["playback: buffering -> playing"]
    pb --> modelDone["turn: model_done"]
    modelDone --> drain["playback: awaiting_drain"]
    drain --> finalized["turn: finalized"]

    committed -.-> engaged["engagement: idle/alert/cooldown -> engaged"]
    playing -.-> speaking["engagement: engaged/alert -> speaking"]
    drain -.-> cooldown["engagement: speaking -> cooldown"]
    cooldown -.-> idle["engagement: cooldown -> idle"]
```

## Internal Robot Event Turn

```mermaid
flowchart TD
    sensors["face/nav/battery/patrol bridge"] --> submit["EventCoalescer.submit"]
    submit --> suppress{"patrol suppressed by engagement?"}
    suppress -->|"yes"| ignored["coalescer ignored: patrol_suppressed"]
    suppress -->|"no"| buffer["coalescer buffer"]
    buffer --> debounce["debounce/max-wait timer"]
    debounce --> recording{"recording active?"}
    recording -->|"yes, internal only"| defer["coalescer ignored: recording_active; restart timer"]
    recording -->|"no"| dedup["dedup_events"]
    dedup --> render["render [INTERNAL EVENT] or [PENDING EVENTS]"]
    render --> enqueue["enqueue_internal_event"]
    enqueue --> textTurn["turn: queued text/internal turn"]
    textTurn --> item["local explicit-input system item"]
    item --> response["response.create(input=[selected items])"]
    response --> model["Realtime model audio/tools"]
```

## Engagement Reducer

```mermaid
stateDiagram-v2
    [*] --> idle
    idle --> alert: face_or_wake / stop + cancel nav
    idle --> engaged: human_input / stop + cancel nav
    alert --> engaged: human_input
    alert --> speaking: agent_output_started or agent_done(has_reply)
    alert --> idle: alert_timeout / force flush + idle callback
    engaged --> speaking: agent_output_started or agent_done(has_reply)
    engaged --> cooldown: agent_done(no reply) or playback_fallback
    speaking --> cooldown: playback_terminal or playback_fallback
    cooldown --> engaged: human_input / stop + cancel nav
    cooldown --> idle: cooldown_timeout / idle callback
```

The reducer returns a decision: old state, new state, reason, and declarative
actions. The runtime wrapper performs the actions, such as publishing a local
`stop` voice command, canceling interruptible navigation, force-flushing the
coalescer, or notifying patrol-resume logic after idle entry.

## Turn And Playback Branches

```mermaid
flowchart TD
    responseRequested["turn: response_requested"] --> waiting["turn: waiting_first_output"]
    waiting --> audioDelta["response.output_audio.delta"]
    audioDelta --> turnPlaying["turn: playing"]
    audioDelta --> playbackBuffering["playback: buffering"]
    playbackBuffering --> playbackPlaying["playback: playing"]
    waiting --> func["function_call.done"]
    func --> tools["turn: waiting_tools"]
    tools --> toolResults["function_call_output items"]
    toolResults --> followup["turn: requesting_followup"]
    followup --> responseRequested
    waiting --> noAudio["response.done without audio"]
    noAudio --> retry["turn: requesting_followup"]
    retry --> responseRequested
    turnPlaying --> done["response.done"]
    done --> modelDone["turn: model_done"]
    modelDone --> awaitingDrain["playback: awaiting_drain"]
    awaitingDrain --> completed["playback: completed"]
    completed --> finalized["turn: finalized"]
    turnPlaying --> interrupt["wake/stop interruption"]
    interrupt --> truncated["playback: stopped_truncated"]
    truncated --> canceled["turn: canceled"]
```

## Observability

Each state move should emit:

```text
component=state event=transition axis=<axis> old_state=<old> new_state=<new>
trigger=<trigger> req_id=<req_id> stream_id=<stream_id>
```

Ignored events should emit:

```text
component=state event=ignored axis=<axis> trigger=<trigger>
ignored_reason=<reason>
```

That is why state names are treated as dashboard-stable API. Tests such as
`tests/argos_src/agent/control/test_state_axes.py`,
`test_engagement_reducer.py`, and `test_coalescing_reducer.py` protect this
shape.

## Keeping This Diagram Fresh

Update this file when a change alters any of these:

- A value in `StateAxis`, `CaptureState`, `TurnState`, `PlaybackState`,
  `EngagementMode`, `TranscriptionState`, or `RobotArbitrationState`.
- A reducer transition, reducer action, or coalescing rule.
- The order of audio commit, response creation, server-event binding, playback
  completion, tool follow-up, interruption, or watchdog recovery.
- The structured state observer contract consumed by logs or dashboards.

Suggested subagent prompt:

```text
Use the state-machine-diagrammer custom agent.

Scope:
- Check the current diff plus docs/realtime_state_machine_diagram.md.
- Read docs/realtime_state_model.md and docs/realtime_turn_flow.md.
- Read touched files under argos_src/agent/control/ and argos_src/runtime/
  that affect state axes, reducers, turn flow, playback, coalescing, or
  robot arbitration.

Task:
- Patch docs/realtime_state_machine_diagram.md if the diagram or source file
  references drift.
- Preserve public state names unless the main task explicitly changed them.
- Report changed paths, evidence checked, and any remaining uncertainty.
```
