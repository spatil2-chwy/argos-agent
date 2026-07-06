# Realtime Turn Flow and Engagement State

Read this with:

- `argos_src/agent/agent_runtime.py`
- `argos_src/agent/agent_audio.py`
- `argos_src/agent/agent_events/`
- `argos_src/agent/orchestrator.py`
- `argos_src/runtime/audio_admission.py`
- `argos_src/display/runtime.py`
- `argos_src/agent/factory.py`

This document explains the live control flow of the Argos realtime runtime:

- how the persistent OpenAI Realtime session is created and updated
- how audio turns differ from internal text turns
- how engagement state changes over time
- how interruptions, tool calls, and edge cases are handled

For the camera attention signal that can open passive mic admission, see
`docs/attention_gate.md`.

## Mental Model

Think of the runtime as one long-lived realtime session with three local control layers around it:

```text
robot sensors / nav / battery / face events
        -> EventCoalescer
        -> RealtimeRobotAgent.enqueue_internal_event(...)

microphone audio
        -> local VAD + wake word + admission policy
        -> input_audio_buffer.append / commit

Realtime API response stream
        -> local playback buffer
        -> engagement state machine
        -> tool execution loop
        -> optional display update worker
```

The Realtime API is stateful, but the robot still owns turn boundaries, audio
admission, engagement state, display state, interruption, and history trimming.

## Main Components

| Component | Responsibility |
|---|---|
| `RealtimeRobotAgent` | Owns the websocket, mic capture, playback, turn queue, tool queue, and history bookkeeping. |
| `RealtimeAgentAudioMixin` | Owns audio stream setup, local admission-driven capture, audio commit, and playback progress callbacks. |
| `agent/agent_events/` | Shared parsing and dispatch helpers for OpenAI Realtime server payloads. |
| `DisplayRuntime` | Optional interaction display facade for Puffle's screen. |
| `EngagementStateMachine` | Tracks `idle -> alert -> engaged -> speaking -> cooldown` and decides when patrol or navigation should be suppressed or resumed. |
| `EventCoalescer` | Debounces rapid internal events, merges them, and flushes them into turns. Human-triggered text flushes immediately. |
| `FacePresenceGate` | Lightweight local cache of face-presence snapshots for audio admission. |
| `FaceEventBridge` | Polls face recognition, publishes presence, and emits proactive face events only when the robot is truly idle. |
| `PatrolLoopBridge` | Schedules the next patrol hop after successful navigation. |
| `NavigationState` / `BatteryStateCache` | Feed interaction context and internal events into the runtime. |

## Realtime Session Lifecycle

The runtime opens exactly one websocket session in `RealtimeRobotAgent.start()`:

```text
create_connection("wss://api.openai.com/v1/realtime?model=...")
  -> server sends `session.created`
  -> client sends `session.update`
  -> server sends `session.updated`
  -> runtime marks `_session_ready`
```

`start()` sends `session.update` and then starts audio streams and worker loops.
The mic callback checks `_session_ready` before it records or sends audio. The
response worker is already running, so startup/internal events should not be
queued before the session is ready unless the caller intentionally handles that
startup race.

`session.update` is where the runtime installs the long-lived session configuration:

- static system prompt (`instructions`)
- voice
- nested input/output audio formats
- `output_modalities=["audio"]`
- `audio.input.turn_detection=None`
- tool schemas
- optional transcription config
- optional input noise reduction

Important detail: local code owns turn detection. The runtime does not ask the Realtime API to auto-segment speech. It explicitly sends `input_audio_buffer.commit` when local end-of-speech fires.

## Where `session.update` vs `response.create` Fit

The split is:

- `session.update`: stable session-level setup
- `response.create`: turn-level trigger plus the effective turn prompt

The effective turn prompt starts with the same static persona/policy prompt and then appends the latest dynamic context blocks for that turn. This preserves a stable prefix for prompt caching while ensuring only the current dynamic block is present.

So the model is not re-created every turn. The same session keeps accumulating history until the runtime trims it.

## Two Turn Families

There are really two inputs into the same model session.

### 1. External Human Speech

```text
audio input -> audio output
```

The user speaks through the microphone. The runtime buffers raw PCM into the Realtime input audio buffer, commits the buffer, then explicitly asks for a response.

### 2. Internal Robot Events

```text
text input -> audio output
```

Face events, battery events, patrol events, and navigation events are turned into short text payloads and inserted into the same conversation as system-role messages before `response.create`.

## What Is an "Item"?

In the Realtime API, an item is one conversation object, not one human-assistant exchange.

Examples of one item:

- one spoken human message
- one assistant reply
- one internal system message
- one function call
- one function-call output

Argos no longer keeps a fixed item-count tail. Realtime history is scoped to the
current resolved owner, and older conversation items are deleted on owner
handoff.

## Human Audio Turn: Exact Flow

### Step 1: Mic callback runs continuously

`_capture_callback()` receives PCM from `sounddevice`.

For every chunk it:

1. Resamples the chunk to 16 kHz.
2. Runs local VAD.
3. Runs local wake-word detection.
4. Reads the current interaction snapshot from `EngagementStateMachine`.
5. Evaluates `resolve_record_admission(...)`.
6. Keeps a short local pre-roll buffer while not actively recording.

### Step 2: Local admission decides whether recording is open

Admission uses:

- face presence
- attention presence, when enabled by profile
- current engagement state
- wake-window state
- whether the robot is already speaking
- whether navigation is active and interruptible

This is why the runtime can behave differently in `idle`, `alert`, `cooldown`, or focused navigation even before the model sees anything.

Attention and face presence are checked only when a recording is not already
active. Once recording starts, admission closing does not stop the capture; local
VAD and `silence_grace_period` decide when the active audio turn ends.

### Step 3: Recording starts locally

When admission is open and voice is detected:

- `_start_recording_locked()` marks recording active
- the optional interaction display moves to the `think` face with `Recording...`
- the runtime sends `input_audio_buffer.clear`
- raw PCM chunks start flowing into `_audio_send_queue`
- a small pre-roll window is prepended so the first syllable is less likely to be clipped
- passive starts require two consecutive VAD-positive chunks; wake-word starts open immediately
- `_audio_sender_loop()` converts each chunk to base64 and sends `input_audio_buffer.append`

Recording gesture updates are dispatched through a serial background worker.
That keeps slow robot/bridge gesture calls out of the realtime audio callback
and out of the `speech_end -> commit` path.

### Step 4: Local end-of-speech commits the turn

When voice has been absent for `silence_grace_period`:

- `_finalize_recording_locked()` ends the local capture
- the optional interaction display shows the centered `Thinking...` message
- `_commit_audio_turn()` waits for `_audio_send_queue` to drain
- the runtime sends `input_audio_buffer.commit`
- a new `QueuedTurn(kind="audio")` is created

OpenAI input transcription is attached to the committed audio item. It is used
for observability, memory, and preference extraction, but it is not what opens,
closes, or commits the local recording.

### Step 5: Pending internal events may be folded into the same turn

Right after commit, `_commit_audio_turn()` calls:

- `coalescer.drain_internal_events_for_audio_turn(...)`

If internal events were waiting, they are packed into `turn.pending_internal_text`.

That means one audio turn can become:

```text
human speech audio
+ optional "[PENDING EVENTS]" text block
+ one `response.create`
```

So the model can answer the human while still seeing the latest robot-side events.

### Step 6: Engagement moves to `ENGAGED`

For committed human audio turns the runtime calls:

- `engagement.on_human_input(req_id)`

This is the main handoff from speech capture into interaction state.

### Step 7: Response worker runs the turn

`_response_loop()` pulls the turn and `_run_turn()` does:

1. Register the turn as active.
2. If this were a text turn, create a system message item.
3. If `pending_internal_text` exists, create a system message item for it.
4. Send `response.create`.
5. Wait for both model completion and playback completion.

Important nuance: for audio turns, the human speech itself is not wrapped in a local `conversation.item.create`. The server creates the audio-backed user item after `input_audio_buffer.commit`.

### Step 8: Server events bind the realtime objects back to the turn

The runtime learns the exact Realtime object ids incrementally:

- `conversation.item.created`
- `conversation.item.input_audio_transcription.completed`
- `response.created`
- `response.output_audio.delta`
- `response.output_audio_transcript.delta`
- `response.output_item.done`
- `response.done`

The binding rules are:

- audio user items are matched using `_pending_audio_turn_req_ids`
- locally created text items are matched using `_pending_local_created_items`
- assistant items and function calls are matched using `response_id`

This is the key bookkeeping that keeps transcripts, playback, tool calls, and history ownership aligned even with overlap and interruption.

### Step 9: First audio delta flips the turn into playback

On the first `response.output_audio.delta`:

- the turn gets `audio_started=True`
- phase becomes `playing`
- `first_audio_latency_s` is logged
- engagement gets `on_agent_output_started(...)`
- engagement also gets `on_playback_event("playback_started", ...)`
- the optional interaction display moves to the `excited` face

Assistant transcript deltas stream as response subtitles. Recording uses a fixed
status subtitle, while the post-recording thinking phase uses a centered message
instead of a face.

### Step 10: Completion waits for both response and playback

The runtime treats "model finished generating" and "human finished hearing it" as different things.

- `response.done` means generation finished
- playback completion means the local speaker drained the buffered audio

Only after both are satisfied does the turn finalize cleanly.

## Internal Event Turn: Exact Flow

Internal events originate from bridges and watchdogs:

- `FACE_EVENT`
- `NAV_EVENT`
- `PATROL_EVENT`
- `BATTERY_EVENT`
- `BATTERY_LOW_EVENT`

Flow:

```text
bridge/watchdog
  -> EventCoalescer.submit(...)
  -> debounce / dedup
  -> RealtimeRobotAgent.enqueue_internal_event(...)
  -> QueuedTurn(kind="text", source_is_internal=True)
  -> conversation.item.create(system input_text)
  -> response.create
  -> model audio reply
```

The coalescer can emit:

- `[INTERNAL EVENT]` for one event
- `[PENDING EVENTS]` for multiple internal events
- `[HUMAN INPUT]` if it ever flushes a mixed text batch

In the current Argos runtime, the common text-only turns are internal robot events, not typed human chat.

## Event Coalescer Rules

The coalescer is intentionally opinionated:

- human text flushes immediately
- internal events are debounced
- internal-only flushes are deferred while recording is active, so they can be drained into the audio turn
- repeated face events for the same person collapse to the newest one
- nav waypoint chatter is dropped if a final goal result is already present
- patrol-resume events are suppressed if a face event or human input is in the same batch
- patrol is suppressed whenever engagement is not `IDLE`

This keeps the model from seeing a noisy stream of low-value internal chatter.

## Engagement State Machine

The engagement states are:

| State | Meaning | Typical entry |
|---|---|---|
| `idle` | No active interaction. Patrol may proceed. | Startup, cooldown timeout. |
| `alert` | A proactive interaction has claimed attention but no human turn is committed yet. | Face event while idle. |
| `engaged` | A human turn has been committed and the robot is waiting to answer. | `on_human_input(...)`. |
| `speaking` | The robot is actively outputting or awaiting the terminal playback event for a spoken answer. | First audio delta or playback start. |
| `cooldown` | The reply just ended; patrol and proactive greetings stay suppressed briefly while the display returns to idle. | Playback completed/stopped or text-only no-reply completion. |

### State Transitions in Practice

```text
IDLE
  -> ALERT      (proactive face event)
  -> ENGAGED    (committed human turn)

ALERT
  -> ENGAGED    (human speaks)
  -> SPEAKING   (agent proactively replies)
  -> IDLE       (alert timeout)

ENGAGED
  -> SPEAKING   (first model audio / playback start)
  -> COOLDOWN   (agent finished without spoken reply)

SPEAKING
  -> COOLDOWN   (playback completed or stopped)

COOLDOWN
  -> ENGAGED    (follow-up human input)
  -> IDLE       (cooldown timeout)
```

### What Actually Triggers the Transitions

- `on_face_or_wake()`
  Moves `IDLE -> ALERT` for proactive face attention. If already speaking and a wake word is heard, it is used together with interruption logic.
- `on_human_input(req_id)`
  Moves `IDLE` / `ALERT` / `COOLDOWN -> ENGAGED`.
- `on_agent_output_started(req_id, stream_id)`
  Moves `ALERT` / `ENGAGED -> SPEAKING`.
- `on_agent_done(has_reply=True, req_id)`
  Arms the machine to wait for a terminal playback event.
- `on_playback_event("playback_completed" | "playback_stopped", ...)`
  Moves `SPEAKING -> COOLDOWN`.

### Timeouts

The watchdog in `EngagementStateMachine` handles:

- `ALERT -> IDLE` after `alert_timeout_sec`
- `COOLDOWN -> IDLE` after `cooldown_sec`
- `SPEAKING/ENGAGED -> COOLDOWN` fallback after `speaking_timeout_sec` if playback terminal signals never arrive

When `ALERT` times out, the coalescer is force-flushed. That keeps a proactive event from getting stuck forever without ever reaching the model.

## Runtime Watchdogs

The engagement watchdog handles user-visible interaction states. The agent also
has turn watchdogs so one stuck response does not hold the runtime forever:

- response creation / first-audio stalls terminate the turn with `response_timeout`
- tool waits terminate the turn with `tool_timeout`
- local playback stalls after model completion force playback completion

These watchdogs are local recovery paths, not Realtime API features.

## Navigation and Patrol Interaction

The engagement machine also decides whether navigation should be interrupted or suppressed.

- proactive face attention can publish a local `stop` voice command and cancel interruptible navigation
- new human input from `IDLE` or `COOLDOWN` can also cancel interruptible navigation
- patrol resumes only after the runtime returns to `IDLE`
- a resolved-owner audio turn can request a short owner-turn motion toward the speaker; tool calls cancel that request

So "engagement state" is not just UX state. It is also a local arbitration layer between conversation and navigation.

## Interruption Path

Interruption is owned locally, not by the model.

Common triggers:

- wake word while the robot is already speaking
- external `/voice_commands` message with `stop`

The audio callback also has playback/echo guards so assistant audio does not
become a new human turn. Wake-word barge-in is the intentional exception.

`interrupt_current_response(...)` calls `_terminate_turn(...)` with:

- `response.cancel`
- local playback clear
- `conversation.item.truncate` using the actual played audio duration

That truncate step is important: it keeps the server-side conversation aligned with what the human truly heard, not what the model had queued.

## Important Edge Cases

### New human turn before the previous answer starts

If the old turn has not produced audio yet, `_supersede_unanswered_turn()` cancels it and lets the new human turn win.

### Model completed without audio

If `response.done` arrives with no audio reply:

- the runtime deletes the silent assistant item
- retries `response.create` once with a delivery hint
- cancels the turn if it happens again

### Model reply was cut off mid-sentence

If the response was incomplete but some audio already played and the transcript looks truncated, the runtime issues one continuation `response.create`.

### Playback stalls locally

If the model is finished but speaker playback stops making progress, the runtime force-completes playback instead of leaving the engagement state stuck forever.

## Current Quirks and Cleanup Opportunities

### History is scoped by resolved owner

The runtime keeps context for consecutive turns from the same resolved
`owner_id`.

When the owner changes, older Realtime conversation items are deleted before
the new response. Anonymous turns share an `anonymous` history until a known
owner appears.

### Audio turns that carry pending internal text are still treated as external turns

That is mostly correct, but it means a mixed audio turn can still feed preference extraction because `source_is_internal` stays `False`.

If you ever want "internal event piggybacked on a human turn" to be excluded from memory extraction, that would need an extra flag.

### `ALERT` is mainly a proactive-face state, not a generic wake-word state

Wake word in `idle` opens recording through admission logic, but it does not explicitly move the engagement machine into `ALERT`. That behavior is fine, but it is worth documenting because "wake word" and `ALERT` are easy to mentally conflate.
