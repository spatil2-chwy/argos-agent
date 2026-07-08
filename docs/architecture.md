# Argos Realtime Architecture

This is the high-level map of the current Argos stack. The deeper component docs are:

- `realtime_turn_flow.md`
- `realtime_control_refactor_plan.md`
- `prompting_and_history.md`
- `robot_tools.md`
- `attention_gate.md`
- `speaker_recognition.md`
- `face_recognition.md`
- `identity_memory.md`
- `observability.md`

## Core Idea

Argos now runs as one persistent OpenAI Realtime session with local control around it:

- local mic admission and end-of-speech detection
- local engagement state machine
- typed realtime state axes and structured transition logging
- local interruption and playback tracking
- local tool execution
- local event coalescing for face/nav/battery/patrol events

There is no separate ASR process and no separate TTS process in the supported path.

## Main Runtime Files

- `run_profile.py`
  Supported single-process launcher.
- `argos_src/agent/factory.py`
  Wires face runtime, nav state, battery cache, display runtime, tools, bridges, coalescer, and engagement state.
- `argos_src/agent/agent_runtime.py`
  Public composition root for the websocket session, queues, providers, and control modules.
- `argos_src/agent/control/audio_runtime.py`
  Owns audio stream setup, local capture admission, audio commit, and playback callbacks.
- `argos_src/agent/control/server_event_runtime.py`
  Applies Realtime server events to turn, transcription, playback, and tool state.
- `argos_src/agent/control/state_runtime.py`
  Owns the remaining history, response binding, and websocket send helper surface.
- `argos_src/agent/control/robot_arbitration.py`
  Centralizes patrol-resume and proactive face-attention allow/suppress decisions.
- `argos_src/agent/control/`
  Contains `EventCoalescer`, `EngagementStateMachine`, typed state axes,
  transition observer protocols, state stores, and pure reducers used to make
  the runtime control plane observable and testable.
- `argos_src/runtime/audio_admission.py`
  Pure local speech admission policy.
- `argos_src/display/runtime.py`
  Optional interaction-screen facade for faces, subtitles, and blocking review UI.
- `argos_src/agent/runtime_context.py`
  Builds dynamic per-turn instruction blocks.

## Repository Layout

- `argos_src/`
  Importable runtime source only.
- `config/profiles/`
  Scenario YAML profiles such as `static_interaction`.
- `resources/`
  Prompt files, wake-word ONNX models, and navigation-location JSON.
- `var/`
  Ignored local runtime state for identity, face, speaker, and other local
  runtime stores. Social/context memory lives in Tailwag.
- `scripts/labs/`
  Operator/lab tools that exercise runtime services without starting the agent.

## Human Context and Memory Path

The face/identity subsystem is not a sidecar prompt helper. It is part of the
realtime control loop:

```text
camera + optional depth
    -> FaceRecognitionService
    -> FacePresenceCache
    -> FaceEventBridge publish / proactive FACE_EVENT
    -> RealtimeRobotAgent turn snapshot + prompt context
    -> preference segment buffering
    -> TailwagMemoryProvider
    -> Tailwag episode + person context
```

Tailwag is the semantic memory writer and person-context source. Argos sends the
full active conversation episode to Tailwag from speaker-owned realtime turn
text. Prompt views still surface as `About` and `Potential Followups`, but the
storage and extraction contract lives outside Argos. See `identity_memory.md`
for the Tailwag-backed provider contract.

Important distinctions:

- `primary_face_person_id`
  Strict face-derived owner candidate. It is present only when exactly one
  depth-gated recognized face is usable.
- `audio_speaker_id`
  Voice-derived speaker match for the completed audio turn.
- `owner_id`
  The final person id for the turn after resolving voice and face evidence.

That means preference extraction is driven by `owner_id`, not by visual scene
order or proximity.

## Architecture Sketch

```text
face/nav/battery/patrol signals
    -> bridges / watchdogs
    -> EventCoalescer
    -> RealtimeRobotAgent text turns

microphone
    -> local VAD + wake word + admission
    -> Realtime input audio buffer
    -> explicit input_audio_buffer.commit
    -> RealtimeRobotAgent audio turns

Realtime model
    -> audio deltas / transcripts / function calls
    -> local playback buffer + tool loop
    -> EngagementStateMachine updates
    -> optional interaction display updates
```

## Boot Flow

1. `run_profile.py` loads `static_interaction`.
2. `factory.py` builds the robot-side runtime pieces.
3. If `display.enabled` is true and a display resource is selected, `interaction_display` is wired through `DisplayRuntime`.
4. `RealtimeRobotAgent.start()` opens the websocket, starts worker threads, and sends `session.update`.
5. After `session.updated`, `_session_ready` is set. Mic capture is gated on that flag; startup/internal event queues should still be treated as startup-sensitive because response workers are already running.

## Two Trigger Families

### Internal robot events

```text
text input -> audio output
```

Used for:

- face events
- navigation events
- patrol resume events
- battery events

### External human speech

```text
audio input -> audio output
```

Used for:

- wake-word turns from `idle`
- passive follow-up turns during `alert`
- attention-gated turns when someone is actively facing the robot

## Prompting Model

Every response combines:

1. static system prompt from `static_interaction_prompt.md`
2. dynamic instructions attached on `response.create`
3. rolling Realtime session history

The important split is:

- stable persona and policy live in `session.update`
- situational context lives in per-turn `response.create`
- conversation items and tool traces live in session history

## Conversation History Model

History includes:

- spoken user turns
- internal-event text turns
- assistant replies
- function call items
- function call outputs
- optional tool artifact messages with images

History does not include:

- static prompt
- dynamic prompt blocks
- raw VAD/wake/admission decisions

The runtime clears older Realtime conversation items on resolved owner handoff
while protecting active unresolved items.

## Tool Calling Model

The Realtime model calls the existing robot tools directly through function schemas.
Use `robot_tools.md` for the public tool IDs, capability requirements, side
effects, and safety notes.

Flow:

1. model emits function call
2. Python executes tool locally
3. runtime inserts `function_call_output`
4. runtime sends a follow-up `response.create` when all tool calls for that turn are done

## Engagement Model

The main states are:

- `idle`
- `alert`
- `engaged`
- `speaking`
- `cooldown`

That state machine does more than UI:

- suppresses patrol during interaction
- cancels interruptible navigation when needed
- controls passive-listening windows
- decides when patrol may resume

Display updates are derived from runtime state, not model tool calls. The
default Puffle display maps idle to `happy`, listening/recording/thinking to
`think`, and assistant playback to `excited`; assistant transcript deltas stream
as subtitles.

## Read Next

- Use `realtime_turn_flow.md` for exact event order, state transitions, interruptions, and edge cases.
- Use `prompting_and_history.md` for prompt layering, history assembly, tool traces, and transcript ownership.
- Use `robot_tools.md` for public tool IDs, provider capabilities, navigation/patrol/battery side effects, and live-robot safety notes.
- Use `speaker_recognition.md` for voice enrollment, audio ownership, strict face ownership, and voice-reference management.
- Use `interaction_display.md` for the Puffle browser display resource and enrollment review UI.
- Use `face_recognition.md` for face detection, identity assignment, proactive alerts, and preference extraction.
- Use `identity_memory.md` for Tailwag-backed person profiles, biometric references, episodes, semantic search, Slack ingestion, and reset boundaries.
