# Argos Realtime Architecture

This is the high-level map of the current Argos stack. The deeper component docs are:

- `realtime_turn_flow.md`
- `prompting_and_history.md`
- `voice.md`
- `speaker_recognition.md`
- `face_recognition.md`
- `memory_store.md`
- `observability.md`

## Core Idea

Argos now runs as one persistent OpenAI Realtime session with local control around it:

- local mic admission and end-of-speech detection
- local engagement state machine
- local interruption and playback tracking
- local tool execution
- local event coalescing for face/nav/battery/patrol events

There is no separate ASR process and no separate TTS process in the supported path.

## Main Runtime Files

- `run_profile.py`
  Supported single-process launcher.
- `argos_src/agent/factory.py`
  Wires face runtime, nav state, battery cache, tools, bridges, coalescer, and engagement state.
- `argos_src/agent/agent_runtime.py`
  Owns the websocket session, turn queue, playback, history bookkeeping, and tool loop.
- `argos_src/agent/agent_audio.py`
  Owns audio stream setup, local capture admission, audio commit, and playback callbacks.
- `argos_src/agent/orchestrator.py`
  Contains `EventCoalescer` and `EngagementStateMachine`.
- `argos_src/runtime/audio_admission.py`
  Pure local speech admission policy.
- `argos_src/agent/runtime_context.py`
  Builds dynamic per-turn instruction blocks.

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
    -> PreferenceExtractor
    -> MemoryStore
```

`PreferenceExtractor` is the semantic memory writer. It asks the LLM to compile
future prompt context from speaker-owned conversation segments and choose the
right memory `kind`. Prompt views are derived from that kind: durable person
kinds become `About`, while `followup` becomes `Potential Followups`. The local
writer keeps the database safe with structural checks, but it does not use
phrase-matching rules to decide whether a memory is semantically good. See
`memory_store.md` for the extraction contract.

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
```

## Boot Flow

1. `run_profile.py` loads `static_interaction`.
2. `factory.py` builds the robot-side runtime pieces.
3. `RealtimeRobotAgent.start()` opens the websocket and sends `session.update`.
4. After `session.updated`, mic capture and realtime turn handling become live.

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
- passive follow-up turns during `alert` / `cooldown`
- face-presence turns when someone is in front of the robot

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

## Read Next

- Use `realtime_turn_flow.md` for exact event order, state transitions, interruptions, and edge cases.
- Use `prompting_and_history.md` for prompt layering, history assembly, tool traces, and transcript ownership.
- Use `speaker_recognition.md` for voice enrollment, audio ownership, strict face ownership, and voice-reference management.
- Use `face_recognition.md` for face detection, identity assignment, proactive alerts, and preference extraction.
- Use `identity_store.md` for shared person records and face/speaker embedding-store management.
