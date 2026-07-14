# Prompting and Conversation History

Read this with:

- the prompt file selected by `realtime.prompt_file` in the active profile
- `argos_src/agent/agent_runtime.py`
- `argos_src/agent/runtime_context.py`
- `argos_src/agent/control/tool_runtime.py`
- `argos_src/agent/control/state_runtime.py`

This document explains what the realtime model actually sees on each turn:

- the static system prompt
- dynamic per-turn instructions
- explicit selected inference history
- tool call items and tool outputs
- transcript side channels used outside the prompt

## The Three Prompt Layers

Every response is shaped by three layers:

```text
1. Static session prompt
   -> sent once in `session.update`

2. Dynamic turn instructions
   -> sent on every `response.create`

3. Explicit selected inference history
   -> sent as `response.create.input` item references/raw items
```

Those layers are deliberately kept separate.

## Layer 1: Static System Prompt

The static system prompt path comes from the selected profile's
`realtime.prompt_file`. In the default `static_interaction` profile, it lives in:

- `resources/prompts/personality_test_prompt.md`

It is loaded by `factory.py` and sent in `RealtimeRobotAgent._configure_session()` as:

```json
{
  "type": "session.update",
  "session": {
    "instructions": "<base system prompt>",
    ...
  }
}
```

That prompt is the stable persona and policy layer:

- identity and speaking style
- response brevity
- event-handling conventions
- social memory behavior
- tool-use policy
- boundaries and truthfulness

It does not change turn to turn.

Important runtime detail:

- `response.create.instructions` is also sent on every turn.
- In practice, the runtime composes that field as:

```text
<static system prompt>

<dynamic turn blocks>
```

- This keeps the static persona/policy prompt at the front of every turn-level request while still appending fresh scene/runtime context after it.
- That ordering is intentional: prompt caching works best when the exact static prefix stays the same and variable context is appended later.

## Layer 2: Dynamic Turn Instructions

Every `response.create` carries fresh instructions from `_build_turn_instructions(turn)`, appended after the static system prompt inside the same `instructions` field.

The runtime currently builds these blocks:

- `[PERSON SPEAKING TO YOU]`, only when `owner_id` is resolved
- `[IDENTITY STATUS]`, only when `owner_id` is not resolved
- `[OTHER PEOPLE IN VIEW]`, only alongside a resolved owner and only as names/counts
- `[CURRENT TIME]`
- `[CURRENT OFFICE LOCATION]`
- `[OFFICE CONTEXT]`
- `[RECENT ENCOUNTERS]`
- `[ROBOT STATE]`
- battery prompt block
- `[SAVED LOCATIONS]`

`[PERSON SPEAKING TO YOU]` may include Tailwag's prompt-ready
`context_markdown` for the resolved turn owner. Argos keeps verified
identity/work lines under `Directory`, then pastes the Tailwag memory block
directly. The current Tailwag block is headed `[PERSON MEMORY]` and may include
subsections such as `Preferences`, `Pets`, `Facts`, `Potential Follow-Ups`, and
`Recent Episodes`.

If `owner_id` is not resolved, no person-specific prompt context is emitted,
even if a recognized person is visible. This avoids addressing a visible
bystander as the speaker when someone else talks off-camera or speaker
recognition is inconclusive. Unknown-owner turns also carry `[IDENTITY STATUS]`
to make the local resolver authoritative: the model must not use a person's name
or infer identity from voice similarity, face runner-up matches, or session
memory unless `[PERSON SPEAKING TO YOU]` is present.

Those lines come from the Tailwag identity-memory client. Tailwag writes
future-facing summaries so the realtime model can use them without seeing the
original conversation.

Two important design choices are hiding here.

### People context is frozen at turn creation

When a turn is created, the runtime captures a `FrozenTurnContext` snapshot:

- cached recognized people and face presence snapshot, used only after an owner
  is resolved for prompt context
- `primary_face_person_id`, when the speech-start scene has exactly one usable recognized face
- `audio_speaker_id`, when voice matching resolves a known speaker
- `owner_id`, the final person id used by memory and person-context lookup

That means owner/person context stays tied to the resolved turn owner. Visible
people who are not the owner are only listed as lightweight names under
`[OTHER PEOPLE IN VIEW]`.

### Robot state is partly regenerated at response time

When `response.create` happens, the runtime recomputes:

- current local time
- latest robot posture
- last tool summary
- latest battery prompt block
- saved locations

So the dynamic instructions are a hybrid:

- frozen human/social context
- live robot/runtime context

## Layer 3: Explicit Inference History

Argos keeps a local inference-history index and sends an explicit `input` list
with every `response.create`. The server default conversation may still contain
items for realtime lifecycle operations, but it is not the model-visible history
boundary.

The runtime adds or observes these item types in history:

| Item type | Who creates it | Purpose |
|---|---|---|
| audio user message | Realtime server after `input_audio_buffer.commit` | Represents spoken human input. |
| system message | Local runtime as an explicit-input item | Internal events and coalesced robot-side context. |
| text user message | Local runtime as an explicit-input item | Tool-artifact prompts and any future explicit text user input. |
| assistant message | Realtime model | Spoken/text response content. |
| function call | Realtime model | Tool invocation request. |
| function_call_output | Local runtime as an explicit-input item | Tool result returned to the model. |

This selected input list is the real history the model conditions on across turns.

## The Two Trigger Paths

The model is always triggered by `response.create`, but the lead-in differs by source.

### Internal Event Path

```text
internal robot event
  -> local system message item(s)
  -> response.create
  -> audio reply
```

### External Human Speech Path

```text
audio append/commit
  -> server-side audio user item
  -> optional pending internal system item
  -> response.create
  -> audio reply
```

So the model does not "auto-answer because audio arrived." The local runtime still explicitly asks for a response after the turn is ready.

## What Exactly Gets Sent on an Internal Event Turn

For a pure internal event turn, the send order is effectively:

```text
local explicit-input item
  role=system
  content=[{"type": "input_text", "text": "...FACE_EVENT / NAV_EVENT / ..."}]

response.create
  instructions="<dynamic blocks>"
  input=[<selected history>, <current system item>]
  output_modalities=["audio"]
```

If the coalescer merged several events, the text payload includes headers such as:

- `[INTERNAL EVENT]`
- `[PENDING EVENTS]`

Those headers are part of a system message item that the runtime adds to the
current response's explicit input list.

## What Exactly Gets Sent on a Human Audio Turn

For live human speech, the runtime sends:

```text
input_audio_buffer.clear           # when local recording starts
input_audio_buffer.append          # repeated for PCM chunks
input_audio_buffer.commit          # when local end-of-speech fires
local explicit-input item?         # only if pending internal text must be injected
response.create(input=[<selected history>, <current audio item>, ...])
```

The human speech itself is represented by the server-side audio user item created after commit.

If internal events were waiting at the same time, the runtime adds one extra system message item before `response.create`, usually with a `[PENDING EVENTS]` block.

## How Tool Calls Become History

Tool use is part of conversation history, not just local side effects.

### Step 1: The model emits a function call

The runtime receives:

- `response.function_call_arguments.delta`
- `response.function_call_arguments.done`

These events are assembled into a `PendingToolCall`.

### Step 2: Python executes the tool locally

`ToolRuntime.execute(...)` invokes the registered tool and updates:

- `_last_tool_name`
- `_last_tool_summary`
- optional robot posture

Those values then influence the next dynamic prompt block.

### Step 3: The tool result is inserted back into explicit input history

The runtime creates a local explicit-input item:

```text
local explicit-input item
  type=function_call_output
  call_id=<same call id>
  output=<stringified tool result>
```

If the tool produced images, the runtime also creates a synthetic user item:

```text
local explicit-input item
  role=user
  content=[
    {"type":"input_text","text":"[TOOL ARTIFACT] ..."},
    {"type":"input_image","image_url":"data:image/png;base64,..."},
    ...
  ]
```

That means useful tool data really is part of the model-visible history:

- the tool call item
- the tool output item
- optional visual artifacts

## Tool Barrier: Why One Human Turn Can Produce Multiple `response.create` Calls

A single human turn can contain:

1. an initial `response.create`
2. one or more model tool calls
3. local `function_call_output` items
4. a follow-up `response.create` after all tool calls finish

The runtime waits until `turn.pending_tool_calls == 0` before sending the follow-up response request.

This assumes the model called a registered tool from the installed schemas.
Unknown tool names are contract bugs: they can leave the turn waiting until the
runtime watchdog cancels it.

So one logical turn may look like:

```text
human/audio input
  -> response.create
  -> function_call
  -> function_call_output
  -> response.create
  -> final spoken answer
```

## What History Items Mean

An item is one Realtime conversation object.

Examples:

- one spoken user message
- one system message
- one assistant reply
- one function call
- one function-call output

Model-visible history is scoped by Argos' local inference index. Consecutive
turns from the same known `owner_id` reuse that owner's selected items, and a
resolved owner change selects a different local scope before the new response.

## How Conversation History Is Tracked

The runtime maintains several id maps so every Realtime item can be tied back to the right turn:

| Structure | Purpose |
|---|---|
| `_response_id_to_req_id` | Binds Realtime response ids to local turn ids. |
| `_item_id_to_req_id` | Binds conversation item ids to local turn ids. |
| `_call_id_to_req_id` | Binds tool call ids to local turn ids. |
| `_pending_audio_turn_req_ids` | Matches the next audio-created user item to the correct turn. |
| `_pending_local_created_items` | Matches server-acknowledged client-created items to the correct turn when that path is used. |

This matters because Realtime events arrive asynchronously and not always in the most convenient order.

Two subtle examples:

- audio transcription can arrive before the runtime has seen the final assistant completion
- function-call argument deltas arrive before the full tool payload is ready

The bookkeeping layer is what keeps that asynchronous stream coherent.

## How User and Assistant Transcripts Are Built

The runtime also keeps transcript side channels on the local `QueuedTurn`:

- `turn.user_transcript`
- `turn.assistant_transcript`

They are populated from:

- `conversation.item.input_audio_transcription.completed`
- `response.output_audio_transcript.delta`
- `response.output_text.delta`
- fallback extraction from `response.done`

These transcripts are used for:

- observability
- debugging
- Tailwag realtime episode ingestion

Episode ingestion records the conversation transcript for Tailwag, but Tailwag
semantic memory extraction is a separate opt-in live-turn setting.

They are not inserted as separate extra history items.

## What Is Not Stored in History

These are intentionally outside the conversation history:

- the static system prompt
- dynamic prompt blocks
- local wake-word decisions
- local VAD decisions
- engagement state transitions
- raw playback/audio buffers

This is one of the biggest simplifications of the rewrite: repeated situational state is re-sent as instructions, not permanently stuffed into history.

## Owner-Scoped Inference Selection

The runtime selects an explicit inference scope before any model response:

- `owner:<person_id>` for recognized owners
- `anonymous:<patch_id>` when no owner is safely resolved

Known-owner scopes are reusable within the agent run, so `owner:A -> owner:B ->
owner:A` can select A's prior permitted items again. Anonymous scopes are
contiguous patches, so `unknown -> unknown` keeps the same patch, while
`owner:A -> unknown -> owner:B -> unknown` creates two separate anonymous
patches.

Before `response.create`, Argos builds an explicit input list from local item
metadata:

```python
selected_item_ids = [
    item.id
    for item in realtime_items
    if item.scope_id == active_scope_id
    and item.status == "done"
    and item.permitted_for_inference
]
```

The current turn's required items are appended to that list. This explicit input
list is the model-visible context; the server default conversation is no longer
the owner privacy boundary. Local `QueuedTurn` transcripts remain available for
observability and Tailwag episode/preference extraction even when an item is not
selected for model inference.

## A Useful Way to Think About "What the Model Sees"

For an internal event turn, the model sees:

```text
base system prompt
+ dynamic blocks for right now
+ recent session history
+ newest internal-event system message
```

For a human audio turn, the model sees:

```text
base system prompt
+ dynamic blocks for right now
+ recent session history
+ newest spoken audio user item
+ optional piggybacked internal-event system item
```

For a tool-follow-up response inside the same turn, the model sees:

```text
base system prompt
+ refreshed dynamic blocks
+ original user turn
+ function call item(s)
+ function_call_output item(s)
+ optional tool artifact message(s)
```

## Current Quirks and Cleanup Opportunities

### Config layout

For Argos, the prompt file and the live model/session knobs now live in the same namespace:

- `realtime.prompt_file`
- `realtime.model`
- `realtime.voice`
- `realtime.max_output_tokens`

That keeps the static prompt source and the realtime session configuration together in one place.

### Transcript side channels are local-only

That is probably the right choice, but it means "what the model saw" and "what the runtime logged as transcript" are related, not identical. For debugging, that distinction is worth keeping in mind.
