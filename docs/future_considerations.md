# Future Considerations

This document records design context and possible follow-up work that should not
be mistaken for current runtime configuration. The canonical description of the
implemented turn lifecycle remains `realtime_turn_flow.md`.

## Tool-Response Speech Policy

### Why this became a problem

A navigation mission exposed several independent behaviors that combined into
one noisy interaction. The user asked the robot to go to a location, describe
what it saw, and report back briefly. Instead, the robot could speak a tool
preamble, an arrival message, a capture acknowledgement, a mission recap, and a
second navigation-completion message.

The main causes were:

- every tool continuation requested another audio-capable model response;
- blocking navigation returned its result through the tool chain and also
  published the same completion as a standalone `NAV_EVENT`;
- proactive face events could start another model turn with the prior mission's
  history and tools;
- assistant transcript text accumulated across response continuations instead
  of being owned by one `response_id`; and
- `capture_scene` captured an image locally, but its LangChain artifact was
  collapsed to text before the follow-up request, so the model received no
  `input_image` even though capture itself reported success.

Prompt rules such as "do not narrate navigation" and "answer in 6-10 words"
could reduce verbosity, but they could not enforce one audible answer while the
runtime independently created several audio responses.

### Current behavior

Argos now treats one human request and all of its tool continuations as one
logical spoken turn:

```text
human request
  -> first tool-bearing model response         # one brief preamble may play
  -> tool result(s) and optional image(s)      # retained for inference
  -> follow-up model response
  -> more tool calls, if needed                # intermediate audio suppressed
  -> terminal response with no tool calls      # released to playback
```

Audio and transcript deltas are buffered by `response_id` until
`response.done`. A response containing any function call is classified as
tool-bearing. The first tool-bearing response may contribute one short spoken
preamble. Every later tool-bearing response is suppressed, while all function
calls, tool outputs, and artifacts remain available to the model. A response
with no function calls is terminal and is released to the speaker.

The follow-up barrier is also response-local. It sends exactly one new
`response.create` only after:

- the source response has reached `response.done`;
- all expected call IDs have produced outputs; and
- no tool execution for the turn remains pending.

This works the same way for one tool, several parallel calls in one response,
and sequential chains where a follow-up response calls another tool. A single
tool can therefore produce "Okay, heading back," run the tool, then finish with
"I'm back." A chained inspection can acknowledge once, navigate, capture,
return silently, and then speak only the final description.

The preamble opportunity is consumed by the first tool-bearing response even if
that response contains no audio. Argos does not treat a later capture or return
response as the initial acknowledgement. It also does not manufacture a canned
preamble when the model did not produce one.

Related isolation rules prevent other turn sources from reopening the mission:

- blocking navigation completion is delivered only as a tool result;
- asynchronous user-facing navigation completion is delivered as a model event;
- patrol navigation completion is runtime-only; and
- standalone face turns are dropped during an active human turn and otherwise
  use fresh history with tools disabled.

For visual tools, content-and-artifact results preserve the artifact and attach
it as an `input_image`. Capture success therefore remains distinct from model
vision success, and both can be verified through usage and request diagnostics.

### Why response buffering is necessary

Realtime output is streamed. Audio deltas can arrive before Argos has received
the complete list of output items, so the runtime may not initially know whether
the response is a final answer or a preamble accompanying a function call.
`response.done` provides the complete response needed for that classification.

Buffering is a local playback decision; it does not change the Realtime API
tool-call protocol. Argos still receives the response, executes each function,
adds a `function_call_output` with the matching call ID, and requests the next
response. Only the decision to play intermediate PCM is different.

The spoken first preamble is included in inference history. Unheard assistant
text from later tool-bearing responses is kept for diagnostics but excluded from
future inference history. Function calls and outputs remain selected. This
keeps model history aligned with what the person actually heard.

## Adopted First-Preamble Policy

The adopted behavior is "first preamble plus terminal answer." `allow_preamble`
is not a public configuration field; it is shorthand for the decision to admit
the first model-generated acknowledgement in a logical tool turn.

Navigation, capture-and-describe, return-to-point, and mixed physical workflows
all use this rule. The first acknowledgement may state intent only. It must not
claim arrival, successful capture, or mission completion before the relevant
tool result exists. No periodic progress speech is generated, even for a long
mission.

A future need may justify making speech policy configurable at the logical-turn
level. Possible modes would be:

- `first_preamble`: the current behavior;
- `terminal_only`: suppress every tool-bearing response and speak only the
  terminal answer; and
- potentially `silent`: execute a background/internal operation without any
  spoken response, where the initiating event contract explicitly permits it.

Such a policy should still belong to the interaction, not be recomputed for each
tool in a chain. Scattered per-tool exceptions would make mixed and dynamically
chosen chains difficult to reason about.

### Why tool count is brittle

Tool count is not a useful proxy for interaction behavior:

- one navigation call can take minutes;
- several parallel lookups can finish almost immediately;
- a sequential chain appears as one tool call in each of several responses;
- retries can change the count without changing the user's intent; and
- the model, not application code, often decides how many calls are needed.

Policy should instead reflect user experience and side effects: whether the
operation moves the robot, how long it may take, whether it is interruptible,
and whether an intermediate statement could become false.

### Current invariants and future guardrails

The implementation and any future policy expansion should preserve these
invariants:

1. The policy is explicit and bound to the logical turn, not inferred from call
   count or scattered per-tool name checks.
2. A preamble can only acknowledge intent; it cannot claim arrival, successful
   capture, or task completion before the corresponding result exists.
3. At most one preamble is audible across a sequential tool chain; long mission
   duration alone does not create progress responses.
4. A spoken preamble becomes permitted inference history; an unheard one remains
   diagnostic-only.
5. Interruption and playback bookkeeping distinguish preamble audio from the
   terminal answer.
6. The tool barrier still emits exactly one follow-up per completed
   tool-bearing response.
7. Terminal speech remains response-based, not "the response with the greatest
   ID" or "the last response observed so far." Response IDs identify objects;
   they do not encode completion order or user-facing priority.
8. Tests cover one tool, parallel tools, sequential tools, unknown tools,
   failures, interruption, and first-preamble playback completion.

Argos uses the first of these approaches:

- release model-generated preamble audio once a function call makes the
  response tool-bearing, with strong prompt and playback guards; or
- keep intermediate model audio suppressed and use a deterministic local cue or
  acknowledgement.

Model-generated speech preserves the natural voice but makes exact preamble
wording partly probabilistic. The prompt therefore requests a tiny
intent-only acknowledgement, while the runtime provides the stronger guarantee:
only the first tool-bearing response and the terminal response can be audible.
A deterministic local cue remains a future alternative if prompt-level brevity
proves unreliable.

## What to measure

Evaluate the current policy and any future variants using:

- time from user speech end to tool start;
- `first_audio_latency_s` for provider generation;
- `terminal_audio_release_latency_s` for audible response availability;
- total tool-chain duration;
- number of audible segments per human turn;
- interruption rate while tools are running; and
- operator/user reports of confusing silence versus excessive narration.

Those measurements can justify future tuning more reliably than the number of
tools selected by the model.
