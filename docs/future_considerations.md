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
  -> model response containing tool call(s)   # buffered, not spoken
  -> tool result(s) and optional image(s)      # retained for inference
  -> follow-up model response
  -> more tool calls, if needed                # also buffered, not spoken
  -> terminal response with no tool calls      # released to playback
```

Audio and transcript deltas are buffered by `response_id` until
`response.done`. A response containing any function call is classified as
tool-bearing. Its audio is suppressed, while its function calls, tool outputs,
and artifacts remain available to the model. A response with no function calls
is terminal and is released to the speaker.

The follow-up barrier is also response-local. It sends exactly one new
`response.create` only after:

- the source response has reached `response.done`;
- all expected call IDs have produced outputs; and
- no tool execution for the turn remains pending.

This works the same way for one tool, several parallel calls in one response,
and sequential chains where a follow-up response calls another tool. A single
tool does not lose its answer: the tool-bearing response stays silent, the tool
runs, its result is returned, and the terminal follow-up is spoken.

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

The unheard assistant preamble is kept for diagnostics but excluded from future
inference history. Function calls and outputs remain selected. This avoids
teaching the model that the person heard words that Argos intentionally did not
play.

## Possible `allow_preamble` Policy

`allow_preamble` is a design idea, not an implemented option or metadata field.
It would mean that a short acknowledgement from a tool-bearing response, such as
"I'll check that," may be played before or while the tool runs. The current
behavior is effectively `terminal_only`: only the response that contains no tool
call is audible.

The default should remain terminal-only for robot missions. Navigation,
capture-and-describe, return-to-point, and mixed physical workflows benefit from
one concise final report, and intermediate speech can become stale or misleading
if motion fails, is interrupted, or changes course.

A future preamble mode may be useful for genuinely slow, conversational,
non-motion operations where silence feels broken. If introduced, it should be an
explicit interaction-level speech policy rather than a rule based on tool count.
Possible modes are:

- `terminal_only`: suppress every tool-bearing response and speak once at the
  terminal response;
- `allow_preamble`: permit one short, non-result acknowledgement, then speak the
  terminal answer; and
- potentially `silent`: execute a background/internal operation without any
  spoken response, where the initiating event contract explicitly permits it.

The strictest applicable policy should win in a mixed chain. For example, a
chain containing navigation and a quick data lookup should remain
terminal-only, even if the lookup alone might allow a preamble.

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

### Requirements before implementing it

If preambles become necessary, the implementation should preserve these
invariants:

1. The policy is explicit and bound to the logical turn, not inferred from call
   count or scattered per-tool name checks.
2. A preamble can only acknowledge intent; it cannot claim arrival, successful
   capture, or task completion before the corresponding result exists.
3. At most one preamble is audible across a sequential tool chain unless a
   separately specified progress-update policy exists.
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
   failures, interruption, and a mixed chain whose strictest policy wins.

There are two plausible implementation approaches:

- release model-generated preamble audio once a function call makes the
  response tool-bearing, with strong prompt and playback guards; or
- keep intermediate model audio suppressed and use a deterministic local cue or
  acknowledgement.

The first preserves the model's natural voice but makes brevity and semantic
safety partly probabilistic. The second is easier to constrain but introduces a
separate audio/UX mechanism. Neither should be added until operator evidence
shows that terminal-only silence during tools is a real usability problem.

## What to measure

Before changing the current policy, compare real mission traces using:

- time from user speech end to tool start;
- `first_audio_latency_s` for provider generation;
- `terminal_audio_release_latency_s` for audible response availability;
- total tool-chain duration;
- number of audible segments per human turn;
- interruption rate while tools are running; and
- operator/user reports of confusing silence versus excessive narration.

Those measurements can justify a policy change more reliably than the number of
tools selected by the model.
