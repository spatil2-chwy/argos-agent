# Realtime Control Refactor Plan

This document is the north star for turning the current POC runtime into a
modular, observable control plane without changing the robot's externally
visible behavior.

Read this with:

- `docs/realtime_state_model.md`
- `docs/realtime_turn_flow.md`
- `docs/prompting_and_history.md`
- `docs/observability.md`
- `argos_src/agent/agent_runtime.py`
- `argos_src/agent/control/`

## Goal

The runtime should be composed from small control-plane modules with explicit
state ownership:

```text
RealtimeControlPlane
  -> Session lifecycle and websocket transport
  -> AudioRuntime
  -> TurnStore / ResponseBindingStore / HistoryStore
  -> TurnRunner
  -> PlaybackRuntime
  -> ToolRuntime
  -> RobotArbitrationPolicy
  -> StateObserver
```

`RealtimeRobotAgent` remains the public entry point. It is now a composition
root that wires the websocket, queues, providers, and control modules instead
of inheriting broad runtime mixins.

## Non-Negotiable Invariants

- Local code owns audio turn boundaries. The Realtime API must not be allowed to
  auto-segment microphone speech.
- Audio turns still send `input_audio_buffer.commit` before `response.create`.
- Internal face/nav/battery/patrol events may be coalesced or folded into a
  human audio turn without breaking that human turn.
- Every turn reaches a terminal state: `finalized`, `canceled`, or `superseded`.
- Late `response.created` events from canceled turns must not bind to later
  turns.
- If a stale response slot times out, the next ambiguous `response.created` is
  canceled as stale rather than being bound to a later turn.
- If that ambiguous cancellation may have consumed the live turn's real
  response, the runtime must immediately reissue `response.create` for the
  still-pending live turn.
- Owner-scoped history rotation must not delete active unresolved turn items.
- Public tool IDs, provider resource IDs, profile fields, prompt contracts,
  identity IDs, and memory kind semantics stay stable.
- Live robot/provider/audio bring-up is never part of automated validation
  without explicit operator approval.

## Completed Foundation

- Typed state axes exist in `argos_src/agent/control/types.py`.
- Structured transition and ignored-event logging exists through
  `StructuredStateObserver`.
- Engagement behavior is backed by a pure reducer.
- Internal event coalescing is backed by pure reducer helpers.
- Capture and turn phase transitions emit `component=state` events.
- A state log report helper exists at `argos_src/observability/state_report.py`.
- Concrete structured state logging lives under `argos_src/observability/`,
  while `argos_src/agent/control/` keeps only the observer protocol and safe
  wrappers.
- Patrol resume paths re-check battery and stale navigation targets.
- Response binding has a dedicated `PendingResponseBindingStore` so stale
  `response.created` events remain isolated from later turns.
- Timed-out stale response slots are treated conservatively: they may force a
  later ambiguous response to be canceled, but they must not corrupt turn
  ownership. When that happens, the live pending turn is reissued instead of
  waiting for the watchdog to cancel it.
- Owner-scoped history item order, ownership, protected-item planning, and
  delete-candidate planning are handled by
  `OwnerScopedHistoryIndex`.
- `EventCoalescer` now lives under `argos_src/agent/control/coalescer.py`.
- `EngagementStateMachine` now lives under
  `argos_src/agent/control/engagement_runtime.py`.
- Tool execution, result insertion, artifact messages, and enrollment
  side-effects now live behind `argos_src/agent/control/tool_runtime.py`.
- Turn response/tool/playback-stall watchdog decisions now live behind
  `argos_src/agent/control/watchdog_runtime.py`.
- Realtime server-event dispatch is wrapped by
  `argos_src/agent/control/event_adapter.py`.
- Audio capture, admission, commit, playback callbacks, speaker resolution at
  commit, and capture-axis transitions now live behind
  `argos_src/agent/control/audio_runtime.py`.
- Speaker-owned preference buffering and extraction scheduling now live behind
  `argos_src/agent/control/preference_runtime.py`.
- Local voice-command suppression and stop-command handling now live behind
  `argos_src/agent/control/voice_command_runtime.py`.
- Async display mode/subtitle updates now live behind
  `argos_src/agent/control/display_controller.py`.
- OpenAI Realtime server-event mutation handling now lives behind
  `argos_src/agent/control/server_event_runtime.py`; the adapter only routes
  event types.
- Patrol resume and proactive face-attention decisions use
  `argos_src/agent/control/robot_arbitration.py`.
- The remaining state/history/transport helper surface has moved from the
  deleted top-level `agent_state.py` into
  `argos_src/agent/control/state_runtime.py`.
- Top-level legacy runtime modules `agent_audio.py`, `agent_state.py`,
  `agent_preferences.py`, `agent_playback.py`, `agent_tools.py`,
  `orchestrator.py`, and `agent_events/dispatch.py` have been removed.

## Target Phases

### Phase 1: State Model and Observability

Status: complete for the current POC runtime.

Keep extending structured state logs until the dashboard can reconstruct:

- session readiness and shutdown
- capture admission and recording
- turn lifecycle
- playback lifecycle
- engagement mode
- robot arbitration decisions
- ignored triggers and suppression reasons

Session, capture, transcription, turn, playback, engagement, robot arbitration,
and coalescer events now emit structured state rows.

Validation:

- `tests/argos_src/agent/control`
- `tests/argos_src/observability/test_state_report.py`

### Phase 2: Turn and Response Stores

Status: complete for current behavior; further split is optional hardening.

Mutable turn bookkeeping is isolated behind stores and the composed
`AgentStateRuntime` control surface:

- response id binding and stale response protection
- owner-scoped history item order and ownership
- audio item binding
- local created item binding
- active turn registry
- terminal turn cleanup

Current modules:

- `argos_src/agent/control/turn_store.py`
- `argos_src/agent/control/history_store.py`

Validation:

- direct store unit tests
- `tests/argos_src/agent/test_agent_runtime.py`
- `tests/argos_src/agent/test_owner_scoped_history.py`

### Phase 3: History and Ownership Boundary

Status: complete for current behavior.

Owner-scoped conversation history lives in `OwnerScopedHistoryIndex`; transport
sends remain behind the composed `AgentStateRuntime` so all websocket writes
still pass through one runtime surface.

The store should answer:

- which items belong to which turn
- which items are protected during owner handoff
- which stale or deleted ids must be forgotten
- what the active owner key is

Validation:

- owner handoff tests
- tool call item protection tests
- active unresolved turn protection tests

### Phase 4: Audio Capture Controller

Status: complete for current behavior; explicit-dependency cleanup can happen
later without changing semantics.

Capture/admission behavior is now isolated in `AudioRuntime`, which owns:

- admission snapshots
- VAD/wake-word candidate tracking
- recording start/finalize
- audio queue drain before commit
- capture-axis state transitions

`AudioRuntime` still shares a few low-level queues and fields with the
composition root for callback compatibility, but capture behavior is no longer
implemented on the root agent.

Keep the sounddevice callback small and non-blocking.

Validation:

- `tests/argos_src/runtime/test_audio_admission.py`
- capture tests in `tests/argos_src/agent/test_agent_runtime.py`

### Phase 5: Turn Runner and Realtime Event Adapter

Status: complete for current behavior.

Split the current response worker into:

- `RealtimeEventAdapter`: parses server events and routes typed callbacks.
- `TurnRunner`: prepares history, sends `response.create`, and waits for model
  completion plus playback completion.
- `TurnWatchdog`: cancels, retries, or finalizes stuck turns.

`TurnWatchdogRuntime`, `RealtimeEventAdapter`, `ServerEventRuntime`, and
`TurnRunner` are extracted. The response worker thread still lives in
`RealtimeRobotAgent`, but it delegates turn execution to `TurnRunner`.

Validation:

- response-created stale binding tests
- no-audio retry tests
- incomplete-audio continuation tests
- response/tool/playback completion ordering tests

### Phase 6: Playback Controller

Status: complete for current behavior.

Extract playback buffering, stream callbacks, subtitle/display deltas, and
completion arming into a controller with explicit `playback` axis transitions.

Playback completion waits, stall force-completion, interruption delegation, and
playback-axis instrumentation live behind
`argos_src/agent/control/playback_runtime.py`. Output audio event mutation lives
in `ServerEventRuntime`; PCM buffer ownership stays on the composition root for
sounddevice callback compatibility.

Validation:

- playback completion tests
- interruption/truncation tests
- no-audio response tests

### Phase 7: Tool Runtime

Status: complete for current behavior.

Move tool queueing, argument assembly, execution, result insertion, and follow-up
`response.create` logic behind a `ToolRuntime`.

Execution, argument parsing, result insertion, artifact messages, side effects,
schema building, and follow-up `response.create` are extracted. Function-call
event routing lives in `ServerEventRuntime`.

Keep public tool schemas and result contracts unchanged.

Validation:

- tool result tests
- provider contract tests when robot-facing tools are touched
- action latency marker tests when available

### Phase 8: Robot Arbitration Policy

Status: complete for current behavior.

Patrol-resume and proactive face-event suppression checks use a single policy
that consumes state snapshots and returns explicit allow/suppress decisions
with reasons.

The priority order remains:

1. Battery safety.
2. Explicit human or wake interaction.
3. Non-interruptible or focused navigation.
4. Active recording or playback.
5. Patrol resume.
6. Proactive face greeting.
7. Idle gestures and display.

Validation:

- bridge tests
- factory wiring tests
- robot safety auditor review

### Phase 9: Composition Root Cleanup

Status: complete for current behavior.

The broad top-level mixins are gone. `RealtimeRobotAgent` now composes control
modules for audio, display, event routing, server-event mutation, playback,
preferences, robot arbitration, state/history, tools, turn running, voice
commands, and watchdog behavior.

Legacy compatibility modules are removed. Thin runtime delegates remain where
tests and neighboring controllers intentionally use the public agent surface.

Validation:

- full `tests/argos_src/agent`
- `tests/argos_src/runtime`
- docs sync audit

## Post-Live-Test Cleanup Candidates

These are intentionally deferred until the refactored runtime has been exercised
on the real Jetson/robot/audio/provider stack. They are not legacy behavior
paths; they are remaining coupling points kept to reduce risk during the first
large refactor.

Do not start this cleanup until a live validation run proves:

- A normal wake-word turn reaches recording, response creation, playback, and
  terminal turn cleanup.
- Passive face/attention admission can open listening while idle and does not
  open while suppressed.
- Barge-in or stop-command interruption truncates playback without corrupting
  the next turn.
- At least one tool call completes, inserts a tool result, and produces the
  follow-up assistant response.
- Owner attribution, history rotation, and preference extraction still work for
  a recognized speaker.
- Internal face/nav/battery/patrol events coalesce correctly and do not steal a
  human audio turn.
- Patrol resume/proactive face attention remain suppressed during human input,
  playback, non-interruptible navigation, and low-battery blocking conditions.
- `logs/latency.log`, `state_report`, and the dashboard reconstruct the same
  sessions, interactions, state transitions, ignored triggers, and failures.
- The non-live validation gate still passes after the live run.

Once that baseline is captured, the next cleanup targets are:

1. Align declared state axes with emitted runtime transitions.
   The typed model includes states such as `candidate_voice`, `queued`,
   `preparing_history`, `buffering`, and `awaiting_drain`, while the current
   runtime emits a smaller practical subset. Either emit the missing transitions
   when they help dashboard diagnosis, or trim/mark them as reserved so the docs
   do not imply stronger observability than the runtime provides.

2. Keep engagement state names single-sourced.
   `EngagementMode` and `EngagementState` intentionally describe the same
   values today. Consolidate them, or add a narrow compatibility check, if future
   edits start touching engagement names frequently enough for drift to become a
   real risk.

3. Route engagement transitions through one clear policy surface.
   The pure reducer owns the main transition table, but a few runtime paths still
   perform direct state changes for playback terminal events and watchdog
   fallback. Keep the direct paths if they remain simpler, but prefer reducer
   coverage when changing those behaviors so the transition policy stays easy to
   audit.

4. Replace host proxy controllers with explicit dependencies.
   `AudioRuntime`, `ServerEventRuntime`, and `AgentStateRuntime` currently use
   host field proxying to keep callback behavior stable. Replace that with
   typed context objects or small `Protocol` interfaces for queues, websocket
   send, state stores, playback buffers, display, logging, and profile access.

5. Move callback-owned buffers into the owning controllers.
   Audio capture buffers, playback buffers, resampling state, capture metadata,
   and playback tracking can move out of `RealtimeRobotAgent` once the live
   sounddevice callbacks are proven stable with the extracted `AudioRuntime`
   and `PlaybackRuntime`.

6. Reduce thin delegate methods on `RealtimeRobotAgent`.
   The public composition root still exposes methods such as
   `_send_response_create`, `_append_text_message_item`,
   `_wait_for_playback_and_complete`, and server-event handlers because nearby
   controllers and tests use that surface. After live validation, call sites can
   depend directly on controller interfaces and the root can shrink further.

7. Split `AgentStateRuntime` into narrower services.
   The current module is the remaining broad state/history/transport helper.
   Candidate splits are `PromptContextRuntime`, `HistoryRuntime`,
   `ResponseCreateRuntime`, and `TransportRuntime`.

8. Replace `Any` host typing with controller protocols.
   Each controller should declare the exact methods and fields it consumes. This
   will make future state-machine changes easier to review and safer to test.

9. Collapse duplicate transition helper patterns.
   Session, capture, transcription, playback, robot arbitration, and turn
   transitions are now observable, but each controller still owns small emit
   wrappers. A shared typed emitter can reduce repetition after the axes are
   proven complete in live logs.

10. Tighten dashboard schema once live logs settle.
   If live runs show missing session ids, weak failure classification, or
   ambiguous interaction grouping, update the log fields before hardening the
   dashboard API contract.

The cleanup is ready only when the live baseline artifacts are saved with:

- the exact profile and provider/robot setup used
- a representative `logs/latency.log`
- dashboard screenshots or exported snapshot JSON for the run
- notes for any manual recovery, robot stop, or operator intervention
- the full non-live validation command and result

## Subagent Strategy

Use subagents at phase boundaries, not for every small edit:

- `repo-structure-auditor`: module boundaries and naming after extraction
  slices.
- `realtime-turn-auditor`: turn queue, response binding, playback,
  interruption, or owner-history changes.
- `robot-safety-auditor`: navigation, patrol, battery, gestures, and proactive
  face-event arbitration.
- `provider-contract-guardian`: provider resources, tool schemas, profile
  fields, or fake/provider parity.
- `docs-sync-auditor`: when public behavior docs or operator flow may drift.
- `test-runner`: after a meaningful slice is implemented and targeted tests are
  selected.

Keep write scopes disjoint when using worker agents. The main thread owns final
integration and must review changed paths before trusting them.

## Default Validation Gate

For each behavior-preserving slice:

```bash
source setup_shell.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest <nearest tests> -q
python3 -B -m compileall -q argos_src/agent argos_src/observability
git diff --check
```

Before handoff, broaden to:

```bash
source setup_shell.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest tests/argos_src/agent tests/argos_src/runtime tests/argos_src/observability tests/scripts/labs -q
```

Do not use live robot/provider/runtime smoke tests unless the operator explicitly
approves that bring-up.
