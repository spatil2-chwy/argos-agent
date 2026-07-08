# Argos Docs

Start with these in order:

1. `launch.md` for operator bring-up, smoke tests, and troubleshooting.
2. `architecture.md` for the high-level runtime shape.
3. `realtime_turn_flow.md` for exact audio/internal-event turn order.
4. `realtime_state_model.md` for the observable control-plane state axes.
5. `realtime_state_machine_diagram.md` for a visual map of the state axes,
   reducers, and turn/playback flow.
6. `prompting_and_history.md` for what the Realtime model sees.
7. `robot_tools.md` for tool IDs, provider capabilities, side effects, and safety notes.

## Runtime Internals

- `architecture.md`: system map, boot flow, prompt/history/tool model, and engagement model.
- `realtime_turn_flow.md`: websocket session lifecycle, turn queue, event coalescing, playback, interruptions, watchdogs, and edge cases.
- `realtime_state_model.md`: typed control-plane axes, states, triggers, and dashboard-facing state event semantics.
- `realtime_state_machine_diagram.md`: Mermaid flowcharts for the state axes, engagement reducer, human audio turns, internal event turns, and tool/playback branches.
- `realtime_control_refactor_plan.md`: completed refactor phases, invariants, validation gates, and remaining hardening notes.
- `prompting_and_history.md`: static prompt, dynamic turn instructions, Realtime history items, tool outputs, and owner-scoped history.
- `observability.md`: latency log format, realtime markers, cost/usage events, CLI helpers, and FastAPI/Vite dashboard.
- `attention_gate.md`: head-pose attention signal and passive microphone admission.

## Identity And Memory

- `face_recognition.md`: face enrollment, recognition preprocessing, strict primary face ownership, and display review.
- `speaker_recognition.md`: turn-owned speaker matching, voice enrollment, and ownership fallback.
- `identity_memory.md`: the package boundary between Argos sensing and Tailwag-owned identity, biometrics, Slack ingestion, and memory.

## Providers, Tools, And Surfaces

- `robot_tools.md`: public tool IDs, runtime tool names, provider capabilities, motion/navigation side effects, patrol, battery, and manual safety.
- `interaction_display.md`: optional browser display resource, HTTP contract, state mapping, and face-enrollment review UI.
- `complete_setup.md`: fresh-machine setup and provider/runtime separation.
- `voice.md`: short entry point for voice interaction docs.

## How To Keep These Coherent

- Treat `realtime_turn_flow.md` as canonical for turn lifecycle and engagement timing.
- Treat `prompting_and_history.md` as canonical for Realtime item roles and model-visible history.
- Treat `robot_tools.md` as canonical for model-visible tool contracts.
- Treat `identity_memory.md` as canonical for identity and memory ownership boundaries.
- Keep setup commands in `launch.md` and `complete_setup.md`; avoid repeating them in subsystem docs unless the command is subsystem-specific.
