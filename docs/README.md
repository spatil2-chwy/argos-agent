# Argos Docs

Start with these in order:

1. `launch.md` for operator bring-up, smoke tests, and troubleshooting.
2. `architecture.md` for the high-level runtime shape.
3. `realtime_turn_flow.md` for exact audio/internal-event turn order.
4. `prompting_and_history.md` for what the Realtime model sees.
5. `robot_tools.md` for tool IDs, provider capabilities, side effects, and safety notes.

## Runtime Internals

- `architecture.md`: system map, boot flow, prompt/history/tool model, and engagement model.
- `realtime_turn_flow.md`: websocket session lifecycle, turn queue, event coalescing, playback, interruptions, watchdogs, and edge cases.
- `prompting_and_history.md`: static prompt, dynamic turn instructions, Realtime history items, tool outputs, and owner-scoped history.
- `observability.md`: latency log format, realtime markers, cost/usage events, and CLI helpers.
- `attention_gate.md`: head-pose attention signal and passive microphone admission.

## Identity And Memory

- `face_recognition.md`: face enrollment, recognition preprocessing, strict primary face ownership, and display review.
- `speaker_recognition.md`: turn-owned speaker matching, voice enrollment, and ownership fallback.
- `identity_store.md`: person records, aliases, face/speaker embeddings, and operator identity commands.
- `memory_provider.md`: Tailwag-backed memory provider, realtime episodes, person context, encounters, and local/external reset boundaries.
- `slack_memory.md`: Tailwag-backed Slack polling, cursor state, identity convergence, and prompt visibility boundaries.
- `employee_directory.md`: local employee-directory validation used during enrollment.

## Providers, Tools, And Surfaces

- `robot_tools.md`: public tool IDs, runtime tool names, provider capabilities, motion/navigation side effects, patrol, battery, and manual safety.
- `interaction_display.md`: optional browser display resource, HTTP contract, state mapping, and face-enrollment review UI.
- `complete_setup.md`: fresh-machine setup and provider/runtime separation.
- `voice.md`: short entry point for voice interaction docs.

## How To Keep These Coherent

- Treat `realtime_turn_flow.md` as canonical for turn lifecycle and engagement timing.
- Treat `prompting_and_history.md` as canonical for Realtime item roles and model-visible history.
- Treat `robot_tools.md` as canonical for model-visible tool contracts.
- Treat `memory_provider.md` as canonical for shared memory semantics; use `slack_memory.md` only for Slack-specific ingestion mechanics.
- Keep setup commands in `launch.md` and `complete_setup.md`; avoid repeating them in subsystem docs unless the command is subsystem-specific.
