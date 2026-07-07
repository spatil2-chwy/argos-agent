---
name: realtime-change-workflow
description: Guide changes to the realtime robot session, turn lifecycle, audio capture/admission, playback, interruption handling, websocket event parsing, tool loop, or owner-scoped history. Use for edits under argos_src/agent, argos_src/runtime/audio_admission.py, realtime tests, or docs/realtime_turn_flow.md.
---

# Realtime Change Workflow

Use this workflow for changes that can alter how a human or internal robot event becomes one realtime model response.

## Load Context

Read the smallest relevant subset:

- `docs/realtime_turn_flow.md`
- `docs/prompting_and_history.md`
- `docs/observability.md`
- `docs/architecture.md`

Then inspect the touched code and nearest tests.

## Preserve

- Local code owns turn boundaries; the Realtime API does not auto-segment speech.
- Audio turns commit with `input_audio_buffer.commit`, then local code sends `response.create`.
- Internal events may be coalesced or folded into the same audio turn without breaking the human turn.
- Playback, tool calls, response completion, and watchdogs must reconcile to a terminal turn state.
- Owner-scoped history and active unresolved items must remain protected during handoff.

## Work

1. Identify the exact turn phase, queue, event binding, or state-machine behavior being changed.
2. Keep concurrency changes small and explicit; avoid hidden waits on audio callback, playback, or tool threads.
3. Preserve log markers that operators use: `recording_started`, `speech_end`, `audio_commit`, `response_create`, `first_audio_latency_s`, and usage events when applicable.
4. Add or update targeted tests for the changed turn behavior.
5. If parallel agents are requested, use `realtime-turn-auditor` and `test-runner`.

## Targeted Tests

Start with the smallest relevant slice:

```bash
python3 -B -m pytest tests/argos_src/agent/test_agent_runtime.py
python3 -B -m pytest tests/argos_src/agent/control
python3 -B -m pytest tests/argos_src/runtime/test_audio_admission.py
python3 -B -m pytest tests/argos_src/agent/test_agent_events.py
```
