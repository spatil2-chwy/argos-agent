---
name: debug-bringup-workflow
description: Guide debugging of launch, realtime bring-up, audio devices, provider transport, latency logs, display connection, OpenAI Realtime setup, face/speaker startup, or operator smoke-test failures.
---

# Debug Bring-Up Workflow

Use this workflow for diagnosing why the robot runtime, realtime session, audio, provider, display, or recognition services are not behaving correctly.

## Load Context

Read:

- `docs/launch.md`
- `docs/observability.md`
- `docs/realtime_turn_flow.md` when a turn fails
- `docs/face_recognition.md` or `docs/speaker_recognition.md` for recognition startup
- `config/profiles/static_interaction.yaml`
- `config/manifests/puffle.yaml`

## Diagnose By Layer

1. Environment: Poetry/setup shell, `OPENAI_API_KEY`, device availability.
2. Provider: transport, resource IDs, capabilities, display URL.
3. Realtime: websocket session, `session.update`, `response_create`.
4. Audio: input/output devices, VAD, wake window, admission policy.
5. Runtime: engagement state, coalescer, tool loop, playback.
6. Background: preference extraction, face/speaker model loading, Slack memory.

## Safety

Do not start `run_profile.py`, provider bring-up, live audio loops, or robot motion commands without explicit user approval. Prefer logs, config inspection, and tests first.

## Useful Checks

```bash
python3 -B -m pytest tests/argos_src/test_argos_profile_config.py
python3 -B -m pytest tests/argos_src/observability/test_pricing.py
python3 -m argos_src.observability.latency_report
```
