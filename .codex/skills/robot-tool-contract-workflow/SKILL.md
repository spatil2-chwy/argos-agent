---
name: robot-tool-contract-workflow
description: Guide changes to robot-facing tools, motion, navigation, gestures, patrol behavior, battery handling, face/nav internal events, display enrollment review, or LangChain tool schemas under argos_src/tools and robot runtime bridges.
---

# Robot Tool Contract Workflow

Use this workflow for code that can cause robot motion, navigation, posture changes, display commands, camera capture, or model-visible tool side effects.

## Load Context

Read the relevant files or docs:

- `docs/architecture.md`
- `docs/launch.md`
- `docs/face_recognition.md` for enrollment/display review
- `docs/realtime_turn_flow.md` for internal events and tool loop
- `config/profiles/static_interaction.yaml`
- `config/manifests/puffle.yaml`

## Preserve

- Tool schemas expose only model-safe inputs; local code owns hardware details and safeguards.
- Motion, navigation, and posture actions remain bounded and cancelable where intended.
- Patrol is suppressed during human interaction and resumes only through explicit engagement rules.
- Enrollment review should not save face state until accepted.
- Live robot commands require explicit user approval.

## Work

1. Identify the model-visible tool contract and the local provider call separately.
2. Keep tool IDs stable unless the user explicitly asks to migrate them.
3. Preserve tool result JSON shape when prompts or tests depend on it.
4. Prefer fake/provider tests before any manual robot smoke test.
5. If parallel agents are requested, use `robot-safety-auditor`, `provider-contract-guardian`, and `test-runner`.

## Targeted Tests

```bash
python3 -B -m pytest tests/argos_src/tools
python3 -B -m pytest tests/argos_src/agent/test_bridges.py
python3 -B -m pytest tests/argos_src/agent/test_factory_gestures.py
python3 -B -m pytest tests/argos_src/face_recognition/test_enrollment_display_review.py
```
