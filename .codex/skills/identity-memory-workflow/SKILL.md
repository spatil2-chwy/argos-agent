---
name: identity-memory-workflow
description: Guide changes to face recognition, speaker recognition, identity records, owner resolution, employee directory lookup, live enrollment, preference extraction, memory storage, memory prompt projection, or Slack memory ingestion.
---

# Identity Memory Workflow

Use this workflow when code may change who a turn belongs to, what gets stored about a person, or what person context reaches the realtime prompt.

## Load Context

Read the relevant docs:

- `docs/face_recognition.md`
- `docs/speaker_recognition.md`
- `docs/identity_memory.md`
- `docs/prompting_and_history.md`

## Preserve

- `primary_face_person_id` is strict: only one usable recognized face.
- `audio_speaker_id` and face evidence combine through owner-resolution policy.
- Enrollment requires verified identity flow and local quality/consent gates.
- Memory kind controls prompt surface. Durable kinds become `About`; `followup` becomes `Potential Followups`.
- Follow-ups must expire. Temporal plans should not silently become durable notes.
- Local runtime state under `var/` must not be reset without explicit user approval.

## Work

1. Decide whether the change affects recognition, ownership, identity storage, memory extraction, or prompt projection.
2. Keep LLM-facing tool schemas small; rehydrate trusted employee/identity metadata locally.
3. Preserve structural validation in memory writes and avoid semantic regex shortcuts for memory classification.
4. Update docs when changing ownership or memory contracts.
5. If parallel agents are requested, use `identity-memory-auditor` and `test-runner`.

## Targeted Tests

```bash
python3 -B -m pytest tests/argos_src/face_recognition
python3 -B -m pytest tests/argos_src/speaker_recognition
python3 -B -m pytest tests/argos_src/tools/common/memory
```
