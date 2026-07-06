# Repository Guidance

This repository is a realtime robot voice-agent stack. Treat changes as potentially affecting live audio, robot motion, identity, memory, and operator bring-up.

## Working Defaults

- Read the closest docs before changing risky runtime paths:
  - `docs/architecture.md`
  - `docs/realtime_turn_flow.md`
  - `docs/prompting_and_history.md`
  - `docs/face_recognition.md`
  - `docs/speaker_recognition.md`
  - `docs/identity_store.md`
  - `docs/memory_store.md`
  - `docs/observability.md`
- Use `source setup_shell.sh` as the canonical local environment setup when running project commands.
- Use `python3`, not `python`, for local Python commands in this environment unless a project script explicitly says otherwise.
- Prefer targeted pytest slices before broad test runs. Start with tests near the changed module.
- Preserve public tool IDs, provider resource IDs, profile fields, prompt contracts, identity IDs, and memory kind semantics unless the user explicitly asks to change them.
- Do not reset or delete `var/` runtime state, identity databases, face/speaker stores, memory stores, logs, or generated knowledge indexes unless the user explicitly requests that exact cleanup.
- Do not launch live robot/provider/runtime commands without explicit user approval. This includes `run_profile.py`, provider bring-up, robot motion commands, and long-running live audio loops.

## Codex Workflow

- Use repo-local skills from `.codex/skills` when their descriptions match the requested work.
- For every non-trivial request, briefly decide whether agents would add useful parallel signal. Do not require the user to explicitly ask for agents.
- Use `.codex/agent-routing.md` to choose subagents when the request is risky, cross-cutting, review-heavy, test-heavy, presentation-heavy, or otherwise benefits from independent parallel work.
- Skills guide normal work in the main thread. Subagents are sidecars for parallel review, audits, targeted test execution, documentation sync checks, presentation QA, or isolated implementation slices.
- Keep subagent tasks concrete and bounded. Assign disjoint file ownership for write-capable subagents.
- After subagents finish, review their findings in the main thread before applying or reporting conclusions.
- Do not expose agent coordination noise to the user. Report what was done, why those agents were used or skipped, tests run, docs updated, and remaining risk.
- Treat repeatable workflow mistakes as guidance bugs. If an agent discovers that a missing instruction caused wasted work, it should report a compact learning update; write-capable agents may patch `.codex/` guidance when that path is explicitly in scope, and read-only agents should propose the exact update for the main thread to apply.

## Validation Bias

- For realtime turn/audio/playback changes, include tests under `tests/argos_src/agent/` and `tests/argos_src/runtime/` when relevant.
- For face/speaker/identity/memory changes, include tests under `tests/argos_src/face_recognition/`, `tests/argos_src/speaker_recognition/`, `tests/argos_src/memory/`, and identity tests when relevant.
- For provider/tool/profile changes, include provider API, tool ID, manifest, and profile config tests when relevant.
- For observability changes, include pricing/log parser/report tests when relevant.
