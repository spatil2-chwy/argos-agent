---
name: agent-routing-workflow
description: Guide selection and coordination of this repo's custom Codex subagents. Use when deciding whether a request should use agents, subagents, parallel review, delegated tests, documentation sync, presentation work, or agent-backed validation.
---

# Agent Routing Workflow

Use this workflow whenever agent use is being considered for this repo. For ordinary small coding requests, do the check quickly and keep working in the main thread. Spawn agents only when they add useful parallel signal.

## Route

Read `.codex/agent-routing.md`, then choose the smallest useful bundle.

Default choices:

- General diff review: `change-reviewer`
- Realtime/audio/playback/history: `realtime-turn-auditor`
- Robot-facing tools/motion/nav/patrol: `robot-safety-auditor`
- Face/speaker/identity/memory: `identity-memory-auditor`
- Provider/profile/tool contract: `provider-contract-guardian`
- Targeted test execution/fixing: `test-runner`
- Logs/latency/bring-up diagnosis: `observability-debugger`
- Documentation drift and source-backed summaries: `docs-sync-auditor`
- HTML decks, weekly briefings, stakeholder presentations: `presentation-creator`, then `presentation-reviewer`

## Coordinate

1. Keep the blocking implementation path in the main thread.
2. Delegate independent review, diagnosis, or test work.
3. Give each agent concrete files, docs, tests, and expected output.
4. Avoid overlapping write scopes. Use read-only auditors for review.
5. Consolidate findings before reporting or editing further.
6. Tell the user which agents were used or skipped only at the outcome level, not through raw coordination logs.
7. Capture repeatable workflow mistakes as learning updates. Read-only agents propose the update; write-capable agents may patch `.codex/` guidance only when that path is explicitly in scope.

## Do Not

- Do not spawn agents just because a task is complex.
- Do not ask several agents the same vague question.
- Do not let a write-capable agent touch live runtime state or robot commands.
- Do not create branches or worktrees for agents unless the user asks or the task truly needs isolated implementations.
- Do not turn one-off errors into permanent process; only record lessons that are concrete, repeatable, and useful for future agents.
