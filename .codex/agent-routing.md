# Agent Routing

Use this file whenever deciding whether a Codex request should use subagents. For ordinary small coding requests, make the decision quickly, use the matching skill in `.codex/skills`, and work in the main thread. The user should not need to explicitly ask for agents.

The goal is a coherent agent ecosystem: the main thread owns the user relationship, implementation plan, final judgment, and synthesis. Subagents provide bounded parallel signal: review, tests, diagnostics, documentation sync, presentation creation/review, or isolated implementation experiments.

## Default Pattern

1. Classify the request: small/local, risky/domain-specific, cross-cutting, test-heavy, review-heavy, documentation-heavy, presentation/reporting, or exploratory.
2. Keep the immediate blocking implementation path in the main thread.
3. Spawn agents only for independent work that can run in parallel and produce useful evidence before the final answer.
4. Give each agent a concrete scope, expected output, relevant docs/tests, and file/module boundary.
5. Use read-only auditors for review. Use workspace-write agents only for targeted tests, isolated fixes, presentation generation, or explicitly bounded implementation slices.
6. Avoid branches/worktrees by default. Use them only for competing implementation experiments, long-running parallel edits, or explicit user requests.
7. Wait for agents when their result is needed, then consolidate, verify, and report the final decision in normal user-facing language.

## Safety Defaults

- Do not let agents launch live robot, provider, or runtime commands without explicit user approval. This includes `run_profile.py`, provider bring-up, robot motion commands, and long-running live audio loops.
- Do not let agents reset, delete, or rewrite `var/` runtime state, identity databases, face/speaker stores, memory stores, logs, generated knowledge indexes, or lab/eval artifacts unless the user explicitly requests that exact cleanup.
- Workspace-write agents must list every changed path. The main thread reviews their changes before applying, merging, or reporting conclusions.

## Spawn Heuristics

Consider agents when at least one of these is true and there is an independent bounded output to gather:

- A change touches live runtime risk: realtime audio, turn lifecycle, robot motion, identity, memory, provider contracts, or operator bring-up.
- The task has separable validation: implementation in main thread, tests in `test-runner`, review in one or more read-only auditors.
- The task crosses ownership boundaries or could silently drift docs, profile fields, public tool IDs, prompt contracts, or presentation claims.
- The user asks for a review, audit, weekly summary, release narrative, presentation, or confidence check that needs independent evidence, generation, or QA.
- Parallel exploration can compare options without causing overlapping writes.

Skip agents when all of these are true:

- The task is small, local, low-risk, and easy to verify directly.
- The likely agent output would duplicate the main thread's work.
- There is no independent file/test/doc boundary to assign.
- Waiting for an agent would add latency without increasing confidence.

## Communication Contract

Subagents should return compact structured results, not long transcripts:

- `Verdict`: pass, concerns, or blocked.
- `Findings`: ordered by severity with file references.
- `Evidence`: tests run, docs read, logs checked, or artifacts inspected.
- `Docs`: required documentation or presentation updates.
- `Learning`: repeatable workflow lesson and the exact guidance file that should be updated, when applicable.
- `Risk`: what remains unverified.

Binary pass/fail is useful only as a top-line verdict. It is not enough for this repo because safety, identity, memory, provider, and presentation claims need evidence and file references.

## Learning Loop

Agents should learn from failed attempts without turning every mistake into permanent process. A learning update is appropriate when all of these are true:

- The agent lost time because local guidance was missing, stale, ambiguous, or contradicted the actual environment.
- The lesson is likely to recur for future agents in this repo.
- The update can be captured as a small, concrete instruction in `AGENTS.md`, `.codex/agent-routing.md`, a skill under `.codex/skills/`, or an agent description under `.codex/agents/`.

Examples include command availability such as using `python3` instead of `python` in this environment, required setup commands, ignored-file behavior, known safe validation slices, or task-specific routing mistakes.

Read-only agents must report the proposed learning update instead of editing files. Workspace-write agents may edit `.codex/` guidance only when the parent task explicitly includes that write scope or when the user asked to improve the agent ecosystem. The main thread reviews all guidance updates before relying on them.

## Model Budgeting

- Default to the agent's configured model and reasoning effort.
- Use higher reasoning for realtime turn flow, robot safety, identity/memory, provider contracts, non-trivial refactors, and final presentation review.
- Use medium effort for log triage, narrow test diagnosis, and mechanical documentation checks unless the evidence is contradictory.
- Reserve explicitly premium models, such as `gpt-5.5`, for synthesis-heavy presentation work, high-stakes architecture review, or tasks where quality of narrative judgment matters more than raw throughput.

## Agent Matrix

| Trigger | Agent | Mode | Pair With |
|---|---|---:|---|
| Any non-trivial diff or risky behavior change | `change-reviewer` | read-only | The closest workflow skill |
| Realtime session, turn queue, audio capture, playback, tool loop, interrupts, history | `realtime-turn-auditor` | read-only | `realtime-change-workflow` |
| Robot motion, navigation, gestures, patrol, battery, face/nav event side effects | `robot-safety-auditor` | read-only | `robot-tool-contract-workflow` |
| Face recognition, speaker recognition, owner resolution, identity store, memory extraction | `identity-memory-auditor` | read-only | `identity-memory-workflow` |
| Provider transports, manifests, resources, profiles, tool IDs, tool schemas | `provider-contract-guardian` | read-only | `provider-profile-workflow` |
| Module boundaries, folder structure, naming, file size, coupling, and refactor shape | `repo-structure-auditor` | read-only | `safe-refactor-workflow` |
| Tests should be selected, run, diagnosed, or fixed in parallel | `test-runner` | workspace-write | Any workflow |
| Public behavior, setup, operator flow, architecture, safety, identity, provider, or memory docs may drift | `docs-sync-auditor` | read-only | Any workflow |
| Latency logs, realtime markers, cost telemetry, bring-up failures | `observability-debugger` | read-only by default | `debug-bringup-workflow` |
| Dashboard UI/API, latency-log indexing, dashboard build, observability dashboard docs | `observability-dashboard-maintainer` | read-only | `safe-refactor-workflow` |
| HTML deck, demo briefing, roadmap deck, weekly status presentation, or stakeholder review package | `presentation-creator` | workspace-write | `argos-html-presentation-workflow` |
| Presentation artifact needs QA, claim checking, browser/layout review, or source alignment | `presentation-reviewer` | read-only | `argos-presentation-review-workflow` |
| VBR-style written review, deck-to-DOCX conversion, or 2-3 page source-backed stakeholder write-up | `vbr-docx-writer` | workspace-write | `argos-html-presentation-workflow` when starting from an Argos deck |

## Common Bundles

- Realtime/audio change: `realtime-turn-auditor`, `test-runner`, optionally `observability-debugger`.
- Identity/memory change: `identity-memory-auditor`, `test-runner`, optionally `docs-sync-auditor` or `change-reviewer`.
- Robot/tool behavior change: `robot-safety-auditor`, `provider-contract-guardian`, `test-runner`, optionally `docs-sync-auditor`.
- Provider/profile/tool contract change: `provider-contract-guardian`, `test-runner`, optionally `docs-sync-auditor`.
- Refactor crossing module boundaries: `repo-structure-auditor`, `change-reviewer`, `test-runner`, plus the domain auditor for the touched path.
- Structure/naming/modularity review: `repo-structure-auditor`, optionally `change-reviewer` if a diff already exists.
- Bring-up/debug session: `observability-debugger`, optionally `provider-contract-guardian` if resources or transports are involved.
- Documentation update: `docs-sync-auditor`, optionally the domain auditor for the subsystem being documented.
- Observability dashboard update: `observability-dashboard-maintainer`, optionally `observability-debugger` when live logs or latency diagnosis are involved.
- Weekly change presentation: main thread owns synthesis and gathers `git log`/diff/source artifacts; optionally use `docs-sync-auditor` for source-backed theme extraction or doc drift, then `presentation-creator` builds the deck and `presentation-reviewer` checks claims and layout.
- VBR DOCX from an existing deck: main thread or `vbr-docx-writer` turns the deck into a concise write-up with tables/figures, then optionally use `docs-sync-auditor` or `presentation-reviewer` for source-backed claim and layout review.
- Final pre-handoff check for risky work: relevant domain auditor, `test-runner`, `docs-sync-auditor`, and `change-reviewer`.

## Prompt Template

```text
Use the <agent-name> custom agent for this bounded review.

Scope:
- Files/modules:
- Behavior to protect:
- Relevant docs:
- Relevant tests:

Return:
- Findings first, with file references.
- Missing tests or docs.
- Learning update, if a repeatable workflow lesson should be captured.
- Any assumptions or residual risks.
```
