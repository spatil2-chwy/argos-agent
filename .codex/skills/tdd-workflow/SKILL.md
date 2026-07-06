---
name: tdd-workflow
description: Guide test-first or red-green-refactor work in this repo. Use when the user asks for TDD, test-first development, regression-first bug fixing, or behavior-first implementation through public seams.
---

# TDD Workflow

Use this workflow when a change has observable behavior and regression protection matters.

## Work

1. Name the public seam: runtime event, tool call, provider method, CLI, profile load, store API, or prompt compiler output.
2. Write one failing behavior test for one expected behavior.
3. Implement the minimum production change to pass that test.
4. Repeat in small slices.
5. Refactor only while tests are green.
6. Keep mocks at the boundary; prefer real local collaborators when the public seam can exercise them.

## Repo Bias

- For realtime behavior, test turn state, emitted events, and queue/tool outcomes rather than private timing details.
- For identity and memory, test ownership decisions and stored/prompt-projected outputs.
- For tools/providers, test model-visible contract and local provider call separately.
- For profiles, test config parsing and resulting enabled behavior.

## Validate

Run the new test first, then the nearest existing regression slice. If parallel agents are requested, use `test-runner` after the first implementation slice.
