---
name: safe-refactor-workflow
description: Guide behavior-preserving refactors, simplification, extraction, modularity cleanup, or large-file decomposition in this realtime robot codebase while keeping tests, docs, tool contracts, and runtime behavior stable.
---

# Safe Refactor Workflow

Use this workflow when the user asks to refactor, clean up, simplify, extract, or reorganize without intended behavior changes.

## Lock The Boundary

Before editing, identify the surfaces that must not change:

- CLI commands and profile fields.
- Tool IDs, tool schemas, and tool result JSON.
- Realtime event ordering and turn terminal states.
- Provider dataclasses, resource IDs, and capability names.
- Identity IDs, memory kinds, and prompt projection shape.
- Tests that patch module-level seams.

## Work

1. Prefer extraction and locality over broad rewrites.
2. Move behavior behind smaller interfaces only when callers get simpler.
3. Keep compatibility shims when tests, scripts, or docs rely on existing names.
4. Update docs only when the conceptual contract changed or the old doc becomes misleading.
5. If parallel agents are requested, use `change-reviewer`, `test-runner`, and the domain auditor for the touched path.

## Validate

Run import/compile checks and the closest tests first. Broaden only when the refactor crosses domains.

```bash
python3 -B -m pytest <nearest-test-file>
```
