---
name: argos-html-presentation-workflow
description: Create or materially edit Argos presentations as polished browser-openable HTML decks, briefings, and companion one-pagers with claims backed by repo docs, lab reports, runtime artifacts, and supplied media.
---

# Argos HTML Presentation Workflow

## Use This Skill When
- Creating an Argos presentation, demo deck, roadmap deck, review package, or stakeholder briefing.
- Turning Argos docs, architecture/turn-flow explanations, lab outputs, eval reports, screenshots, logs, or operator notes into a visual narrative.
- Editing an existing HTML deck or converting a presentation plan into a browser-openable artifact.

## Default Output
- Prefer an editable static HTML presentation over PPTX unless the user explicitly asks for PowerPoint.
- Save final repo deliverables under `docs/presentations/` when no path is provided.
- Keep scratch renders, drafts, generated thumbnails, and bulky raw media outside the repo unless the user asks to commit them.

## Source Map
Load only the sources needed for the claims being made:
- Overall system: `README.md`, `docs/architecture.md`, `docs/realtime_turn_flow.md`, `docs/launch.md`.
- Voice/realtime behavior: `docs/voice.md`, `docs/prompting_and_history.md`, `docs/observability.md`.
- Human identity and memory: `docs/face_recognition.md`, `docs/speaker_recognition.md`, `docs/identity_memory.md`.
- Display and attention: `docs/interaction_display.md`, `docs/attention_gate.md`.
- Evidence and diagnostics: `scripts/labs/README.md`, `var/labs/**`, `var/eval/**`, `data_collection/**`, plus user-supplied media or reports.
- Implementation detail: scoped files under `argos_src/`, `config/profiles/`, `resources/`, and `scripts/labs/`.

## Workflow
1. Identify audience, purpose, date, target length, output path, and must-use source material. If absent, choose a concise technical-stakeholder HTML deck.
2. Build a claim spine before layout: one visible takeaway per slide, each tied to a proof object such as a doc section, code path, lab report, screenshot, metric, or supplied note.
3. Choose the smallest deck shape that works: title/context, architecture or flow, evidence, risks/unknowns, and next actions. Move dense details to appendix sections.
4. Use real Argos visuals first: architecture/turn-flow diagrams, screenshots, lab report tables, eval charts, interaction display screenshots, camera/audio examples, or annotated code-path diagrams.
5. For architecture or flow slides, keep terminology aligned with current docs: persistent OpenAI Realtime session, local audio admission, engagement state machine, event coalescer, tool execution, face/voice identity, source-aware memory, provider API, and Unitree Go2 capabilities.
6. Implement static HTML that opens from `file://` when possible. Use relative asset paths, escaped text, responsive layout, print-friendly styles, and no external CDN dependency unless explicitly allowed.
7. Keep the design presentation-like, not a marketing landing page: strong slide rhythm, readable typography, clear hierarchy, purposeful visuals, and enough whitespace for live narration.
8. Render or open the deck locally when feasible and inspect desktop plus narrow viewport behavior for clipping, overlap, broken media, and unreadable diagrams.
9. Record validation notes: sources checked, render/browser checks run, unsupported assumptions, and any evidence gaps.

## HTML Deck Guidance
- Use semantic sections for slides and a small amount of JavaScript only when it improves navigation, presenter mode, or responsive scaling.
- Keep text concise; speaker detail can live in notes, hidden appendix content, or a companion Markdown brief.
- Include provenance on evidence-heavy slides: source path, run id, report path, config/profile, date, or supplied-media note.
- Use generated diagrams only as derived explanations of checked source material. Do not present synthetic visuals as live robot evidence.
- Prefer inline CSS for portable one-file decks unless local reusable assets are intentional.

## Verification
- Every substantive claim has a cited or clearly named source.
- The deck opens locally without a server unless documented otherwise.
- Text, diagrams, tables, and media are legible on desktop and narrow viewports.
- Links and media resolve relative to the final deck location.
- Argos terminology, file paths, commands, and dates match the checked repo or supplied source material.
- Non-trivial decks should be reviewed with `presentation-reviewer` when subagents are available.

## Don'ts
- Do not invent robot capabilities, eval results, customer/site facts, employee data, dates, or deployment status.
- Do not imply lab or eval output exists unless it is present or provided by the user.
- Do not replace canonical docs, configs, logs, or lab artifacts with the HTML deck as source of truth.
- Do not leave large generated media or scratch files in the repo deliverable folder.
