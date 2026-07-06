---
name: argos-presentation-review-workflow
description: Review Argos HTML decks, presentation drafts, and briefing artifacts for narrative clarity, visual quality, source-backed claims, browser usability, and alignment with current Argos docs and runtime evidence.
---

# Argos Presentation Review Workflow

## Use This Skill When
- Reviewing an Argos HTML deck, PPTX, one-pager, or briefing before handoff.
- Checking a presentation after substantial content, design, diagram, media, or evidence edits.
- Auditing whether presentation claims match current Argos docs, code paths, lab reports, eval outputs, or supplied source material.

## Inputs
- Presentation source and, when available, rendered screenshots/contact sheets.
- Source material named by the creator or user.
- Relevant Argos docs, code, lab outputs, screenshots, or media only for claims under review.

## Review Workflow
1. Establish scope: audience, output path, expected format, and whether the task is QA-only.
2. Inspect the artifact itself before broad source reading. For HTML, check source plus browser/rendered output when feasible. For PPTX, inspect rendered slide previews if available.
3. Trace each major claim to evidence. Use the smallest relevant subset of repo docs and artifacts.
4. Review narrative: title promise, slide order, one takeaway per slide, audience fit, transitions, and whether risks or unknowns are stated plainly.
5. Review visual quality: hierarchy, spacing, typography, diagram readability, media cropping, alignment, contrast, rhythm, and whether charts/tables can be read live.
6. Review HTML usability when applicable: `file://` behavior, relative links, broken media, narrow viewport, print/export sanity, keyboard navigation if present, and absence of unsafe external dependencies.
7. Report findings first, ordered by severity. Include file paths and concrete fixes.

## Argos Accuracy Checks
- Realtime architecture claims align with `docs/architecture.md` and `docs/realtime_turn_flow.md`.
- Launch, setup, and operator commands align with `docs/launch.md` and `README.md`.
- Voice, prompting, history, and observability claims align with their corresponding docs.
- Face, speaker, identity, employee-directory, and memory claims distinguish actual implemented behavior from planned work.
- Lab/eval claims cite real artifacts under `scripts/labs/README.md`, `var/labs/**`, `var/eval/**`, `data_collection/**`, or supplied files.
- Robot capability claims are scoped to current provider/tool code and profile config; do not infer live performance from static implementation alone.

## Findings Format
Lead with findings. Keep summaries secondary.

For each finding include:
- Severity.
- Affected slide/section/file.
- What is wrong or risky.
- Why it matters.
- Concrete fix or evidence needed.

Then include:
- Unsupported or weak claims.
- Visual/layout issues.
- Browser/export issues.
- Sources reviewed and remaining risks.

## Verification Checklist
- Claims are concise, visible, and backed by proof objects.
- No invented metrics, dates, deployment states, employee data, robot capabilities, or live-demo results.
- Text is readable and not clipped or overlapping.
- Diagrams and screenshots materially support the slide takeaway.
- HTML artifacts open locally, use relative paths, and do not depend on external networks unless explicitly intended.
- Final deliverables live in the requested path, typically `docs/presentations/`.

## Don'ts
- Do not rewrite the deck during a QA-only review unless the user asks.
- Do not perform repo-wide archaeology when the reviewed claims are narrow.
- Do not accept a polished-looking deck if the evidence is missing or stale.
- Do not treat the presentation as canonical source material for Argos behavior.
