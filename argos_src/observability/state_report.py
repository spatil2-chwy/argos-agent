#!/usr/bin/env python3
"""Summarize structured control-plane state events from latency.log."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from argos_src.observability.dashboard_data import parse_latency_line


def parse_line(line: str) -> dict[str, str]:
    return parse_latency_line(line)


def summarize_state_rows(rows: Iterable[dict[str, str]]) -> dict[str, Counter]:
    transitions: Counter[str] = Counter()
    ignored: Counter[str] = Counter()
    reasons: Counter[str] = Counter()

    for row in rows:
        if row.get("component") != "state":
            continue
        axis = row.get("axis", "unknown")
        event = row.get("event", "")
        if event == "transition":
            old_state = row.get("old_state", "?")
            new_state = row.get("new_state", "?")
            trigger = row.get("trigger", "?")
            transitions[f"{axis}:{old_state}->{new_state}:{trigger}"] += 1
        elif event == "ignored":
            reason = row.get("ignored_reason", "unknown")
            trigger = row.get("trigger", "?")
            ignored[f"{axis}:{trigger}:{reason}"] += 1
            reasons[reason] += 1

    return {
        "transitions": transitions,
        "ignored": ignored,
        "ignored_reasons": reasons,
    }


def _print_counter(title: str, counter: Counter[str], *, limit: int) -> None:
    print(title)
    if not counter:
        print("  none")
        return
    for label, count in counter.most_common(limit):
        print(f"  {count:>5}  {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Argos state transition logs.")
    parser.add_argument("--log-path", default="logs/latency.log")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    path = Path(args.log_path)
    if not path.exists():
        raise SystemExit(f"Log file not found: {path}")

    rows = [parse_line(line) for line in path.read_text(encoding="utf-8").splitlines()]
    summary = summarize_state_rows(rows)
    _print_counter("State transitions", summary["transitions"], limit=max(args.limit, 1))
    print()
    _print_counter("Ignored state events", summary["ignored"], limit=max(args.limit, 1))
    print()
    _print_counter("Ignored reasons", summary["ignored_reasons"], limit=max(args.limit, 1))


if __name__ == "__main__":
    main()
