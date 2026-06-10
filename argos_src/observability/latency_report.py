#!/usr/bin/env python3
"""Summarize latency.log into mean/p50/p95/max tables per metric."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


def parse_line(line: str) -> Dict[str, str]:
    parts = [p.strip() for p in line.strip().split("|")]
    out: Dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def fmt(value: float) -> str:
    return f"{value:.3f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize latency logs (mean/p50/p95/max).")
    parser.add_argument(
        "--log-path",
        default="logs/latency.log",
        help="Path to latency log file (default: logs/latency.log).",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        row = parse_line(raw)
        metric = row.get("metric")
        duration = row.get("duration_s")
        if not metric or not duration:
            continue
        try:
            value = float(duration)
        except ValueError:
            continue
        component = row.get("component", "unknown")
        tool = row.get("tool")
        key = (component, f"{metric}[{tool}]" if tool else metric)
        groups[key].append(value)

    if not groups:
        print("No latency records found.")
        return

    col_w = max(len(f"{c}/{m}") for c, m in groups) + 2
    header = f"{'component/metric':<{col_w}} {'count':>5}  {'mean':>8}  {'p50':>8}  {'p95':>8}  {'max':>8}"
    print(header)
    print("-" * len(header))
    for (component, metric), values in sorted(groups.items()):
        label = f"{component}/{metric}"
        print(
            f"{label:<{col_w}} {len(values):>5}  {fmt(mean(values)):>8}  "
            f"{fmt(percentile(values, 50)):>8}  {fmt(percentile(values, 95)):>8}  {fmt(max(values)):>8}"
        )


if __name__ == "__main__":
    main()

