#!/usr/bin/env python3
"""Tail latency logs with optional req_id / component filtering."""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional


def parse_line(line: str) -> Dict[str, str]:
    row: Dict[str, str] = {}
    for part in [p.strip() for p in line.strip().split("|")]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        row[key.strip()] = value.strip()
    return row


def should_print(row: Dict[str, str], req_id: Optional[str], component: Optional[str]) -> bool:
    if req_id and row.get("req_id") != req_id:
        return False
    if component and row.get("component") != component:
        return False
    return True


def render(line: str) -> str:
    row = parse_line(line)
    ts = row.get("ts", "-")
    comp = row.get("component", "-")
    req = row.get("req_id", "-")
    label = row.get("metric") or row.get("event") or "-"
    dur = row.get("duration_s")
    metric_str = f"  {dur}s" if dur is not None else ""
    tool_str = f"  tool={row['tool']}" if row.get("tool") else ""
    extra = ""
    for key in (
        "text_preview",
        "speech_end_unix_s",
        "status",
        "estimated_cost_usd",
        "session_total_cost_usd",
        "model",
    ):
        if row.get(key):
            extra += f"  {key}={row[key]}"
    return f"{ts} | {comp:<6} | req={req} | {label}{metric_str}{tool_str}{extra}".rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail latency logs in a compact format.")
    parser.add_argument("--log-path", default="logs/latency.log", help="Path to latency log file.")
    parser.add_argument("--req-id", default=None, help="Filter to one req_id.")
    parser.add_argument("--component", default=None, help="Filter to one component.")
    parser.add_argument("--follow", action="store_true", help="Keep watching for new lines.")
    parser.add_argument("--last", type=int, default=40, help="Show last N lines before follow.")
    args = parser.parse_args()

    path = Path(args.log_path)
    if not path.exists():
        raise SystemExit(f"Log file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines[-max(args.last, 0):]:
        row = parse_line(line)
        if should_print(row, args.req_id, args.component):
            print(render(line))

    if not args.follow:
        return

    with path.open("r", encoding="utf-8") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.3)
                continue
            row = parse_line(line)
            if should_print(row, args.req_id, args.component):
                print(render(line))


if __name__ == "__main__":
    main()
