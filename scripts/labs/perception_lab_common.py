#!/usr/bin/env python3
"""Shared output helpers for perception lab capture scripts."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAB_ROOT = REPO_ROOT / "var" / "labs"


def make_run_id(prefix: str = "") -> str:
    rendered_prefix = _safe_path_part(prefix)
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{rendered_prefix}_{suffix}" if rendered_prefix else suffix


def _safe_path_part(value: str) -> str:
    rendered = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(value or "").strip()
    ).strip("_")
    return rendered or ""


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(payload), sort_keys=True, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            rendered = line.strip()
            if not rendered:
                continue
            payload = json.loads(rendered)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{lineno} must contain a JSON object")
            rows.append(payload)
    return rows


def current_git_commit(repo_root: str | Path = REPO_ROOT) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


class LabRunWriter:
    """Owns the standard file layout for one perception lab run."""

    def __init__(
        self,
        *,
        component: str,
        mode: str,
        root: str | Path = DEFAULT_LAB_ROOT,
        run_id: str | None = None,
    ) -> None:
        self.component = _safe_path_part(component) or "perception"
        self.mode = _safe_path_part(mode) or "run"
        self.run_id = _safe_path_part(run_id or make_run_id()) or make_run_id()
        self.run_dir = Path(root) / self.component / self.mode / self.run_id
        self.artifacts_dir = self.run_dir / "artifacts"
        self.reports_dir = self.run_dir / "reports"
        self.samples_path = self.run_dir / "samples.jsonl"
        self.labels_path = self.run_dir / "labels.todo.jsonl"
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.samples_path.write_text("", encoding="utf-8")
        self.labels_path.write_text("", encoding="utf-8")

    def write_manifest(self, payload: dict[str, Any]) -> None:
        manifest = {
            "component": self.component,
            "mode": self.mode,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "created_at_unix_s": round(time.time(), 3),
            "git_commit": current_git_commit(),
            "command": list(sys.argv),
            **dict(payload),
        }
        write_json(self.manifest_path, manifest)

    def append_sample(self, sample: dict[str, Any], label_template: dict[str, Any]) -> None:
        append_jsonl(self.samples_path, sample)
        append_jsonl(self.labels_path, label_template)

    def write_quick_summary(self, lines: Iterable[str]) -> Path:
        path = self.reports_dir / "quick_summary.md"
        rendered = "\n".join(str(line).rstrip() for line in lines).rstrip() + "\n"
        path.write_text(rendered, encoding="utf-8")
        return path


def yes_no_label(value: Any) -> bool | None:
    rendered = str(value or "").strip().lower()
    if rendered in {"yes", "y", "true", "1"}:
        return True
    if rendered in {"no", "n", "false", "0"}:
        return False
    return None
