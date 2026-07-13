"""Local latency observability helpers for argos_src only."""

import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

_DEFAULT_LOG_PATH = "logs/latency.log"
_LOG_PATH_ENV = "GO2_LATENCY_LOG_PATH"
_LOG_TO_CONSOLE_ENV = "GO2_LATENCY_CONSOLE"

_file_lock = threading.Lock()
_ctx_local = threading.local()


def perf_now() -> float:
    return time.perf_counter()


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _to_str(value: Any) -> str:
    if isinstance(value, float):
        if value != 0.0 and abs(value) < 0.01:
            return f"{value:.8f}"
        return f"{value:.3f}"
    return str(value)


def set_request_context(**kwargs: Any) -> None:
    ctx = dict(getattr(_ctx_local, "value", {}))
    ctx.update({k: v for k, v in kwargs.items() if v is not None})
    _ctx_local.value = ctx


def get_request_context() -> Dict[str, Any]:
    return dict(getattr(_ctx_local, "value", {}))


def clear_request_context() -> None:
    _ctx_local.value = {}


class LatencyLogger:
    def __init__(self, component: str):
        self.component = component
        self._logger = logging.getLogger(f"argos.latency.{component}")

    def _log_path(self) -> Path:
        return Path(os.getenv(_LOG_PATH_ENV, _DEFAULT_LOG_PATH))

    def _console_enabled(self) -> bool:
        value = os.getenv(_LOG_TO_CONSOLE_ENV, "1").lower()
        return value not in {"0", "false", "no"}

    def emit(self, **fields: Any) -> None:
        console_omit_fields = {
            str(field)
            for field in fields.pop("_console_omit_fields", ()) or ()
            if field is not None
        }
        row: Dict[str, Any] = {}
        row.update(get_request_context())
        row.update({k: v for k, v in fields.items() if v is not None})
        row.setdefault("component", self.component)

        for internal in ("speech_end_perf_s", "transcript_perf_s", "communication_id"):
            row.pop(internal, None)

        preferred = ["component", "event", "metric", "duration_s", "tool", "req_id"]

        timestamp = _now_ts()

        def _format_line(log_row: Dict[str, Any]) -> str:
            row_copy = dict(log_row)
            parts = [f"ts={timestamp}"]
            for key in preferred:
                if key in row_copy:
                    parts.append(f"{key}={_to_str(row_copy.pop(key))}")
            for key in sorted(row_copy):
                parts.append(f"{key}={_to_str(row_copy[key])}")
            return " | ".join(parts)

        line = _format_line(row)

        console_row = row
        if console_omit_fields:
            console_row = {
                key: value for key, value in row.items() if key not in console_omit_fields
            }
        console_line = _format_line(console_row)

        path = self._log_path()
        with _file_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        if self._console_enabled():
            print(console_line)
        else:
            self._logger.info(line)

    def timing(self, metric: str, duration_s: float, **fields: Any) -> None:
        self.emit(metric=metric, duration_s=duration_s, **fields)
