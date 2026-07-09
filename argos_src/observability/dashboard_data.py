"""Build dashboard-ready snapshots from Argos latency logs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence


DEFAULT_LOG_PATH = Path("logs/latency.log")
DEFAULT_SESSION_ID = "local-log"
ERROR_HINTS = ("error", "failed", "failure", "exception", "timeout", "cancel")
HUMAN_REQ_PREFIX = "rt-"
INTERNAL_REQ_PREFIX = "evt-"


def parse_latency_line(line: str) -> dict[str, str]:
    """Parse one pipe-separated latency row."""

    row: dict[str, str] = {}
    for part in [piece.strip() for piece in line.strip().split("|")]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        row[key.strip()] = value.strip()
    return row


def read_latency_rows(path: str | Path = DEFAULT_LOG_PATH) -> list[dict[str, str]]:
    log_path = Path(path)
    if not log_path.exists():
        return []
    return [
        parse_latency_line(line)
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat(timespec="milliseconds")


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _event_label(row: dict[str, str]) -> str:
    return row.get("event") or row.get("metric") or "log"


def _session_id(row: dict[str, str]) -> str:
    return row.get("session_id") or row.get("session") or DEFAULT_SESSION_ID


def _session_key(row: dict[str, str], req_sessions: dict[str, str]) -> str:
    return row.get("run_id") or req_sessions.get(row.get("req_id", "")) or _session_id(row)


def _openai_session_id(row: dict[str, str]) -> str:
    return row.get("openai_session_id") or row.get("session_id") or row.get("session") or ""


def _owner_key(owner_id: Any) -> str:
    rendered = str(owner_id or "").strip()
    if rendered:
        return f"owner:{rendered}"
    return "anonymous"


def _owner_label(owner_id: Any) -> str:
    rendered = str(owner_id or "").strip()
    if rendered:
        return rendered
    return "Unknown speaker"


def _turn_kind(row: dict[str, str]) -> str:
    rendered = str(row.get("turn_kind") or "").strip()
    if rendered:
        return rendered
    req_id = str(row.get("req_id") or "").strip()
    if req_id.startswith(INTERNAL_REQ_PREFIX):
        return "internal_text"
    if req_id.startswith(HUMAN_REQ_PREFIX):
        return "human_audio"
    if _event_label(row) in {"recording_started", "speech_end", "audio_commit"}:
        return "human_audio"
    return ""


def _is_human_exchange_row(row: dict[str, str]) -> bool:
    kind = _turn_kind(row)
    if kind in {"human_audio", "human_text"}:
        return True
    req_id = str(row.get("req_id") or "").strip()
    return req_id.startswith(HUMAN_REQ_PREFIX)


def _exchange_key(row: dict[str, str], req_exchanges: dict[str, str]) -> str:
    if not _is_human_exchange_row(row):
        return ""
    req_id = str(row.get("req_id") or "").strip()
    return str(row.get("exchange_id") or req_exchanges.get(req_id, "") or req_id).strip()


def _status_from_rows(rows: Sequence[dict[str, str]]) -> str:
    for row in rows:
        terminal_status = str(row.get("terminal_status") or "").lower()
        if terminal_status in {"error", "canceled", "cancelled", "superseded"}:
            return "error" if terminal_status == "error" else terminal_status
        label = _event_label(row).lower()
        if any(hint in label for hint in ERROR_HINTS):
            return "error"
        ignored_reason = row.get("ignored_reason", "").lower()
        if any(hint in ignored_reason for hint in ERROR_HINTS):
            return "error"
    if any(
        row.get("event")
        in {"exchange_complete", "playback_completed", "response_done", "response_usage"}
        for row in rows
    ):
        return "complete"
    if rows:
        return "active"
    return "unknown"


def _date_label(value: datetime | None) -> str:
    if value is None:
        return "Unknown time"
    return value.strftime("%b %-d %I:%M:%S %p")


def _stage_details(row: dict[str, str]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for key in (
        "trigger",
        "admission_reason",
        "interaction_state",
        "primary_face_person_id",
        "visible_face_person_ids",
        "audio_speaker_id",
        "owner_id",
        "owner_source",
        "owner_confidence",
        "old_owner_key",
        "new_owner_key",
        "deleted_items",
        "protected_items",
        "history_action",
        "memory_segment_id",
        "memory_person_id",
        "memory_turn_count",
        "memory_flush_reason",
        "memory_extraction_enabled",
        "memory_extraction_scheduled",
        "tailwag_episode_id",
        "tailwag_episode_extract_memory",
        "tailwag_memory_result_count",
        "tailwag_memory_created_count",
        "tailwag_memory_addressed_count",
        "tailwag_memory_supported_count",
        "tailwag_memory_error_count",
        "tailwag_episode_error",
        "biometric_update_modality",
        "biometric_update_person_id",
        "biometric_update_accepted",
        "biometric_update_status",
        "biometric_update_reason",
        "biometric_update_reference_id",
        "biometric_update_sample_count",
        "biometric_update_target_sample_count",
        "biometric_update_similarity",
        "audio_score",
        "audio_runner_up_score",
        "audio_score_margin",
        "face_match_status",
        "face_match_reason",
        "face_match_name",
        "face_match_person_id",
        "face_score",
        "face_score_threshold",
        "face_runner_up_score",
        "face_score_margin",
        "face_margin_threshold",
        "speaker_visible",
        "tool",
        "call_id",
        "tool_arguments_json",
        "tool_success",
        "tool_result_preview",
        "response_status",
        "terminal_status",
        "terminal_reason",
        "error_source",
        "error_type",
        "error_code",
        "error_message",
        "server_error_type",
        "server_error_code",
        "server_error_message",
        "pending_internal_events",
        "capture_vad_positive_blocks",
        "audio_duration_s",
        "estimated_cost_usd",
        "session_total_cost_usd",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "cache_hit_ratio",
    ):
        value = row.get(key)
        if value not in (None, ""):
            details[key] = value
    duration = _float(row, "duration_s")
    if duration is not None:
        details["duration_s"] = duration
    return details


def _stage_from_row(row: dict[str, str]) -> dict[str, Any] | None:
    label = _event_label(row)
    stage_map = {
        "recording_started": ("recording", "Recording started"),
        "speech_end": ("speech_end", "Speech ended"),
        "audio_commit": ("audio_commit", "Audio committed"),
        "exchange_context": ("identity", "Speaker and owner resolved"),
        "owner_handoff": ("owner_handoff", "Owner handoff"),
        "memory_segment_flushed": ("memory_flushed", "Memory segment flushed"),
        "tailwag_episode_recorded": ("tailwag_episode_recorded", "Tailwag episode recorded"),
        "tailwag_episode_failed": ("tailwag_episode_failed", "Tailwag episode failed"),
        "tailwag_episode_skipped": ("tailwag_episode_skipped", "Tailwag episode skipped"),
        "adaptive_biometric_update": (
            "biometric_update",
            "Biometric reference update",
        ),
        "response_create": ("model_requested", "Model requested"),
        "first_audio_latency_s": ("first_audio", "First reply audio"),
        "tool_call_requested": ("tool_requested", "Tool requested"),
        "tool_result": ("tool_finished", "Tool finished"),
        "response_done": ("response_done", "Response complete"),
        "response_usage": ("response_usage", "Usage recorded"),
        "playback_completed": ("playback_completed", "Playback complete"),
        "playback_stopped": ("playback_stopped", "Playback stopped"),
        "exchange_complete": ("exchange_complete", "Exchange complete"),
        "exchange_terminal": ("exchange_terminal", "Exchange ended"),
    }
    mapped = stage_map.get(label)
    if mapped is None:
        return None
    key, title = mapped
    if label == "adaptive_biometric_update":
        title = _biometric_update_title(row)
    if label == "tailwag_episode_recorded":
        created = str(row.get("tailwag_memory_created_count") or "0").strip()
        errors = str(row.get("tailwag_memory_error_count") or "0").strip()
        title = f"Tailwag episode recorded: {created} memories, {errors} errors"
    if label == "tailwag_episode_failed":
        reason = str(row.get("tailwag_episode_error") or "unknown").strip()
        title = f"Tailwag episode failed: {reason}"
    return {
        "key": key,
        "title": title,
        "ts": row.get("ts"),
        "label": label,
        "component": row.get("component", "unknown"),
        "details": _stage_details(row),
    }


def _biometric_update_title(row: dict[str, str]) -> str:
    modality = str(row.get("biometric_update_modality") or "reference").strip()
    status = str(row.get("biometric_update_status") or "unknown").strip()
    reason = str(row.get("biometric_update_reason") or "").strip()
    sample_count = str(row.get("biometric_update_sample_count") or "").strip()
    target_count = str(row.get("biometric_update_target_sample_count") or "").strip()
    count_suffix = (
        f" ({sample_count}/{target_count})"
        if sample_count and target_count and target_count != "0"
        else ""
    )
    if status == "updated":
        return f"Biometric {modality} update accepted{count_suffix}"
    if status == "complete":
        return f"Biometric {modality} update complete{count_suffix}"
    if reason:
        return f"Biometric {modality} update {status}: {reason}{count_suffix}"
    return f"Biometric {modality} update {status}{count_suffix}"


@dataclass
class InteractionAccumulator:
    req_id: str
    session_id: str
    exchange_id: str = ""
    exchange_index: int = 0
    rows: list[dict[str, str]] = field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None

    def add(self, row: dict[str, str]) -> None:
        self.rows.append(row)
        if not self.exchange_id:
            self.exchange_id = str(row.get("exchange_id") or "").strip()
        if not self.req_id:
            self.req_id = str(row.get("req_id") or "").strip()
        if self.session_id == DEFAULT_SESSION_ID and _session_id(row) != DEFAULT_SESSION_ID:
            self.session_id = _session_id(row)
        if not self.exchange_index:
            try:
                self.exchange_index = int(row.get("exchange_index") or 0)
            except ValueError:
                self.exchange_index = 0
        ts = _parse_ts(row.get("ts"))
        if ts is None:
            return
        if self.started_at is None or ts < self.started_at:
            self.started_at = ts
        if self.ended_at is None or ts > self.ended_at:
            self.ended_at = ts

    def to_payload(self) -> dict[str, Any]:
        events: Counter[str] = Counter()
        metrics: dict[str, float] = {}
        state_transitions: list[dict[str, Any]] = []
        ignored_state_events: list[dict[str, Any]] = []
        tools: Counter[str] = Counter()
        costs: dict[str, float] = {}
        timeline: list[dict[str, Any]] = []
        lifecycle: list[dict[str, Any]] = []
        context: dict[str, Any] = {}
        openai_session_ids: set[str] = set()
        state_by_axis: dict[str, dict[str, Any]] = {}
        tool_calls_by_key: dict[str, dict[str, Any]] = {}
        counted_tool_keys: set[str] = set()
        estimated_exchange_cost = 0.0

        for index, row in enumerate(self.rows):
            label = _event_label(row)
            openai_id = _openai_session_id(row)
            if openai_id:
                openai_session_ids.add(openai_id)
            if row.get("event"):
                events[label] += 1
            if row.get("metric"):
                duration = _float(row, "duration_s")
                if duration is not None:
                    metrics[label] = duration
            if row.get("tool"):
                tool_key = str(row.get("call_id") or row.get("tool") or f"tool-{index}")
                if tool_key not in counted_tool_keys:
                    tools[row["tool"]] += 1
                    counted_tool_keys.add(tool_key)
            for cost_key in ("estimated_cost_usd", "session_total_cost_usd"):
                cost = _float(row, cost_key)
                if cost is not None:
                    costs[cost_key] = cost
                    if cost_key == "estimated_cost_usd":
                        estimated_exchange_cost += cost
            for context_key in (
                "trigger",
                "admission_reason",
                "interaction_state",
                "primary_face_person_id",
                "visible_face_person_ids",
                "audio_speaker_id",
                "owner_id",
                "owner_source",
                "owner_confidence",
                "audio_score",
                "audio_runner_up_score",
                "audio_score_margin",
                "face_match_status",
                "face_match_reason",
                "face_match_name",
                "face_match_person_id",
                "face_score",
                "face_score_threshold",
                "face_runner_up_score",
                "face_score_margin",
                "face_margin_threshold",
                "speaker_visible",
                "turn_kind",
                "terminal_status",
                "terminal_reason",
                "error_source",
                "error_type",
                "error_code",
                "error_message",
                "server_error_type",
                "server_error_code",
                "server_error_message",
            ):
                value = row.get(context_key)
                if value not in (None, ""):
                    context[context_key] = value
            if row.get("component") == "state" and row.get("event") == "transition":
                transition = {
                    "axis": row.get("axis", "unknown"),
                    "old_state": row.get("old_state"),
                    "new_state": row.get("new_state"),
                    "trigger": row.get("trigger"),
                    "ts": row.get("ts"),
                }
                state_transitions.append(transition)
                axis = str(transition["axis"] or "unknown")
                state_by_axis.setdefault(
                    axis,
                    {"axis": axis, "transitions": [], "ignored": []},
                )["transitions"].append(transition)
            if row.get("component") == "state" and row.get("event") == "ignored":
                ignored = {
                    "axis": row.get("axis", "unknown"),
                    "trigger": row.get("trigger"),
                    "ignored_reason": row.get("ignored_reason", "unknown"),
                    "ts": row.get("ts"),
                }
                ignored_state_events.append(ignored)
                axis = str(ignored["axis"] or "unknown")
                state_by_axis.setdefault(
                    axis,
                    {"axis": axis, "transitions": [], "ignored": []},
                )["ignored"].append(ignored)
            if label in {"tool_call_requested", "tool_result"} or row.get("tool"):
                call_key = str(row.get("call_id") or row.get("tool") or f"tool-{index}")
                tool_call = tool_calls_by_key.setdefault(
                    call_key,
                    {
                        "call_id": row.get("call_id") or "",
                        "tool": row.get("tool") or "",
                        "requested_at": None,
                        "finished_at": None,
                        "arguments_json": "",
                        "result_preview": "",
                        "success": None,
                    },
                )
                if row.get("tool") and not tool_call["tool"]:
                    tool_call["tool"] = row.get("tool")
                if row.get("call_id") and not tool_call["call_id"]:
                    tool_call["call_id"] = row.get("call_id")
                if label == "tool_call_requested":
                    tool_call["requested_at"] = row.get("ts")
                    tool_call["arguments_json"] = row.get("tool_arguments_json", "")
                if label == "tool_result":
                    tool_call["finished_at"] = row.get("ts")
                    tool_call["result_preview"] = row.get("tool_result_preview", "")
                    tool_call["success"] = row.get("tool_success")
            stage = _stage_from_row(row)
            if stage is not None:
                lifecycle.append(stage)
            timeline.append(
                {
                    "index": index,
                    "ts": row.get("ts"),
                    "component": row.get("component", "unknown"),
                    "kind": "metric" if row.get("metric") else "event",
                    "label": label,
                    "axis": row.get("axis"),
                    "old_state": row.get("old_state"),
                    "new_state": row.get("new_state"),
                    "trigger": row.get("trigger"),
                    "duration_s": _float(row, "duration_s"),
                    "tool": row.get("tool"),
                    "call_id": row.get("call_id"),
                    "tool_arguments_json": row.get("tool_arguments_json"),
                    "tool_success": row.get("tool_success"),
                    "tool_result_preview": row.get("tool_result_preview"),
                    "ignored_reason": row.get("ignored_reason"),
                }
            )

        duration_s: float | None = None
        if self.started_at is not None and self.ended_at is not None:
            duration_s = max((self.ended_at - self.started_at).total_seconds(), 0.0)

        exchange_id = self.exchange_id or self.req_id
        label_index = self.exchange_index
        display_label = (
            f"Exchange {label_index}"
            if label_index
            else f"{format_time_for_label(self.started_at)} human -> Argos"
        )
        if estimated_exchange_cost > 0:
            costs["estimated_exchange_cost_usd"] = estimated_exchange_cost

        return {
            "exchange_id": exchange_id,
            "exchange_index": self.exchange_index,
            "label": display_label,
            "req_id": self.req_id,
            "raw_req_ids": sorted({str(row.get("req_id") or "") for row in self.rows if row.get("req_id")}),
            "session_id": self.session_id,
            "openai_session_ids": sorted(openai_session_ids),
            "started_at": _iso(self.started_at),
            "ended_at": _iso(self.ended_at),
            "duration_s": duration_s,
            "status": _status_from_rows(self.rows),
            "context": context,
            "lifecycle": lifecycle,
            "event_counts": dict(events),
            "metrics": metrics,
            "state_transitions": state_transitions,
            "ignored_state_events": ignored_state_events,
            "state_by_axis": [
                state_by_axis[axis]
                for axis in sorted(
                    state_by_axis,
                    key=lambda value: (
                        (
                            "capture",
                            "turn",
                            "playback",
                            "engagement",
                            "transcription",
                            "robot_arbitration",
                            "coalescer",
                            "session",
                        ).index(value)
                        if value
                        in {
                            "capture",
                            "turn",
                            "playback",
                            "engagement",
                            "transcription",
                            "robot_arbitration",
                            "coalescer",
                            "session",
                        }
                        else 99
                    ),
                )
            ],
            "tools": dict(tools),
            "tool_calls": list(tool_calls_by_key.values()),
            "costs": costs,
            "first_audio_latency_s": metrics.get("first_audio_latency_s"),
            "timeline": timeline,
        }


@dataclass
class SessionAccumulator:
    session_id: str
    rows: list[dict[str, str]] = field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None

    def add(self, row: dict[str, str]) -> None:
        self.rows.append(row)
        ts = _parse_ts(row.get("ts"))
        if ts is None:
            return
        if self.started_at is None or ts < self.started_at:
            self.started_at = ts
        if self.ended_at is None or ts > self.ended_at:
            self.ended_at = ts


def _conversation_segments_for_interactions(
    interaction_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive consecutive owner-scoped conversation segments per session."""

    by_session: dict[str, list[dict[str, Any]]] = {}
    for item in interaction_payloads:
        by_session.setdefault(str(item.get("session_id") or DEFAULT_SESSION_ID), []).append(item)

    segments: list[dict[str, Any]] = []
    for session_id, session_items in by_session.items():
        ordered = sorted(
            session_items,
            key=lambda item: (
                str(item.get("started_at") or ""),
                int(item.get("exchange_index") or 0),
                str(item.get("req_id") or ""),
            ),
        )
        active: dict[str, Any] | None = None
        previous_owner_key = ""
        segment_index = 0
        for item in ordered:
            context = item.get("context") if isinstance(item.get("context"), dict) else {}
            owner_id = context.get("owner_id") if context is not None else ""
            owner_key = _owner_key(owner_id)
            owner_source = str((context or {}).get("owner_source") or "unknown")
            if active is None or active["owner_key"] != owner_key:
                if active is not None:
                    _finalize_conversation_segment(active)
                    segments.append(active)
                    previous_owner_key = str(active["owner_key"])
                segment_index += 1
                active = {
                    "segment_id": f"{session_id}:conversation:{segment_index}",
                    "session_id": session_id,
                    "segment_index": segment_index,
                    "owner_key": owner_key,
                    "owner_id": str(owner_id or ""),
                    "owner_label": _owner_label(owner_id),
                    "started_at": item.get("started_at"),
                    "ended_at": item.get("ended_at"),
                    "duration_s": None,
                    "exchange_count": 0,
                    "exchange_ids": [],
                    "exchange_indexes": [],
                    "first_exchange_id": "",
                    "latest_exchange_id": "",
                    "status": "unknown",
                    "status_counts": Counter(),
                    "owner_source_counts": Counter(),
                    "owner_sources": [],
                    "avg_first_audio_latency_s": None,
                    "total_exchange_cost_usd": None,
                    "handoff_from_owner_key": previous_owner_key,
                    "handoff_to_owner_key": owner_key,
                    "boundary_reason": "session_start" if not previous_owner_key else "owner_handoff",
                    "_latencies": [],
                    "_costs": [],
                }

            if active is None:
                continue
            exchange_id = str(item.get("exchange_id") or item.get("req_id") or "")
            active["exchange_count"] += 1
            active["exchange_ids"].append(exchange_id)
            exchange_index = item.get("exchange_index")
            if exchange_index:
                active["exchange_indexes"].append(exchange_index)
            if not active["first_exchange_id"]:
                active["first_exchange_id"] = exchange_id
            active["latest_exchange_id"] = exchange_id
            if item.get("started_at") and (
                not active["started_at"] or str(item["started_at"]) < str(active["started_at"])
            ):
                active["started_at"] = item["started_at"]
            if item.get("ended_at") and (
                not active["ended_at"] or str(item["ended_at"]) > str(active["ended_at"])
            ):
                active["ended_at"] = item["ended_at"]
            active["status_counts"][str(item.get("status") or "unknown")] += 1
            if owner_source:
                active["owner_source_counts"][owner_source] += 1
            latency = item.get("first_audio_latency_s")
            if isinstance(latency, (int, float)):
                active["_latencies"].append(float(latency))
            costs = item.get("costs") if isinstance(item.get("costs"), dict) else {}
            exchange_cost = None
            if isinstance(costs, dict):
                exchange_cost = costs.get("estimated_exchange_cost_usd", costs.get("estimated_cost_usd"))
            if isinstance(exchange_cost, (int, float)):
                active["_costs"].append(float(exchange_cost))

            item["conversation_segment_id"] = active["segment_id"]
            item["conversation_segment_index"] = active["exchange_count"]
            item["owner_key"] = owner_key

        if active is not None:
            _finalize_conversation_segment(active)
            segments.append(active)

    segments.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
    return segments


def _finalize_conversation_segment(segment: dict[str, Any]) -> None:
    started = _parse_ts(str(segment.get("started_at") or ""))
    ended = _parse_ts(str(segment.get("ended_at") or ""))
    if started is not None and ended is not None:
        segment["duration_s"] = max((ended - started).total_seconds(), 0.0)
    status_counts = segment.pop("status_counts")
    source_counts = segment.pop("owner_source_counts")
    latencies = segment.pop("_latencies")
    costs = segment.pop("_costs")
    segment["status_counts"] = dict(status_counts)
    segment["owner_source_counts"] = dict(source_counts)
    segment["owner_sources"] = [source for source, _ in source_counts.most_common()]
    if status_counts.get("error", 0):
        segment["status"] = "error"
    elif status_counts.get("active", 0):
        segment["status"] = "active"
    elif status_counts:
        segment["status"] = status_counts.most_common(1)[0][0]
    segment["avg_first_audio_latency_s"] = mean(latencies) if latencies else None
    segment["total_exchange_cost_usd"] = sum(costs) if costs else None


def format_time_for_label(value: datetime | None) -> str:
    if value is None:
        return "Unknown time"
    return value.strftime("%I:%M:%S %p").lstrip("0")


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    index = (len(clean) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(clean) - 1)
    weight = index - lower
    return clean[lower] * (1 - weight) + clean[upper] * weight


def build_dashboard_snapshot(
    rows: Iterable[dict[str, str]],
    *,
    source: str = str(DEFAULT_LOG_PATH),
) -> dict[str, Any]:
    row_list = [row for row in rows if row]
    req_sessions: dict[str, str] = {}
    req_exchanges: dict[str, str] = {}
    for row in row_list:
        req_id = str(row.get("req_id") or "").strip()
        if not req_id:
            continue
        run_id = str(row.get("run_id") or "").strip()
        if run_id:
            req_sessions[req_id] = run_id
        elif req_id not in req_sessions and _session_id(row) != DEFAULT_SESSION_ID:
            req_sessions[req_id] = _session_id(row)
        exchange_id = str(row.get("exchange_id") or "").strip()
        if exchange_id:
            req_exchanges[req_id] = exchange_id
    sessions: dict[str, SessionAccumulator] = {}
    interactions: dict[str, InteractionAccumulator] = {}
    system_events: list[dict[str, Any]] = []
    pending_human_rows: list[dict[str, str]] = []
    components: Counter[str] = Counter()
    state_axes: Counter[str] = Counter()
    ignored_reasons: Counter[str] = Counter()
    error_rows: list[dict[str, Any]] = []
    latest_total_cost: tuple[datetime, int, float] | None = None

    for row_index, row in enumerate(row_list):
        session_id = _session_key(row, req_sessions)
        session = sessions.setdefault(session_id, SessionAccumulator(session_id=session_id))
        session.add(row)
        components[row.get("component", "unknown")] += 1

        if row.get("component") == "state" and row.get("axis"):
            state_axes[row["axis"]] += 1
        if row.get("ignored_reason"):
            ignored_reasons[row["ignored_reason"]] += 1

        label = _event_label(row)
        if any(hint in label.lower() for hint in ERROR_HINTS):
            error_rows.append(
                {
                    "row_index": row_index,
                    "ts": row.get("ts"),
                    "session_id": session_id,
                    "component": row.get("component", "unknown"),
                    "label": label,
                    "req_id": row.get("req_id"),
                }
            )
        session_total_cost = _float(row, "session_total_cost_usd")
        ts = _parse_ts(row.get("ts")) or datetime.min
        if session_total_cost is not None:
            candidate = (ts, row_index, session_total_cost)
            if latest_total_cost is None or candidate[:2] > latest_total_cost[:2]:
                latest_total_cost = candidate

        key = _exchange_key(row, req_exchanges)
        if key:
            interaction = interactions.setdefault(
                key,
                InteractionAccumulator(
                    req_id=str(row.get("req_id") or ""),
                    session_id=session_id,
                    exchange_id=str(row.get("exchange_id") or key),
                ),
            )
            if pending_human_rows and _event_label(row) == "audio_commit":
                for pending in pending_human_rows:
                    interaction.add(pending)
                pending_human_rows = []
            interaction.add(row)
        else:
            if (
                _event_label(row) in {"recording_started", "speech_end"}
                and not row.get("req_id")
            ):
                pending_human_rows.append(row)
                continue
            system_events.append(
                {
                    "row_index": row_index,
                    "ts": row.get("ts"),
                    "session_id": session_id,
                    "component": row.get("component", "unknown"),
                    "label": label,
                    "axis": row.get("axis"),
                    "trigger": row.get("trigger"),
                    "ignored_reason": row.get("ignored_reason"),
                }
            )

    interaction_payloads = sorted(
        (interaction.to_payload() for interaction in interactions.values()),
        key=lambda item: (item["started_at"] or "", item["req_id"]),
        reverse=True,
    )
    conversation_segments = _conversation_segments_for_interactions(interaction_payloads)
    latency_values = [
        item["first_audio_latency_s"]
        for item in interaction_payloads
        if item.get("first_audio_latency_s") is not None
    ]
    tool_metrics = [
        item["metrics"][metric_name]
        for item in interaction_payloads
        for metric_name in ("tool_dispatch_s", "memory_query_s")
        if metric_name in item["metrics"]
    ]
    status_counts = Counter(item["status"] for item in interaction_payloads)
    latest_cost_by_session: dict[str, tuple[str, float]] = {}
    for item in interaction_payloads:
        session_total = item["costs"].get("session_total_cost_usd")
        if session_total is None:
            continue
        ended_at = str(item.get("ended_at") or item.get("started_at") or "")
        current = latest_cost_by_session.get(item["session_id"])
        if current is None or ended_at >= current[0]:
            latest_cost_by_session[item["session_id"]] = (ended_at, float(session_total))

    all_sessions_payload = []
    for session in sessions.values():
        session_interactions = [
            item for item in interaction_payloads if item["session_id"] == session.session_id
        ]
        latencies = [
            item["first_audio_latency_s"]
            for item in session_interactions
            if item.get("first_audio_latency_s") is not None
        ]
        latest_session_cost = latest_cost_by_session.get(session.session_id)
        standalone_session_errors = sum(
            1
            for row in error_rows
            if row.get("session_id") == session.session_id and row.get("req_id") is None
        )
        all_sessions_payload.append(
            {
                "session_id": session.session_id,
                "label": _date_label(session.started_at),
                "started_at": _iso(session.started_at),
                "ended_at": _iso(session.ended_at),
                "row_count": len(session.rows),
                "exchange_count": len(session_interactions),
                "interaction_count": len(session_interactions),
                "error_count": (
                    sum(1 for item in session_interactions if item["status"] == "error")
                    + standalone_session_errors
                ),
                "avg_first_audio_latency_s": mean(latencies) if latencies else None,
                "session_total_cost_usd": (
                    latest_session_cost[1] if latest_session_cost is not None else None
                ),
            }
        )
    all_sessions_payload.sort(key=lambda item: item["started_at"] or "", reverse=True)
    sessions_payload = [item for item in all_sessions_payload if item["exchange_count"] > 0]

    return {
        "source": source,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "row_count": len(row_list),
            "session_count": len(sessions_payload),
            "raw_session_count": len(all_sessions_payload),
            "exchange_count": len(interaction_payloads),
            "interaction_count": len(interaction_payloads),
            "conversation_segment_count": len(conversation_segments),
            "system_event_count": len(system_events),
            "error_count": (
                status_counts.get("error", 0)
                + sum(1 for row in error_rows if row.get("req_id") is None)
            ),
            "status_counts": dict(status_counts),
            "component_counts": dict(components),
            "state_axis_counts": dict(state_axes),
            "ignored_reason_counts": dict(ignored_reasons),
            "first_audio_latency_avg_s": mean(latency_values) if latency_values else None,
            "first_audio_latency_p50_s": _percentile(latency_values, 0.5),
            "first_audio_latency_p95_s": _percentile(latency_values, 0.95),
            "first_audio_latency_max_s": max(latency_values) if latency_values else None,
            "tool_latency_avg_s": mean(tool_metrics) if tool_metrics else None,
            "total_logged_cost_usd": (
                sum(cost for _, cost in latest_cost_by_session.values())
                if latest_cost_by_session
                else None
            ),
            "latest_session_total_cost_usd": (
                latest_total_cost[2] if latest_total_cost is not None else None
            ),
        },
        "sessions": sessions_payload,
        "conversation_segments": conversation_segments,
        "exchanges": interaction_payloads,
        "interactions": interaction_payloads,
        "system_events": system_events[-200:],
        "errors": error_rows[-100:],
    }


def load_dashboard_snapshot(path: str | Path = DEFAULT_LOG_PATH) -> dict[str, Any]:
    log_path = Path(path)
    return build_dashboard_snapshot(read_latency_rows(log_path), source=str(log_path))
