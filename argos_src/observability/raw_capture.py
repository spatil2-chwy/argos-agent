"""Opt-in raw audio and face-detection artifact capture for POC sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import queue
import re
import threading
import time
import wave
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

DEFAULT_RAW_DATA_DIR = Path("data_collection/raw_sessions")


def _utc_stamp(value: float | None = None) -> str:
    ts = float(value if value is not None else time.time())
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _safe_name(value: Any, *, fallback: str = "unknown") -> str:
    rendered = str(value or "").strip()
    if not rendered:
        rendered = fallback
    rendered = re.sub(r"[^A-Za-z0-9_.-]+", "_", rendered)
    return rendered.strip("._") or fallback


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True)
class RawFaceSnapshot:
    """Latest face-loop frame and detector output captured around a turn."""

    image: Any
    faces: list[dict[str, Any]]
    camera_resource_id: str = ""
    captured_at_unix_s: float = 0.0


@dataclass
class _CaptureSessionState:
    run_id: str = ""
    last_owner_key: str = ""
    conversation_index: int = 0
    conversation_by_exchange: dict[str, tuple[str, str]] = field(default_factory=dict)


class RawDataCaptureSink:
    """Write raw POC artifacts without coupling the live runtime to disk speed."""

    def __init__(
        self,
        root_dir: str | Path = DEFAULT_RAW_DATA_DIR,
        *,
        queue_maxsize: int = 256,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self._queue: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue(
            maxsize=max(1, int(queue_maxsize))
        )
        self._state = _CaptureSessionState()
        self._state_lock = threading.Lock()
        self._dropped = 0
        self._closed = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="argos-raw-data-capture",
            daemon=True,
        )
        self._worker.start()

    @property
    def run_id(self) -> str:
        return self._state.run_id

    def start_session(self, *, run_id: str, metadata: dict[str, Any] | None = None) -> None:
        run_id = _safe_name(run_id, fallback="run")
        with self._state_lock:
            self._state.run_id = run_id
        self._enqueue(
            "session_start",
            {
                "run_id": run_id,
                "started_at_unix_s": time.time(),
                "metadata": dict(metadata or {}),
            },
        )

    def save_exchange(
        self,
        *,
        exchange_id: str,
        exchange_index: int,
        owner_id: str = "",
        owner_source: str = "",
        audio_pcm16: bytes = b"",
        sample_rate_hz: int = 16000,
        face_snapshot: RawFaceSnapshot | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.run_id:
            return
        owner_key = f"owner:{owner_id}" if owner_id else "anonymous"
        conversation_id, conversation_dir = self._conversation_for_exchange(
            exchange_id=exchange_id,
            owner_key=owner_key,
        )
        self._enqueue(
            "exchange",
            {
                "exchange_id": str(exchange_id or ""),
                "exchange_index": int(exchange_index or 0),
                "conversation_id": conversation_id,
                "conversation_dir": conversation_dir,
                "owner_key": owner_key,
                "owner_id": str(owner_id or ""),
                "owner_source": str(owner_source or ""),
                "audio_pcm16": bytes(audio_pcm16 or b""),
                "sample_rate_hz": int(sample_rate_hz or 16000),
                "face_snapshot": face_snapshot,
                "metadata": dict(metadata or {}),
                "captured_at_unix_s": time.time(),
            },
        )

    def close(self, *, timeout: float = 2.0) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=max(0.0, float(timeout)))

    def _conversation_for_exchange(
        self,
        *,
        exchange_id: str,
        owner_key: str,
    ) -> tuple[str, str]:
        exchange_id = str(exchange_id or "").strip()
        owner_key = str(owner_key or "anonymous").strip() or "anonymous"
        with self._state_lock:
            if exchange_id and exchange_id in self._state.conversation_by_exchange:
                return self._state.conversation_by_exchange[exchange_id]
            if owner_key != self._state.last_owner_key:
                self._state.conversation_index += 1
                self._state.last_owner_key = owner_key
            conversation_id = f"conversation-{self._state.conversation_index:03d}"
            conversation_dir = (
                f"{conversation_id}_{_safe_name(owner_key.replace(':', '_'))}"
            )
            if exchange_id:
                self._state.conversation_by_exchange[exchange_id] = (
                    conversation_id,
                    conversation_dir,
                )
            return conversation_id, conversation_dir

    def _enqueue(self, kind: str, payload: dict[str, Any]) -> None:
        if self._closed.is_set():
            return
        try:
            self._queue.put_nowait((kind, payload))
        except queue.Full:
            self._dropped += 1
            if self._dropped in {1, 10, 100} or self._dropped % 1000 == 0:
                logger.warning("Raw data capture queue full; dropped=%s", self._dropped)

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                kind, payload = item
                if kind == "session_start":
                    self._write_session_start(payload)
                elif kind == "exchange":
                    self._write_exchange(payload)
            except Exception:
                logger.exception("Failed to write raw data capture artifact")
            finally:
                self._queue.task_done()

    def _session_dir(self) -> Path:
        run_id = self.run_id or "run-unassigned"
        return self.root_dir / _safe_name(run_id, fallback="run")

    def _write_session_start(self, payload: dict[str, Any]) -> None:
        session_dir = self._session_dir()
        session_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": payload.get("run_id"),
            "started_at_unix_s": payload.get("started_at_unix_s"),
            "started_at_utc": _utc_stamp(payload.get("started_at_unix_s")),
            "layout": "session/conversations/exchanges",
            "metadata": _json_ready(payload.get("metadata") or {}),
        }
        _write_json(session_dir / "session.json", manifest)

    def _write_exchange(self, payload: dict[str, Any]) -> None:
        exchange_index = int(payload.get("exchange_index") or 0)
        exchange_id = str(payload.get("exchange_id") or "")
        exchange_dir = (
            self._session_dir()
            / "conversations"
            / str(payload.get("conversation_dir") or "conversation-000_anonymous")
            / "exchanges"
            / f"{exchange_index:04d}_{_safe_name(exchange_id, fallback='exchange')}"
        )
        exchange_dir.mkdir(parents=True, exist_ok=True)

        audio_pcm16 = bytes(payload.get("audio_pcm16") or b"")
        audio_file = ""
        if audio_pcm16:
            audio_file = "input_audio_16khz_mono.wav"
            _write_wav(
                exchange_dir / audio_file,
                audio_pcm16,
                sample_rate_hz=int(payload.get("sample_rate_hz") or 16000),
            )

        face_file = ""
        face_json = ""
        snapshot = payload.get("face_snapshot")
        if isinstance(snapshot, RawFaceSnapshot):
            face_file = "face_at_recording_start.jpg"
            face_json = "face_at_recording_start.json"
            _write_image(exchange_dir / face_file, snapshot.image)
            _write_json(
                exchange_dir / face_json,
                {
                    "camera_resource_id": snapshot.camera_resource_id,
                    "captured_at_unix_s": snapshot.captured_at_unix_s,
                    "captured_at_utc": _utc_stamp(snapshot.captured_at_unix_s),
                    "image_file": face_file,
                    "faces": _json_ready(snapshot.faces),
                },
            )

        _write_json(
            exchange_dir / "manifest.json",
            {
                "run_id": self.run_id,
                "conversation_id": payload.get("conversation_id"),
                "owner_key": payload.get("owner_key"),
                "owner_id": payload.get("owner_id"),
                "owner_source": payload.get("owner_source"),
                "exchange_id": exchange_id,
                "exchange_index": exchange_index,
                "captured_at_unix_s": payload.get("captured_at_unix_s"),
                "captured_at_utc": _utc_stamp(payload.get("captured_at_unix_s")),
                "audio_file": audio_file,
                "face_image_file": face_file,
                "face_detection_file": face_json,
                "metadata": _json_ready(payload.get("metadata") or {}),
            },
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_wav(path: Path, audio_pcm16: bytes, *, sample_rate_hz: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate_hz))
        wav.writeframes(audio_pcm16)


def _write_image(path: Path, image: Any) -> None:
    if image is None:
        return
    import cv2

    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image artifact: {path}")
    path.write_bytes(encoded.tobytes())
