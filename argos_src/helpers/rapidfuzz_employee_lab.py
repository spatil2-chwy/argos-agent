#!/usr/bin/env python3
"""Standalone employee-directory registration probe.

Run from `argos_src`:

    source setup_shell.sh
    poetry run python -m argos_src.helpers.rapidfuzz_employee_lab --sites bos1,bos3
    poetry run python -m argos_src.helpers.rapidfuzz_employee_lab --sites bos1 bos3 --loop

This helper is intentionally narrow. It tests the registration name-matching
path in a standalone way:

1. capture one spoken utterance from the microphone
2. transcribe it with the configured OpenAI transcription model
3. run a narrow Realtime pass with the agent prompt and
   `resolve_employee_identity` tool schema
4. inspect the exact tool-call args the model emitted
5. run the shared employee-directory matcher with those same args
"""

from __future__ import annotations

import argparse
import audioop
from collections import deque
from dataclasses import asdict, dataclass
import io
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time
import unicodedata
import wave
from typing import Any, Callable, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from argos_src.employee_directory import EmployeeDirectoryService
from argos_src.employee_directory.service import load_directory_records_by_site
from argos_src.helpers.face_lab_common import configure_logging, json_print
from argos_src.openai_realtime import (
    realtime_auth_headers,
    realtime_response_payload,
    realtime_text_session_payload,
    realtime_websocket_url,
)
from argos_src.profile_config import load_scenario_profile, resolve_prompt_file
from argos_src.prompts.loader import load_system_prompt
from argos_src.tools.unitree_go2.vision.resolve_employee_identity import (
    get_resolve_employee_identity_tool,
)

AUDIO_CHANNELS = 1
AUDIO_DTYPE = "int16"
VAD_SAMPLE_RATE = 16000
DEFAULT_LISTEN_TIMEOUT_SEC = 10.0
DEFAULT_MAX_RECORD_SEC = 6.0
DEFAULT_RMS_THRESHOLD = 350.0
_SITE_SPLIT_RE = re.compile(r"[\s,]+")


@dataclass(frozen=True)
class SpokenName:
    shared_name: str
    shared_first_name: str
    shared_last_name: str


@dataclass(frozen=True)
class LoadedDirectory:
    site_code: str
    service: Any
    record_count: int
    load_error: str = ""


@dataclass(frozen=True)
class RuntimeSettings:
    profile_name: str
    fallback_site: str
    input_device: str
    input_sample_rate: int
    input_block_size: int
    vad_threshold: float
    silence_grace_period: float
    transcription_model: str
    realtime_model: str


class _RmsVAD:
    """Small fallback VAD when Silero is unavailable locally."""

    def __init__(self, rms_threshold: float) -> None:
        self.rms_threshold = float(rms_threshold)

    def __call__(
        self,
        audio_data: np.ndarray,
        input_parameters: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        samples = np.asarray(audio_data, dtype=np.float32)
        if samples.size == 0:
            rms = 0.0
        else:
            rms = float(np.sqrt(np.mean(np.square(samples))))
        result = dict(input_parameters)
        result["rms_vad"] = {
            "rms": round(rms, 4),
            "threshold": round(self.rms_threshold, 4),
        }
        return rms >= self.rms_threshold, result


class MicrophoneRecorder:
    def __init__(
        self,
        *,
        input_device: str,
        sample_rate: int,
        block_size: int,
        silence_grace_period: float,
        listen_timeout_sec: float,
        max_record_sec: float,
        vad: Callable[[np.ndarray, dict[str, Any]], tuple[bool, dict[str, Any]]],
    ) -> None:
        self.input_device = input_device
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.silence_grace_period = float(silence_grace_period)
        self.listen_timeout_sec = float(listen_timeout_sec)
        self.max_record_sec = float(max_record_sec)
        self.vad = vad

    def record_one(self) -> dict[str, Any]:
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - local env dependent.
            return {
                "success": False,
                "failure_reason": "missing_sounddevice",
                "message": f"sounddevice is required for microphone mode: {exc}",
            }

        lock = threading.Lock()
        done_event = threading.Event()
        pre_roll_chunks: deque[np.ndarray] = deque(maxlen=5)
        state: dict[str, Any] = {
            "listening_started_at": time.time(),
            "recording_started_at": 0.0,
            "last_voice_at": 0.0,
            "recording": False,
            "timed_out": False,
            "done": False,
            "stop_reason": "",
            "status_flags": [],
            "chunks": [],
            "vad_frames": 0,
        }
        resample_state = None

        def _finish(stop_reason: str) -> None:
            with lock:
                if state["done"]:
                    return
                state["done"] = True
                state["stop_reason"] = stop_reason
            done_event.set()

        def _callback(
            indata: np.ndarray,
            frames: int,
            time_info: Any,
            status: Any,
        ) -> None:
            del frames, time_info
            nonlocal resample_state
            raw_chunk = np.asarray(indata.copy(), dtype=np.int16).reshape(-1)
            pre_roll_chunks.append(raw_chunk)
            if status:
                with lock:
                    state["status_flags"].append(str(status))
            try:
                resampled, resample_state = audioop.ratecv(
                    raw_chunk.tobytes(),
                    np.dtype(np.int16).itemsize,
                    AUDIO_CHANNELS,
                    self.sample_rate,
                    VAD_SAMPLE_RATE,
                    resample_state,
                )
                audio_16k = np.frombuffer(resampled, dtype=np.int16)
            except Exception:
                audio_16k = raw_chunk

            try:
                voice_detected, _ = self.vad(audio_16k, {})
            except Exception:
                voice_detected = False

            now = time.time()
            with lock:
                if state["done"]:
                    return
                if not state["recording"]:
                    if voice_detected:
                        state["recording"] = True
                        state["recording_started_at"] = now
                        state["last_voice_at"] = now
                        state["chunks"] = list(pre_roll_chunks)
                        state["vad_frames"] = 1
                        return
                    if (now - state["listening_started_at"]) >= self.listen_timeout_sec:
                        state["timed_out"] = True
                        state["done"] = True
                        state["stop_reason"] = "listen_timeout"
                        done_event.set()
                    return

                state["chunks"].append(raw_chunk)
                if voice_detected:
                    state["last_voice_at"] = now
                    state["vad_frames"] = int(state["vad_frames"]) + 1
                elif (now - float(state["last_voice_at"])) >= self.silence_grace_period:
                    state["done"] = True
                    state["stop_reason"] = "silence"
                    done_event.set()
                    return

                if (now - float(state["recording_started_at"])) >= self.max_record_sec:
                    state["done"] = True
                    state["stop_reason"] = "max_record_sec"
                    done_event.set()

        wait_budget_sec = self.listen_timeout_sec + self.max_record_sec + 5.0
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=AUDIO_CHANNELS,
                dtype=AUDIO_DTYPE,
                callback=_callback,
                device=self.input_device or None,
            ):
                print("Listening...")
                if not done_event.wait(timeout=wait_budget_sec):
                    _finish("capture_timeout")
        except Exception as exc:  # pragma: no cover - local env dependent.
            return {
                "success": False,
                "failure_reason": "microphone_open_failed",
                "message": f"Could not open the microphone input stream: {exc}",
            }

        with lock:
            status_flags = list(state["status_flags"])
            timed_out = bool(state["timed_out"])
            stop_reason = str(state["stop_reason"] or "")
            recording_started_at = float(state["recording_started_at"] or 0.0)
            vad_frames = int(state["vad_frames"] or 0)
            chunks = [np.asarray(chunk, dtype=np.int16) for chunk in state["chunks"]]

        if timed_out or not chunks:
            return {
                "success": False,
                "failure_reason": stop_reason or "listen_timeout",
                "message": (
                    "No speech was detected before the listen timeout. "
                    "Try again and start speaking after 'Listening...'."
                ),
                "status_flags": status_flags,
            }

        audio = np.concatenate(chunks).astype(np.int16, copy=False)
        duration_sec = float(audio.size) / float(self.sample_rate)
        return {
            "success": True,
            "audio": audio,
            "stop_reason": stop_reason or "silence",
            "duration_sec": round(duration_sec, 3),
            "recording_started_at_unix_s": round(recording_started_at, 3),
            "status_flags": status_flags,
            "vad_frames": vad_frames,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use the microphone to test the real Argos registration name-extraction "
            "+ employee-directory matching path without launching the full agent."
        )
    )
    parser.add_argument(
        "--profile",
        default="static_interaction",
        help="Argos profile name or YAML path. Default: static_interaction.",
    )
    parser.add_argument(
        "--sites",
        nargs="*",
        default=[],
        help=(
            "Site codes to load. Accepts spaces or commas, for example "
            "--sites bos1 bos3 or --sites bos1,bos3. Defaults to the profile site."
        ),
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep listening and matching until Ctrl-C.",
    )
    parser.add_argument(
        "--language",
        default="",
        help=(
            "Optional transcription language hint, for example en. Leave empty to "
            "let the transcription model auto-detect."
        ),
    )
    parser.add_argument(
        "--transcription-model",
        default="",
        help="Override the profile realtime.transcription_model.",
    )
    parser.add_argument(
        "--input-device",
        default="",
        help="Override the profile realtime.input_device.",
    )
    parser.add_argument(
        "--input-sample-rate",
        type=int,
        default=None,
        help="Override the profile realtime.input_sample_rate.",
    )
    parser.add_argument(
        "--input-block-size",
        type=int,
        default=None,
        help="Override the profile realtime.input_block_size.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="Override the profile realtime.vad_threshold for Silero VAD.",
    )
    parser.add_argument(
        "--silence-grace-period",
        type=float,
        default=None,
        help="Override the profile realtime.silence_grace_period.",
    )
    parser.add_argument(
        "--listen-timeout-sec",
        type=float,
        default=DEFAULT_LISTEN_TIMEOUT_SEC,
        help=f"How long to wait for speech before timing out. Default: {DEFAULT_LISTEN_TIMEOUT_SEC}.",
    )
    parser.add_argument(
        "--max-record-sec",
        type=float,
        default=DEFAULT_MAX_RECORD_SEC,
        help=f"Hard cap for one utterance. Default: {DEFAULT_MAX_RECORD_SEC}.",
    )
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=DEFAULT_RMS_THRESHOLD,
        help=(
            "Fallback RMS threshold when Silero VAD cannot initialize locally. "
            f"Default: {DEFAULT_RMS_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def parse_site_codes(raw_sites: Sequence[str], *, fallback_site: str = "") -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_value in raw_sites:
        for piece in _SITE_SPLIT_RE.split(str(raw_value or "").strip()):
            cleaned = piece.strip().upper()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            ordered.append(cleaned)
    fallback = str(fallback_site or "").strip().upper()
    if not ordered and fallback:
        ordered.append(fallback)
    return tuple(ordered)


def _clean_transcript(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _contains_non_latin_letters(value: str) -> bool:
    for character in str(value or ""):
        if not character.isalpha():
            continue
        if "LATIN" not in unicodedata.name(character, ""):
            return True
    return False


def _build_runtime_settings(args: argparse.Namespace) -> RuntimeSettings:
    profile = load_scenario_profile(args.profile)
    realtime = profile.realtime
    return RuntimeSettings(
        profile_name=profile.name,
        fallback_site=profile.employee_directory.site_code,
        input_device=args.input_device or realtime.input_device,
        input_sample_rate=(
            int(args.input_sample_rate)
            if args.input_sample_rate is not None
            else int(realtime.input_sample_rate)
        ),
        input_block_size=(
            int(args.input_block_size)
            if args.input_block_size is not None
            else int(realtime.input_block_size)
        ),
        vad_threshold=(
            float(args.vad_threshold)
            if args.vad_threshold is not None
            else float(realtime.vad_threshold)
        ),
        silence_grace_period=(
            float(args.silence_grace_period)
            if args.silence_grace_period is not None
            else float(realtime.silence_grace_period)
        ),
        transcription_model=(
            str(args.transcription_model or realtime.transcription_model or "").strip()
        ),
        realtime_model=str(realtime.model or "").strip(),
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return {}
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(cleaned[start : end + 1])
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _coerce_spoken_name_payload(payload: dict[str, Any]) -> SpokenName:
    shared_name = _clean_transcript(str(payload.get("shared_name", "") or ""))
    shared_first_name = _clean_transcript(
        str(payload.get("shared_first_name", "") or "")
    )
    shared_last_name = _clean_transcript(str(payload.get("shared_last_name", "") or ""))
    if not shared_name:
        shared_name = " ".join(
            part for part in (shared_first_name, shared_last_name) if part
        ).strip()
    return SpokenName(
        shared_name=shared_name,
        shared_first_name=shared_first_name,
        shared_last_name=shared_last_name,
    )


def _build_tool_schema(tool: Any) -> dict[str, Any]:
    schema_source = getattr(tool, "args_schema", None)
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    if schema_source is not None:
        try:
            parameters = dict(schema_source.model_json_schema())
        except Exception:
            try:
                parameters = dict(schema_source.schema())
            except Exception:
                parameters = {"type": "object", "properties": {}}
    parameters.pop("title", None)
    return {
        "type": "function",
        "name": str(getattr(tool, "name", "") or ""),
        "description": str(getattr(tool, "description", "") or ""),
        "parameters": parameters,
    }


def _response_request_payload(
    *,
    instructions: str,
    max_output_tokens: int | None,
) -> dict[str, Any]:
    return realtime_response_payload(
        instructions=instructions,
        output_modalities=["text"],
        max_output_tokens=max_output_tokens,
    )


def _run_agent_realtime_name_probe(
    *,
    transcript: str,
    prompt_text: str,
    scenario_profile: Any,
    schema_directory_service: Any,
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for agent-style Realtime extraction.")

    try:
        import websocket
    except ImportError as exc:
        raise RuntimeError(
            f"websocket-client is required for agent-style Realtime extraction: {exc}"
        ) from exc

    realtime = scenario_profile.realtime
    headers = realtime_auth_headers(api_key)
    url = realtime_websocket_url(realtime.model)
    ws = websocket.create_connection(url, header=headers, timeout=30)
    pending_function_args: dict[str, dict[str, str]] = {}
    assistant_text = ""
    response_done = False
    response_status = ""
    response_status_details: dict[str, Any] | None = None
    try:
        tool = get_resolve_employee_identity_tool(schema_directory_service)
        session = realtime_text_session_payload(
            profile=realtime,
            instructions=prompt_text,
            tools=[_build_tool_schema(tool)],
        )

        ws.send(json.dumps({"type": "session.update", "session": session}))
        session_deadline = time.time() + 15.0
        while time.time() < session_deadline:
            event = json.loads(ws.recv())
            event_type = str(event.get("type", "") or "")
            if event_type == "session.updated":
                break
            if event_type == "error":
                raise RuntimeError(str(event.get("error", {}) or event))

        ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": transcript}],
                    },
                }
            )
        )
        ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": _response_request_payload(
                        instructions=prompt_text,
                        max_output_tokens=getattr(realtime, "max_output_tokens", None),
                    ),
                }
            )
        )

        deadline = time.time() + 30.0
        while time.time() < deadline:
            event = json.loads(ws.recv())
            event_type = str(event.get("type", "") or "")
            if event_type == "response.output_text.delta":
                assistant_text += str(event.get("delta", "") or "")
                continue
            if event_type == "response.function_call_arguments.delta":
                item_id = str(event.get("item_id", "") or "")
                if not item_id:
                    continue
                bucket = pending_function_args.setdefault(item_id, {})
                if event.get("call_id") is not None:
                    bucket["call_id"] = str(event.get("call_id") or "")
                if event.get("name") is not None:
                    bucket["name"] = str(event.get("name") or "")
                bucket["arguments"] = bucket.get("arguments", "") + str(
                    event.get("delta", "") or ""
                )
                continue
            if event_type == "response.function_call_arguments.done":
                item_id = str(event.get("item_id", "") or "")
                cached = pending_function_args.pop(item_id, None) if item_id else None
                tool_name = str(event.get("name", "") or "")
                call_id = str(event.get("call_id", "") or "")
                arguments_json = str(event.get("arguments", "") or "")
                if cached:
                    tool_name = tool_name or cached.get("name", "")
                    call_id = call_id or cached.get("call_id", "")
                    arguments_json = arguments_json or cached.get("arguments", "")
                arguments = _extract_json_object(arguments_json)
                spoken_name = _coerce_spoken_name_payload(arguments)
                return {
                    "success": True,
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "arguments": arguments,
                    "spoken_name": spoken_name,
                    "assistant_text": assistant_text.strip(),
                }
            if event_type == "response.done":
                response = event.get("response", {}) or {}
                response_done = True
                response_status = str(response.get("status", "") or "")
                details = response.get("status_details")
                response_status_details = details if isinstance(details, dict) else None
                break
            if event_type == "error":
                raise RuntimeError(str(event.get("error", {}) or event))

        return {
            "success": False,
            "tool_name": "",
            "call_id": "",
            "arguments": {},
            "spoken_name": SpokenName("", "", ""),
            "assistant_text": assistant_text.strip(),
            "response_done": response_done,
            "response_status": response_status,
            "response_status_details": response_status_details,
        }
    finally:
        try:
            ws.close()
        except Exception:
            pass


def _build_vad(
    *,
    vad_threshold: float,
    rms_threshold: float,
) -> tuple[Callable[[np.ndarray, dict[str, Any]], tuple[bool, dict[str, Any]]], dict[str, Any]]:
    try:
        from argos_src.audio import SileroVAD

        return SileroVAD(VAD_SAMPLE_RATE, vad_threshold), {
            "mode": "silero",
            "threshold": float(vad_threshold),
        }
    except Exception as exc:  # pragma: no cover - local env/cache dependent.
        logging.getLogger(__name__).warning(
            "Silero VAD unavailable, falling back to RMS VAD: %s",
            exc,
        )
        return _RmsVAD(rms_threshold), {
            "mode": "rms",
            "rms_threshold": float(rms_threshold),
            "fallback_reason": str(exc),
        }


def _load_directories(site_codes: Sequence[str]) -> list[LoadedDirectory]:
    loaded: list[LoadedDirectory] = []
    try:
        records_by_site = load_directory_records_by_site(list(site_codes))
    except Exception as exc:
        load_error = str(exc)
        for site_code in site_codes:
            service = EmployeeDirectoryService(site_code=site_code, env_loader=lambda: None)
            service.mark_load_failed(load_error)
            loaded.append(
                LoadedDirectory(
                    site_code=site_code,
                    service=service,
                    record_count=0,
                    load_error=load_error,
                )
            )
        return loaded

    for site_code in site_codes:
        service = EmployeeDirectoryService(site_code=site_code, env_loader=lambda: None)
        records = list(records_by_site.get(site_code, []))
        service.set_loaded_records(records)
        loaded.append(
            LoadedDirectory(
                site_code=site_code,
                service=service,
                record_count=len(records),
            )
        )
    return loaded


def resolve_against_directories(
    spoken_name: SpokenName,
    directories: Sequence[LoadedDirectory],
) -> dict[str, Any]:
    per_site_results: list[dict[str, Any]] = []
    best_candidates: list[dict[str, Any]] = []

    for directory in directories:
        result = directory.service.resolve_identity(
            shared_first_name=spoken_name.shared_first_name,
            shared_last_name=spoken_name.shared_last_name,
            shared_name=spoken_name.shared_name,
        )
        site_candidates: list[dict[str, Any]] = []
        for candidate in result.get("data", {}).get("candidates", []):
            enriched = dict(candidate)
            enriched["site_code"] = directory.site_code
            enriched["site_status"] = result.get("status", "")
            site_candidates.append(enriched)
        best_candidates.extend(site_candidates)
        per_site_results.append(
            {
                "site_code": directory.site_code,
                "record_count": int(directory.record_count),
                "load_error": directory.load_error,
                "success": bool(result.get("success", False)),
                "status": str(result.get("status", "") or ""),
                "message": str(result.get("message", "") or ""),
                "candidate_count": len(site_candidates),
                "candidates": site_candidates,
            }
        )

    best_candidates.sort(
        key=lambda item: (
            -float(item.get("match_score", 0.0) or 0.0),
            str(item.get("official_name", "")).casefold(),
            str(item.get("site_code", "")).casefold(),
        )
    )
    return {
        "best_candidate": best_candidates[0] if best_candidates else None,
        "best_candidates": best_candidates,
        "candidate_count": len(best_candidates),
        "per_site_results": per_site_results,
    }


def process_transcript(
    transcript: str,
    spoken_name: SpokenName,
    directories: Sequence[LoadedDirectory],
    *,
    agent_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cleaned_transcript = _clean_transcript(transcript)
    resolution = resolve_against_directories(spoken_name, directories)

    diagnostic_hint = ""
    if not cleaned_transcript:
        diagnostic_hint = "No transcript text was available to match."
    elif agent_probe is not None and not bool(agent_probe.get("success")):
        diagnostic_hint = (
            "The agent-style Realtime probe did not emit resolve_employee_identity. "
            "Check agent_probe for the assistant reply or response status."
        )
    elif not spoken_name.shared_name:
        diagnostic_hint = (
            "The agent-style Realtime probe returned empty name fields."
        )
    elif not spoken_name.shared_last_name:
        diagnostic_hint = (
            "The agent-style Realtime probe did not provide both first and last name."
        )
    elif not resolution["best_candidates"] and _contains_non_latin_letters(cleaned_transcript):
        diagnostic_hint = (
            "The transcript contains non-Latin letters. If the employee directory stores "
            "Latin-script names, rerun with --language en so transcription stays closer "
            "to the directory spelling."
        )
    elif not resolution["best_candidates"]:
        diagnostic_hint = (
            "No plausible employee match was found for the agent-emitted name args."
        )

    payload = {
        "transcript": cleaned_transcript,
        "spoken_name": asdict(spoken_name),
        "name_extraction_mode": "agent",
        **resolution,
        "diagnostic_hint": diagnostic_hint,
    }
    if agent_probe is not None:
        payload["agent_probe"] = agent_probe
    return payload


def _transcribe_audio_with_openai(
    *,
    audio: np.ndarray,
    sample_rate: int,
    model_name: str,
    language: str,
) -> str:
    if not str(model_name or "").strip():
        raise RuntimeError(
            "No transcription model is configured. Set realtime.transcription_model "
            "in the profile or pass --transcription-model."
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for microphone transcription in this helper."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(f"openai package is required for transcription: {exc}") from exc

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(AUDIO_CHANNELS)
        wav_file.setsampwidth(np.dtype(np.int16).itemsize)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(np.asarray(audio, dtype=np.int16).tobytes())
    buffer.seek(0)
    buffer.name = "rapidfuzz_employee_lab.wav"

    request: dict[str, Any] = {
        "model": str(model_name).strip(),
        "file": buffer,
    }
    if str(language or "").strip():
        request["language"] = str(language).strip()

    client = OpenAI()
    response = client.audio.transcriptions.create(**request)
    return str(response.text or "").strip()


def _shutdown_directories(directories: Sequence[LoadedDirectory]) -> None:
    for directory in directories:
        try:
            directory.service.shutdown()
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to stop employee directory service for site=%s",
                directory.site_code,
            )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    scenario_profile = load_scenario_profile(args.profile)
    settings = _build_runtime_settings(args)
    site_codes = parse_site_codes(args.sites, fallback_site=settings.fallback_site)
    if not site_codes:
        parser.error(
            "No site codes were provided and the selected profile does not define "
            "employee_directory.site_code."
        )

    prompt_text = load_system_prompt(
        resolve_prompt_file(scenario_profile.realtime.prompt_file)
    )
    vad, vad_info = _build_vad(
        vad_threshold=settings.vad_threshold,
        rms_threshold=args.rms_threshold,
    )
    directories = _load_directories(site_codes)

    json_print(
        {
            "mode": "microphone_loop" if args.loop else "microphone_once",
            "profile": settings.profile_name,
            "sites": list(site_codes),
            "input_device": settings.input_device,
            "input_sample_rate": settings.input_sample_rate,
            "input_block_size": settings.input_block_size,
            "silence_grace_period": settings.silence_grace_period,
            "transcription_model": settings.transcription_model,
            "realtime_model": settings.realtime_model,
            "name_extraction_mode": "agent",
            "language": str(args.language or "").strip() or None,
            "vad": vad_info,
        }
    )
    for directory in directories:
        json_print(
            {
                "event": "directory_ready",
                "site_code": directory.site_code,
                "record_count": directory.record_count,
                "load_error": directory.load_error or None,
            }
        )

    recorder = MicrophoneRecorder(
        input_device=settings.input_device,
        sample_rate=settings.input_sample_rate,
        block_size=settings.input_block_size,
        silence_grace_period=settings.silence_grace_period,
        listen_timeout_sec=float(args.listen_timeout_sec),
        max_record_sec=float(args.max_record_sec),
        vad=vad,
    )

    try:
        while True:
            capture = recorder.record_one()
            if not capture.get("success"):
                capture["captured_at_unix_s"] = round(time.time(), 3)
                json_print(capture)
                if not args.loop:
                    return 2
                continue

            audio = np.asarray(capture.pop("audio"), dtype=np.int16)
            try:
                transcript = _transcribe_audio_with_openai(
                    audio=audio,
                    sample_rate=settings.input_sample_rate,
                    model_name=settings.transcription_model,
                    language=str(args.language or "").strip(),
                )
            except Exception as exc:
                json_print(
                    {
                        "success": False,
                        "failure_reason": "transcription_failed",
                        "message": str(exc),
                        "capture": capture,
                        "captured_at_unix_s": round(time.time(), 3),
                    }
                )
                if not args.loop:
                    return 2
                continue

            try:
                probe = _run_agent_realtime_name_probe(
                    transcript=transcript,
                    prompt_text=prompt_text,
                    scenario_profile=scenario_profile,
                    schema_directory_service=directories[0].service,
                )
            except Exception as exc:
                probe = {
                    "success": False,
                    "tool_name": "",
                    "call_id": "",
                    "arguments": {},
                    "spoken_name": SpokenName("", "", ""),
                    "assistant_text": "",
                    "error": str(exc),
                }

            spoken_name = probe.get("spoken_name")
            if not isinstance(spoken_name, SpokenName):
                spoken_name = SpokenName("", "", "")
            agent_probe = {
                key: (asdict(value) if isinstance(value, SpokenName) else value)
                for key, value in probe.items()
            }
            payload = process_transcript(
                transcript,
                spoken_name,
                directories,
                agent_probe=agent_probe,
            )
            payload["capture"] = capture
            payload["captured_at_unix_s"] = round(time.time(), 3)
            json_print(payload)
            if not args.loop:
                return 0
    except KeyboardInterrupt:
        print("Stopped.")
        return 0
    finally:
        _shutdown_directories(directories)


if __name__ == "__main__":
    raise SystemExit(main())
