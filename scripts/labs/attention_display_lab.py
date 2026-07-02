#!/usr/bin/env python3
"""Live attention display lab.

Run from the repo root:

    source setup_shell.sh
    poetry run python -m scripts.labs.attention_display_lab

This starts the same background face loop used by the realtime agent, polls the
exported face-presence cache, and displays attention changes on Puffle's screen.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from argos_src.face_recognition.attention_gate import (
    AttentionSmoother,
    AttentionSmoothingSettings,
)
from argos_src.profile_config import load_scenario_profile
from scripts.labs.enrollment_collection_common import create_display_runtime_for_profile
from scripts.labs.face_lab_common import (
    add_enrollment_policy_args,
    add_profile_args,
    build_enrollment_policy,
    build_face_service,
    configure_logging,
    json_print,
)


class AttentionPublisher(Protocol):
    def publish(self, text: str) -> bool:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class AttentionDisplayState:
    status: str
    text: str
    signature: tuple[Any, ...]


class ConsolePublisher:
    def publish(self, text: str) -> bool:
        print(text, flush=True)
        return True

    def close(self) -> None:
        return None


class RuntimeDisplayPublisher:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def publish(self, text: str) -> bool:
        show_message = getattr(self._runtime, "show_message", None)
        if not callable(show_message):
            return False
        return bool(show_message(text))

    def close(self) -> None:
        shutdown = getattr(self._runtime, "shutdown", None)
        if callable(shutdown):
            shutdown()


def _attention_display_state(snapshot: dict[str, Any]) -> AttentionDisplayState:
    status = str(snapshot.get("attention_status") or "none").strip() or "none"
    faces = int(snapshot.get("faces_detected") or 0)
    attention_count = int(snapshot.get("attention_count") or 0)
    recognized_names = _recognized_names_for_display(snapshot)

    if faces <= 0:
        label = "Not Detected"
    elif status == "attentive" or attention_count > 0:
        label = "Detected | Attentive"
    else:
        label = "Detected | Non-Attentive"

    lines = [label]
    if faces > 0 and recognized_names:
        lines.append(f"recognized: {', '.join(recognized_names)}")

    return AttentionDisplayState(
        status=status,
        text="\n".join(lines),
        signature=(label, tuple(recognized_names)),
    )


def _recognized_names_for_display(snapshot: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("primary_attention_name", "primary_face_name"):
        rendered = str(snapshot.get(key) or "").strip()
        if rendered:
            candidates.append(rendered)
    for name in snapshot.get("recognized_names") or ():
        rendered = str(name or "").strip()
        if rendered:
            candidates.append(rendered)

    unique_names: list[str] = []
    for name in candidates:
        if name not in unique_names:
            unique_names.append(name)
    return unique_names


def _disable_attention_smoothing(service: Any) -> bool:
    gate = getattr(service, "_attention_gate", None)
    settings = getattr(gate, "settings", None)
    if gate is None or settings is None:
        return False

    smoothing = AttentionSmoothingSettings(
        window_sec=0.001,
        min_observations=1,
        hold_sec=0.0,
    )
    gate.settings = replace(settings, smoothing=smoothing)
    gate._smoother = AttentionSmoother(smoothing)
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the production face attention loop and display detected / "
            "attentive state changes for head-range testing."
        )
    )
    add_profile_args(parser)
    add_enrollment_policy_args(parser)
    parser.add_argument(
        "--display",
        choices=("profile", "off"),
        default="profile",
        help=(
            "Where to publish attention changes. 'profile' uses the configured "
            "interaction_display resource; 'off' prints changes to the console."
        ),
    )
    parser.add_argument(
        "--loop-interval",
        type=float,
        default=None,
        help="Face-loop interval. Defaults to face_recognition.loop_interval_sec.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="How often to poll the presence cache for display changes.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Stop automatically after this many seconds. Default 0 runs until Ctrl-C.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print each changed snapshot as JSON in addition to the display message.",
    )
    parser.add_argument(
        "--raw-attention",
        action="store_true",
        help=(
            "Lab-only mode: bypass temporal smoothing so the display follows the "
            "raw per-frame attention decision."
        ),
    )
    parser.add_argument(
        "--show-startup-message",
        action="store_true",
        help="Publish a startup message before the first attention state is available.",
    )
    return parser


def _create_publisher(args: argparse.Namespace) -> AttentionPublisher:
    if args.display == "off":
        return ConsolePublisher()

    profile = load_scenario_profile(args.profile)
    runtime = create_display_runtime_for_profile(profile)
    if runtime is None:
        print(
            "Profile display is unavailable; falling back to console output.",
            file=sys.stderr,
            flush=True,
        )
        return ConsolePublisher()
    return RuntimeDisplayPublisher(runtime)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    enrollment_policy = build_enrollment_policy(args)
    service, config = build_face_service(args, enrollment_policy=enrollment_policy)
    if args.raw_attention and not _disable_attention_smoothing(service):
        print(
            "Raw attention requested, but the attention gate was unavailable.",
            file=sys.stderr,
            flush=True,
        )
    publisher = _create_publisher(args)
    loop_interval_sec = (
        float(args.loop_interval)
        if args.loop_interval is not None
        else float(config["loop_interval_sec"])
    )
    poll_interval_sec = max(0.01, float(args.poll_interval))
    duration_sec = max(0.0, float(args.duration_sec))
    deadline = time.monotonic() + duration_sec if duration_sec > 0.0 else None
    last_signature: tuple[Any, ...] | None = None

    try:
        if args.show_startup_message:
            publisher.publish("Attention lab starting...")
        service.start_loop(
            camera_resource_id=config["camera_resource_id"],
            interval=loop_interval_sec,
        )
        json_print(
            {
                "mode": "attention_display",
                "profile": config["profile_name"],
                "camera_resource_id": config["camera_resource_id"],
                "loop_interval_sec": loop_interval_sec,
                "poll_interval_sec": poll_interval_sec,
                "display": args.display,
                "attention_mode": "raw" if args.raw_attention else "smoothed",
            }
        )
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                return 0
            snapshot = service.get_presence_snapshot()
            state = _attention_display_state(snapshot)
            if state.signature != last_signature:
                last_signature = state.signature
                publisher.publish(state.text)
                if args.print_json:
                    payload = dict(snapshot)
                    payload["display_text"] = state.text
                    payload["captured_at_unix_s"] = round(time.time(), 3)
                    json_print(payload)
            time.sleep(poll_interval_sec)
    except KeyboardInterrupt:
        print("Stopped.")
        return 0
    finally:
        try:
            service.shutdown()
        finally:
            robot_client = config.get("robot_client") if "config" in locals() else None
            if robot_client is not None:
                robot_client.shutdown()
            publisher.close()


if __name__ == "__main__":
    raise SystemExit(main())
