"""Single-process profile-driven launcher for the Argos realtime runtime."""

from __future__ import annotations

import argparse
import faulthandler
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from argos_src.agent import create_agent
from argos_src.agent.startup import RobotStartupPreparationError
from argos_src.logging_config import configure_argos_logging
from argos_src.profile_config import (
    ProfileValidationError,
    apply_audio_cli_overrides,
    load_scenario_profile,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the full Argos realtime stack from a scenario profile.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="Scenario profile name under config/profiles/ or an explicit YAML path",
    )
    parser.add_argument("--map-file", type=str, default=None)
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--wake-word", type=str, default=None)
    parser.add_argument("--wake-threshold", type=float, default=None)
    parser.add_argument("--wake-window-sec", type=float, default=None)
    parser.add_argument("--face-presence-topic", type=str, default=None)
    parser.add_argument("--silence-grace-period", type=float, default=None)
    parser.add_argument("--patrol-route", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    faulthandler.enable(all_threads=True)
    configure_argos_logging()
    args = parse_arguments()

    patrol_route = None
    if args.patrol_route:
        patrol_route = [
            name.strip() for name in args.patrol_route.split(",") if name.strip()
        ]
        if not patrol_route:
            raise ProfileValidationError(
                "Invalid --patrol-route value. Provide at least one location."
            )

    scenario = apply_audio_cli_overrides(
        load_scenario_profile(args.profile, require_explicit=True),
        map_file=args.map_file,
        patrol_route=patrol_route,
        prompt_file=args.prompt_file,
        wake_word=args.wake_word,
        wake_threshold=args.wake_threshold,
        wake_window_sec=args.wake_window_sec,
        face_presence_topic=args.face_presence_topic,
        silence_grace_period=args.silence_grace_period,
    )

    launcher_label = "Spot" if scenario.robot_family == "spot" else "Go2"
    print("=" * 64)
    print(f"{launcher_label} Realtime Scenario Launcher")
    print("=" * 64)
    print(f"Profile: {scenario.name}")
    print(f"Profile file: {scenario.source_path}")
    if scenario.navigation.locations_file:
        print(f"Locations file: {scenario.navigation.locations_file}")
    if scenario.navigation.startup_patrol_route:
        print(f"Startup patrol route: {list(scenario.navigation.startup_patrol_route)}")
    if scenario.realtime.prompt_file:
        print(f"Prompt file: {scenario.realtime.prompt_file}")
    print(f"Realtime model: {scenario.realtime.model}")
    print(f"Voice: {scenario.realtime.voice}")
    print()

    agent = None
    try:
        agent = create_agent(scenario_profile=scenario)
        agent.start()
        print("Realtime agent running. Press Ctrl+C to stop.")
        agent.wait_until_shutdown()
        return 0
    except RobotStartupPreparationError as exc:
        print(f"Robot startup preparation failed: {exc}")
        return 2
    except KeyboardInterrupt:
        return 0
    finally:
        if agent is not None:
            agent.shutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProfileValidationError as exc:
        print(f"Invalid profile/config: {exc}")
        raise SystemExit(2)
