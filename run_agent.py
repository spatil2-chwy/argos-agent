"""Direct realtime speech-first entrypoint for the Argos companion."""

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
from argos_src.profile_config import (
    ProfileValidationError,
    apply_agent_cli_overrides,
    apply_audio_cli_overrides,
    load_scenario_profile,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the profile-driven realtime Argos social agent",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help=(
            "Scenario profile name under argos_src/config/profiles/ or an "
            "explicit YAML path. Defaults to the shipped static_interaction profile."
        ),
    )
    parser.add_argument(
        "--map-file",
        type=str,
        default=None,
        help=(
            "Map-specific locations file name (must end with .json) under "
            "argos_src/nav_support/locations/. Example: office_map.json"
        ),
    )
    parser.add_argument(
        "--startup-patrol-route",
        type=str,
        default=None,
        help=(
            "Comma-separated saved locations for auto patrol start. "
            "Example: lab_a,lab_b,lab_c"
        ),
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help=(
            "Optional system prompt file. Bare filenames are resolved under "
            "argos_src/prompts/. Example: static_interaction_prompt.md"
        ),
    )
    parser.add_argument("--wake-word", type=str, default=None)
    parser.add_argument("--wake-threshold", type=float, default=None)
    parser.add_argument("--wake-window-sec", type=float, default=None)
    parser.add_argument("--face-presence-topic", type=str, default=None)
    parser.add_argument("--silence-grace-period", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable(all_threads=True)
    args = parse_arguments(argv)
    agent = None

    startup_patrol_route = None
    if args.startup_patrol_route:
        startup_patrol_route = [
            name.strip()
            for name in args.startup_patrol_route.split(",")
            if name.strip()
        ]
        if not startup_patrol_route:
            print(
                "Invalid --startup-patrol-route value. Provide at least one "
                "comma-separated location name."
            )
            return 2

    try:
        scenario = apply_agent_cli_overrides(
            load_scenario_profile(args.profile),
            map_file=args.map_file,
            startup_patrol_route=startup_patrol_route,
            prompt_file=args.prompt_file,
        )
        scenario = apply_audio_cli_overrides(
            scenario,
            wake_word=args.wake_word,
            wake_threshold=args.wake_threshold,
            wake_window_sec=args.wake_window_sec,
            face_presence_topic=args.face_presence_topic,
            silence_grace_period=args.silence_grace_period,
        )
        profile_label = "Spot" if scenario.robot_family == "spot" else "Go2"
        print(f"Starting {profile_label} Realtime Companion...")
        print("  Live transport: audio in -> realtime agent -> audio out")
        print(f"  Realtime model: {scenario.realtime.model}")
        print(f"  Voice: {scenario.realtime.voice}")
        print(f"  Input device: {scenario.realtime.input_device}")
        print(f"  Output device: {scenario.realtime.output_device}")
        if scenario.navigation.locations_file:
            print(f"  Navigation locations file: {scenario.navigation.locations_file}")
        if scenario.navigation.startup_patrol_route:
            print(
                "  Startup patrol route: "
                f"{list(scenario.navigation.startup_patrol_route)}"
            )
        if scenario.realtime.prompt_file:
            print(f"  Prompt file: {scenario.realtime.prompt_file}")
        print(f"  Profile: {scenario.name}")
        print()

        agent = create_agent(scenario_profile=scenario)
        agent.start()
        print("Realtime agent running. Press Ctrl+C to stop.")
        agent.wait_until_shutdown()
        return 0
    except (ProfileValidationError, ValueError) as exc:
        print(f"Invalid profile/config: {exc}")
        return 2
    except RobotStartupPreparationError as exc:
        print(f"Robot startup preparation failed: {exc}")
        return 2
    except KeyboardInterrupt:
        return 0
    finally:
        if agent is not None:
            agent.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
