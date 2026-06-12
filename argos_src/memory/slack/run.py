"""Run Slack memory ingestion without starting the realtime robot agent."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import signal
import sqlite3
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from argos_src.identity import IdentityStore
from argos_src.memory import MemoryStore
from argos_src.memory.slack import SlackMemoryService
from argos_src.profile_config import ProfileValidationError, load_scenario_profile


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Argos Slack memory ingestion from a scenario profile.",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Scenario profile name under config/profiles/ or explicit YAML path.",
    )
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit.")
    parser.add_argument(
        "--print-llm-output",
        action="store_true",
        help="Print structured Slack extraction JSON before post-processing/writes.",
    )
    parser.add_argument(
        "--print-llm-prompt",
        action="store_true",
        help="Print the exact Slack extraction prompt sent to the LLM.",
    )
    parser.add_argument(
        "--reset-checkpoints",
        action="store_true",
        help="Clear Slack channel checkpoints before running, so lookback_minutes is used.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    scenario = load_scenario_profile(args.profile, require_explicit=True)
    if not scenario.slack_memory.enabled:
        logging.getLogger(__name__).warning(
            "slack_memory.enabled is false in profile %s; nothing to run.",
            scenario.name,
        )
        return 0

    memory_store = MemoryStore(db_path=scenario.memory_store.db_path)
    identity_store = IdentityStore(db_path=scenario.identity_store.db_path)
    service = SlackMemoryService(
        profile=scenario.slack_memory,
        memory_store=memory_store,
        identity_store=identity_store,
        default_site_code=scenario.employee_directory.site_code,
        debug_llm_prompt=args.print_llm_prompt,
        debug_llm_output=args.print_llm_output,
    )
    if args.reset_checkpoints:
        _reset_slack_checkpoints(memory_store)

    if args.once:
        service.run_once()
        return 0

    def _handle_signal(_signum, _frame) -> None:
        service.shutdown()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    logging.getLogger(__name__).info(
        "Starting Slack memory loop profile=%s interval_sec=%s channels=%s",
        scenario.name,
        scenario.slack_memory.poll_interval_sec,
        [channel.name for channel in scenario.slack_memory.channels],
    )
    service.run_forever()
    return 0


def _reset_slack_checkpoints(memory_store: MemoryStore) -> None:
    db_path = str(getattr(memory_store, "db_path", "") or "").strip()
    if not db_path:
        return
    with sqlite3.connect(db_path, timeout=30.0) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS slack_channel_checkpoints (
                channel_id TEXT PRIMARY KEY,
                last_ts TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute("DELETE FROM slack_channel_checkpoints")
    logging.getLogger(__name__).info("Reset Slack channel checkpoints.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProfileValidationError as exc:
        print(f"Invalid profile/config: {exc}")
        raise SystemExit(2)
