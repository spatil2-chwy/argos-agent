"""Console logging setup for Argos launchers."""

from __future__ import annotations

import logging
import os


def configure_argos_logging(default_level: str = "INFO") -> None:
    """Install Argos-owned console logging with optional colors."""
    level_name = os.getenv("ARGOS_LOG_LEVEL", default_level).strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    try:
        import coloredlogs

        coloredlogs.install(
            level=level,
            fmt=fmt,
            datefmt=datefmt,
            milliseconds=True,
        )
    except Exception:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s",
            datefmt=datefmt,
        )

    for noisy_logger in (
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "posthog",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
