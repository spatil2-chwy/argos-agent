"""Derived timeout budgets for blocking navigation and docking."""

from __future__ import annotations

import math
import threading

from argos_src.provider_api.deadlines import PROVIDER_RESPONSE_GRACE_SEC


NAV_TIMEOUT_BASE_SEC = 30.0
NAV_TIMEOUT_SEC_PER_METER = 10.0
TOOL_WATCHDOG_GRACE_SEC = 10.0
DOCK_ALIGNMENT_TIMEOUT_SEC = 60.0


def estimate_navigation_timeout_sec(distance_m: float) -> float:
    """Return the navigation execution budget for a map-frame distance."""
    rendered_distance_m = float(distance_m)
    if not math.isfinite(rendered_distance_m) or rendered_distance_m < 0.0:
        raise ValueError("Navigation distance must be a finite non-negative number.")
    timeout_sec = NAV_TIMEOUT_BASE_SEC + (
        NAV_TIMEOUT_SEC_PER_METER * rendered_distance_m
    )
    return _validated_timeout(timeout_sec, extra_sec=PROVIDER_RESPONSE_GRACE_SEC)


def navigation_tool_timeout_sec(navigation_timeout_sec: float) -> float:
    """Return the outer tool deadline derived from a provider navigation budget."""
    navigation_timeout_sec = _validated_timeout(
        navigation_timeout_sec,
        extra_sec=PROVIDER_RESPONSE_GRACE_SEC,
    )
    return _validated_timeout(
        float(navigation_timeout_sec)
        + PROVIDER_RESPONSE_GRACE_SEC
        + TOOL_WATCHDOG_GRACE_SEC
    )


def charging_tool_timeout_sec(
    navigation_timeout_sec: float,
    *,
    alignment_timeout_sec: float = DOCK_ALIGNMENT_TIMEOUT_SEC,
) -> float:
    """Return the outer deadline for approach navigation plus final alignment."""
    navigation_timeout_sec = _validated_timeout(
        navigation_timeout_sec,
        extra_sec=PROVIDER_RESPONSE_GRACE_SEC,
    )
    alignment_timeout_sec = _validated_timeout(
        alignment_timeout_sec,
        extra_sec=PROVIDER_RESPONSE_GRACE_SEC,
    )
    return _validated_timeout(
        float(navigation_timeout_sec)
        + PROVIDER_RESPONSE_GRACE_SEC
        + float(alignment_timeout_sec)
        + PROVIDER_RESPONSE_GRACE_SEC
        + TOOL_WATCHDOG_GRACE_SEC
    )


def _validated_timeout(timeout_sec: float, *, extra_sec: float = 0.0) -> float:
    rendered_timeout_sec = float(timeout_sec)
    total_sec = rendered_timeout_sec + float(extra_sec)
    if (
        not math.isfinite(rendered_timeout_sec)
        or rendered_timeout_sec <= 0.0
        or not math.isfinite(total_sec)
        or total_sec > threading.TIMEOUT_MAX
    ):
        raise ValueError("Timeout is unsupported on this platform.")
    return rendered_timeout_sec
