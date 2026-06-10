"""Lazy exports for runtime helpers shared across the Argos runtime."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "BatteryStateCache",
    "CHARGING_READY_PCT",
    "CURRENT_CHARGE_A",
    "CURRENT_DISCHARGE_A",
    "FACE_PRESENCE_TOPIC",
    "LOW_BATTERY_NAVIGATION_MSG",
    "LOW_BATTERY_PCT",
    "reports_charging",
]

_LAZY_EXPORTS = {
    "BatteryStateCache": ("argos_src.runtime.battery_state", "BatteryStateCache"),
    "CHARGING_READY_PCT": ("argos_src.runtime.battery_state", "CHARGING_READY_PCT"),
    "CURRENT_CHARGE_A": ("argos_src.runtime.battery_state", "CURRENT_CHARGE_A"),
    "CURRENT_DISCHARGE_A": ("argos_src.runtime.battery_state", "CURRENT_DISCHARGE_A"),
    "LOW_BATTERY_NAVIGATION_MSG": (
        "argos_src.runtime.battery_state",
        "LOW_BATTERY_NAVIGATION_MSG",
    ),
    "LOW_BATTERY_PCT": ("argos_src.runtime.battery_state", "LOW_BATTERY_PCT"),
    "reports_charging": ("argos_src.runtime.battery_state", "reports_charging"),
    "FACE_PRESENCE_TOPIC": (
        "argos_src.runtime.interaction_topics",
        "FACE_PRESENCE_TOPIC",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
