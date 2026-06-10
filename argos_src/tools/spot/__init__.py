"""Lazy exports for Boston Dynamics Spot tools."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "SPOT_MOBILITY_TOOL_NAMES",
    "SPOT_SYSTEM_TOOL_NAMES",
    "get_spot_tools",
]

_LAZY_EXPORTS = {
    "SPOT_MOBILITY_TOOL_NAMES": (
        "argos_src.tools.spot.mobility.toolset",
        "SPOT_MOBILITY_TOOL_NAMES",
    ),
    "SPOT_SYSTEM_TOOL_NAMES": (
        "argos_src.tools.spot.system.toolset",
        "SPOT_SYSTEM_TOOL_NAMES",
    ),
    "get_spot_tools": ("argos_src.tools.spot.registry", "get_spot_tools"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
