"""Navigation state helpers and canonical tool exports."""

from __future__ import annotations

from importlib import import_module

from .locations import (
    LocationStore,
    load_locations,
    resolve_map_locations_path,
    save_locations,
)

__all__ = [
    "LocationStore",
    "load_locations",
    "resolve_map_locations_path",
    "save_locations",
    "get_navigation_tools",
]


def __getattr__(name: str):
    if name != "get_navigation_tools":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("argos_src.tools.unitree_go2.navigation")
    value = getattr(module, name)
    globals()[name] = value
    return value
