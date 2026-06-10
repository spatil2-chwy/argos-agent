"""Lazy exports for Argos observability helpers."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "LatencyLogger",
    "clear_request_context",
    "get_request_context",
    "perf_now",
    "set_request_context",
]

_LAZY_EXPORTS = {
    "LatencyLogger": ("argos_src.observability.observability", "LatencyLogger"),
    "clear_request_context": (
        "argos_src.observability.observability",
        "clear_request_context",
    ),
    "get_request_context": (
        "argos_src.observability.observability",
        "get_request_context",
    ),
    "perf_now": ("argos_src.observability.observability", "perf_now"),
    "set_request_context": (
        "argos_src.observability.observability",
        "set_request_context",
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
