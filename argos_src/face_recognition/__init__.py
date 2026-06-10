"""Face recognition module for Go2 robot social interactions."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "FacePresenceSnapshot",
    "FaceRecognitionService",
    "FaceRecognitionStore",
    "FaceTurnTarget",
    "PersonContext",
]

_LAZY_EXPORTS = {
    "FacePresenceSnapshot": (
        "argos_src.face_recognition.models",
        "FacePresenceSnapshot",
    ),
    "FaceRecognitionService": (
        "argos_src.face_recognition.face_recognition_service",
        "FaceRecognitionService",
    ),
    "FaceRecognitionStore": (
        "argos_src.face_recognition.store",
        "FaceRecognitionStore",
    ),
    "FaceTurnTarget": ("argos_src.face_recognition.models", "FaceTurnTarget"),
    "PersonContext": ("argos_src.face_recognition.models", "PersonContext"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
