"""Argos realtime agent subpackage."""

def create_agent(*args, **kwargs):
    """Lazily import the transport-neutral factory to avoid package import cycles."""
    from .factory import create_agent as _create_agent

    return _create_agent(*args, **kwargs)

__all__ = ["create_agent"]
