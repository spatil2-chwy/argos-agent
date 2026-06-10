"""Common knowledge tool exports."""

from .whoami_query import (
    SUPPORTED_KNOWLEDGE_BASE_KINDS,
    build_knowledge_tool,
    get_default_chewy_knowledge_profile,
)


def get_chewy_knowledge_tool():
    """Build the legacy default Chewy knowledge-base tool from the common package."""
    return build_knowledge_tool(get_default_chewy_knowledge_profile())

__all__ = [
    "SUPPORTED_KNOWLEDGE_BASE_KINDS",
    "build_knowledge_tool",
    "get_chewy_knowledge_tool",
    "get_default_chewy_knowledge_profile",
]
