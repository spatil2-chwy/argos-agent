"""Provider error types shared by transport implementations."""

from __future__ import annotations


class ProviderError(RuntimeError):
    """Raised when the provider rejects or fails a capability request."""


class ProviderTimeout(ProviderError):
    """Raised when the provider does not answer a request in time."""


def is_provider_error(exc: BaseException) -> bool:
    """Whether an exception is an expected provider/capability failure."""
    return isinstance(exc, ProviderError)


__all__ = [
    "ProviderError",
    "ProviderTimeout",
    "is_provider_error",
]
