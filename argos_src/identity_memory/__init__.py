"""Argos identity-memory integration boundary."""

from .models import (
    BiometricCandidate,
    BiometricEnrollmentResult,
    BiometricSearchResult,
    OwnerResolution,
    PersonMemoryContext,
    PersonProfile,
)
from .noop import NoopIdentityMemoryClient
from .tailwag_package import TailwagPackageIdentityMemoryClient

__all__ = [
    "BiometricCandidate",
    "BiometricEnrollmentResult",
    "BiometricSearchResult",
    "NoopIdentityMemoryClient",
    "OwnerResolution",
    "PersonMemoryContext",
    "PersonProfile",
    "TailwagPackageIdentityMemoryClient",
]
