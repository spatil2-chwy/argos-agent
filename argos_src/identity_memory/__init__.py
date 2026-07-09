"""Argos identity-memory integration boundary."""

from .biometric_updates import (
    AdaptiveBiometricObservation,
    AdaptiveBiometricUpdateCoordinator,
)
from .models import (
    BiometricCandidate,
    BiometricEnrollmentResult,
    BiometricSearchResult,
    BiometricUpdateResult,
    OwnerResolution,
    PersonMemoryContext,
    PersonProfile,
)
from .noop import NoopIdentityMemoryClient
from .tailwag_package import TailwagPackageIdentityMemoryClient

__all__ = [
    "BiometricCandidate",
    "AdaptiveBiometricObservation",
    "AdaptiveBiometricUpdateCoordinator",
    "BiometricEnrollmentResult",
    "BiometricSearchResult",
    "BiometricUpdateResult",
    "NoopIdentityMemoryClient",
    "OwnerResolution",
    "PersonMemoryContext",
    "PersonProfile",
    "TailwagPackageIdentityMemoryClient",
]
