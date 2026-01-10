"""
OTS Privacy Module.

Provides anonymization utilities for sharing trajectories while
preserving learning signal and protecting sensitive information.
"""

from ots.privacy.protocols import (
    AnonymizationPolicy,
    DefaultAnonymizationPolicy,
    LearningPreservingPolicy,
)
from ots.privacy.anonymization import (
    anonymize_trajectory,
    hash_identifier,
)

__all__ = [
    # Protocols
    "AnonymizationPolicy",
    "DefaultAnonymizationPolicy",
    "LearningPreservingPolicy",
    # Functions
    "anonymize_trajectory",
    "hash_identifier",
]
