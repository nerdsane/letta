"""
OTS Management Module.

Provides decision management utilities for extracting, embedding,
and searching decisions from trajectories.
"""

from ots.management.decision_manager import (
    DecisionManager,
    DecisionRecord,
    DecisionSearchResult,
)

__all__ = [
    "DecisionManager",
    "DecisionRecord",
    "DecisionSearchResult",
]
