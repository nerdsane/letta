"""
Adapter base classes for OTS.

Adapters convert between framework-specific trajectory formats and OTS.
Each agent framework implements its own adapter.
"""

# Re-export protocol for convenience
from ots.protocols import TrajectoryAdapter

__all__ = [
    "TrajectoryAdapter",
]
