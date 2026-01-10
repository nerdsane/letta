"""
Letta OTS Integration Package.

Provides Letta-specific implementations for the OTS (Open Trajectory Specification) library:
- LettaAdapter: Converts Letta trajectories to/from OTS format
- LettaStorageBackend: Uses Letta's PostgreSQL infrastructure for trajectory storage
- DSFEntityExtractor: Extracts Deep Sci-Fi specific entities from trajectories

Usage:
    from ots import TrajectoryStore, ContextLearning
    from letta.ots import LettaAdapter, LettaStorageBackend, DSFEntityExtractor

    # Use Letta's storage
    store = TrajectoryStore(backend=LettaStorageBackend(trajectory_manager))

    # Convert Letta trajectories
    adapter = LettaAdapter()
    ots_traj = adapter.to_ots(letta_trajectory)

    # Domain-specific entity extraction
    store.register_extractor(DSFEntityExtractor())
"""

from letta.ots.adapter import LettaAdapter
from letta.ots.backend import LettaStorageBackend
from letta.ots.dsf_extractor import DSFEntityExtractor

__all__ = [
    "LettaAdapter",
    "LettaStorageBackend",
    "DSFEntityExtractor",
]
