"""
In-memory storage backend for OTS trajectories.

Useful for testing and development. Does not persist data.
"""

from typing import Dict, List, Optional

from ots.models import OTSTrajectory
from ots.protocols import SearchResult


class MemoryBackend:
    """
    In-memory storage backend.

    Stores trajectories in a dictionary. Does not persist across restarts.
    Useful for testing and development.

    Example:
        backend = MemoryBackend()
        store = TrajectoryStore(backend=backend)
    """

    def __init__(self) -> None:
        self._trajectories: Dict[str, OTSTrajectory] = {}

    async def store(self, trajectory: OTSTrajectory) -> str:
        """Store a trajectory in memory."""
        self._trajectories[trajectory.trajectory_id] = trajectory
        return trajectory.trajectory_id

    async def get(self, trajectory_id: str) -> Optional[OTSTrajectory]:
        """Get a trajectory by ID."""
        return self._trajectories.get(trajectory_id)

    async def search(
        self,
        query: str,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search trajectories by simple text matching.

        Note: Memory backend uses basic text matching, not semantic search.
        For semantic search, use SQLiteBackend with embeddings.
        """
        results = []

        query_lower = query.lower()
        for trajectory in self._trajectories.values():
            # Simple text matching on task description and tags
            score = 0.0
            task_desc = (trajectory.metadata.task_description or "").lower()

            if query_lower in task_desc:
                score = 0.8

            for tag in trajectory.metadata.tags:
                if query_lower in tag.lower():
                    score = max(score, 0.7)

            if score > 0 and (min_score is None or score >= min_score):
                results.append(SearchResult(
                    trajectory=trajectory,
                    similarity=score,
                ))

        # Sort by score descending
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[OTSTrajectory]:
        """List trajectories with optional filtering."""
        trajectories = list(self._trajectories.values())

        # Filter by domain
        if domain:
            trajectories = [t for t in trajectories if t.metadata.domain == domain]

        # Filter by tags
        if tags:
            tag_set = set(tags)
            trajectories = [
                t for t in trajectories
                if tag_set.intersection(set(t.metadata.tags))
            ]

        # Sort by timestamp descending
        trajectories.sort(
            key=lambda t: t.metadata.timestamp_start,
            reverse=True,
        )

        # Paginate
        return trajectories[offset:offset + limit]

    async def delete(self, trajectory_id: str) -> bool:
        """Delete a trajectory."""
        if trajectory_id in self._trajectories:
            del self._trajectories[trajectory_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all trajectories (for testing)."""
        self._trajectories.clear()

    def __len__(self) -> int:
        """Return number of stored trajectories."""
        return len(self._trajectories)
