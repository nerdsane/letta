"""
File-based storage backend for OTS trajectories.

Stores trajectories as JSON files in a directory.
"""

import json
from pathlib import Path
from typing import List, Optional

from ots.models import OTSTrajectory
from ots.protocols import SearchResult


class FileBackend:
    """
    File-based storage backend using JSON files.

    Each trajectory is stored as a separate JSON file.
    Useful for simple persistence without database dependencies.

    Example:
        backend = FileBackend(Path("./trajectories"))
        store = TrajectoryStore(backend=backend)
    """

    def __init__(self, directory: Path) -> None:
        """
        Initialize file backend.

        Args:
            directory: Directory to store trajectory files
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _trajectory_path(self, trajectory_id: str) -> Path:
        """Get file path for a trajectory."""
        # Sanitize ID for filename
        safe_id = trajectory_id.replace("/", "_").replace("\\", "_")
        return self.directory / f"{safe_id}.json"

    async def store(self, trajectory: OTSTrajectory) -> str:
        """Store a trajectory as a JSON file."""
        path = self._trajectory_path(trajectory.trajectory_id)
        data = trajectory.to_dict()

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return trajectory.trajectory_id

    async def get(self, trajectory_id: str) -> Optional[OTSTrajectory]:
        """Get a trajectory by ID."""
        path = self._trajectory_path(trajectory_id)

        if not path.exists():
            return None

        with open(path, "r") as f:
            data = json.load(f)

        return OTSTrajectory.from_dict(data)

    async def search(
        self,
        query: str,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search trajectories by simple text matching.

        Note: File backend uses basic text matching, not semantic search.
        For semantic search, use SQLiteBackend with embeddings.
        """
        results = []
        query_lower = query.lower()

        for path in self.directory.glob("*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)

                trajectory = OTSTrajectory.from_dict(data)

                # Simple text matching
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

            except (json.JSONDecodeError, Exception):
                # Skip invalid files
                continue

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
        trajectories = []

        for path in self.directory.glob("*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)

                trajectory = OTSTrajectory.from_dict(data)

                # Filter by domain
                if domain and trajectory.metadata.domain != domain:
                    continue

                # Filter by tags
                if tags:
                    tag_set = set(tags)
                    if not tag_set.intersection(set(trajectory.metadata.tags)):
                        continue

                trajectories.append(trajectory)

            except (json.JSONDecodeError, Exception):
                continue

        # Sort by timestamp descending
        trajectories.sort(
            key=lambda t: t.metadata.timestamp_start,
            reverse=True,
        )

        # Paginate
        return trajectories[offset:offset + limit]

    async def delete(self, trajectory_id: str) -> bool:
        """Delete a trajectory file."""
        path = self._trajectory_path(trajectory_id)

        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> None:
        """Clear all trajectory files (for testing)."""
        for path in self.directory.glob("*.json"):
            path.unlink()
