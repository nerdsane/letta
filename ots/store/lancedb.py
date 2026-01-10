"""
LanceDB storage backend for OTS trajectories.

Provides persistent storage with native vector search capabilities.
LanceDB is an embedded vector database - no server required.

This is the recommended default backend for context learning.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ots.models import OTSTrajectory
from ots.protocols import EmbeddingProvider, SearchResult

logger = logging.getLogger(__name__)


def _sanitize_id(value: str) -> str:
    """Sanitize ID value for use in LanceDB WHERE clause to prevent injection."""
    # Only allow alphanumeric, hyphens, underscores
    if not re.match(r'^[\w\-]+$', value):
        raise ValueError(f"Invalid ID format: {value}")
    return value


class LanceDBBackend:
    """
    LanceDB storage backend with native vector search.

    LanceDB is an embedded vector database (like SQLite, but for vectors):
    - Local files, no server required
    - No API key needed
    - Native approximate nearest neighbor (ANN) search
    - Can scale to millions of vectors

    This is the recommended backend for context learning, as it provides
    efficient semantic search over trajectory embeddings.

    Example:
        from ots.store.lancedb import LanceDBBackend
        from ots.embeddings import OpenAIEmbeddingProvider

        backend = LanceDBBackend(
            Path("./trajectories"),
            embedding_provider=OpenAIEmbeddingProvider(),
        )

        # Store trajectory
        await backend.store(trajectory)

        # Semantic search
        results = await backend.search("debug memory issues", limit=5)
    """

    def __init__(
        self,
        db_path: Path,
        embedding_provider: Optional[EmbeddingProvider] = None,
        table_name: str = "trajectories",
    ) -> None:
        """
        Initialize LanceDB backend.

        Args:
            db_path: Path to LanceDB database directory
            embedding_provider: Required for semantic search. Without this,
                               only basic CRUD operations work.
            table_name: Name of the table to store trajectories
        """
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "LanceDB is required for LanceDBBackend. "
                "Install with: pip install ots[lancedb]"
            )

        self.db_path = Path(db_path)
        self.embedding_provider = embedding_provider
        self.table_name = table_name

        # Connect to LanceDB
        self.db = lancedb.connect(str(self.db_path))
        self._init_table()

    def _init_table(self) -> None:
        """Initialize table if it doesn't exist."""
        import pyarrow as pa

        # Check if table exists
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)
            return

        # Determine embedding dimension
        embedding_dim = 1536  # Default for text-embedding-3-small
        if self.embedding_provider:
            embedding_dim = self.embedding_provider.dimension

        # Create schema
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("data", pa.string()),  # JSON-serialized trajectory
            pa.field("domain", pa.string()),
            pa.field("task_description", pa.string()),
            pa.field("tags", pa.string()),
            pa.field("timestamp_start", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), embedding_dim)),
        ])

        # Create empty table with schema
        self.table = self.db.create_table(
            self.table_name,
            schema=schema,
        )

    async def store(self, trajectory: OTSTrajectory) -> str:
        """
        Store a trajectory with its embedding.

        Args:
            trajectory: OTS trajectory to store

        Returns:
            Trajectory ID
        """
        data = trajectory.to_dict()

        # Generate embedding if provider configured
        embedding: Optional[List[float]] = None
        if self.embedding_provider:
            embed_text = self._trajectory_to_embed_text(trajectory)
            embedding = await self.embedding_provider.embed(embed_text)

        # Prepare row
        row = {
            "id": trajectory.trajectory_id,
            "data": json.dumps(data, default=str),
            "domain": trajectory.metadata.domain or "",
            "task_description": trajectory.metadata.task_description,
            "tags": ",".join(trajectory.metadata.tags),
            "timestamp_start": trajectory.metadata.timestamp_start.isoformat(),
            "vector": embedding or [0.0] * 1536,  # Placeholder if no embedding
        }

        # Check if trajectory exists (upsert behavior)
        try:
            safe_id = _sanitize_id(trajectory.trajectory_id)
            existing = self.table.search().where(f"id = '{safe_id}'").to_list()
            if existing:
                # Delete existing row
                self.table.delete(f"id = '{safe_id}'")
        except ValueError as e:
            logger.warning(f"Invalid trajectory ID: {e}")
            raise
        except Exception as e:
            logger.debug(f"No existing trajectory found (table may be empty): {e}")

        # Add new row
        self.table.add([row])

        return trajectory.trajectory_id

    async def get(self, trajectory_id: str) -> Optional[OTSTrajectory]:
        """
        Get a trajectory by ID.

        Args:
            trajectory_id: ID of the trajectory

        Returns:
            Trajectory or None if not found
        """
        try:
            safe_id = _sanitize_id(trajectory_id)
            results = self.table.search().where(f"id = '{safe_id}'").limit(1).to_list()
            if not results:
                return None

            data = json.loads(results[0]["data"])
            return OTSTrajectory.from_dict(data)
        except ValueError as e:
            logger.warning(f"Invalid trajectory ID format: {e}")
            return None
        except Exception as e:
            logger.debug(f"Failed to get trajectory {trajectory_id}: {e}")
            return None

    async def search(
        self,
        query: str,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search trajectories by semantic similarity.

        Uses LanceDB's native vector search for efficient ANN queries.

        Args:
            query: Search query (natural language)
            limit: Maximum results to return
            min_score: Minimum similarity score threshold (0-1)

        Returns:
            List of search results with trajectories and scores
        """
        if not self.embedding_provider:
            return await self._text_search(query, limit, min_score)

        # Get query embedding
        query_embedding = await self.embedding_provider.embed(query)

        # Perform vector search
        try:
            results = (
                self.table.search(query_embedding)
                .limit(limit * 2)  # Get extra for filtering
                .to_list()
            )
        except Exception as e:
            logger.debug(f"Vector search failed: {e}")
            return []

        search_results = []
        for row in results:
            # LanceDB returns _distance, convert to similarity
            # Lower distance = higher similarity
            distance = row.get("_distance", 0)
            similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity

            if min_score is not None and similarity < min_score:
                continue

            try:
                trajectory = OTSTrajectory.from_dict(json.loads(row["data"]))
                search_results.append(SearchResult(
                    trajectory=trajectory,
                    similarity=similarity,
                ))
            except Exception as e:
                logger.debug(f"Failed to parse trajectory from search result: {e}")
                continue

        return search_results[:limit]

    async def _text_search(
        self,
        query: str,
        limit: int,
        min_score: Optional[float],
    ) -> List[SearchResult]:
        """Fallback text search when no embedding provider."""
        results = []
        query_lower = query.lower()

        try:
            rows = self.table.search().limit(1000).to_list()
        except Exception as e:
            logger.debug(f"Text search failed to retrieve rows: {e}")
            return []

        for row in rows:
            score = 0.0
            task_desc = (row.get("task_description") or "").lower()

            if query_lower in task_desc:
                score = 0.8

            tags = (row.get("tags") or "").split(",")
            for tag in tags:
                if query_lower in tag.lower():
                    score = max(score, 0.7)

            if score > 0 and (min_score is None or score >= min_score):
                try:
                    trajectory = OTSTrajectory.from_dict(json.loads(row["data"]))
                    results.append(SearchResult(
                        trajectory=trajectory,
                        similarity=score,
                    ))
                except Exception as e:
                    logger.debug(f"Failed to parse trajectory in text search: {e}")
                    continue

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[OTSTrajectory]:
        """
        List trajectories with optional filtering.

        Args:
            limit: Maximum results
            offset: Pagination offset
            domain: Filter by domain
            tags: Filter by tags (any match)

        Returns:
            List of trajectories
        """
        # Build filter
        filters = []
        if domain:
            filters.append(f"domain = '{domain}'")

        # LanceDB doesn't support complex OR in WHERE, so we'll filter in Python
        where_clause = " AND ".join(filters) if filters else None

        try:
            query = self.table.search()
            if where_clause:
                query = query.where(where_clause)

            # Get extra rows to handle offset and tag filtering
            rows = query.limit(limit + offset + 100).to_list()
        except Exception as e:
            logger.debug(f"List query failed: {e}")
            return []

        trajectories = []
        for row in rows:
            # Apply tag filter in Python
            if tags:
                row_tags = (row.get("tags") or "").split(",")
                if not any(tag in row_tags for tag in tags):
                    continue

            try:
                trajectory = OTSTrajectory.from_dict(json.loads(row["data"]))
                trajectories.append(trajectory)
            except Exception as e:
                logger.debug(f"Failed to parse trajectory in list: {e}")
                continue

        # Apply offset and limit
        return trajectories[offset:offset + limit]

    async def delete(self, trajectory_id: str) -> bool:
        """
        Delete a trajectory.

        Args:
            trajectory_id: ID of the trajectory to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            safe_id = _sanitize_id(trajectory_id)
            # Check if exists first
            existing = self.table.search().where(f"id = '{safe_id}'").to_list()
            if not existing:
                return False

            self.table.delete(f"id = '{safe_id}'")
            return True
        except ValueError as e:
            logger.warning(f"Invalid trajectory ID format for delete: {e}")
            return False
        except Exception as e:
            logger.debug(f"Delete failed for trajectory {trajectory_id}: {e}")
            return False

    def _trajectory_to_embed_text(self, trajectory: OTSTrajectory) -> str:
        """Convert trajectory to text for embedding."""
        parts = [trajectory.metadata.task_description]

        if trajectory.metadata.domain:
            parts.append(f"Domain: {trajectory.metadata.domain}")

        if trajectory.metadata.tags:
            parts.append(f"Tags: {', '.join(trajectory.metadata.tags)}")

        # Include first turn messages
        if trajectory.turns:
            for msg in trajectory.turns[0].messages[:3]:
                if msg.content.text:
                    parts.append(msg.content.text[:500])

        return " | ".join(parts)

    def clear(self) -> None:
        """Clear all trajectories (for testing)."""
        # Drop and recreate table
        try:
            self.db.drop_table(self.table_name)
        except Exception as e:
            logger.debug(f"Could not drop table during clear (may not exist): {e}")
        self._init_table()

    @property
    def count(self) -> int:
        """Return count of stored trajectories."""
        try:
            return len(self.table.search().limit(100000).to_list())
        except Exception as e:
            logger.debug(f"Count query failed: {e}")
            return 0
