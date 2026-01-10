"""
PostgreSQL storage backend for OTS trajectories.

Provides production-ready storage with pgvector for vector search.
Requires: pip install ots[postgres]
"""

import json
from typing import Any, Dict, List, Optional

from ots.models import OTSTrajectory
from ots.protocols import EmbeddingProvider, SearchResult


class PostgresBackend:
    """
    PostgreSQL storage backend with pgvector for vector search.

    This is the recommended backend for production deployments:
    - Scalable to millions of trajectories
    - Native pgvector support for efficient ANN search
    - Full ACID compliance
    - Supports concurrent access

    Requires:
        - PostgreSQL with pgvector extension
        - pip install ots[postgres]

    Example:
        from ots.store.postgres import PostgresBackend
        from ots.embeddings import OpenAIEmbeddingProvider

        backend = await PostgresBackend.create(
            connection_string="postgresql://user:pass@localhost/ots",
            embedding_provider=OpenAIEmbeddingProvider(),
        )

        await backend.store(trajectory)
        results = await backend.search("debug memory issues", limit=5)
    """

    def __init__(
        self,
        pool: Any,  # asyncpg.Pool
        embedding_provider: Optional[EmbeddingProvider] = None,
        table_name: str = "trajectories",
        embedding_dimension: int = 1536,
    ) -> None:
        """
        Initialize PostgreSQL backend.

        Use PostgresBackend.create() instead of __init__ directly.

        Args:
            pool: asyncpg connection pool
            embedding_provider: Optional embedding provider for semantic search
            table_name: Name of the table to store trajectories
            embedding_dimension: Dimension of embeddings (default: 1536 for OpenAI)
        """
        self._pool = pool
        self.embedding_provider = embedding_provider
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension

    @classmethod
    async def create(
        cls,
        connection_string: str,
        embedding_provider: Optional[EmbeddingProvider] = None,
        table_name: str = "trajectories",
        min_connections: int = 2,
        max_connections: int = 10,
    ) -> "PostgresBackend":
        """
        Create and initialize PostgreSQL backend.

        Args:
            connection_string: PostgreSQL connection string
            embedding_provider: Optional embedding provider for semantic search
            table_name: Name of the table to store trajectories
            min_connections: Minimum pool connections
            max_connections: Maximum pool connections

        Returns:
            Initialized PostgresBackend
        """
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg is required for PostgresBackend. "
                "Install with: pip install ots[postgres]"
            )

        # Determine embedding dimension
        embedding_dimension = 1536
        if embedding_provider:
            embedding_dimension = embedding_provider.dimension

        # Create connection pool
        pool = await asyncpg.create_pool(
            connection_string,
            min_size=min_connections,
            max_size=max_connections,
        )

        backend = cls(
            pool=pool,
            embedding_provider=embedding_provider,
            table_name=table_name,
            embedding_dimension=embedding_dimension,
        )

        await backend._init_schema()
        return backend

    async def _init_schema(self) -> None:
        """Initialize database schema."""
        async with self._pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    domain TEXT,
                    task_description TEXT,
                    tags TEXT[],
                    timestamp_start TIMESTAMPTZ,
                    embedding vector({self.embedding_dimension}),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indices
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_domain
                ON {self.table_name}(domain)
            """)

            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp
                ON {self.table_name}(timestamp_start DESC)
            """)

            # Create vector index for efficient ANN search
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding
                ON {self.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

    async def store(self, trajectory: OTSTrajectory) -> str:
        """
        Store a trajectory.

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

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table_name}
                (id, data, domain, task_description, tags, timestamp_start, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO UPDATE SET
                    data = EXCLUDED.data,
                    domain = EXCLUDED.domain,
                    task_description = EXCLUDED.task_description,
                    tags = EXCLUDED.tags,
                    timestamp_start = EXCLUDED.timestamp_start,
                    embedding = EXCLUDED.embedding
                """,
                trajectory.trajectory_id,
                json.dumps(data, default=str),
                trajectory.metadata.domain,
                trajectory.metadata.task_description,
                trajectory.metadata.tags,
                trajectory.metadata.timestamp_start,
                embedding,
            )

        return trajectory.trajectory_id

    async def get(self, trajectory_id: str) -> Optional[OTSTrajectory]:
        """
        Get a trajectory by ID.

        Args:
            trajectory_id: ID of the trajectory

        Returns:
            Trajectory or None if not found
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT data FROM {self.table_name} WHERE id = $1",
                trajectory_id,
            )

        if not row:
            return None

        data = json.loads(row["data"])
        return OTSTrajectory.from_dict(data)

    async def search(
        self,
        query: str,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search trajectories by semantic similarity.

        Uses pgvector's efficient ANN search.

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

        # Perform vector search using pgvector
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT data, 1 - (embedding <=> $1) as similarity
                FROM {self.table_name}
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1
                LIMIT $2
                """,
                query_embedding,
                limit * 2,  # Get extra for filtering
            )

        results = []
        for row in rows:
            similarity = float(row["similarity"])

            if min_score is not None and similarity < min_score:
                continue

            trajectory = OTSTrajectory.from_dict(json.loads(row["data"]))
            results.append(SearchResult(
                trajectory=trajectory,
                similarity=similarity,
            ))

        return results[:limit]

    async def _text_search(
        self,
        query: str,
        limit: int,
        min_score: Optional[float],
    ) -> List[SearchResult]:
        """Fallback text search when no embedding provider."""
        query_lower = query.lower()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT data, task_description, tags
                FROM {self.table_name}
                WHERE LOWER(task_description) LIKE $1
                   OR $2 = ANY(tags)
                LIMIT $3
                """,
                f"%{query_lower}%",
                query_lower,
                limit,
            )

        results = []
        for row in rows:
            score = 0.8  # Default score for text match

            trajectory = OTSTrajectory.from_dict(json.loads(row["data"]))
            if min_score is None or score >= min_score:
                results.append(SearchResult(
                    trajectory=trajectory,
                    similarity=score,
                ))

        return results

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
        # Build query
        conditions = []
        params: List[Any] = []
        param_idx = 1

        if domain:
            conditions.append(f"domain = ${param_idx}")
            params.append(domain)
            param_idx += 1

        if tags:
            conditions.append(f"tags && ${param_idx}")
            params.append(tags)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        params.extend([limit, offset])

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT data FROM {self.table_name}
                WHERE {where_clause}
                ORDER BY timestamp_start DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
                """,
                *params,
            )

        return [OTSTrajectory.from_dict(json.loads(row["data"])) for row in rows]

    async def delete(self, trajectory_id: str) -> bool:
        """
        Delete a trajectory.

        Args:
            trajectory_id: ID of the trajectory to delete

        Returns:
            True if deleted, False if not found
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.table_name} WHERE id = $1",
                trajectory_id,
            )
            return result.split()[-1] != "0"

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

    async def close(self) -> None:
        """Close the connection pool."""
        await self._pool.close()

    async def clear(self) -> None:
        """Clear all trajectories (for testing)."""
        async with self._pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self.table_name}")
