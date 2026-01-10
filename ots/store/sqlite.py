"""
SQLite storage backend for OTS trajectories.

Provides persistent storage with optional semantic search via embeddings.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from ots.models import OTSTrajectory
from ots.protocols import EmbeddingProvider, SearchResult


class SQLiteBackend:
    """
    SQLite storage backend with optional semantic search.

    Stores trajectories in SQLite database. When an embedding provider
    is configured, enables semantic search via vector similarity.

    Example:
        # Basic usage (text search only)
        backend = SQLiteBackend(Path("./trajectories.db"))

        # With semantic search
        from ots.embeddings import OpenAIEmbeddingProvider
        backend = SQLiteBackend(
            Path("./trajectories.db"),
            embedding_provider=OpenAIEmbeddingProvider(),
        )
    """

    def __init__(
        self,
        db_path: Path,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        """
        Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database file
            embedding_provider: Optional embedding provider for semantic search
        """
        self.db_path = Path(db_path)
        self.embedding_provider = embedding_provider
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    id TEXT PRIMARY KEY,
                    data JSON NOT NULL,
                    domain TEXT,
                    task_description TEXT,
                    tags TEXT,
                    timestamp_start TEXT,
                    embedding BLOB,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indices for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trajectories_domain
                ON trajectories(domain)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trajectories_timestamp
                ON trajectories(timestamp_start DESC)
            """)

            conn.commit()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    async def store(self, trajectory: OTSTrajectory) -> str:
        """Store a trajectory in SQLite."""
        data = trajectory.to_dict()

        # Generate embedding if provider configured
        embedding_bytes: Optional[bytes] = None
        if self.embedding_provider:
            # Embed task description + first turn content
            embed_text = self._trajectory_to_embed_text(trajectory)
            embedding = await self.embedding_provider.embed(embed_text)
            embedding_bytes = self._embedding_to_bytes(embedding)

        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trajectories
                (id, data, domain, task_description, tags, timestamp_start, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trajectory.trajectory_id,
                    json.dumps(data, default=str),
                    trajectory.metadata.domain,
                    trajectory.metadata.task_description,
                    ",".join(trajectory.metadata.tags),
                    trajectory.metadata.timestamp_start.isoformat(),
                    embedding_bytes,
                ),
            )
            conn.commit()

        return trajectory.trajectory_id

    async def get(self, trajectory_id: str) -> Optional[OTSTrajectory]:
        """Get a trajectory by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT data FROM trajectories WHERE id = ?",
                (trajectory_id,),
            ).fetchone()

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
        Search trajectories.

        If embedding provider is configured, uses semantic search.
        Otherwise falls back to text matching.
        """
        if self.embedding_provider:
            return await self._semantic_search(query, limit, min_score)
        else:
            return await self._text_search(query, limit, min_score)

    async def _semantic_search(
        self,
        query: str,
        limit: int,
        min_score: Optional[float],
    ) -> List[SearchResult]:
        """Semantic search using embeddings."""
        query_embedding = await self.embedding_provider.embed(query)

        results = []
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id, data, embedding FROM trajectories WHERE embedding IS NOT NULL"
            ).fetchall()

        for row in rows:
            if not row["embedding"]:
                continue

            trajectory = OTSTrajectory.from_dict(json.loads(row["data"]))
            stored_embedding = self._bytes_to_embedding(row["embedding"])

            similarity = self._cosine_similarity(query_embedding, stored_embedding)

            if min_score is None or similarity >= min_score:
                results.append(SearchResult(
                    trajectory=trajectory,
                    similarity=similarity,
                ))

        # Sort by similarity descending
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    async def _text_search(
        self,
        query: str,
        limit: int,
        min_score: Optional[float],
    ) -> List[SearchResult]:
        """Simple text search fallback."""
        results = []
        query_lower = query.lower()

        with self._connection() as conn:
            rows = conn.execute(
                "SELECT data, task_description, tags FROM trajectories"
            ).fetchall()

        for row in rows:
            score = 0.0
            task_desc = (row["task_description"] or "").lower()

            if query_lower in task_desc:
                score = 0.8

            tags = (row["tags"] or "").split(",")
            for tag in tags:
                if query_lower in tag.lower():
                    score = max(score, 0.7)

            if score > 0 and (min_score is None or score >= min_score):
                trajectory = OTSTrajectory.from_dict(json.loads(row["data"]))
                results.append(SearchResult(
                    trajectory=trajectory,
                    similarity=score,
                ))

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
        query = "SELECT data FROM trajectories WHERE 1=1"
        params: List[Any] = []

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        if tags:
            # Match any tag
            tag_conditions = " OR ".join(["tags LIKE ?" for _ in tags])
            query += f" AND ({tag_conditions})"
            params.extend([f"%{tag}%" for tag in tags])

        query += " ORDER BY timestamp_start DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [OTSTrajectory.from_dict(json.loads(row["data"])) for row in rows]

    async def delete(self, trajectory_id: str) -> bool:
        """Delete a trajectory."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM trajectories WHERE id = ?",
                (trajectory_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

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

    def _embedding_to_bytes(self, embedding: List[float]) -> bytes:
        """Convert embedding to bytes for storage."""
        import struct
        return struct.pack(f"{len(embedding)}f", *embedding)

    def _bytes_to_embedding(self, data: bytes) -> List[float]:
        """Convert bytes back to embedding."""
        import struct
        count = len(data) // 4  # 4 bytes per float
        return list(struct.unpack(f"{count}f", data))

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            # Handle dimension mismatch by truncating to shorter
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def clear(self) -> None:
        """Clear all trajectories (for testing)."""
        with self._connection() as conn:
            conn.execute("DELETE FROM trajectories")
            conn.commit()
