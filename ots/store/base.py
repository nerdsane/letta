"""
TrajectoryStore - Main facade for trajectory storage and retrieval.
"""

from pathlib import Path
from typing import List, Optional, Union

from ots.models import OTSEntity, OTSTrajectory
from ots.protocols import EmbeddingProvider, EntityExtractor, SearchResult, StorageBackend


class TrajectoryStore:
    """
    Main facade for storing and retrieving OTS trajectories.

    Provides a unified interface over different storage backends,
    with optional entity extraction on store.

    Recommended Backends:
        - LanceDBBackend: For context learning (semantic search) - pip install ots[lancedb]
        - SQLiteBackend: For simple storage without semantic search
        - PostgreSQLBackend: For production scale - pip install ots[postgres]

    Example:
        # For context learning (recommended)
        from ots.store.lancedb import LanceDBBackend
        from ots.embeddings import OpenAIEmbeddingProvider

        store = TrajectoryStore(
            backend=LanceDBBackend(
                Path("./trajectories"),
                embedding_provider=OpenAIEmbeddingProvider(),
            )
        )

        # Simple storage (no semantic search)
        store = TrajectoryStore()  # Uses SQLite

        # With entity extraction
        store.register_extractor(MyEntityExtractor())
    """

    def __init__(
        self,
        backend: Optional[StorageBackend] = None,
        db_path: Optional[Union[str, Path]] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        """
        Initialize trajectory store.

        Args:
            backend: Custom storage backend (optional)
            db_path: Path for default backend (default: ./trajectories.db or ./trajectories/)
            embedding_provider: Embedding provider for semantic search.
                               If provided and LanceDB is available, uses LanceDB.
                               Otherwise falls back to SQLite.
        """
        if backend is not None:
            self._backend = backend
        elif embedding_provider is not None:
            # Try to use LanceDB for semantic search
            self._backend = self._create_backend_with_embeddings(
                db_path, embedding_provider
            )
        else:
            # Default to SQLite (simple storage, no semantic search)
            from ots.store.sqlite import SQLiteBackend
            path = Path(db_path) if db_path else Path("./trajectories.db")
            self._backend = SQLiteBackend(path)

        self._extractors: List[EntityExtractor] = []

    def _create_backend_with_embeddings(
        self,
        db_path: Optional[Union[str, Path]],
        embedding_provider: EmbeddingProvider,
    ) -> StorageBackend:
        """Create the best available backend for semantic search."""
        # Try LanceDB first (preferred for semantic search)
        try:
            from ots.store.lancedb import LanceDBBackend
            path = Path(db_path) if db_path else Path("./trajectories")
            return LanceDBBackend(path, embedding_provider=embedding_provider)
        except ImportError:
            pass

        # Fall back to SQLite with embeddings (less efficient but works)
        from ots.store.sqlite import SQLiteBackend
        path = Path(db_path) if db_path else Path("./trajectories.db")
        return SQLiteBackend(path, embedding_provider=embedding_provider)

    def register_extractor(self, extractor: EntityExtractor) -> None:
        """
        Register an entity extractor.

        Extractors are run on store to populate trajectory.context.entities.

        Args:
            extractor: Entity extractor to register
        """
        self._extractors.append(extractor)

    async def store(self, trajectory: OTSTrajectory) -> str:
        """
        Store a trajectory.

        Runs registered extractors to populate entities before storing.

        Args:
            trajectory: OTS trajectory to store

        Returns:
            Trajectory ID
        """
        # Run extractors
        if self._extractors:
            all_entities: List[OTSEntity] = []
            existing_ids = {e.id for e in trajectory.context.entities}

            for extractor in self._extractors:
                entities = extractor.extract(trajectory)
                for entity in entities:
                    if entity.id not in existing_ids:
                        all_entities.append(entity)
                        existing_ids.add(entity.id)

            trajectory.context.entities.extend(all_entities)

        return await self._backend.store(trajectory)

    async def get(self, trajectory_id: str) -> Optional[OTSTrajectory]:
        """
        Get a trajectory by ID.

        Args:
            trajectory_id: ID of the trajectory

        Returns:
            Trajectory or None if not found
        """
        return await self._backend.get(trajectory_id)

    async def search(
        self,
        query: str,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search trajectories.

        Args:
            query: Search query (natural language)
            limit: Maximum results
            min_score: Minimum similarity threshold

        Returns:
            List of search results
        """
        return await self._backend.search(query, limit, min_score)

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
            tags: Filter by tags

        Returns:
            List of trajectories
        """
        return await self._backend.list(limit, offset, domain, tags)

    async def delete(self, trajectory_id: str) -> bool:
        """
        Delete a trajectory.

        Args:
            trajectory_id: ID of trajectory to delete

        Returns:
            True if deleted, False if not found
        """
        return await self._backend.delete(trajectory_id)

    @property
    def backend(self) -> StorageBackend:
        """Access the underlying storage backend."""
        return self._backend
