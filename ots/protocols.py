"""
Protocol definitions for OTS extensibility.

These protocols define the interfaces that frameworks and users can implement
to customize OTS behavior for their specific needs.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, runtime_checkable

from ots.models import OTSEntity, OTSTrajectory


@dataclass
class SearchResult:
    """Result from a trajectory search."""
    trajectory: OTSTrajectory
    similarity: float
    metadata: Optional[dict] = None


# === Trajectory Adapter ===


@runtime_checkable
class TrajectoryAdapter(Protocol):
    """
    Converts between framework-specific trajectory format and OTS.

    Each agent framework (Letta, LangChain, etc.) implements this
    to convert their internal trajectory representation to/from OTS.

    Example:
        class LettaAdapter(TrajectoryAdapter):
            def to_ots(self, letta_trajectory) -> OTSTrajectory:
                # Convert Letta's format to OTS
                ...

            def from_ots(self, ots_trajectory: OTSTrajectory) -> LettaTrajectory:
                # Convert OTS back to Letta's format
                ...
    """

    def to_ots(self, framework_trajectory: Any) -> OTSTrajectory:
        """Convert framework trajectory to OTS format."""
        ...

    def from_ots(self, ots_trajectory: OTSTrajectory) -> Any:
        """Convert OTS trajectory back to framework format."""
        ...


# === Storage Backend ===


@runtime_checkable
class StorageBackend(Protocol):
    """
    Storage backend for trajectories.

    Implement this to provide custom storage (PostgreSQL, MongoDB, S3, etc.).
    OTS ships with SQLite, File, and Memory backends.

    Example:
        class PostgreSQLBackend(StorageBackend):
            async def store(self, trajectory: OTSTrajectory) -> str:
                # Store in PostgreSQL
                ...
    """

    async def store(self, trajectory: OTSTrajectory) -> str:
        """
        Store a trajectory.

        Args:
            trajectory: OTS trajectory to store

        Returns:
            Trajectory ID
        """
        ...

    async def get(self, trajectory_id: str) -> Optional[OTSTrajectory]:
        """
        Retrieve a trajectory by ID.

        Args:
            trajectory_id: ID of the trajectory

        Returns:
            Trajectory or None if not found
        """
        ...

    async def search(
        self,
        query: str,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search trajectories by semantic similarity.

        Args:
            query: Search query (natural language)
            limit: Maximum results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of search results with trajectories and scores
        """
        ...

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
        ...

    async def delete(self, trajectory_id: str) -> bool:
        """
        Delete a trajectory.

        Args:
            trajectory_id: ID of the trajectory to delete

        Returns:
            True if deleted, False if not found
        """
        ...


# === Entity Extractor ===


@runtime_checkable
class EntityExtractor(Protocol):
    """
    Extracts domain-specific entities from trajectories.

    Implement this for domain-specific entity extraction.
    OTS ships with ToolEntityExtractor for generic tool-based extraction.

    Example:
        class DSFEntityExtractor(EntityExtractor):
            def extract(self, trajectory: OTSTrajectory) -> List[OTSEntity]:
                # Extract worlds, stories, rules from DSF tool calls
                ...
    """

    def extract(self, trajectory: OTSTrajectory) -> List[OTSEntity]:
        """
        Extract entities from a trajectory.

        Args:
            trajectory: OTS trajectory to extract entities from

        Returns:
            List of extracted entities
        """
        ...


# === Embedding Provider ===


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Provides text embeddings for semantic search.

    Implement this for custom embedding providers (local models, Anthropic, etc.).
    OTS ships with OpenAIEmbeddingProvider.

    Example:
        class LocalEmbeddingProvider(EmbeddingProvider):
            async def embed(self, text: str) -> List[float]:
                # Generate embeddings using local model
                ...
    """

    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


# === LLM Client ===


@runtime_checkable
class LLMClient(Protocol):
    """
    LLM client for decision enrichment.

    Used by DecisionExtractor to extract rationale and alternatives
    from agent reasoning. Optional - decisions can be extracted
    programmatically without LLM.

    Example:
        class AnthropicClient(LLMClient):
            async def generate(self, prompt: str, response_format: str = "text") -> str:
                # Call Claude API
                ...
    """

    async def generate(
        self,
        prompt: str,
        response_format: str = "text",
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Prompt to send to LLM
            response_format: "text" or "json"

        Returns:
            Generated text
        """
        ...
