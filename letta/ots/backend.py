"""
LettaStorageBackend - Uses Letta's PostgreSQL infrastructure for OTS storage.

Implements the ots.StorageBackend protocol to enable OTS trajectories
to be stored in Letta's existing trajectory infrastructure.
"""

from typing import Any, Dict, List, Optional

from ots import OTSTrajectory, SearchResult

from letta.log import get_logger
from letta.ots.adapter import LettaAdapter
from letta.schemas.trajectory import TrajectoryCreate, TrajectorySearchRequest
from letta.schemas.user import User as PydanticUser
from letta.services.trajectory_manager import TrajectoryManager

logger = get_logger(__name__)


class LettaStorageBackend:
    """
    Storage backend using Letta's PostgreSQL + TrajectoryManager.

    Implements the ots.StorageBackend protocol.

    This backend provides:
    - Persistent storage in PostgreSQL with JSONB
    - Semantic search via pgvector embeddings
    - Async LLM processing for summaries and scoring
    - Cross-organization sharing with anonymization

    Example:
        from ots import TrajectoryStore
        from letta.ots import LettaStorageBackend

        backend = LettaStorageBackend(trajectory_manager, actor)
        store = TrajectoryStore(backend=backend)
    """

    def __init__(
        self,
        trajectory_manager: Optional[TrajectoryManager] = None,
        actor: Optional[PydanticUser] = None,
        auto_process: bool = True,
    ) -> None:
        """
        Initialize Letta storage backend.

        Args:
            trajectory_manager: Letta TrajectoryManager instance
            actor: User context for storage operations
            auto_process: Whether to trigger async LLM processing on store
        """
        self.trajectory_manager = trajectory_manager or TrajectoryManager()
        self.actor = actor
        self.auto_process = auto_process
        self.adapter = LettaAdapter()

    def set_actor(self, actor: PydanticUser) -> None:
        """Set the actor for storage operations."""
        self.actor = actor

    async def store(self, trajectory: OTSTrajectory) -> str:
        """
        Store an OTS trajectory in Letta's storage.

        Args:
            trajectory: OTS trajectory to store

        Returns:
            Trajectory ID
        """
        if not self.actor:
            raise ValueError("Actor must be set before storing trajectories")

        # Convert OTS to Letta storage format
        letta_data = self.adapter.from_ots(trajectory)

        # Create trajectory in Letta storage
        trajectory_create = TrajectoryCreate(
            agent_id=trajectory.metadata.agent_id,
            data=letta_data,
        )

        letta_trajectory = await self.trajectory_manager.create_and_process_async(
            trajectory_create,
            self.actor,
            auto_process=self.auto_process,
        )

        logger.info(f"Stored OTS trajectory as {letta_trajectory.id}")
        return str(letta_trajectory.id)

    async def get(self, trajectory_id: str) -> Optional[OTSTrajectory]:
        """
        Retrieve an OTS trajectory by ID.

        Args:
            trajectory_id: ID of the trajectory

        Returns:
            OTS trajectory or None if not found
        """
        if not self.actor:
            raise ValueError("Actor must be set before retrieving trajectories")

        letta_trajectory = await self.trajectory_manager.get_trajectory_async(
            trajectory_id,
            self.actor,
        )

        if not letta_trajectory:
            return None

        return self.adapter.to_ots(letta_trajectory)

    async def search(
        self,
        query: str,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search trajectories using semantic similarity.

        Uses Letta's pgvector-based search with OpenAI embeddings.

        Args:
            query: Natural language search query
            limit: Maximum results to return
            min_score: Minimum trajectory quality score threshold

        Returns:
            List of search results with trajectories and similarity scores
        """
        if not self.actor:
            raise ValueError("Actor must be set before searching trajectories")

        search_request = TrajectorySearchRequest(
            query=query,
            min_score=min_score,
            limit=limit,
        )

        results = await self.trajectory_manager.search_trajectories_async(
            search_request,
            self.actor,
        )

        return [
            SearchResult(
                trajectory=self.adapter.to_ots(result.trajectory),
                similarity=result.similarity,
                metadata={
                    "letta_id": str(result.trajectory.id),
                    "processing_status": result.trajectory.processing_status,
                },
            )
            for result in results
        ]

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
            domain: Filter by domain type
            tags: Filter by tags

        Returns:
            List of OTS trajectories
        """
        if not self.actor:
            raise ValueError("Actor must be set before listing trajectories")

        # Use trajectory manager's list with filters
        letta_trajectories = await self.trajectory_manager.list_trajectories_async(
            self.actor,
            domain_type=domain,
            tags=tags,
            limit=limit,
            offset=offset,
        )

        return [
            self.adapter.to_ots(traj)
            for traj in letta_trajectories
        ]

    async def delete(self, trajectory_id: str) -> bool:
        """
        Delete a trajectory.

        Args:
            trajectory_id: ID of the trajectory to delete

        Returns:
            True if deleted, False if not found
        """
        if not self.actor:
            raise ValueError("Actor must be set before deleting trajectories")

        try:
            await self.trajectory_manager.delete_trajectory_async(
                trajectory_id,
                self.actor,
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to delete trajectory {trajectory_id}: {e}")
            return False

    async def search_by_domain(
        self,
        query: str,
        domain_type: str,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search trajectories within a specific domain.

        Args:
            query: Natural language search query
            domain_type: Domain type filter (e.g., "story_agent")
            limit: Maximum results
            min_score: Minimum quality score threshold

        Returns:
            List of search results
        """
        if not self.actor:
            raise ValueError("Actor must be set before searching trajectories")

        search_request = TrajectorySearchRequest(
            query=query,
            domain_type=domain_type,
            min_score=min_score,
            limit=limit,
        )

        results = await self.trajectory_manager.search_trajectories_async(
            search_request,
            self.actor,
        )

        return [
            SearchResult(
                trajectory=self.adapter.to_ots(result.trajectory),
                similarity=result.similarity,
            )
            for result in results
        ]

    async def search_cross_org(
        self,
        query: str,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search trajectories across organizations (anonymized).

        Returns anonymized results from trajectories marked for cross-org sharing.

        Args:
            query: Natural language search query
            limit: Maximum results
            min_score: Minimum quality score threshold

        Returns:
            List of anonymized search results
        """
        if not self.actor:
            raise ValueError("Actor must be set before searching trajectories")

        search_request = TrajectorySearchRequest(
            query=query,
            min_score=min_score,
            limit=limit,
            include_cross_org=True,
        )

        results = await self.trajectory_manager.search_trajectories_async(
            search_request,
            self.actor,
        )

        return [
            SearchResult(
                trajectory=self.adapter.to_ots(result.trajectory),
                similarity=result.similarity,
                metadata={
                    "anonymized": result.trajectory.organization_id != self.actor.organization_id,
                },
            )
            for result in results
        ]

    async def update_sharing(
        self,
        trajectory_id: str,
        share_cross_org: bool,
        domain_type: Optional[str] = None,
    ) -> bool:
        """
        Update cross-organization sharing settings.

        Args:
            trajectory_id: ID of the trajectory
            share_cross_org: Whether to share across organizations
            domain_type: Optional domain type for grouping

        Returns:
            True if updated successfully
        """
        if not self.actor:
            raise ValueError("Actor must be set before updating trajectories")

        from letta.schemas.trajectory import TrajectoryUpdate

        try:
            await self.trajectory_manager.update_trajectory_async(
                trajectory_id,
                TrajectoryUpdate(
                    share_cross_org=share_cross_org,
                    domain_type=domain_type,
                ),
                self.actor,
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to update sharing for {trajectory_id}: {e}")
            return False
