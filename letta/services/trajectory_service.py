"""
Trajectory service for managing agent execution traces with LLM-based processing.

This service handles:
- CRUD operations for trajectories
- LLM-based summary generation (for search)
- LLM-based outcome scoring (for filtering successes/failures)
- Embedding generation (for similarity search)
- Pgvector-based semantic search
"""

import json
from typing import List, Optional

from sqlalchemy import delete, desc, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from letta.orm.trajectory import Trajectory as TrajectoryModel
from letta.schemas.trajectory import (
    Trajectory,
    TrajectoryCreate,
    TrajectorySearchRequest,
    TrajectorySearchResult,
    TrajectoryUpdate,
)
from letta.services.trajectory_processing import TrajectoryProcessor
from letta.settings import DatabaseChoice, settings


class TrajectoryService:
    """Service for managing trajectories with LLM-powered processing."""

    def __init__(self, db: AsyncSession, user_id: str):
        self.db = db
        self.user_id = user_id
        self.processor = TrajectoryProcessor()

    async def create_trajectory(self, trajectory_create: TrajectoryCreate) -> Trajectory:
        """
        Create a new trajectory.

        Processing (summary, scoring, embedding) happens asynchronously after creation.
        """
        # Create ORM object
        trajectory_orm = TrajectoryModel(
            agent_id=trajectory_create.agent_id,
            data=trajectory_create.data,
            # LLM fields populated later
            searchable_summary=None,
            outcome_score=None,
            score_reasoning=None,
            embedding=None,
        )

        self.db.add(trajectory_orm)
        await self.db.commit()
        await self.db.refresh(trajectory_orm)

        # TODO: Trigger async processing job
        # For MVP, we'll process synchronously in the /process endpoint

        return self._orm_to_pydantic(trajectory_orm)

    async def get_trajectory(self, trajectory_id: str) -> Optional[Trajectory]:
        """Get a single trajectory by ID."""
        result = await self.db.execute(select(TrajectoryModel).where(TrajectoryModel.id == trajectory_id))
        trajectory_orm = result.scalar_one_or_none()

        if not trajectory_orm:
            return None

        return self._orm_to_pydantic(trajectory_orm)

    async def list_trajectories(
        self,
        agent_id: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Trajectory]:
        """
        List trajectories with filtering.
        """
        query = select(TrajectoryModel).order_by(desc(TrajectoryModel.created_at))

        # Apply filters
        if agent_id:
            query = query.where(TrajectoryModel.agent_id == agent_id)
        if min_score is not None:
            query = query.where(TrajectoryModel.outcome_score >= min_score)
        if max_score is not None:
            query = query.where(TrajectoryModel.outcome_score <= max_score)

        # Pagination
        query = query.limit(limit).offset(offset)

        result = await self.db.execute(query)
        trajectories_orm = result.scalars().all()

        return [self._orm_to_pydantic(t) for t in trajectories_orm]

    async def update_trajectory(self, trajectory_id: str, trajectory_update: TrajectoryUpdate) -> Optional[Trajectory]:
        """
        Update an existing trajectory.
        """
        result = await self.db.execute(select(TrajectoryModel).where(TrajectoryModel.id == trajectory_id))
        trajectory_orm = result.scalar_one_or_none()

        if not trajectory_orm:
            return None

        # Update fields
        if trajectory_update.data is not None:
            trajectory_orm.data = trajectory_update.data
        if trajectory_update.searchable_summary is not None:
            trajectory_orm.searchable_summary = trajectory_update.searchable_summary
        if trajectory_update.outcome_score is not None:
            trajectory_orm.outcome_score = trajectory_update.outcome_score
        if trajectory_update.score_reasoning is not None:
            trajectory_orm.score_reasoning = trajectory_update.score_reasoning

        await self.db.commit()
        await self.db.refresh(trajectory_orm)

        return self._orm_to_pydantic(trajectory_orm)

    async def delete_trajectory(self, trajectory_id: str) -> bool:
        """Delete a trajectory."""
        result = await self.db.execute(delete(TrajectoryModel).where(TrajectoryModel.id == trajectory_id))
        await self.db.commit()
        return result.rowcount > 0

    async def search_trajectories(self, search_request: TrajectorySearchRequest) -> List[TrajectorySearchResult]:
        """
        Search for similar trajectories using semantic similarity (pgvector).

        This is the core retrieval mechanism for context learning.
        """
        # Generate embedding for the search query
        query_embedding = await self.processor.generate_embedding(search_request.query)

        if not query_embedding:
            # Fallback: return recent trajectories if embedding generation fails
            trajectories = await self.list_trajectories(
                agent_id=search_request.agent_id, min_score=search_request.min_score, max_score=search_request.max_score, limit=search_request.limit
            )
            return [TrajectorySearchResult(trajectory=t, similarity=0.0) for t in trajectories]

        # Build similarity search query
        if settings.database_engine == DatabaseChoice.POSTGRES:
            # Use pgvector cosine similarity (<=> operator)
            # Note: The embedding column needs to exist and have pgvector index for this to work
            from pgvector.sqlalchemy import Vector

            # Calculate cosine similarity (1 - cosine distance)
            similarity_expr = TrajectoryModel.embedding.cosine_distance(query_embedding).label("distance")

            query = (
                select(TrajectoryModel, (1 - similarity_expr).label("similarity"))
                .where(TrajectoryModel.embedding.isnot(None))  # Only search trajectories with embeddings
                .order_by(similarity_expr)  # Order by distance (ascending = most similar first)
            )

            # Apply filters
            if search_request.agent_id:
                query = query.where(TrajectoryModel.agent_id == search_request.agent_id)
            if search_request.min_score is not None:
                query = query.where(TrajectoryModel.outcome_score >= search_request.min_score)
            if search_request.max_score is not None:
                query = query.where(TrajectoryModel.outcome_score <= search_request.max_score)

            # Limit results
            query = query.limit(search_request.limit)

            # Execute query
            result = await self.db.execute(query)
            rows = result.all()

            # Convert to search results
            results = [
                TrajectorySearchResult(trajectory=self._orm_to_pydantic(trajectory_orm), similarity=float(similarity)) for trajectory_orm, similarity in rows
            ]

            return results

        else:
            # SQLite fallback: no pgvector support
            # Return recent trajectories with placeholder similarity
            trajectories = await self.list_trajectories(
                agent_id=search_request.agent_id, min_score=search_request.min_score, max_score=search_request.max_score, limit=search_request.limit
            )
            return [TrajectorySearchResult(trajectory=t, similarity=0.0) for t in trajectories]

    async def process_trajectory(self, trajectory_id: str) -> Optional[Trajectory]:
        """
        Process a trajectory with LLM to generate summary, score, and embedding.

        This is the core LLM-powered processing:
        1. Generate searchable summary from trajectory data
        2. Score the trajectory (0-1, with reasoning)
        3. Generate embedding from summary
        """
        result = await self.db.execute(select(TrajectoryModel).where(TrajectoryModel.id == trajectory_id))
        trajectory_orm = result.scalar_one_or_none()

        if not trajectory_orm:
            return None

        # Process with LLM
        summary, score, reasoning, embedding = await self.processor.process_trajectory(trajectory_orm.data)

        # Update trajectory with processed data
        trajectory_orm.searchable_summary = summary
        trajectory_orm.outcome_score = score
        trajectory_orm.score_reasoning = reasoning
        trajectory_orm.embedding = embedding

        await self.db.commit()
        await self.db.refresh(trajectory_orm)

        return self._orm_to_pydantic(trajectory_orm)

    def _orm_to_pydantic(self, trajectory_orm: TrajectoryModel) -> Trajectory:
        """Convert ORM model to Pydantic schema."""
        return Trajectory(
            id=trajectory_orm.id,
            agent_id=trajectory_orm.agent_id,
            data=trajectory_orm.data,
            searchable_summary=trajectory_orm.searchable_summary,
            outcome_score=trajectory_orm.outcome_score,
            score_reasoning=trajectory_orm.score_reasoning,
            created_at=trajectory_orm.created_at,
            updated_at=trajectory_orm.updated_at,
            organization_id=trajectory_orm.organization_id,
        )
