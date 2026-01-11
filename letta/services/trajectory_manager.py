"""
Trajectory manager for managing agent execution traces with LLM-based processing.

This manager handles:
- CRUD operations for trajectories
- LLM-based summary generation (for search)
- LLM-based outcome scoring (for filtering successes/failures)
- Embedding generation (for similarity search)
- Pgvector-based semantic search
- Async background processing with retry logic
- Cross-organization sharing with anonymization
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, desc, func, select, text

# OTS imports for anonymization
from ots.privacy import hash_identifier as ots_hash_identifier

from letta.log import get_logger
from letta.orm.agent import Agent as AgentModel
from letta.orm.trajectory import Trajectory as TrajectoryModel
from letta.schemas.trajectory import (
    AnonymizedTrajectory,
    Trajectory,
    TrajectoryCreate,
    TrajectorySearchRequest,
    TrajectorySearchResult,
    TrajectoryUpdate,
    TrajectoryVisibility,
)
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.services.trajectory_processing import TrajectoryProcessor
from letta.settings import DatabaseChoice, settings

logger = get_logger(__name__)

# Salt for anonymization hashing - rotate periodically for security
ANONYMIZATION_SALT = os.environ.get("TRAJECTORY_ANONYMIZATION_SALT", "letta-trajectory-anon-salt-v1")


class TrajectoryManager:
    """Manager for trajectories with LLM-powered processing."""

    def __init__(self):
        self.processor = TrajectoryProcessor()
        self._processing_tasks: Dict[str, asyncio.Task] = {}  # Track background tasks

    async def create_trajectory_async(self, trajectory_create: TrajectoryCreate, actor: PydanticUser) -> Trajectory:
        """
        Create a new trajectory.

        Processing (summary, scoring, embedding) happens asynchronously after creation.
        Domain type is inherited from the agent for cross-org sharing support.
        """
        async with db_registry.async_session() as session:
            async with session.begin():
                # Fetch agent to get domain_type
                agent_domain_type = None
                if trajectory_create.agent_id:
                    agent = await session.get(AgentModel, trajectory_create.agent_id)
                    if agent:
                        agent_domain_type = agent.domain_type

                # Create ORM object
                trajectory_orm = TrajectoryModel(
                    agent_id=trajectory_create.agent_id,
                    organization_id=actor.organization_id,
                    domain_type=agent_domain_type,  # Inherit from agent
                    share_cross_org=False,  # Default opt-out for privacy
                    data=trajectory_create.data,
                    # LLM fields populated later
                    searchable_summary=None,
                    outcome_score=None,
                    score_reasoning=None,
                    embedding=None,
                )

                session.add(trajectory_orm)
                await session.flush()
                await session.refresh(trajectory_orm)

                return self._orm_to_pydantic(trajectory_orm)

    async def get_trajectory_async(self, trajectory_id: str, actor: PydanticUser) -> Optional[Trajectory]:
        """Get a single trajectory by ID."""
        async with db_registry.async_session() as session:
            result = await session.execute(
                select(TrajectoryModel).where(
                    TrajectoryModel.id == trajectory_id, TrajectoryModel.organization_id == actor.organization_id
                )
            )
            trajectory_orm = result.scalar_one_or_none()

            if not trajectory_orm:
                return None

            return self._orm_to_pydantic(trajectory_orm)

    async def list_trajectories_async(
        self,
        actor: PydanticUser,
        agent_id: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Trajectory]:
        """
        List trajectories with filtering.
        """
        async with db_registry.async_session() as session:
            query = select(TrajectoryModel).where(TrajectoryModel.organization_id == actor.organization_id).order_by(desc(TrajectoryModel.created_at))

            # Apply filters
            if agent_id:
                query = query.where(TrajectoryModel.agent_id == agent_id)
            if min_score is not None:
                query = query.where(TrajectoryModel.outcome_score >= min_score)
            if max_score is not None:
                query = query.where(TrajectoryModel.outcome_score <= max_score)

            # Pagination
            query = query.limit(limit).offset(offset)

            result = await session.execute(query)
            trajectories_orm = result.scalars().all()

            return [self._orm_to_pydantic(t) for t in trajectories_orm]

    async def update_trajectory_async(
        self, trajectory_id: str, trajectory_update: TrajectoryUpdate, actor: PydanticUser
    ) -> Optional[Trajectory]:
        """
        Update an existing trajectory.
        """
        async with db_registry.async_session() as session:
            async with session.begin():
                result = await session.execute(
                    select(TrajectoryModel).where(
                        TrajectoryModel.id == trajectory_id, TrajectoryModel.organization_id == actor.organization_id
                    )
                )
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
                if trajectory_update.tags is not None:
                    trajectory_orm.tags = trajectory_update.tags
                if trajectory_update.task_category is not None:
                    trajectory_orm.task_category = trajectory_update.task_category
                if trajectory_update.complexity_level is not None:
                    trajectory_orm.complexity_level = trajectory_update.complexity_level
                if trajectory_update.trajectory_metadata is not None:
                    trajectory_orm.trajectory_metadata = trajectory_update.trajectory_metadata

                await session.flush()
                await session.refresh(trajectory_orm)

                return self._orm_to_pydantic(trajectory_orm)

    async def delete_trajectory_async(self, trajectory_id: str, actor: PydanticUser) -> bool:
        """Delete a trajectory."""
        async with db_registry.async_session() as session:
            async with session.begin():
                result = await session.execute(
                    delete(TrajectoryModel).where(
                        TrajectoryModel.id == trajectory_id, TrajectoryModel.organization_id == actor.organization_id
                    )
                )
                return result.rowcount > 0

    async def search_trajectories_async(self, search_request: TrajectorySearchRequest, actor: PydanticUser) -> List[TrajectorySearchResult]:
        """
        Search for similar trajectories using semantic similarity (pgvector).

        This is the core retrieval mechanism for context learning.
        Supports cross-organization search with anonymization when include_cross_org=True.
        """
        # Generate embedding for the search query
        query_embedding = await self.processor.generate_embedding(search_request.query)

        if not query_embedding:
            # Fallback: return recent trajectories if embedding generation fails
            trajectories = await self.list_trajectories_async(
                actor=actor,
                agent_id=search_request.agent_id,
                min_score=search_request.min_score,
                max_score=search_request.max_score,
                limit=search_request.limit,
            )
            return [TrajectorySearchResult(trajectory=t, similarity=0.0, visibility=TrajectoryVisibility.FULL) for t in trajectories]

        results = []

        async with db_registry.async_session() as session:
            # Build similarity search query
            if settings.database_engine == DatabaseChoice.POSTGRES:
                # Use pgvector cosine similarity (<=> operator)
                from pgvector.sqlalchemy import Vector

                # Calculate cosine similarity (1 - cosine distance)
                similarity_expr = TrajectoryModel.embedding.cosine_distance(query_embedding).label("distance")

                # 1. Same-org query (full detail)
                same_org_query = (
                    select(TrajectoryModel, (1 - similarity_expr).label("similarity"))
                    .where(
                        TrajectoryModel.embedding.isnot(None),
                        TrajectoryModel.organization_id == actor.organization_id,
                    )
                    .order_by(similarity_expr)
                )

                # Apply filters to same-org query
                if search_request.agent_id:
                    same_org_query = same_org_query.where(TrajectoryModel.agent_id == search_request.agent_id)
                if search_request.domain_type:
                    same_org_query = same_org_query.where(TrajectoryModel.domain_type == search_request.domain_type)
                if search_request.min_score is not None:
                    same_org_query = same_org_query.where(TrajectoryModel.outcome_score >= search_request.min_score)
                if search_request.max_score is not None:
                    same_org_query = same_org_query.where(TrajectoryModel.outcome_score <= search_request.max_score)
                if search_request.task_category:
                    same_org_query = same_org_query.where(TrajectoryModel.task_category == search_request.task_category)
                if search_request.complexity_level:
                    same_org_query = same_org_query.where(TrajectoryModel.complexity_level == search_request.complexity_level)
                if search_request.tags:
                    for tag in search_request.tags:
                        same_org_query = same_org_query.where(TrajectoryModel.tags.contains([tag]))

                same_org_query = same_org_query.limit(search_request.limit)
                same_org_result = await session.execute(same_org_query)

                for trajectory_orm, similarity in same_org_result.all():
                    results.append(TrajectorySearchResult(
                        trajectory=self._orm_to_pydantic(trajectory_orm),
                        similarity=float(similarity),
                        visibility=TrajectoryVisibility.FULL,
                    ))

                # 2. Cross-org query (anonymized) - only if requested AND domain_type specified
                if search_request.include_cross_org and search_request.domain_type:
                    cross_org_query = (
                        select(TrajectoryModel, (1 - similarity_expr).label("similarity"))
                        .where(
                            TrajectoryModel.embedding.isnot(None),
                            TrajectoryModel.organization_id != actor.organization_id,
                            TrajectoryModel.domain_type == search_request.domain_type,
                            TrajectoryModel.share_cross_org == True,
                        )
                        .order_by(similarity_expr)
                    )

                    # Apply score/category/complexity filters to cross-org
                    if search_request.min_score is not None:
                        cross_org_query = cross_org_query.where(TrajectoryModel.outcome_score >= search_request.min_score)
                    if search_request.max_score is not None:
                        cross_org_query = cross_org_query.where(TrajectoryModel.outcome_score <= search_request.max_score)
                    if search_request.task_category:
                        cross_org_query = cross_org_query.where(TrajectoryModel.task_category == search_request.task_category)
                    if search_request.complexity_level:
                        cross_org_query = cross_org_query.where(TrajectoryModel.complexity_level == search_request.complexity_level)
                    if search_request.tags:
                        for tag in search_request.tags:
                            cross_org_query = cross_org_query.where(TrajectoryModel.tags.contains([tag]))

                    cross_org_query = cross_org_query.limit(search_request.limit)
                    cross_org_result = await session.execute(cross_org_query)

                    for trajectory_orm, similarity in cross_org_result.all():
                        # Create anonymized trajectory
                        anonymized = self._anonymize_trajectory(trajectory_orm)
                        # Wrap in Trajectory for API consistency
                        results.append(TrajectorySearchResult(
                            trajectory=Trajectory(
                                id=anonymized.id,
                                agent_id=anonymized.agent_id,
                                domain_type=anonymized.domain_type,
                                share_cross_org=True,
                                data=anonymized.data,
                                searchable_summary=anonymized.searchable_summary,
                                outcome_score=anonymized.outcome_score,
                                score_reasoning=anonymized.score_reasoning,
                                tags=anonymized.tags,
                                task_category=anonymized.task_category,
                                complexity_level=anonymized.complexity_level,
                                trajectory_metadata=anonymized.trajectory_metadata,
                                processing_status="completed",
                                created_at=anonymized.created_at,
                                organization_id=anonymized.source_organization_hash,
                            ),
                            similarity=float(similarity),
                            visibility=TrajectoryVisibility.ANONYMIZED,
                        ))

                # Sort all results by similarity and limit
                results.sort(key=lambda r: r.similarity, reverse=True)
                return results[:search_request.limit]

            else:
                # SQLite fallback: no pgvector support
                # Return recent trajectories with placeholder similarity
                trajectories = await self.list_trajectories_async(
                    actor=actor,
                    agent_id=search_request.agent_id,
                    min_score=search_request.min_score,
                    max_score=search_request.max_score,
                    limit=search_request.limit,
                )
                return [TrajectorySearchResult(trajectory=t, similarity=0.0, visibility=TrajectoryVisibility.FULL) for t in trajectories]

    async def process_trajectory_async(self, trajectory_id: str, actor: PydanticUser) -> Optional[Trajectory]:
        """
        Process a trajectory with LLM to generate summary, score, labels, metadata, embedding, and OTS data.

        This is the core LLM-powered processing:
        1. Generate searchable summary from trajectory data (gpt-4o-mini)
        2. Score the trajectory (0-1, with reasoning) (gpt-4o-mini)
        3. Extract labels and metadata (tags, category, complexity, patterns) (gpt-4o-mini)
        4. Generate embedding from summary (text-embedding-3-small)
        5. Extract OTS decisions and entities (gpt-4o-mini)
        """
        async with db_registry.async_session() as session:
            async with session.begin():
                result = await session.execute(
                    select(TrajectoryModel).where(
                        TrajectoryModel.id == trajectory_id, TrajectoryModel.organization_id == actor.organization_id
                    )
                )
                trajectory_orm = result.scalar_one_or_none()

                if not trajectory_orm:
                    return None

                # Process with LLM (includes OTS extraction with gpt-4o-mini)
                (
                    summary,
                    score,
                    reasoning,
                    tags,
                    task_category,
                    complexity_level,
                    trajectory_metadata,
                    embedding,
                    ots_decisions,
                    ots_entities,
                ) = await self.processor.process_trajectory(trajectory_orm.data)

                # Update trajectory with processed data
                trajectory_orm.searchable_summary = summary
                trajectory_orm.outcome_score = score
                trajectory_orm.score_reasoning = reasoning
                trajectory_orm.tags = tags
                trajectory_orm.task_category = task_category
                trajectory_orm.complexity_level = complexity_level
                trajectory_orm.trajectory_metadata = trajectory_metadata
                trajectory_orm.embedding = embedding
                trajectory_orm.ots_decisions = ots_decisions
                trajectory_orm.ots_entities = ots_entities

                await session.flush()
                await session.refresh(trajectory_orm)

                return self._orm_to_pydantic(trajectory_orm)

    async def create_and_process_async(
        self,
        trajectory_create: TrajectoryCreate,
        actor: PydanticUser,
        auto_process: bool = True
    ) -> Trajectory:
        """
        Create a trajectory and optionally process it asynchronously in the background.

        This is the recommended way to create trajectories - it returns immediately
        while processing (summary, scoring, embedding) happens in the background.

        Args:
            trajectory_create: Trajectory data
            actor: User creating the trajectory
            auto_process: If True, spawn background task to process (default: True)

        Returns:
            Trajectory with processing_status='pending' (will be 'completed' later)
        """
        # Create trajectory (fast, <100ms)
        trajectory = await self.create_trajectory_async(trajectory_create, actor)

        # Spawn background processing task (non-blocking!)
        if auto_process:
            task = asyncio.create_task(
                self._process_trajectory_background(trajectory.id, actor)
            )
            self._processing_tasks[trajectory.id] = task
            logger.info(f"Spawned background processing task for trajectory {trajectory.id}")

        return trajectory

    async def _process_trajectory_background(
        self,
        trajectory_id: str,
        actor: PydanticUser,
        max_retries: int = 3,
        initial_delay: float = 2.0
    ) -> None:
        """
        Background task to process trajectory with exponential backoff retry logic.

        Processing includes:
        1. Generate searchable summary (LLM call, ~3-5 seconds)
        2. Score outcome quality (LLM call, ~3-5 seconds)
        3. Extract labels and metadata (LLM call, ~3-5 seconds)
        4. Generate embedding (API call, ~1-2 seconds)

        Total time: 10-20 seconds

        Args:
            trajectory_id: ID of trajectory to process
            actor: User for database access
            max_retries: Maximum number of retry attempts (default: 3)
            initial_delay: Initial delay between retries in seconds (default: 2.0)
        """
        retry_count = 0
        delay = initial_delay

        while retry_count <= max_retries:
            try:
                # Mark as processing on first attempt
                if retry_count == 0:
                    async with db_registry.async_session() as session:
                        async with session.begin():
                            result = await session.execute(
                                select(TrajectoryModel).where(
                                    TrajectoryModel.id == trajectory_id,
                                    TrajectoryModel.organization_id == actor.organization_id
                                )
                            )
                            trajectory_orm = result.scalar_one_or_none()
                            if trajectory_orm:
                                trajectory_orm.processing_status = "processing"
                                trajectory_orm.processing_started_at = datetime.utcnow()

                # Process with LLM (this is the slow part: 7-15 seconds)
                logger.info(f"Processing trajectory {trajectory_id} (attempt {retry_count + 1}/{max_retries + 1})")
                await self.process_trajectory_async(trajectory_id, actor)

                # Mark as completed
                async with db_registry.async_session() as session:
                    async with session.begin():
                        result = await session.execute(
                            select(TrajectoryModel).where(
                                TrajectoryModel.id == trajectory_id,
                                TrajectoryModel.organization_id == actor.organization_id
                            )
                        )
                        trajectory_orm = result.scalar_one_or_none()
                        if trajectory_orm:
                            trajectory_orm.processing_status = "completed"
                            trajectory_orm.processing_completed_at = datetime.utcnow()
                            trajectory_orm.processing_error = None

                logger.info(f"Successfully processed trajectory {trajectory_id}")

                # Remove from tracking dict
                self._processing_tasks.pop(trajectory_id, None)
                return  # Success!

            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                logger.error(f"Failed to process trajectory {trajectory_id} (attempt {retry_count}/{max_retries + 1}): {error_msg}")

                if retry_count > max_retries:
                    # All retries exhausted, mark as failed
                    try:
                        async with db_registry.async_session() as session:
                            async with session.begin():
                                result = await session.execute(
                                    select(TrajectoryModel).where(
                                        TrajectoryModel.id == trajectory_id,
                                        TrajectoryModel.organization_id == actor.organization_id
                                    )
                                )
                                trajectory_orm = result.scalar_one_or_none()
                                if trajectory_orm:
                                    trajectory_orm.processing_status = "failed"
                                    trajectory_orm.processing_error = f"Failed after {max_retries} retries: {error_msg}"
                                    trajectory_orm.processing_completed_at = datetime.utcnow()
                    except Exception as update_error:
                        logger.error(f"Failed to update trajectory status to failed: {update_error}")

                    # Remove from tracking dict
                    self._processing_tasks.pop(trajectory_id, None)
                    return  # Give up
                else:
                    # Wait before retry (exponential backoff)
                    logger.info(f"Retrying trajectory {trajectory_id} in {delay}s...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff: 2s, 4s, 8s

    async def get_processing_stats_async(self, actor: PydanticUser) -> Dict[str, int]:
        """
        Get processing statistics for trajectories.

        Returns counts by processing status for monitoring and debugging.
        """
        async with db_registry.async_session() as session:
            # Count by status
            result = await session.execute(
                select(
                    TrajectoryModel.processing_status,
                    func.count(TrajectoryModel.id).label("count")
                )
                .where(TrajectoryModel.organization_id == actor.organization_id)
                .group_by(TrajectoryModel.processing_status)
            )
            rows = result.all()

            # Build stats dict
            stats = {
                "total": 0,
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
            }

            for status, count in rows:
                stats[status] = count
                stats["total"] += count

            return stats

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status for active background processing tasks.

        Returns information about in-flight processing tasks for monitoring.
        """
        active_tasks = []
        for trajectory_id, task in self._processing_tasks.items():
            active_tasks.append({
                "trajectory_id": trajectory_id,
                "done": task.done(),
                "cancelled": task.cancelled(),
            })

        return {
            "active_count": len(active_tasks),
            "tasks": active_tasks,
        }

    def _orm_to_pydantic(self, trajectory_orm: TrajectoryModel) -> Trajectory:
        """Convert ORM model to Pydantic schema."""
        return Trajectory(
            id=trajectory_orm.id,
            agent_id=trajectory_orm.agent_id,
            domain_type=trajectory_orm.domain_type,
            share_cross_org=trajectory_orm.share_cross_org,
            data=trajectory_orm.data,
            searchable_summary=trajectory_orm.searchable_summary,
            outcome_score=trajectory_orm.outcome_score,
            score_reasoning=trajectory_orm.score_reasoning,
            tags=trajectory_orm.tags,
            task_category=trajectory_orm.task_category,
            complexity_level=trajectory_orm.complexity_level,
            trajectory_metadata=trajectory_orm.trajectory_metadata,
            ots_decisions=trajectory_orm.ots_decisions,
            ots_entities=trajectory_orm.ots_entities,
            processing_status=trajectory_orm.processing_status,
            processing_started_at=trajectory_orm.processing_started_at,
            processing_completed_at=trajectory_orm.processing_completed_at,
            processing_error=trajectory_orm.processing_error,
            created_at=trajectory_orm.created_at,
            updated_at=trajectory_orm.updated_at,
            organization_id=trajectory_orm.organization_id,
        )

    def _hash_identifier(self, value: str) -> str:
        """
        Hash an identifier for privacy in cross-org sharing.

        Uses OTS hash_identifier for consistent hashing across the ecosystem.
        """
        h = ots_hash_identifier(value, salt=ANONYMIZATION_SALT)
        return f"[REDACTED:{h[:8]}]"

    def _anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize trajectory data for cross-org sharing.

        Preserves structural metadata, redacts message content.
        """
        result = {}

        # Preserve metadata structure, anonymize content
        if "metadata" in data:
            result["metadata"] = {
                # Keep structural metrics
                "step_count": data["metadata"].get("step_count"),
                "message_count": data["metadata"].get("message_count"),
                "input_tokens": data["metadata"].get("input_tokens"),
                "output_tokens": data["metadata"].get("output_tokens"),
                "total_tokens": data["metadata"].get("total_tokens"),
                "tools_used": data["metadata"].get("tools_used"),  # Tool names are public
                "models": data["metadata"].get("models"),  # Model names are public
                "duration_ns": data["metadata"].get("duration_ns"),
            }

        # Anonymize turns - keep structure, redact content
        if "turns" in data:
            result["turns"] = [
                {
                    "model": turn.get("model"),
                    "input_tokens": turn.get("input_tokens"),
                    "output_tokens": turn.get("output_tokens"),
                    "message_count": len(turn.get("messages", [])),
                    "tool_calls_count": sum(
                        len(m.get("tool_calls", []) or [])
                        for m in turn.get("messages", [])
                    ),
                    # Messages completely redacted
                    "messages": "[REDACTED]"
                }
                for turn in data.get("turns", [])
            ]

        # Preserve outcome structure (already LLM-generated, abstract)
        if "outcome" in data:
            result["outcome"] = data["outcome"]

        return result

    def _anonymize_trajectory(self, trajectory_orm: TrajectoryModel) -> AnonymizedTrajectory:
        """
        Create anonymized view of trajectory for cross-org sharing.

        Preserves learning signal while protecting privacy:
        - Preserved: summary, score, tags, embeddings, structural metadata
        - Redacted: message content, tool arguments, identifiers
        """
        return AnonymizedTrajectory(
            id=self._hash_identifier(trajectory_orm.id),
            agent_id=self._hash_identifier(trajectory_orm.agent_id),
            domain_type=trajectory_orm.domain_type or "",
            searchable_summary=trajectory_orm.searchable_summary,
            outcome_score=trajectory_orm.outcome_score,
            score_reasoning=trajectory_orm.score_reasoning,
            tags=trajectory_orm.tags,
            task_category=trajectory_orm.task_category,
            complexity_level=trajectory_orm.complexity_level,
            trajectory_metadata=trajectory_orm.trajectory_metadata,
            data=self._anonymize_data(trajectory_orm.data),
            visibility=TrajectoryVisibility.ANONYMIZED,
            source_organization_hash=self._hash_identifier(trajectory_orm.organization_id or ""),
            created_at=trajectory_orm.created_at,
        )

    async def set_trajectory_sharing_async(
        self,
        trajectory_id: str,
        share_cross_org: bool,
        actor: PydanticUser
    ) -> Optional[Trajectory]:
        """
        Enable or disable cross-organization sharing for a trajectory.

        When enabled, the trajectory becomes searchable by other organizations,
        but they only see an anonymized version (PII redacted).
        """
        async with db_registry.async_session() as session:
            async with session.begin():
                result = await session.execute(
                    select(TrajectoryModel).where(
                        TrajectoryModel.id == trajectory_id,
                        TrajectoryModel.organization_id == actor.organization_id
                    )
                )
                trajectory_orm = result.scalar_one_or_none()

                if not trajectory_orm:
                    return None

                trajectory_orm.share_cross_org = share_cross_org
                await session.flush()
                await session.refresh(trajectory_orm)

                return self._orm_to_pydantic(trajectory_orm)

    async def list_trajectories_by_domain_async(
        self,
        domain_type: str,
        actor: PydanticUser,
        include_cross_org: bool = False,
        limit: int = 50,
    ) -> List[TrajectorySearchResult]:
        """
        List trajectories by domain type with optional cross-org inclusion.

        Returns same-org trajectories as full detail, cross-org as anonymized.
        """
        results = []

        async with db_registry.async_session() as session:
            # Same-org: full detail
            same_org_query = (
                select(TrajectoryModel)
                .where(
                    TrajectoryModel.organization_id == actor.organization_id,
                    TrajectoryModel.domain_type == domain_type,
                )
                .order_by(desc(TrajectoryModel.created_at))
                .limit(limit)
            )
            same_org_result = await session.execute(same_org_query)
            for traj in same_org_result.scalars().all():
                results.append(TrajectorySearchResult(
                    trajectory=self._orm_to_pydantic(traj),
                    similarity=1.0,  # Not similarity-based
                    visibility=TrajectoryVisibility.FULL,
                ))

            # Cross-org: anonymized (if requested)
            if include_cross_org:
                cross_org_query = (
                    select(TrajectoryModel)
                    .where(
                        TrajectoryModel.organization_id != actor.organization_id,
                        TrajectoryModel.domain_type == domain_type,
                        TrajectoryModel.share_cross_org == True,
                    )
                    .order_by(desc(TrajectoryModel.created_at))
                    .limit(limit)
                )
                cross_org_result = await session.execute(cross_org_query)
                for traj in cross_org_result.scalars().all():
                    # Return anonymized trajectory wrapped in the result
                    # Note: We create a Trajectory from the anonymized data for API compatibility
                    anonymized = self._anonymize_trajectory(traj)
                    # Convert to Trajectory for consistent return type
                    results.append(TrajectorySearchResult(
                        trajectory=Trajectory(
                            id=anonymized.id,
                            agent_id=anonymized.agent_id,
                            domain_type=anonymized.domain_type,
                            share_cross_org=True,
                            data=anonymized.data,
                            searchable_summary=anonymized.searchable_summary,
                            outcome_score=anonymized.outcome_score,
                            score_reasoning=anonymized.score_reasoning,
                            tags=anonymized.tags,
                            task_category=anonymized.task_category,
                            complexity_level=anonymized.complexity_level,
                            trajectory_metadata=anonymized.trajectory_metadata,
                            processing_status="completed",
                            created_at=anonymized.created_at,
                            organization_id=anonymized.source_organization_hash,
                        ),
                        similarity=1.0,
                        visibility=TrajectoryVisibility.ANONYMIZED,
                    ))

        return results[:limit]
