"""
Letta Decision Manager - thin wrapper over OTS DecisionManager.

Handles Letta-specific concerns:
- ORM persistence to trajectories_decisions table
- Integration with Letta's embedding provider
- Organization-scoped queries
- API response formatting

The core decision extraction and embedding logic is in OTS.
"""

from typing import Any, Dict, List, Optional

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

# OTS imports for core decision management
from ots.management import (
    DecisionManager as OTSDecisionManager,
    DecisionRecord,
    DecisionSearchResult,
    DecisionFilter,
)
from ots.models import (
    OTSTrajectory,
    OTSTurn,
    OTSDecision,
    OTSChoice,
    OTSConsequence,
    OTSMetadata,
    OTSContext,
    DecisionType,
    OutcomeType,
)

from letta.log import get_logger
from letta.orm.trajectory import Trajectory as TrajectoryORM
from letta.orm.trajectory_decision import TrajectoryDecision as TrajectoryDecisionORM
from letta.schemas.trajectory_decision import (
    TrajectoryDecision,
    TrajectoryDecisionCreate,
    TrajectoryDecisionSearchRequest,
    TrajectoryDecisionSearchResult,
    TrajectoryDecisionSearchResponse,
)
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry

logger = get_logger(__name__)


class LettaDecisionManager:
    """
    Letta-specific decision manager wrapping OTS core.

    Usage:
        manager = LettaDecisionManager(embedding_provider=EmbeddingClient())

        # Extract and store decisions from a trajectory
        decisions = await manager.extract_and_store(
            trajectory_id="traj-123",
            actor=current_user,
        )

        # Search decisions
        results = await manager.search_decisions(
            query="world_manager save operation",
            actor=current_user,
        )
    """

    def __init__(self, embedding_provider=None):
        """
        Initialize Letta DecisionManager.

        Args:
            embedding_provider: Provider for generating embeddings.
                               Should implement OTS EmbeddingProvider protocol.
        """
        self.ots_manager = OTSDecisionManager(embedding_provider)

    async def extract_and_store_async(
        self,
        trajectory_id: str,
        actor: PydanticUser,
    ) -> List[TrajectoryDecision]:
        """
        Extract decisions from trajectory and persist to DB.

        Args:
            trajectory_id: ID of trajectory to process
            actor: User performing the action

        Returns:
            List of created TrajectoryDecision records
        """
        async with db_registry.async_session() as session:
            async with session.begin():
                # Get trajectory data
                trajectory_orm = await self._get_trajectory(
                    trajectory_id,
                    actor.organization_id,
                    session,
                )
                if not trajectory_orm:
                    logger.warning(f"Trajectory not found: {trajectory_id}")
                    return []

                # Convert to OTS format
                ots_trajectory = self._letta_to_ots(trajectory_orm)

                # Use OTS to extract decisions
                records = self.ots_manager.extract_decisions(ots_trajectory)

                if not records:
                    logger.info(f"No decisions found in trajectory: {trajectory_id}")
                    return []

                # Generate embeddings
                records = await self.ots_manager.embed_decisions(records)

                # Persist to Letta ORM
                return await self._store_decisions(
                    records,
                    actor.organization_id,
                    session,
                )

    async def search_decisions_async(
        self,
        request: TrajectoryDecisionSearchRequest,
        actor: PydanticUser,
    ) -> TrajectoryDecisionSearchResponse:
        """
        Search decisions by semantic similarity.

        Args:
            request: Search request with query and filters
            actor: User performing the search

        Returns:
            Search response with matching decisions
        """
        if not self.ots_manager.embedding_provider:
            raise ValueError("Embedding provider required for search")

        # Generate query embedding
        query_embedding = await self.ots_manager.embedding_provider.embed(request.query)

        async with db_registry.async_session() as session:
            results = await self._search_by_embedding(
                embedding=query_embedding,
                organization_id=actor.organization_id,
                action=request.action,
                success=request.success,
                limit=request.limit,
                min_similarity=request.min_similarity,
                session=session,
            )

            return TrajectoryDecisionSearchResponse(
                results=results,
                query=request.query,
                total_candidates=len(results),  # Would need separate count query for actual total
            )

    async def get_decisions_for_trajectory_async(
        self,
        trajectory_id: str,
        actor: PydanticUser,
    ) -> List[TrajectoryDecision]:
        """
        Get all decisions for a trajectory.

        Args:
            trajectory_id: ID of trajectory
            actor: User performing the action

        Returns:
            List of decisions for the trajectory
        """
        async with db_registry.async_session() as session:
            result = await session.execute(
                select(TrajectoryDecisionORM)
                .where(
                    TrajectoryDecisionORM.trajectory_id == trajectory_id,
                    TrajectoryDecisionORM.organization_id == actor.organization_id,
                )
                .order_by(
                    TrajectoryDecisionORM.turn_index,
                    TrajectoryDecisionORM.decision_index,
                )
            )
            decisions_orm = result.scalars().all()
            return [self._orm_to_pydantic(d) for d in decisions_orm]

    async def _get_trajectory(
        self,
        trajectory_id: str,
        organization_id: str,
        session: AsyncSession,
    ) -> Optional[TrajectoryORM]:
        """Get trajectory by ID."""
        result = await session.execute(
            select(TrajectoryORM).where(
                TrajectoryORM.id == trajectory_id,
                TrajectoryORM.organization_id == organization_id,
            )
        )
        return result.scalar_one_or_none()

    async def _store_decisions(
        self,
        records: List[DecisionRecord],
        organization_id: str,
        session: AsyncSession,
    ) -> List[TrajectoryDecision]:
        """Store decision records to database."""
        created = []

        for record in records:
            decision_orm = TrajectoryDecisionORM(
                id=record.decision_id,
                trajectory_id=record.trajectory_id,
                organization_id=organization_id,
                turn_index=record.turn_index,
                decision_index=record.decision_index,
                action=record.action,
                decision_type=record.decision_type.value if record.decision_type else None,
                rationale=record.rationale,
                success=record.success,
                result_summary=record.result_summary,
                error_type=record.error_type,
                searchable_text=record.searchable_text,
                embedding=record.embedding,
            )
            session.add(decision_orm)
            created.append(self._orm_to_pydantic(decision_orm))

        await session.flush()
        return created

    async def _search_by_embedding(
        self,
        embedding: List[float],
        organization_id: str,
        action: Optional[str],
        success: Optional[bool],
        limit: int,
        min_similarity: float,
        session: AsyncSession,
    ) -> List[TrajectoryDecisionSearchResult]:
        """Search decisions by embedding similarity using pgvector."""
        # Build query with filters
        # Note: Uses pgvector cosine distance (1 - cosine_similarity)
        # Convert to similarity: 1 - distance

        filters = [TrajectoryDecisionORM.organization_id == organization_id]

        if action:
            filters.append(TrajectoryDecisionORM.action == action)

        if success is not None:
            filters.append(TrajectoryDecisionORM.success == success)

        # pgvector similarity search
        # Using <=> operator for cosine distance
        query = (
            select(
                TrajectoryDecisionORM,
                (1 - TrajectoryDecisionORM.embedding.cosine_distance(embedding)).label(
                    "similarity"
                ),
            )
            .where(*filters)
            .where(TrajectoryDecisionORM.embedding.isnot(None))
            .order_by(TrajectoryDecisionORM.embedding.cosine_distance(embedding))
            .limit(limit)
        )

        result = await session.execute(query)
        rows = result.all()

        results = []
        for decision_orm, similarity in rows:
            if similarity >= min_similarity:
                results.append(
                    TrajectoryDecisionSearchResult(
                        decision=self._orm_to_pydantic(decision_orm),
                        similarity=float(similarity),
                        trajectory_summary=None,  # Could join to get summary
                    )
                )

        return results

    def _letta_to_ots(self, trajectory_orm: TrajectoryORM) -> OTSTrajectory:
        """
        Convert Letta trajectory ORM to OTS format.

        Note: Letta's trajectory data structure differs from OTS.
        This does a best-effort conversion for decision extraction.
        """
        data = trajectory_orm.data or {}
        turns = data.get("turns", [])

        ots_turns = []
        for turn_idx, turn in enumerate(turns):
            # Extract decisions from Letta turn format
            ots_decisions = []
            messages = turn.get("messages", [])

            for msg_idx, msg in enumerate(messages):
                # Convert tool calls to decisions
                tool_calls = msg.get("tool_calls") or []
                for tc_idx, tc in enumerate(tool_calls):
                    # Determine success from tool response
                    success = self._extract_tool_success(messages, tc.get("id"))

                    ots_decisions.append(
                        OTSDecision(
                            decision_id=f"{trajectory_orm.id}-{turn_idx}-{msg_idx}-{tc_idx}",
                            decision_type=DecisionType.TOOL_SELECTION,
                            choice=OTSChoice(
                                action=tc.get("function", {}).get("name", "unknown"),
                                arguments=tc.get("function", {}).get("arguments"),
                                rationale=None,  # Could extract from reasoning
                            ),
                            consequence=OTSConsequence(
                                success=success if success is not None else True,
                                result_summary=None,
                                error_type=None,
                            ),
                        )
                    )

            from datetime import datetime, timezone

            ots_turns.append(
                OTSTurn(
                    turn_id=turn_idx,
                    timestamp=datetime.now(timezone.utc),
                    duration_ms=None,
                    messages=[],  # Not converting messages
                    decisions=ots_decisions,
                )
            )

        # Create OTS trajectory
        from datetime import datetime, timezone

        return OTSTrajectory(
            trajectory_id=trajectory_orm.id,
            metadata=OTSMetadata(
                task_description=trajectory_orm.searchable_summary or "",
                domain=trajectory_orm.domain_type,
                timestamp_start=trajectory_orm.created_at or datetime.now(timezone.utc),
                agent_id=trajectory_orm.agent_id,
                outcome=OutcomeType.SUCCESS
                if (trajectory_orm.outcome_score or 0) > 0.5
                else OutcomeType.FAILURE,
                tags=trajectory_orm.tags or [],
            ),
            context=OTSContext(),
            turns=ots_turns,
            final_reward=trajectory_orm.outcome_score,
        )

    def _extract_tool_success(
        self,
        messages: List[Dict[str, Any]],
        tool_call_id: Optional[str],
    ) -> Optional[bool]:
        """Extract tool success from messages by finding tool response."""
        if not tool_call_id:
            return None

        for msg in messages:
            if msg.get("role") == "tool" and msg.get("tool_call_id") == tool_call_id:
                content = msg.get("content", "")
                # Simple heuristic: errors typically contain "error", "failed", "exception"
                if any(err in content.lower() for err in ["error", "failed", "exception"]):
                    return False
                return True

        return None

    def _orm_to_pydantic(
        self,
        decision_orm: TrajectoryDecisionORM,
    ) -> TrajectoryDecision:
        """Convert ORM model to Pydantic schema."""
        return TrajectoryDecision(
            id=decision_orm.id,
            trajectory_id=decision_orm.trajectory_id,
            organization_id=decision_orm.organization_id,
            turn_index=decision_orm.turn_index,
            decision_index=decision_orm.decision_index,
            action=decision_orm.action,
            decision_type=decision_orm.decision_type,
            rationale=decision_orm.rationale,
            success=decision_orm.success,
            result_summary=decision_orm.result_summary,
            error_type=decision_orm.error_type,
            searchable_text=decision_orm.searchable_text,
            created_at=decision_orm.created_at,
        )
