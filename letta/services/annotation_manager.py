"""
Annotation manager for managing trajectory evaluations.

This manager handles CRUD operations for trajectory annotations,
supporting trajectory-level, turn-level, and decision-level evaluations.
"""

from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import delete, desc, func, select

from letta.log import get_logger
from letta.orm.trajectory import Trajectory as TrajectoryModel
from letta.orm.trajectory_annotation import TrajectoryAnnotation as AnnotationModel
from letta.schemas.trajectory_annotation import (
    AnnotationAggregation,
    AnnotationSearchRequest,
    EvaluatorType,
    TrajectoryAnnotation,
    TrajectoryAnnotationCreate,
    TrajectoryAnnotationUpdate,
)
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.utils import enforce_types

logger = get_logger(__name__)


class AnnotationManager:
    """Manager for trajectory annotations with CRUD operations."""

    @enforce_types
    async def create_annotation_async(
        self,
        annotation_create: TrajectoryAnnotationCreate,
        actor: PydanticUser
    ) -> TrajectoryAnnotation:
        """
        Create a new annotation for a trajectory, turn, or decision.

        Args:
            annotation_create: Annotation data
            actor: User creating the annotation

        Returns:
            Created annotation
        """
        async with db_registry.async_session() as session:
            async with session.begin():
                # Verify trajectory exists and belongs to user's organization
                result = await session.execute(
                    select(TrajectoryModel).where(
                        TrajectoryModel.id == annotation_create.trajectory_id,
                        TrajectoryModel.organization_id == actor.organization_id
                    )
                )
                trajectory = result.scalar_one_or_none()
                if not trajectory:
                    raise ValueError(f"Trajectory {annotation_create.trajectory_id} not found")

                # Create annotation ORM
                annotation_orm = AnnotationModel(
                    trajectory_id=annotation_create.trajectory_id,
                    turn_id=annotation_create.turn_id,
                    decision_id=annotation_create.decision_id,
                    evaluator_id=annotation_create.evaluator_id,
                    evaluator_type=annotation_create.evaluator_type.value,
                    evaluator_version=annotation_create.evaluator_version,
                    score=annotation_create.score,
                    label=annotation_create.label,
                    feedback=annotation_create.feedback,
                    organization_id=actor.organization_id,
                )

                session.add(annotation_orm)
                await session.flush()
                await session.refresh(annotation_orm)

                return self._orm_to_pydantic(annotation_orm)

    @enforce_types
    async def get_annotation_async(
        self,
        annotation_id: str,
        actor: PydanticUser
    ) -> Optional[TrajectoryAnnotation]:
        """Get a single annotation by ID."""
        async with db_registry.async_session() as session:
            result = await session.execute(
                select(AnnotationModel).where(
                    AnnotationModel.id == annotation_id,
                    AnnotationModel.organization_id == actor.organization_id
                )
            )
            annotation_orm = result.scalar_one_or_none()

            if not annotation_orm:
                return None

            return self._orm_to_pydantic(annotation_orm)

    @enforce_types
    async def list_annotations_async(
        self,
        actor: PydanticUser,
        trajectory_id: Optional[str] = None,
        turn_id: Optional[int] = None,
        decision_id: Optional[str] = None,
        evaluator_id: Optional[str] = None,
        evaluator_type: Optional[EvaluatorType] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        label: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TrajectoryAnnotation]:
        """
        List annotations with filtering.

        Args:
            actor: User requesting annotations
            trajectory_id: Filter by trajectory
            turn_id: Filter by turn
            decision_id: Filter by decision
            evaluator_id: Filter by evaluator
            evaluator_type: Filter by evaluator type
            min_score: Minimum score threshold
            max_score: Maximum score threshold
            label: Filter by label
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of matching annotations
        """
        async with db_registry.async_session() as session:
            query = (
                select(AnnotationModel)
                .where(AnnotationModel.organization_id == actor.organization_id)
                .order_by(desc(AnnotationModel.created_at))
            )

            # Apply filters
            if trajectory_id:
                query = query.where(AnnotationModel.trajectory_id == trajectory_id)
            if turn_id is not None:
                query = query.where(AnnotationModel.turn_id == turn_id)
            if decision_id:
                query = query.where(AnnotationModel.decision_id == decision_id)
            if evaluator_id:
                query = query.where(AnnotationModel.evaluator_id == evaluator_id)
            if evaluator_type:
                query = query.where(AnnotationModel.evaluator_type == evaluator_type.value)
            if min_score is not None:
                query = query.where(AnnotationModel.score >= min_score)
            if max_score is not None:
                query = query.where(AnnotationModel.score <= max_score)
            if label:
                query = query.where(AnnotationModel.label == label)

            # Pagination
            query = query.limit(limit).offset(offset)

            result = await session.execute(query)
            annotations = result.scalars().all()

            return [self._orm_to_pydantic(a) for a in annotations]

    @enforce_types
    async def search_annotations_async(
        self,
        search_request: AnnotationSearchRequest,
        actor: PydanticUser
    ) -> List[TrajectoryAnnotation]:
        """Search annotations using the request object."""
        return await self.list_annotations_async(
            actor=actor,
            trajectory_id=search_request.trajectory_id,
            turn_id=search_request.turn_id,
            decision_id=search_request.decision_id,
            evaluator_id=search_request.evaluator_id,
            evaluator_type=search_request.evaluator_type,
            min_score=search_request.min_score,
            max_score=search_request.max_score,
            label=search_request.label,
            limit=search_request.limit,
            offset=search_request.offset,
        )

    @enforce_types
    async def update_annotation_async(
        self,
        annotation_id: str,
        annotation_update: TrajectoryAnnotationUpdate,
        actor: PydanticUser
    ) -> Optional[TrajectoryAnnotation]:
        """
        Update an existing annotation.

        Args:
            annotation_id: ID of annotation to update
            annotation_update: Update data
            actor: User performing the update

        Returns:
            Updated annotation or None if not found
        """
        async with db_registry.async_session() as session:
            async with session.begin():
                result = await session.execute(
                    select(AnnotationModel).where(
                        AnnotationModel.id == annotation_id,
                        AnnotationModel.organization_id == actor.organization_id
                    )
                )
                annotation_orm = result.scalar_one_or_none()

                if not annotation_orm:
                    return None

                # Update fields
                if annotation_update.score is not None:
                    annotation_orm.score = annotation_update.score
                if annotation_update.label is not None:
                    annotation_orm.label = annotation_update.label
                if annotation_update.feedback is not None:
                    annotation_orm.feedback = annotation_update.feedback

                await session.flush()
                await session.refresh(annotation_orm)

                return self._orm_to_pydantic(annotation_orm)

    @enforce_types
    async def delete_annotation_async(
        self,
        annotation_id: str,
        actor: PydanticUser
    ) -> bool:
        """Delete an annotation."""
        async with db_registry.async_session() as session:
            async with session.begin():
                result = await session.execute(
                    delete(AnnotationModel).where(
                        AnnotationModel.id == annotation_id,
                        AnnotationModel.organization_id == actor.organization_id
                    )
                )
                return result.rowcount > 0

    @enforce_types
    async def get_trajectory_annotations_async(
        self,
        trajectory_id: str,
        actor: PydanticUser,
        include_turn_level: bool = True,
        include_decision_level: bool = True,
    ) -> List[TrajectoryAnnotation]:
        """
        Get all annotations for a trajectory at all granularity levels.

        Args:
            trajectory_id: ID of the trajectory
            actor: User requesting annotations
            include_turn_level: Include turn-level annotations
            include_decision_level: Include decision-level annotations

        Returns:
            List of annotations at all requested levels
        """
        async with db_registry.async_session() as session:
            query = (
                select(AnnotationModel)
                .where(
                    AnnotationModel.trajectory_id == trajectory_id,
                    AnnotationModel.organization_id == actor.organization_id
                )
            )

            if not include_turn_level and not include_decision_level:
                # Only trajectory-level (both turn_id and decision_id are null)
                query = query.where(AnnotationModel.turn_id.is_(None))

            if not include_decision_level:
                # Exclude decision-level
                query = query.where(AnnotationModel.decision_id.is_(None))

            query = query.order_by(
                AnnotationModel.turn_id.nulls_first(),
                AnnotationModel.decision_id.nulls_first(),
                desc(AnnotationModel.created_at)
            )

            result = await session.execute(query)
            annotations = result.scalars().all()

            return [self._orm_to_pydantic(a) for a in annotations]

    @enforce_types
    async def get_aggregated_scores_async(
        self,
        trajectory_id: str,
        actor: PydanticUser,
        evaluator_id: Optional[str] = None
    ) -> AnnotationAggregation:
        """
        Get aggregated annotation statistics for a trajectory.

        Args:
            trajectory_id: ID of the trajectory
            actor: User requesting aggregation
            evaluator_id: Optional filter by evaluator

        Returns:
            Aggregated statistics
        """
        async with db_registry.async_session() as session:
            query = select(
                func.count(AnnotationModel.id).label("annotation_count"),
                func.avg(AnnotationModel.score).label("avg_score"),
                func.min(AnnotationModel.score).label("min_score"),
                func.max(AnnotationModel.score).label("max_score"),
                func.count(func.distinct(AnnotationModel.evaluator_id)).label("evaluator_count"),
            ).where(
                AnnotationModel.trajectory_id == trajectory_id,
                AnnotationModel.organization_id == actor.organization_id
            )

            if evaluator_id:
                query = query.where(AnnotationModel.evaluator_id == evaluator_id)

            result = await session.execute(query)
            row = result.one()

            # Get label counts
            label_query = select(
                AnnotationModel.label,
                func.count(AnnotationModel.id).label("count")
            ).where(
                AnnotationModel.trajectory_id == trajectory_id,
                AnnotationModel.organization_id == actor.organization_id,
                AnnotationModel.label.isnot(None)
            ).group_by(AnnotationModel.label)

            if evaluator_id:
                label_query = label_query.where(AnnotationModel.evaluator_id == evaluator_id)

            label_result = await session.execute(label_query)
            labels = {label: count for label, count in label_result.all()}

            return AnnotationAggregation(
                trajectory_id=trajectory_id,
                annotation_count=row.annotation_count or 0,
                avg_score=float(row.avg_score) if row.avg_score else None,
                min_score=float(row.min_score) if row.min_score else None,
                max_score=float(row.max_score) if row.max_score else None,
                evaluator_count=row.evaluator_count or 0,
                labels=labels,
            )

    @enforce_types
    async def batch_create_annotations_async(
        self,
        annotations: List[TrajectoryAnnotationCreate],
        actor: PydanticUser
    ) -> List[TrajectoryAnnotation]:
        """
        Create multiple annotations in a single transaction.

        Args:
            annotations: List of annotation data
            actor: User creating the annotations

        Returns:
            List of created annotations
        """
        async with db_registry.async_session() as session:
            async with session.begin():
                # Verify all trajectories exist
                trajectory_ids = {a.trajectory_id for a in annotations}
                result = await session.execute(
                    select(TrajectoryModel.id).where(
                        TrajectoryModel.id.in_(trajectory_ids),
                        TrajectoryModel.organization_id == actor.organization_id
                    )
                )
                found_ids = {row[0] for row in result.all()}
                missing = trajectory_ids - found_ids
                if missing:
                    raise ValueError(f"Trajectories not found: {missing}")

                # Create all annotations
                created = []
                for annotation_create in annotations:
                    annotation_orm = AnnotationModel(
                        trajectory_id=annotation_create.trajectory_id,
                        turn_id=annotation_create.turn_id,
                        decision_id=annotation_create.decision_id,
                        evaluator_id=annotation_create.evaluator_id,
                        evaluator_type=annotation_create.evaluator_type.value,
                        evaluator_version=annotation_create.evaluator_version,
                        score=annotation_create.score,
                        label=annotation_create.label,
                        feedback=annotation_create.feedback,
                        organization_id=actor.organization_id,
                    )
                    session.add(annotation_orm)
                    created.append(annotation_orm)

                await session.flush()

                # Refresh all
                for annotation_orm in created:
                    await session.refresh(annotation_orm)

                return [self._orm_to_pydantic(a) for a in created]

    @enforce_types
    async def get_annotations_by_evaluator_async(
        self,
        evaluator_id: str,
        actor: PydanticUser,
        min_score: Optional[float] = None,
        limit: int = 100,
    ) -> List[TrajectoryAnnotation]:
        """
        Get all annotations by a specific evaluator.

        Useful for analyzing evaluator performance or finding evaluator-specific patterns.

        Args:
            evaluator_id: ID of the evaluator
            actor: User requesting annotations
            min_score: Optional minimum score filter
            limit: Maximum results

        Returns:
            List of annotations by the evaluator
        """
        async with db_registry.async_session() as session:
            query = (
                select(AnnotationModel)
                .where(
                    AnnotationModel.evaluator_id == evaluator_id,
                    AnnotationModel.organization_id == actor.organization_id
                )
                .order_by(desc(AnnotationModel.created_at))
            )

            if min_score is not None:
                query = query.where(AnnotationModel.score >= min_score)

            query = query.limit(limit)

            result = await session.execute(query)
            annotations = result.scalars().all()

            return [self._orm_to_pydantic(a) for a in annotations]

    def _orm_to_pydantic(self, annotation_orm: AnnotationModel) -> TrajectoryAnnotation:
        """Convert ORM model to Pydantic schema."""
        return TrajectoryAnnotation(
            id=annotation_orm.id,
            trajectory_id=annotation_orm.trajectory_id,
            turn_id=annotation_orm.turn_id,
            decision_id=annotation_orm.decision_id,
            evaluator_id=annotation_orm.evaluator_id,
            evaluator_type=EvaluatorType(annotation_orm.evaluator_type),
            evaluator_version=annotation_orm.evaluator_version,
            score=annotation_orm.score,
            label=annotation_orm.label,
            feedback=annotation_orm.feedback,
            created_at=annotation_orm.created_at,
            organization_id=annotation_orm.organization_id,
        )
