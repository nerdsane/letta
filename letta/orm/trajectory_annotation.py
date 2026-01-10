"""
Trajectory Annotation ORM model for linked evaluations.

Annotations are separate from trajectories for:
- Multiple evaluators can annotate the same trajectory
- Annotations can be added retroactively without modifying trajectories
- Different retention policies for trajectories vs. annotations
- Cleaner schema evolution—annotation format can change independently
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase

if TYPE_CHECKING:
    from letta.orm.organization import Organization
    from letta.orm.trajectory import Trajectory


class TrajectoryAnnotation(SqlalchemyBase, OrganizationMixin):
    """
    Linked annotation for trajectory, turn, or decision evaluation.

    Annotations follow the Open Trajectory Specification (OTS) pattern
    where evaluations are linked entities rather than embedded in trajectories.

    Granularity:
    - trajectory_id only: Trajectory-level annotation
    - trajectory_id + turn_id: Turn-level annotation
    - trajectory_id + turn_id + decision_id: Decision-level annotation
    """

    __tablename__ = "trajectory_annotations"
    __table_args__ = (
        Index("ix_annotations_trajectory_id", "trajectory_id"),
        Index("ix_annotations_evaluator_id", "evaluator_id"),
        Index("ix_annotations_score", "score"),
        Index("ix_annotations_evaluator_score", "evaluator_id", "score"),
        Index("ix_annotations_created_at", "created_at"),
        Index("ix_annotations_organization_id", "organization_id"),
    )

    # Primary key with annotation- prefix
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: f"annotation-{uuid.uuid4()}"
    )

    # Foreign key to trajectory
    trajectory_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("trajectories.id", ondelete="CASCADE"),
        nullable=False,
        doc="The trajectory being annotated"
    )

    # Granularity specifiers (null = parent level)
    turn_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Turn index within trajectory (null = trajectory-level annotation)"
    )
    decision_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Decision ID within turn (null = turn-level annotation)"
    )

    # Evaluator information
    evaluator_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        doc="Identifier for the evaluator (e.g., 'human:user123', 'model:gpt-4', 'heuristic:consistency_check')"
    )
    evaluator_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Type of evaluator: 'human', 'model', 'heuristic'"
    )
    evaluator_version: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        doc="Version of the evaluator (e.g., model version, heuristic version)"
    )

    # Evaluation content
    score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Quality score 0-1 (higher is better)"
    )
    label: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        doc="Categorical label (e.g., 'correct', 'incorrect', 'ambiguous')"
    )
    feedback: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Detailed feedback or explanation"
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow
    )

    # Relationships
    trajectory: Mapped["Trajectory"] = relationship(
        "Trajectory",
        back_populates="annotations"
    )
    organization: Mapped[Optional["Organization"]] = relationship(
        "Organization",
        back_populates="trajectory_annotations"
    )
