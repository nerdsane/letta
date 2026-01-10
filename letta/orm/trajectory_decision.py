"""
TrajectoryDecision ORM model for decision-level embedding persistence.

Stores individual decisions from trajectories with embeddings for
fine-grained semantic search across decisions.

While trajectory-level search finds similar "agent runs", decision-level
search finds similar "choice points" - enabling more precise context learning.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.constants import MAX_EMBEDDING_DIM
from letta.orm.custom_columns import CommonVector
from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.settings import DatabaseChoice, settings

if TYPE_CHECKING:
    from letta.orm.organization import Organization
    from letta.orm.trajectory import Trajectory


class TrajectoryDecision(SqlalchemyBase, OrganizationMixin):
    """
    Individual decisions extracted from trajectories for fine-grained search.

    Each decision captures a choice point: the state, the action taken,
    the reasoning, and the outcome. Embeddings enable semantic search
    across decisions to find relevant past examples.

    Use Cases:
    - "How did I handle similar tool errors before?"
    - "What parameters worked for world_manager.save?"
    - "Find decisions where the agent recovered from failure"
    """

    __tablename__ = "trajectories_decisions"
    __table_args__ = (
        # Primary lookup indexes
        Index("ix_td_trajectory_id", "trajectory_id"),
        Index("ix_td_organization_id", "organization_id"),
        Index("ix_td_action", "action"),
        Index("ix_td_success", "success"),
        # Compound indexes for common queries
        Index("ix_td_org_action", "organization_id", "action"),
        Index("ix_td_org_success", "organization_id", "success"),
        Index("ix_td_traj_turn_decision", "trajectory_id", "turn_index", "decision_index"),
    )

    # Primary key
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: f"decision-{uuid.uuid4()}"
    )

    # Foreign key to parent trajectory
    trajectory_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("trajectories.id", ondelete="CASCADE"),
        nullable=False,
        doc="Parent trajectory containing this decision"
    )

    # Position within trajectory
    turn_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Zero-indexed turn number within trajectory"
    )
    decision_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Zero-indexed decision number within turn"
    )

    # Decision content
    action: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        doc="Tool or action name (e.g., 'world_manager', 'story_manager')"
    )
    decision_type: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        doc="Type: tool_selection, parameter_choice, reasoning_step, response_formulation"
    )
    rationale: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Agent's reasoning for this decision"
    )

    # Outcome
    success: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
        doc="Whether the decision succeeded"
    )
    result_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Summary of the decision outcome"
    )
    error_type: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        doc="Error type if decision failed"
    )

    # Searchable text (for embedding generation)
    searchable_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Combined text representation for embedding"
    )

    # Vector embedding for similarity search
    if settings.database_engine is DatabaseChoice.POSTGRES:
        from pgvector.sqlalchemy import Vector

        embedding: Mapped[Optional[list]] = mapped_column(
            Vector(MAX_EMBEDDING_DIM),
            nullable=True,
            doc="Vector embedding for decision-level similarity search"
        )
    else:
        embedding: Mapped[Optional[list]] = mapped_column(
            CommonVector,
            nullable=True,
            doc="Vector embedding for decision-level similarity search"
        )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        doc="When this decision record was created"
    )

    # Relationships
    trajectory: Mapped["Trajectory"] = relationship(
        "Trajectory",
        back_populates="decisions"
    )
    organization: Mapped[Optional["Organization"]] = relationship(
        "Organization",
        back_populates="trajectory_decisions"
    )
